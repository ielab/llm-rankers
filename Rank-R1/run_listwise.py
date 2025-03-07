from pyserini.search.lucene import LuceneSearcher
from pyserini.search._base import get_topics
from llmrankers.listwise import OpenAiListwiseLlmRanker, ListwiseLlmRanker, receive_permutation, create_permutation_instruction_chat
from llmrankers.rankers import SearchResult
from tqdm import tqdm
import argparse
import sys
import json
import time
import random
import copy
from typing import List, Tuple
from openai import OpenAI
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
import re
import toml
import os
import json

random.seed(929)

def write_log_file(log_file, qid, query, completions):
    with open(log_file, 'a+') as f:
        completions = [c.to_dict() for c in completions]
        f.write(json.dumps({'qid': qid, 'query': query, 'completions': completions}) + '\n')

def load_run_file(path, query_map, ranker, docstore, hits, passage_length):
    first_stage_rankings = []
    with open(path, 'r') as f:
        current_qid = None
        current_ranking = []
        for line in tqdm(f):
            qid, _, docid, _, score, _ = line.strip().split()
            if qid not in query_map:
                continue
            if qid != current_qid:
                if current_qid is not None:
                    first_stage_rankings.append((current_qid, query_map[current_qid], current_ranking[:hits]))
                current_ranking = []
                current_qid = qid
            if len(current_ranking) >= hits:
                continue
            data = json.loads(docstore.doc(docid).raw())
            text = data['contents']
            if 'title' in data:
                text = f'{data["title"]} {text}'
            text = ranker.truncate(text, passage_length)
            current_ranking.append(SearchResult(docid=docid, score=float(score), text=text))
        if current_qid is not None:
            first_stage_rankings.append((current_qid, query_map[current_qid], current_ranking[:hits]))
    return first_stage_rankings


def parse_args(parser, commands):
    # Divide argv by commands
    split_argv = [[]]
    for c in sys.argv[1:]:
        if c in commands.choices:
            split_argv.append([c])
        else:
            split_argv[-1].append(c)
    # Initialize namespace
    args = argparse.Namespace()
    for c in commands.choices:
        setattr(args, c, None)
    # Parse each command
    parser.parse_args(split_argv[0], namespace=args)  # Without command
    for argv in split_argv[1:]:  # Commands
        n = argparse.Namespace()
        setattr(args, argv[0], n)
        parser.parse_args(argv, namespace=n)
    return args


def write_run_file(path, results, tag):
    with open(path, 'a+') as f:
        for qid, _, ranking in results:
            rank = 1
            for doc in ranking:
                docid = doc.docid
                score = doc.score
                f.write(f"{qid}\tQ0\t{docid}\t{rank}\t{score}\t{tag}\n")
                rank += 1


class R1ListwiseLlmRanker(ListwiseLlmRanker):
    CHARACTERS = [f'[{i + 1}]' for i in range(20)]

    def __init__(self, model_name_or_path,
                 tokenizer_name_or_path,
                 prompt,
                 window_size,
                 step_size,
                 lora_path=None,
                 scoring='generation',
                 num_repeat=1, cache_dir=None):
        self.prompt = prompt
        self.lora_path = lora_path
        self.sampling_params = SamplingParams(temperature=0.0,
                                              max_tokens=2048)
        if tokenizer_name_or_path is None:
            tokenizer_name_or_path = model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, cache_dir=cache_dir)
        self.llm = LLM(model=model_name_or_path,
                       tokenizer=tokenizer_name_or_path,
                       enable_lora=True if lora_path is not None else False,
                       max_lora_rank=32,
                       )

        self.window_size = window_size
        self.step_size = step_size
        self.num_repeat = num_repeat

        self.total_compare = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def compare(self, query: str, docs: List):
        self.total_compare += 1

        passages = [doc.text for doc in docs]

        passages = "\n".join([f'{self.CHARACTERS[i]} {passages[i]}' for i in range(len(passages))])
        system_message = self.prompt["prompt_system"]
        user_message = self.prompt['prompt_user'].format(query=query,
                                                         num=len(docs),
                                                         docs=passages)
        input_text = [
            {'role': "system", 'content': system_message},
            {'role': "user", 'content': user_message}
        ]

        outputs = self.llm.chat(input_text,
                                sampling_params=self.sampling_params,
                                use_tqdm=False,
                                lora_request=LoRARequest("R1adapter",
                                                         1,
                                                         self.lora_path)
                                if self.lora_path is not None else None,
                                )
        self.total_completion_tokens += len(outputs[0].outputs[0].token_ids)
        self.total_prompt_tokens += len(outputs[0].prompt_token_ids)
        completion = outputs[0].outputs[0].text

        pattern = rf'{self.prompt["pattern"]}'
        match = re.search(pattern, completion.lower(), re.DOTALL)
        if match:
            result = match.group(1).strip()
        else:
            result = 'None'
            print('Input for no match:', input_text)
            print('Completion for no match:', completion)
        return result


def main(args):
    prompt = toml.load(args.run.prompt_file)
    if args.run.lora_path_or_name is not None:
        # check if the path exists
        if not os.path.exists(args.run.lora_path_or_name):
            # download the model
            lora_path = snapshot_download(args.run.lora_path_or_name)
        else:
            lora_path = args.run.lora_path_or_name
    else:
        lora_path = None

    ranker = R1ListwiseLlmRanker(model_name_or_path=args.run.model_name_or_path,
                                 tokenizer_name_or_path=args.run.tokenizer_name_or_path,
                                 window_size=args.listwise.window_size,
                                 lora_path=lora_path,
                                 step_size=args.listwise.step_size,
                                 num_repeat=args.listwise.num_repeat,
                                 prompt=prompt,
                                 cache_dir=args.run.cache_dir)

    query_map = {}
    if args.run.query_file is not None:
        with open(args.run.query_file, 'r') as f:
            for line in f:
                qid, query = line.strip().split('\t')
                query_map[qid] = ranker.truncate(query, args.run.query_length)
    elif args.run.pyserini_dataset is not None:
        topics = get_topics(args.run.pyserini_dataset)
        for topic_id in list(topics.keys()):
            text = topics[topic_id]['title']
            query_map[str(topic_id)] = ranker.truncate(text, args.run.query_length)
    else:
        raise ValueError('Either query_file or pyserini_dataset must be provided.')

    # if the index is a path, load it
    if os.path.exists(args.run.pyserini_index):
        print(f'Loading index from {args.run.pyserini_index}')
        docstore = LuceneSearcher(args.run.pyserini_index)
    else:
        docstore = LuceneSearcher.from_prebuilt_index(args.run.pyserini_index)


    first_stage_rankings = load_run_file(args.run.run_path, query_map, ranker, docstore,
                                         args.run.hits, args.run.passage_length)

    # if save_path file exists, load it
    ranked_qids = set()
    if os.path.exists(args.run.save_path):
        print(f'{args.run.save_path} exists. Continue ranking')
        reranked_rankings = load_run_file(args.run.save_path, query_map, ranker, docstore,
                                          args.run.hits, args.run.passage_length)
        for qid, _, _ in reranked_rankings:
            ranked_qids.add(qid)

    tic = time.time()
    for qid, query, ranking in tqdm(first_stage_rankings):
        if qid in ranked_qids:
            continue
        if args.run.shuffle_ranking is not None:
            if args.run.shuffle_ranking == 'random':
                random.shuffle(ranking)
            elif args.run.shuffle_ranking == 'inverse':
                ranking = ranking[::-1]
            else:
                raise ValueError(f'Invalid shuffle ranking method: {args.run.shuffle_ranking}.')

        reranked = ranker.rerank(query, ranking)
        write_run_file(args.run.save_path, [(qid, query, reranked)], 'LLMRankers')

        # if args.run.log_file is not None:
        #     write_log_file(args.run.log_file, qid, query, completions)

    toc = time.time()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(title='sub-commands')

    run_parser = commands.add_parser('run')
    run_parser.add_argument('--prompt_file', type=str, help='')
    run_parser.add_argument('--run_path', type=str, help='Path to the first stage run file (TREC format) to rerank.')
    run_parser.add_argument('--save_path', type=str, help='Path to save the reranked run file (TREC format).')
    run_parser.add_argument('--model_name_or_path', type=str,
                            help='Path to the pretrained model or model identifier from huggingface.co/models')
    run_parser.add_argument('--lora_path_or_name', type=str, default=None,
                            help='Path to the lora_path_or_name model')
    run_parser.add_argument('--tokenizer_name_or_path', type=str, default=None,
                            help='Path to the pretrained tokenizer or tokenizer identifier from huggingface.co/tokenizers')
    run_parser.add_argument('--pyserini_index', type=str, default=None)
    run_parser.add_argument('--pyserini_dataset', type=str, default=None)
    run_parser.add_argument('--query_file', type=str, default=None)
    run_parser.add_argument('--hits', type=int, default=100)
    run_parser.add_argument('--query_length', type=int, default=128)
    run_parser.add_argument('--passage_length', type=int, default=128)
    run_parser.add_argument('--device', type=str, default='cuda')
    run_parser.add_argument('--cache_dir', type=str, default=None)
    run_parser.add_argument('--log_file', type=str, default=None)
    run_parser.add_argument('--scoring', type=str, default='generation', choices=['generation', 'likelihood'])
    run_parser.add_argument('--shuffle_ranking', type=str, default=None, choices=['inverse', 'random'])
    listwise_parser = commands.add_parser('listwise')
    listwise_parser.add_argument('--window_size', type=int, default=3)
    listwise_parser.add_argument('--step_size', type=int, default=1)
    listwise_parser.add_argument('--num_repeat', type=int, default=1)

    args = parse_args(parser, commands)
    main(args)


