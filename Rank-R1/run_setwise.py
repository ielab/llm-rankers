from pyserini.search.lucene import LuceneSearcher
from pyserini.search._base import get_topics
from llmrankers.setwise import SetwiseLlmRanker
from llmrankers.rankers import SearchResult
from tqdm import tqdm
import argparse
import sys
import time
import random
import copy
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM
import toml
import os
import json
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import re
from collections import Counter
from huggingface_hub import snapshot_download
import os

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


class R1SetwiseLlmRanker(SetwiseLlmRanker):
    CHARACTERS = [f'[{i+1}]' for i in range(20)]

    def __init__(self,
                 model_name_or_path,
                 lora_path,
                 prompt,
                 tokenizer_name_or_path=None,
                 num_child=3,
                 k=10,
                 scoring='generation',
                 method="heapsort",
                 num_permutation=1,
                 cache_dir=None):

        self.prompt = prompt
        self.lora_path = lora_path
        self.num_child = num_child
        self.num_permutation = num_permutation
        self.k = k
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

        self.scoring = scoring
        self.method = method
        self.total_compare = 0
        self.total_completion_tokens = 0
        self.total_prompt_tokens = 0

    def compare(self, query: str, docs: List):
        self.total_compare += 1 if self.num_permutation == 1 else self.num_permutation

        id_passage = [(i, p) for i, p in enumerate(docs)]
        labels = [self.CHARACTERS[i] for i in range(len(docs))]
        batch_data = []
        for _ in range(self.num_permutation):
            batch_data.append([random.sample(id_passage, len(id_passage)),
                               labels])

        batch_ref = []
        input_text = []
        for batch in batch_data:
            ref = []
            passages = []
            characters = []
            for p, c in zip(batch[0], batch[1]):
                ref.append(p[0])
                passages.append(p[1].text)
                characters.append(c)
            batch_ref.append((ref, characters))
            passages = "\n".join([f'{characters[i]} {passages[i]}' for i in range(len(passages))])
            system_message = self.prompt["prompt_system"]
            user_message = self.prompt['prompt_user'].format(query=query,
                                                             docs=passages)
            input_text.append([
                {'role': "system", 'content': system_message},
                {'role': "user", 'content': user_message}
            ])
        outputs = self.llm.chat(input_text,
                                sampling_params=self.sampling_params,
                                use_tqdm=False,
                                lora_request=LoRARequest("R1adapter",
                                                         1,
                                                         self.lora_path)
                                if self.lora_path is not None else None,
                                )
        results = []
        for output, input in zip(outputs, input_text):
            self.total_completion_tokens += len(output.outputs[0].token_ids)
            self.total_prompt_tokens += len(output.prompt_token_ids)

            completion = output.outputs[0].text
            pattern = rf'{self.prompt["pattern"]}'
            match = re.search(pattern, completion.lower(), re.DOTALL)
            if match:
                results.append(match.group(1).strip())
            else:
                results.append(f'input_text:\n{input}, completion:\n{completion}')

        # vote
        candidates = []
        for ref, result in zip(batch_ref, results):
            result = result.strip()
            docids, characters = ref
            if result not in characters:
                print(f"Unexpected output: {result}")
                continue
            win_doc = docids[characters.index(result)]
            candidates.append(win_doc)

        if len(candidates) == 0:
            print(f"Unexpected voting: {results}")
            output = "Unexpected voting."
        else:
            # handle tie
            candidate_counts = Counter(candidates)
            max_count = max(candidate_counts.values())
            most_common_candidates = [candidate for candidate, count in candidate_counts.items() if
                                      count == max_count]
            if len(most_common_candidates) == 1:
                output = self.CHARACTERS[most_common_candidates[0]]
            else:
                output = self.CHARACTERS[random.choice(most_common_candidates)]

        if output in self.CHARACTERS:
            pass
        else:
            print(f"Unexpected output: {output}")

        return output


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

    ranker = R1SetwiseLlmRanker(model_name_or_path=args.run.model_name_or_path,
                                tokenizer_name_or_path=args.run.tokenizer_name_or_path,
                                lora_path=lora_path,
                                num_child=args.setwise.num_child,
                                method=args.setwise.method,
                                k=args.setwise.k,
                                prompt=prompt)

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

    total_ranked = 0
    total_comparisons = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
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

        total_comparisons += ranker.total_compare
        total_prompt_tokens += ranker.total_prompt_tokens
        total_completion_tokens += ranker.total_completion_tokens
        total_ranked += 1

        reranked = ranker.rerank(query, ranking)
        write_run_file(args.run.save_path, [(qid, query, reranked)], 'LLMRankers')


        # if args.run.log_file is not None:
        #     write_log_file(args.run.log_file, qid, query, completions)

    toc = time.time()

    print(f'Avg comparisons: {total_comparisons/total_ranked}')
    print(f'Avg prompt tokens: {total_prompt_tokens/total_ranked}')
    print(f'Avg completion tokens: {total_completion_tokens/total_ranked}')
    print(f'Avg time per query: {(toc-tic)/total_ranked}')


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
                            help='Path to the lora_path model')
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
    run_parser.add_argument('--openai_key', type=str, default=None)
    run_parser.add_argument('--scoring', type=str, default='generation', choices=['generation', 'likelihood'])
    run_parser.add_argument('--shuffle_ranking', type=str, default=None, choices=['inverse', 'random'])
    setwise_parser = commands.add_parser('setwise')
    setwise_parser.add_argument('--num_child', type=int, default=3)
    setwise_parser.add_argument('--method', type=str, default='heapsort',
                                choices=['heapsort', 'bubblesort'])
    setwise_parser.add_argument('--k', type=int, default=10)
    setwise_parser.add_argument('--num_permutation', type=int, default=1)

    args = parse_args(parser, commands)
    main(args)


