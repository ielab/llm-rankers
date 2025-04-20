import logging

from pyserini.search.lucene import LuceneSearcher
from pyserini.search._base import get_topics
from llmrankers import SearchResult
from llmrankers import SetwiseLlmRanker, SetwiseT5Ranker, ListwiseLlmRanker
from llmrankers.arguments import (
ExperimentArguments, RankerArguments, SetwiseArguments,
PairwiseArguments, PointwiseArguments, ListwiseArguments)
from transformers import HfArgumentParser, set_seed
import os
from tqdm import tqdm
import json
import time
import random
logger = logging.getLogger(__name__)


def load_run_file(path, query_map, docstore, hits):
    first_stage_rankings = []
    with open(path, 'r') as f:
        current_qid = None
        current_ranking = []
        for line in tqdm(f, desc='Loading run file'):
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
            current_ranking.append(SearchResult(docid=docid, score=float(score), text=text))
        if current_qid is not None:
            first_stage_rankings.append((current_qid, query_map[current_qid], current_ranking[:hits]))
    return first_stage_rankings


def write_run_file(path, results, tag):
    with open(path, 'a+') as f:
        for qid, _, ranking in results:
            rank = 1
            for doc in ranking:
                docid = doc.docid
                score = doc.score
                f.write(f"{qid}\tQ0\t{docid}\t{rank}\t{score}\t{tag}\n")
                rank += 1


def main():
    parser = HfArgumentParser((ExperimentArguments, RankerArguments))
    exp_args, ranker_args, method_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    if exp_args.method == 'setwise':
        method_args_cls = SetwiseArguments
    elif exp_args.method == 'pairwise':
        method_args_cls = PairwiseArguments
    elif exp_args.method == 'pointwise':
        method_args_cls = PointwiseArguments
    elif exp_args.method == 'listwise':
        method_args_cls = ListwiseArguments
    else:
        raise ValueError(f'Invalid method: {exp_args.method}.')

    parser = HfArgumentParser(method_args_cls)
    method_args, = parser.parse_args_into_dataclasses(method_args)

    if exp_args.method == 'setwise':
        if 't5' in ranker_args.model_name_or_path:
            ranker = SetwiseT5Ranker(ranker_args, method_args)
        else:
            ranker = SetwiseLlmRanker(ranker_args, method_args)
    elif exp_args.method == 'pairwise':
        raise NotImplementedError('Pairwise ranking is not implemented yet.')
    elif exp_args.method == 'pointwise':
        raise NotImplementedError('Pointwise ranking is not implemented yet.')
    elif exp_args.method == 'listwise':
        ranker = ListwiseLlmRanker(ranker_args, method_args)

    query_map = {}
    topics = get_topics(exp_args.pyserini_topic)
    for topic_id in list(topics.keys()):
        text = topics[topic_id]['title']
        query_map[str(topic_id)] = text
    docstore = LuceneSearcher.from_prebuilt_index(exp_args.pyserini_index)

    logger.info(f'Loading first stage run from {exp_args.run_path}.')
    first_stage_rankings = []
    with open(exp_args.run_path, 'r') as f:
        current_qid = None
        current_ranking = []
        for line in tqdm(f):
            qid, _, docid, _, score, _ = line.strip().split()
            if qid != current_qid:
                if current_qid is not None:
                    first_stage_rankings.append((current_qid, query_map[current_qid], current_ranking[:exp_args.hits]))
                current_ranking = []
                current_qid = qid
            if len(current_ranking) >= exp_args.hits:
                continue
            data = json.loads(docstore.doc(docid).raw())
            text = data['contents']
            if 'title' in data:
                text = f'{data["title"]} {text}'
            current_ranking.append(SearchResult(docid=docid, score=float(score), text=text))
        first_stage_rankings.append((current_qid, query_map[current_qid], current_ranking[:exp_args.hits]))

    # if save_path file exists, load it
    ranked_qids = set()
    if os.path.exists(exp_args.save_path):
        print(f'{exp_args.save_path} exists. Continue ranking')
        reranked_rankings = load_run_file(exp_args.save_path, query_map, docstore, exp_args.hits)
        for qid, _, _ in reranked_rankings:
            ranked_qids.add(qid)

    set_seed(ranker_args.seed)
    total_ranked = 0
    total_comparisons = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0

    tic = time.time()
    for qid, query, ranking in tqdm(first_stage_rankings, desc='Ranking'):
        if qid in ranked_qids:
            continue
        if exp_args.shuffle_ranking is not None:
            if exp_args.run.shuffle_ranking == 'random':
                random.shuffle(ranking)
            elif exp_args.shuffle_ranking == 'inverse':
                ranking = ranking[::-1]
            else:
                raise ValueError(f'Invalid shuffle ranking method: {exp_args.shuffle_ranking}.')
        reranked = ranker.rerank(query, ranking)
        total_comparisons += ranker.total_compare
        total_prompt_tokens += ranker.total_prompt_tokens
        total_completion_tokens += ranker.total_completion_tokens
        total_ranked += 1
        write_run_file(exp_args.save_path, [(qid, query, reranked)], 'LLMRankers')
    toc = time.time()

    if total_ranked > 0:
        print(f'Avg comparisons: {total_comparisons/total_ranked}')
        print(f'Avg prompt tokens: {total_prompt_tokens/total_ranked}')
        print(f'Avg completion tokens: {total_completion_tokens/total_ranked}')
        print(f'Avg time per query: {(toc-tic)/total_ranked}')



if __name__ == '__main__':
    main()
