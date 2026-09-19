"""Re-rank a first-stage TREC run with Jev (pointwise / pairwise / setwise / listwise).

    python jev/run_jev.py \
      run --provider vercel \
          --run_path jev/runs/bm25/run.rank_llm.bm25.dl19.top100.txt \
          --docs_file jev/runs/bm25/docs.dl19.top100.tsv \
          --ir_dataset_name msmarco-passage/trec-dl-2019/judged \
          --save_path jev/runs/jev/dl19.setwise.heapsort.c10.txt \
      setwise --num_child 10 --k 10

Same two-level CLI as ../run.py. Queries come from ir_datasets, passage texts from the docid<TAB>text file written by
prepare_data.sh. Independent queries run concurrently (--query_workers); --max_rps caps the total request rate.
"""
import argparse
import json
import os
import queue
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

import ir_datasets  # noqa: E402
from llmrankers.rankers import SearchResult  # noqa: E402
from jev_rankers import (JevClient, JevListwiseLlmRanker, JevPairwiseLlmRanker,  # noqa: E402
                         JevPointwiseLlmRanker, JevSetwiseLlmRanker)


def parse_args(parser, commands):
    split_argv = [[]]
    for c in sys.argv[1:]:
        split_argv.append([c]) if c in commands.choices else split_argv[-1].append(c)
    args = argparse.Namespace(**{c: None for c in commands.choices})
    parser.parse_args(split_argv[0], namespace=args)
    for argv in split_argv[1:]:
        setattr(args, argv[0], parser.parse_args(argv, namespace=argparse.Namespace()))
    return args


def build_ranker(args, client):
    w = args.run.num_workers
    if args.pointwise:
        return JevPointwiseLlmRanker(client, method=args.pointwise.method, num_workers=w), 'pointwise'
    if args.pairwise:
        return JevPairwiseLlmRanker(client, k=args.pairwise.k, num_workers=w), 'pairwise'
    if args.setwise:
        return JevSetwiseLlmRanker(client, num_child=args.setwise.num_child, k=args.setwise.k, num_workers=w), 'setwise'
    if args.listwise:
        return JevListwiseLlmRanker(client, window_size=args.listwise.window_size, step_size=args.listwise.step_size,
                                    num_repeat=args.listwise.num_repeat, mode=args.listwise.mode, num_workers=w), 'listwise'
    raise ValueError('specify one of: pointwise, pairwise, setwise, listwise')


def load_first_stage(run_path, query_map, texts, hits, truncate, passage_length, max_queries=None):
    per_query, order = {}, []
    with open(run_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            qid, docid, s = parts[0], parts[2], float(parts[4])
            if qid not in per_query:
                per_query[qid] = []
                order.append(qid)
            if len(per_query[qid]) < hits:
                per_query[qid].append((docid, s))
    order = [q for q in order if q in query_map][:max_queries]
    return [(qid, query_map[qid], [SearchResult(docid=d, score=s, text=truncate(texts[d], passage_length))
                                   for d, s in per_query[qid]]) for qid in order]


def main(args):
    client = JevClient(provider=args.run.provider, api_key=args.run.api_key, model=args.run.model,
                       max_rps=args.run.max_rps or None, verbose=args.run.verbose)
    ranker, family = build_ranker(args, client)
    print(f'provider {client.provider}, model {client.model}, method {family}')

    dataset = ir_datasets.load(args.run.ir_dataset_name)
    query_map = {q.query_id: ranker.truncate(q.text, args.run.query_length) for q in dataset.queries_iter()}
    with open(args.run.docs_file) as f:
        texts = dict(line.rstrip('\n').split('\t', 1) for line in f)
    first_stage = load_first_stage(args.run.run_path, query_map, texts, args.run.hits, ranker.truncate,
                                   args.run.passage_length, args.run.max_queries)
    print(f'{len(first_stage)} queries, top-{args.run.hits} candidates each')

    bar = tqdm(total=len(first_stage), desc='re-ranking', unit='query')
    last_refresh = [0.0]

    def on_call(elapsed):  # show progress inside long queries, but redraw at most twice a second (redraws are slow)
        bar.set_postfix_str(f'calls={client.total_calls} last={elapsed:.2f}s '
                            f'mean={sum(client.latencies) / client.total_calls:.2f}s '
                            f'retries={client.total_retries} ~${client.estimated_cost_usd():.3f}', refresh=False)
        if time.time() - last_refresh[0] > 0.5:
            last_refresh[0] = time.time()
            bar.refresh()

    client.on_call = on_call

    rankers = queue.Queue()  # one ranker per worker: rankers keep per-query state, the client is shared
    rankers.put(ranker)
    for _ in range(args.run.query_workers - 1):
        rankers.put(build_ranker(args, client)[0])

    def process(item):
        qid, query, ranking = item
        r = rankers.get()
        try:
            t0 = time.time()
            result = r.rerank(query, ranking)
            stats = (r.total_compare, time.time() - t0)
        finally:
            rankers.put(r)
        bar.update(1)
        return (qid, result), stats

    tic = time.time()
    with ThreadPoolExecutor(max_workers=args.run.query_workers) as ex:
        outputs = list(ex.map(process, first_stage))
    bar.close()
    n = len(outputs)
    stats = {
        'provider': client.provider, 'model': client.model, 'method': family,
        'args': {k: vars(v) for k, v in vars(args).items() if v is not None},
        'num_queries': n,
        'avg_comparisons': sum(o[1][0] for o in outputs) / n,
        'avg_api_calls': client.total_calls / n,
        'avg_input_tokens': client.total_input_tokens / n,
        'avg_query_latency_s': sum(o[1][1] for o in outputs) / n,
        'total_wall_time_s': time.time() - tic,
        'total_retries': client.total_retries,
        'request_latency': client.latency_summary(),
        'estimated_cost_usd': client.estimated_cost_usd(),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.run.save_path)), exist_ok=True)
    with open(args.run.save_path, 'w') as f:
        for qid, ranking in (o[0] for o in outputs):
            for rank, doc in enumerate(ranking, 1):
                f.write(f'{qid}\tQ0\t{doc.docid}\t{rank}\t{doc.score}\tJev-{family}\n')
    with open(args.run.save_path + '.stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    lat = stats['request_latency']
    print(f"avg API calls/query {stats['avg_api_calls']:.1f}, input tokens {client.total_input_tokens} "
          f"(~${stats['estimated_cost_usd']:.3f}), request latency mean {lat.get('mean_s', 0):.3f}s "
          f"p95 {lat.get('p95_s', 0):.3f}s, retries {client.total_retries}, wall {stats['total_wall_time_s']:.0f}s")
    print(f'run written to {args.run.save_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(title='sub-commands')
    run = commands.add_parser('run')
    run.add_argument('--run_path', required=True, help='first-stage TREC run file')
    run.add_argument('--docs_file', required=True, help='docid<TAB>text file for the candidates (prepare_data.sh)')
    run.add_argument('--save_path', required=True, help='re-ranked TREC run file to write')
    run.add_argument('--ir_dataset_name', default='msmarco-passage/trec-dl-2019/judged')
    run.add_argument('--hits', type=int, default=100)
    run.add_argument('--query_length', type=int, default=32)
    run.add_argument('--passage_length', type=int, default=128)
    run.add_argument('--provider', default='vercel', choices=list(JevClient.PROVIDERS))
    run.add_argument('--model', default=None, help="model id at the provider (default: the provider's Jev id)")
    run.add_argument('--api_key', default=None, help='default: environment or repo-root .env')
    run.add_argument('--num_workers', type=int, default=4, help='concurrent requests inside one query (pointwise)')
    run.add_argument('--query_workers', type=int, default=12, help='queries re-ranked concurrently')
    run.add_argument('--max_rps', type=float, default=20.0, help='global requests/second cap (0 = none)')
    run.add_argument('--max_queries', type=int, default=None, help='only re-rank the first N queries')
    run.add_argument('--verbose', action='store_true')
    commands.add_parser('pointwise').add_argument('--method', default='noul', choices=['noul', 'score'])
    commands.add_parser('pairwise').add_argument('--k', type=int, default=10)
    setwise = commands.add_parser('setwise')
    setwise.add_argument('--num_child', type=int, default=10)
    setwise.add_argument('--k', type=int, default=10)
    listwise = commands.add_parser('listwise')
    listwise.add_argument('--window_size', type=int, default=20)
    listwise.add_argument('--step_size', type=int, default=10)
    listwise.add_argument('--num_repeat', type=int, default=1)
    listwise.add_argument('--mode', default='choice', choices=['choice', 'score'])

    args = parse_args(parser, commands)
    if args.run is None or sum(getattr(args, m) is not None for m in ('pointwise', 'pairwise', 'setwise', 'listwise')) != 1:
        parser.error('usage: run ... <pointwise|pairwise|setwise|listwise> ...')
    main(args)
