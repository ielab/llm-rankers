"""Evaluate TREC run files with ir_measures (no pyserini / trec_eval binary needed).

    python jev/eval_run.py --dataset msmarco-passage/trec-dl-2019/judged jev/runs/bm25/*.dl19.txt jev/runs/jev/dl19.*.txt

nDCG@10 here equals `trec_eval -c -m ndcg_cut.10` (graded gains, queries missing from the run count as 0).
"""
import argparse
import collections
import glob

import ir_datasets
import ir_measures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('runs', nargs='+', help='TREC run files (globs allowed).')
    parser.add_argument('--dataset', default='msmarco-passage/trec-dl-2019/judged', help='ir_datasets id with qrels.')
    parser.add_argument('--measures', nargs='+', default=['nDCG@10'],
                        help='ir_measures names, e.g. nDCG@10 RR(rel=2)@10 R(rel=2)@100')
    args = parser.parse_args()

    qrels = list(ir_datasets.load(args.dataset).qrels_iter())
    qids = {q.query_id for q in qrels}
    measures = [ir_measures.parse_measure(m) for m in args.measures]

    paths = [p for pattern in args.runs for p in sorted(glob.glob(pattern))] or args.runs
    header = f'{"run":70s}  #q  ' + '  '.join(f'{str(m):>14s}' for m in measures)
    print(header)
    for path in paths:
        run = list(ir_measures.read_trec_run(path))
        per_query = collections.defaultdict(dict)
        for m in ir_measures.iter_calc(measures, qrels, run):
            per_query[m.measure][m.query_id] = m.value
        run_qids = {r.query_id for r in run} & qids
        cells = []
        for m in measures:
            values = per_query[m]
            cells.append(f'{sum(values.get(q, 0.0) for q in qids) / len(qids):14.4f}')  # trec_eval -c semantics
        print(f'{path:70s}  {len(run_qids):3d}  ' + '  '.join(cells))


if __name__ == '__main__':
    main()
