"""Compare two TREC run files produced by the same method (run-to-run variance of Jev).

    python jev/compare_runs.py jev/runs/jev/dl19.pointwise.noul.txt jev/runs/rerun/dl19.pointwise.noul.txt \
        --dataset msmarco-passage/trec-dl-2019/judged
"""
import argparse

import ir_datasets
import ir_measures


def load(path):
    d = {}
    with open(path) as f:
        for line in f:
            qid, _, docid, rank, score, _ = line.split()
            d[(qid, docid)] = (int(rank), float(score))
    return d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_a')
    parser.add_argument('run_b')
    parser.add_argument('--dataset', default='msmarco-passage/trec-dl-2019/judged')
    args = parser.parse_args()

    a, b = load(args.run_a), load(args.run_b)
    keys = [k for k in a if k in b]
    same_score = sum(a[k][1] == b[k][1] for k in keys)
    same_rank = sum(a[k][0] == b[k][0] for k in keys)
    diffs = [abs(a[k][1] - b[k][1]) for k in keys]
    print(f'{len(keys)} query-passage pairs: identical score {same_score / len(keys):.1%}, identical rank '
          f'{same_rank / len(keys):.1%}, mean |diff| {sum(diffs) / len(diffs):.4f}, max |diff| {max(diffs):.3f}')

    qrels = list(ir_datasets.load(args.dataset).qrels_iter())
    qids = {q.query_id for q in qrels}
    measure = ir_measures.parse_measure('nDCG@10')
    for path in (args.run_a, args.run_b):
        per_q = {m.query_id: m.value for m in ir_measures.iter_calc([measure], qrels, ir_measures.read_trec_run(path))}
        print(f'{path}: nDCG@10 {sum(per_q.get(q, 0.0) for q in qids) / len(qids):.4f}')


if __name__ == '__main__':
    main()
