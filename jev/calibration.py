"""How calibrated are Jev's probabilities? Bin the pointwise scores stored in the run files and compare each bin
with the NIST graded relevance judgments (0-3) of DL19 + DL20.   python jev/calibration.py

* pointwise `noul` run files store P("the passage answers the query")
* pointwise `score` run files store the expected level over the 4 TREC-DL graded levels
Unjudged passages are skipped.
"""
import collections
import os

import ir_datasets

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs', 'jev')
DATASETS = {'dl19': 'msmarco-passage/trec-dl-2019/judged', 'dl20': 'msmarco-passage/trec-dl-2020/judged'}


def judged_scores(key):
    pairs = []
    for ds, ds_id in DATASETS.items():
        path = os.path.join(RUNS, f'{ds}.{key}.txt')
        if not os.path.exists(path):
            continue
        qrels = {(q.query_id, q.doc_id): q.relevance for q in ir_datasets.load(ds_id).qrels_iter()}
        with open(path) as f:
            for line in f:
                qid, _, docid, _, s, _ = line.split()
                if (qid, docid) in qrels:
                    pairs.append((float(s), qrels[(qid, docid)]))
    return pairs


def table(pairs, edges, label):
    bins = collections.defaultdict(lambda: [0, 0, 0.0])
    for s, rel in pairs:
        b = min(max(int((s - edges[0]) / (edges[1] - edges[0])), 0), len(edges) - 2)
        bins[b][0] += 1
        bins[b][1] += rel >= 2
        bins[b][2] += rel
    print(f'| {label} | passages | share with grade >= 2 | mean grade (0-3) |')
    print('|---|---|---|---|')
    for b in sorted(bins):
        n, nr, sg = bins[b]
        print(f'| {edges[b]:.1f} – {edges[b + 1]:.1f} | {n} | {nr / n:.2f} | {sg / n:.2f} |')


def main():
    noul = judged_scores('pointwise.noul')
    print(f'Pointwise noul, P(yes) vs. judgment ({len(noul)} judged pairs):\n')
    table(noul, [i / 10 for i in range(11)], 'P(yes)')
    sc = judged_scores('pointwise.score')
    print(f'\nPointwise score, expected level vs. judgment ({len(sc)} pairs):\n')
    table(sc, [i / 2 for i in range(8)], 'expected level')


if __name__ == '__main__':
    main()
