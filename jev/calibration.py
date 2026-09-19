"""How calibrated are Jev's pointwise scores? Compare the scores stored in the pointwise run files with the NIST
graded relevance judgments (0-3) of DL19 + DL20.

    python jev/calibration.py                     # all pointwise runs found in jev/runs/jev
    python jev/calibration.py --methods score trec

For P(yes)-type runs (noul, cookbook) the score is binned into tenths; for graded-score runs (score, cookbook_score,
trec, umbrela) the expected level is binned into halves and, in addition, its rounded value is compared with the
human grade (confusion matrix, exact / within-one agreement, mean absolute error). Unjudged passages are skipped.
"""
import argparse
import collections
import glob
import os

import ir_datasets

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs', 'jev')
DATASETS = {'dl19': 'msmarco-passage/trec-dl-2019/judged', 'dl20': 'msmarco-passage/trec-dl-2020/judged'}
_QRELS = {}


def qrels(ds_id):
    if ds_id not in _QRELS:
        _QRELS[ds_id] = {(q.query_id, q.doc_id): q.relevance for q in ir_datasets.load(ds_id).qrels_iter()}
    return _QRELS[ds_id]


def judged_scores(key):
    pairs = []
    for ds, ds_id in DATASETS.items():
        path = os.path.join(RUNS, f'{ds}.pointwise.{key}.txt')
        if not os.path.exists(path):
            continue
        q = qrels(ds_id)
        with open(path) as f:
            for line in f:
                qid, _, docid, _, s, _ = line.split()
                if (qid, docid) in q:
                    pairs.append((float(s), q[(qid, docid)]))
    return pairs


def bin_table(pairs, edges, label):
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


def confusion(pairs):
    """Rounded expected level vs. human grade."""
    m = collections.Counter((min(3, max(0, int(round(s)))), rel) for s, rel in pairs)
    n = len(pairs)
    print('| predicted level \\ human grade | 0 | 1 | 2 | 3 |')
    print('|---|---|---|---|---|')
    for p in range(4):
        print(f'| {p} | ' + ' | '.join(str(m[(p, g)]) for g in range(4)) + ' |')
    exact = sum(v for (p, g), v in m.items() if p == g) / n
    within1 = sum(v for (p, g), v in m.items() if abs(p - g) <= 1) / n
    mae = sum(abs(s - rel) for s, rel in pairs) / n
    bias = sum(s - rel for s, rel in pairs) / n
    print(f'\nexact agreement {exact:.1%}, within one level {within1:.1%}, mean |expected level - grade| {mae:.2f}, '
          f'mean (expected level - grade) {bias:+.2f}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--methods', nargs='*', default=None, help='pointwise method keys (default: all found)')
    args = parser.parse_args()
    methods = args.methods or sorted({os.path.basename(p).split('.')[2] for p in glob.glob(os.path.join(RUNS, '*.pointwise.*.txt'))})
    for key in methods:
        pairs = judged_scores(key)
        if not pairs:
            continue
        graded = max(s for s, _ in pairs) > 1.0
        print(f'\n## pointwise `{key}` ({len(pairs)} judged query-passage pairs, DL19 + DL20)\n')
        if graded:
            bin_table(pairs, [i / 2 for i in range(8)], 'expected level')
            print()
            confusion(pairs)
        else:
            bin_table(pairs, [i / 10 for i in range(11)], 'P(yes)')


if __name__ == '__main__':
    main()
