"""Oracle tests for the sorting code in jev_rankers.py: with a perfect, transitive comparison a correct top-k
algorithm must return exactly the true top-k.   Run:  python jev/test_sorters.py"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jev_rankers import SearchResult, heapsort_pairwise, heapsort_setwise, sliding_window  # noqa: E402


def make_docs(n, seed):
    rel = list(range(n))
    random.Random(seed).shuffle(rel)
    return [SearchResult(docid=str(i), score=0.0, text=str(rel[i])) for i in range(n)]


def true_order(docs):
    return [d.docid for d in sorted(docs, key=lambda d: -int(d.text))]


def ok(out, docs, top):
    return [d.docid for d in out[:top]] == true_order(docs)[:top] and sorted(d.docid for d in out) == sorted(d.docid for d in docs)


def main():
    seeds, failures = range(200), 0
    for n, k in [(100, 10), (20, 10), (10, 10), (7, 3), (2, 5)]:
        for s in seeds:
            docs = make_docs(n, s)
            failures += not ok(heapsort_pairwise(docs, k, lambda a, b: int(a.text) > int(b.text)), docs, k)
    for n, c, k in [(100, 2, 10), (100, 10, 10), (100, 19, 10), (100, 99, 10), (20, 2, 10), (8, 2, 8), (3, 4, 3)]:
        for s in seeds:
            docs = make_docs(n, s)
            best = lambda w: max(range(len(w)), key=lambda i: int(w[i].text))  # noqa: E731
            failures += not ok(heapsort_setwise(docs, k, c, best), docs, k)
    for n, w, st in [(100, 20, 10), (100, 4, 2), (43, 20, 10), (100, 20, 15), (10, 20, 10), (100, 100, 100), (5, 2, 1)]:
        for s in seeds:
            docs = make_docs(n, s)
            order = lambda win: sorted(range(len(win)), key=lambda i: -int(win[i].text))  # noqa: E731
            guaranteed = n if n <= w else max(1, w - st)  # one pass guarantees the top (window - step)
            failures += not ok(sliding_window(docs, w, st, 1, order), docs, guaranteed)
    print('ALL OK' if failures == 0 else f'FAILED ({failures})')
    sys.exit(0 if failures == 0 else 1)


if __name__ == '__main__':
    main()
