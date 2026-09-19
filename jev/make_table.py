"""Print the Jev rows of the README table from the run files in jev/runs/jev (written by run_dl.sh):
nDCG@10 on DL19 / DL20, API calls per query, mean HTTP latency per request, list-price cost per query.

    python jev/make_table.py
"""
import json
import os

import ir_datasets
import ir_measures

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs', 'jev')
DATASETS = {'dl19': 'msmarco-passage/trec-dl-2019/judged', 'dl20': 'msmarco-passage/trec-dl-2020/judged'}
METHODS = [
    ('pointwise.noul', 'Jev pointwise, noul'),
    ('pointwise.score', 'Jev pointwise, score'),
    ('pairwise.heapsort', 'Jev pairwise, heapsort'),
    ('setwise.heapsort.c10', 'Jev setwise, heapsort c=10'),
    ('listwise.choice.w20s10', 'Jev listwise w20/s10, choice'),
    ('listwise.score.w20s10', 'Jev listwise w20/s10, score'),
    ('listwise.choice.w100', 'Jev listwise 100-in-1, choice'),
    ('listwise.score.w100', 'Jev listwise 100-in-1, score'),
]


def ndcg10(dataset_id, run_path, _cache={}):
    if dataset_id not in _cache:
        _cache[dataset_id] = list(ir_datasets.load(dataset_id).qrels_iter())
    qrels = _cache[dataset_id]
    qids = {q.query_id for q in qrels}
    measure = ir_measures.parse_measure('nDCG@10')
    per_q = {m.query_id: m.value for m in ir_measures.iter_calc([measure], qrels, ir_measures.read_trec_run(run_path))}
    return sum(per_q.get(q, 0.0) for q in qids) / len(qids)  # trec_eval -c semantics


def main():
    print('| Method | DL19 | DL20 | Calls / query | API latency (s) | Cost / query |')
    print('|---|---|---|---|---|---|')
    for key, label in METHODS:
        ndcg, calls, lat, lat_n, cost, nq = {}, 0.0, 0.0, 0, 0.0, 0
        for ds, ds_id in DATASETS.items():
            path = os.path.join(RUNS, f'{ds}.{key}.txt')
            if not os.path.exists(path):
                continue
            ndcg[ds] = ndcg10(ds_id, path)
            with open(path + '.stats.json') as f:
                s = json.load(f)
            calls += s['avg_api_calls'] * s['num_queries']
            cost += s['estimated_cost_usd']
            nq += s['num_queries']
            lat += s['request_latency']['mean_s'] * s['request_latency']['calls']
            lat_n += s['request_latency']['calls']
        f = lambda v, spec: '–' if v is None else format(v, spec)  # noqa: E731
        print(f"| {label} | {f(ndcg.get('dl19'), '.3f')} | {f(ndcg.get('dl20'), '.3f')} | "
              f"{f(calls / nq if nq else None, '.0f')} | {f(lat / lat_n if lat_n else None, '.2f')} | "
              f"{'–' if not nq else '$' + format(cost / nq, '.4f')} |")


if __name__ == '__main__':
    main()
