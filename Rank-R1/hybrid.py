import argparse
from tqdm import tqdm


def read_trec_run(file):
    run = {}
    with open(file, 'r') as f:
        for line in f:
            qid, _, docid, rank, score, _ = line.strip().split()
            # score = -float(rank)
            if qid not in run:
                run[qid] = {'docs': {}, 'max_score': float(score), 'min_score': float(score)}
            run[qid]['docs'][docid] = float(score)
            run[qid]['min_score'] = float(score)
    return run


def write_trec_run(run, file, name='fusion'):
    with open(file, 'w') as f:
        for qid in run:
            doc_score = run[qid]
            if 'docs' in doc_score:
                doc_score = doc_score['docs']
            # sort by score
            doc_score = dict(sorted(doc_score.items(), key=lambda item: item[1], reverse=True))
            for i, (doc, score) in enumerate(doc_score.items()):
                f.write(f'{qid} Q0 {doc} {i+1} {score} {name}\n')


def fuse(runs, weights):
    fused_run = {}
    qids = set()
    for run in runs:
        qids.update(run.keys())
    for qid in qids:
        fused_run[qid] = {}
        for run in runs:
            for doc in run[qid]['docs']:
                if doc not in fused_run[qid]:
                    score = 0
                    for temp_run, weight in zip(runs, weights):
                        if doc in temp_run[qid]['docs']:
                            min_score = temp_run[qid]['min_score']
                            max_score = temp_run[qid]['max_score']
                            denominator = max_score - min_score
                            denominator = max(denominator, 1e-9)
                            score += weight * ((temp_run[qid]['docs'][doc] - min_score) / denominator)
                        else:
                            score += 0
                    fused_run[qid][doc] = score
    return fused_run


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_1", type=str)
    parser.add_argument("--run_2", type=str)
    parser.add_argument("--alpha", default=0.5, type=float, help="Weight for the --run_1")
    parser.add_argument("--save_path", type=str)

    args = parser.parse_args()

    run1 = read_trec_run(args.run_1)
    run2 = read_trec_run(args.run_2)

    # handle queries that are not in both runs
    qids = set(run1.keys()).union(set(run2.keys()))
    for qid in qids:
        if qid not in run1:
            run1[qid] = run2[qid]
        if qid not in run2:
            run2[qid] = run1[qid]

    print('fusing runs')
    fusion_run = fuse(
        runs=[run1, run2],
        weights=[args.alpha, (1 - args.alpha)]
    )
    write_trec_run(fusion_run, args.save_path)