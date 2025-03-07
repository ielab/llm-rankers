import argparse
from datasets import load_dataset

argparser = argparse.ArgumentParser()
argparser.add_argument('--run', type=str, required=True)
argparser.add_argument('--split', type=str, required=True)
args = argparser.parse_args()

bright_queries = load_dataset("xlangai/BRIGHT", 'examples')[args.split]

# Load the run file
run = {}
with open(args.run, 'r') as f:
    for line in f:
        qid, _, docid, rank, score, _ = line.strip().split()
        if qid not in run:
            run[qid] = []
        run[qid].append((docid, float(score)))

# drop the line if the docid in the excluded_ids
for query in bright_queries:
    query_id = query['id']
    excluded_ids = query['excluded_ids']
    if query_id in run:
        run[query_id] = [(docid, score) for docid, score in run[query_id] if docid not in excluded_ids]

# Write the filtered run file
with open(args.run.replace('.trec', '.filtered.trec'), 'w') as f:
    for qid, docid_scores in run.items():
        for rank, (docid, score) in enumerate(docid_scores):
            f.write(f'{qid}\tQ0\t{docid}\t{rank+1}\t{score}\tfiltered\n')