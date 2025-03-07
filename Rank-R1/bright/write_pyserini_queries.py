from datasets import load_dataset
from tqdm import tqdm
import os

bright_queries = load_dataset("xlangai/BRIGHT", 'examples')
splits = list(bright_queries.keys())

os.makedirs('data/pyserini_queries', exist_ok=True)
for split in splits:
    queries = bright_queries[split]
    print(split)
    with open(f'data/pyserini_queries/{split}.tsv', 'w') as f:
        for query in tqdm(queries):
            query_id = query['id']
            query = query['query']
            # replace newlines and tabs with spaces including CR and LF
            query = query.replace('\n', ' ').replace('\t', ' ').replace('\r', ' ').replace('\f', ' ')
            f.write(f'{query_id}\t{query}\n')
