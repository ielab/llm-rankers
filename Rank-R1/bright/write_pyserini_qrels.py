from datasets import load_dataset
from tqdm import tqdm
import os

bright_queries = load_dataset("xlangai/BRIGHT", 'examples')
splits = list(bright_queries.keys())

os.makedirs('data/pyserini_qrels', exist_ok=True)
for split in splits:
    queries = bright_queries[split]
    print(split)
    with open(f'data/pyserini_qrels/{split}.tsv', 'w') as f:
        for query in tqdm(queries):
            query_id = query['id']
            positive_doc_ids = query['gold_ids']
            for doc_id in set(positive_doc_ids):
                doc_id = doc_id.replace(' ', '_')
                f.write(f'{query_id}\t0\t{doc_id}\t1\n')
            
