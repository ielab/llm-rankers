from datasets import load_dataset
import json
from tqdm import tqdm
import os

bright_corpus = load_dataset("xlangai/BRIGHT", 'documents')
splits = list(bright_corpus.keys())
print(splits)

for split in splits:
    exist_docids = set()
    os.makedirs(f'data/pyserini_corpus/{split}', exist_ok=True)
    corpus = bright_corpus[split]
    print(split)
    with open(f'data/pyserini_corpus/{split}/{split}.jsonl', 'w') as f:
        for doc in tqdm(corpus):
            id_ = doc['id']
            # replace spaces with underscores
            id_ = id_.replace(' ', '_')
            contents = doc['content']
            if id_ in exist_docids:
                print(f'Duplicate document id found: {id_}')
                continue
            exist_docids.add(id_)
            f.write(json.dumps({'id': id_, 'contents': contents}) + '\n')