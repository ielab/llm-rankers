from datasets import load_dataset
import random
import toml



dataset = load_dataset("Tevatron/msmarco-passage", split="train")
prompt = toml.load('prompts/prompt_setwise.toml')

def add_prefix(example):
    query = example['query']
    rel_doc = f"{example['positive_passages'][0]['title']} {example['positive_passages'][0]['text']}"
    random.shuffle(example['negative_passages'])
    neg_docs = example['negative_passages']
    neg_docs = neg_docs[:19]
    neg_docs = [f"{doc['title']} {doc['text']}" for doc in neg_docs]
    docs = [rel_doc] + neg_docs
    labels = [1] + [0] * len(neg_docs)

    indices = list(range(len(labels)))
    random.shuffle(indices)
    docs = [docs[i] for i in indices]
    labels = [labels[i] for i in indices]
    docs = [f"[{i+1}] {doc}" for i, doc in enumerate(docs)]
    docs_text = '\n'.join(docs)
    ground_truth = f'[{labels.index(1) + 1}]'
    example['prompt'] = [
        {'role': 'system',
         'content': prompt['prompt_system']},
        {'role': 'user',
         'content': prompt['prompt_user'].format(query=query, docs=docs_text)},
    ]
    example['completion'] = [
        {'role': 'assistant',
         'content': f'<answer>{ground_truth}</answer>'},
    ]
    return example

dataset = dataset.map(add_prefix)

dataset = dataset.remove_columns(['query', 'query_id', 'positive_passages', 'negative_passages'])
dataset.save_to_disk("./msmarco-passage-setwise-sft")
