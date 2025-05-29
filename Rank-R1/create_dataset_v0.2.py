from datasets import load_dataset
import random
import toml
from transformers import AutoTokenizer
random.seed(929)

dataset = load_dataset("Tevatron/reasonir-data-hn", split="train", num_proc=20)
# split dataset to train and test

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-32B")
prompt = toml.load('prompts/prompt_setwise-R1-v0.2.toml')

def add_prefix(example):
    query = example['query']
    rel_docs = example['positive_passages']
    # ramdomly select one relevant document
    rel_doc = random.choice(rel_docs)
    # rel_doc = f"{rel_doc['title']} {rel_doc['text']}".strip()
    rel_doc = rel_doc['text'].strip()
    random.shuffle(example['negative_passages'])
    neg_docs = example['negative_passages']

    # maximun 14 neg_docs
    neg_docs = neg_docs[:9]
    # random sample num negatives, larger number has higher probability
    nums = list(range(1, len(neg_docs) + 1))
    num = random.choices(nums, weights=nums, k=1)[0]
    neg_docs = neg_docs[:num]
    # neg_docs = [f"{doc['title']} {doc['text']}" for doc in neg_docs]
    neg_docs = [doc['text'].strip() for doc in neg_docs]
    docs = [rel_doc] + neg_docs

    # truncate documents to 512 tokens
    docs = [tokenizer.tokenize(doc)[:512] for doc in docs]
    docs = [tokenizer.convert_tokens_to_string(doc) for doc in docs]
    labels = [1] + [0] * len(neg_docs)
    indices = list(range(len(labels)))
    random.shuffle(indices)
    docs = [docs[i] for i in indices]
    labels = [labels[i] for i in indices]
    docs = [f"{prompt['doc_prefix'].format(num=i+1)}{doc}" for i, doc in enumerate(docs)]
    docs_text = prompt['doc_separator'].join(docs)
    ground_truth = prompt['ground_truth'].format(num=labels.index(1) + 1)
    example['ground_truth'] = ground_truth

    example['prompt'] = [
        {'role': 'system',
         'content': prompt['prompt_system']},
        {'role': 'user',
         'content': prompt['prompt_user'].format(query=query, docs=docs_text)},
    ]
    return example

dataset = dataset.map(add_prefix, num_proc=20)

dataset = dataset.remove_columns(['query', 'query_id', 'positive_passages', 'negative_passages'])
dataset = dataset.train_test_split(test_size=1000, seed=929)
train_dataset = dataset['train']
test_dataset = dataset['test']
train_dataset.save_to_disk("./reasonir-setwise-r1-v0.2/train")
test_dataset.save_to_disk("./reasonir-setwise-r1-v0.2/test")

