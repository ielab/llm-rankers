# Rank-R1: Enhancing Reasoning in LLM-based Document Rerankers via Reinforcement Learning

In this work, we introduce Rank-R1, a Setwise reranker with reasoning abilities. 
We trained Rank-R1 on the MSMARCO dataset using the GRPO RL algorithm.

---
## Installation
For training and inference Rank-R1, first follow the README.md in the root directory to install llm-rankers. Then
```bash
pip install vllm
pip install trl
pip install peft deepspeed
```
---
## Inference Rank-R1
### Python example:

```Python
from llmrankers.setwise import RankR1SetwiseLlmRanker
from llmrankers.rankers import SearchResult

docs = [SearchResult(docid=i, text=f'this is passage {i}', score=None) for i in range(20)]
query = 'Give me passage 6'

ranker = RankR1SetwiseLlmRanker(
    model_name_or_path='Qwen/Qwen2.5-7B-Instruct',
    lora_name_or_path='ielabgroup/Rank-R1-7B-v0.1',
    prompt_file='prompts/prompt_setwise-R1.toml',
    num_child=19,
    k=1,
    verbose=True
)

print(ranker.rerank(query, docs)[0])
```
Internally, the above code is equivalent to the following:
```Python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

def get_model(peft_model_name):
    config = PeftConfig.from_pretrained(peft_model_name)
    base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
    model = PeftModel.from_pretrained(base_model, peft_model_name)
    model = model.merge_and_unload()
    return model

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
model = get_model('ielabgroup/Rank-R1-7B-v0.1').to('cuda:0').eval()

prompt_system = "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>."
prompt_user = '''Given the query: "{query}", which of the following documents is most relevant?
{docs}
After completing the reasoning process, please provide only the label of the most relevant document to the query, enclosed in square brackets, within the answer tags. For example, if the third document is the most relevant, the answer should be: <think> reasoning process here </think> <answer>[3]</answer>.'''

query = 'Give me passage 6'
docs = [f'[{i+1}] this is passage {i+1}' for i in range(20)]
docs = '\n'.join(docs)

messages = [
    {'role': "system", 'content': prompt_system},
    {'role': "user", 'content': prompt_user.format(query=query, docs=docs)}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to('cuda:0')

generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=2048,
    do_sample=False,
)
generated_ids = [
    output_ids[len(input_ids)-1:] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
'''
<think> The query "Give me passage 6" is asking for the specific document that contains the text "this is passage 6". By looking at the list provided, we can see that the document labeled [6] matches this description exactly. </think>
<answer>[6]</answer>
'''

# extract the answer
import re
pattern = '<think>.*?</think>\s*<answer>(.*?)</answer>'
answer = re.search(pattern, response, re.DOTALL).group(1) # answer = '[6]'
```
> Note that our Rank-R1 Setwise rerankers are trained with the prompt format shown above, which includes 20 documents. Other numbers of documents should also work fine, but this would represent a "zero-shot" setting for the model.
---
### Released Models (LoRA adapters)
| Resource                                                                       | Description                       |
|:-------------------------------------------------------------------------------|:----------------------------------|
| [Rank-R1-3B-v0.1](https://huggingface.co/ielabgroup/Rank-R1-3B-v0.1)           | Trained from Qwen2.5-3B-Instruct  |
| [Rank-R1-7B-v0.1](https://huggingface.co/ielabgroup/Rank-R1-7B-v0.1)           | Trained from Qwen2.5-7B-Instruct  |
| [Rank-R1-14B-v0.1](https://huggingface.co/ielabgroup/Rank-R1-14B-v0.1)         | Trained from Qwen2.5-14B-Instruct |
| [Setwise-SFT-3B-v0.1](https://huggingface.co/ielabgroup/Setwise-SFT-3B-v0.1)   | Trained from Qwen2.5-3B-Instruct  |
| [Setwise-SFT-7B-v0.1](https://huggingface.co/ielabgroup/Setwise-SFT-7B-v0.1)   | Trained from Qwen2.5-7B-Instruct  |
| [Setwise-SFT-14B-v0.1](https://huggingface.co/ielabgroup/Setwise-SFT-14B-v0.1) | Trained from Qwen2.5-14B-Instruct |
---
### TREC DL examples:
<details>
<summary>Rank-R1</summary>

```bash
# Rank-R1
for dataset in dl19 dl20; do
  for size in 3 7 14; do
    model_name_or_path=Qwen/Qwen2.5-${size}B-Instruct
    lora_path=ielabgroup/Rank-R1-${size}B-v0.1
    mkdir -p runs/${lora_path}
    
    CUDA_VISIBLE_DEVICES=0 python3 run_setwise.py \
    run --model_name_or_path ${model_name_or_path} \
      --lora_path ${lora_path} \
      --prompt_file prompts/prompt_setwise-R1.toml \
      --run_path runs/pyserini_bm25/run.msmarco-v1-passage.bm25-default.${dataset}.txt \
      --save_path runs/${lora_path}/${dataset}.txt \
      --pyserini_dataset ${dataset}-passage \
      --pyserini_index msmarco-v1-passage \
      --hits 100 \
      --query_length 32 \
      --passage_length 512 \
      --scoring generation \
      setwise --num_child 19 \
               --method heapsort \
               --k 10
           
  python -m pyserini.eval.trec_eval -c -l 2 -m ndcg_cut.10 ${dataset}-passage runs/${lora_path}/${dataset}.txt > runs/${lora_path}/${dataset}.eval
  done
done
```
</details>

<details>
<summary>Rank-R1 zeroshot</summary>

```bash
# Rank-R1 zeroshot
for dataset in dl19 dl20; do
  for size in 3 7 14; do
    model_name_or_path=Qwen/Qwen2.5-${size}B-Instruct
    
    mkdir -p runs/${model_name_or_path}/zeroshot-R1
    
    CUDA_VISIBLE_DEVICES=0 python3 run_setwise.py \
    run --model_name_or_path ${model_name_or_path} \
      --prompt_file prompts/prompt_setwise-R1.toml \
      --run_path runs/pyserini_bm25/run.msmarco-v1-passage.bm25-default.${dataset}.txt \
      --save_path runs/${model_name_or_path}/zeroshot-R1/${dataset}.txt \
      --pyserini_dataset ${dataset}-passage \
      --pyserini_index msmarco-v1-passage \
      --hits 100 \
      --query_length 32 \
      --passage_length 512 \
      --scoring generation \
      setwise --num_child 19 \
               --method heapsort \
               --k 10
           
  python -m pyserini.eval.trec_eval -c -l 2 -m ndcg_cut.10 ${dataset}-passage runs/${model_name_or_path}/zeroshot-R1/${dataset}.txt > runs/${model_name_or_path}/zeroshot-R1/${dataset}.eval
  done
done
```
</details>

<details>
<summary>Setwise SFT</summary>

```bash
# Setwise SFT
for dataset in dl19 dl20; do
  for size in 3 7 14; do
    model_name_or_path=Qwen/Qwen2.5-${size}B-Instruct
    lora_path=ielabgroup/Setwise-SFT-${size}B-v0.1
    
    mkdir -p runs/${lora_path}
    
    CUDA_VISIBLE_DEVICES=0 python3 run_setwise.py \
    run --model_name_or_path ${model_name_or_path} \
      --lora_path ${lora_path} \
      --prompt_file prompts/prompt_setwise.toml \
      --run_path runs/pyserini_bm25/run.msmarco-v1-passage.bm25-default.${dataset}.txt \
      --save_path runs/${lora_path}/${dataset}.txt \
      --pyserini_dataset ${dataset}-passage \
      --pyserini_index msmarco-v1-passage \
      --hits 100 \
      --query_length 32 \
      --passage_length 512 \
      --scoring generation \
      setwise --num_child 19 \
               --method heapsort \
               --k 10
               
    python -m pyserini.eval.trec_eval -c -l 2 -m ndcg_cut.10 ${dataset}-passage runs/${lora_path}/${dataset}.txt > runs/${lora_path}/${dataset}.eval
    done
done
```
</details>

<details>
<summary>Setwise zeroshot</summary>

```bash
# Setwise zeroshot
for dataset in dl19 dl20; do
  for size in 3 7 14; do
    model_name_or_path=Qwen/Qwen2.5-${size}B-Instruct
    
    mkdir -p runs/${model_name_or_path}/zeroshot
    
    CUDA_VISIBLE_DEVICES=0 python3 run_setwise.py \
    run --model_name_or_path ${model_name_or_path} \
      --prompt_file prompts/prompt_setwise.toml \
      --run_path runs/pyserini_bm25/run.msmarco-v1-passage.bm25-default.${dataset}.txt \
      --save_path runs/${model_name_or_path}/zeroshot/${dataset}.txt \
      --pyserini_dataset ${dataset}-passage \
      --pyserini_index msmarco-v1-passage \
      --hits 100 \
      --query_length 32 \
      --passage_length 512 \
      --scoring generation \
      setwise --num_child 19 \
               --method heapsort \
               --k 10
               
    python -m pyserini.eval.trec_eval -c -l 2 -m ndcg_cut.10 ${dataset}-passage runs/${model_name_or_path}/zeroshot/${dataset}.txt > runs/${model_name_or_path}/zeroshot/${dataset}.eval
    done
done
```
</details>

<details>
<summary>Rankzephyr</summary>

```bash
# Rankzephyr
for dataset in dl19 dl20; do
    model_name_or_path=castorini/rank_zephyr_7b_v1_full
    
    mkdir -p runs/${model_name_or_path}
    
    CUDA_VISIBLE_DEVICES=0 python3 run_listwise.py \
      run --model_name_or_path castorini/rank_zephyr_7b_v1_full \
      --prompt_file prompts/prompt_listwise_rankzephyr.toml \
      --run_path runs/pyserini_bm25/run.msmarco-v1-passage.bm25-default.${dataset}.txt \
      --save_path runs/${model_name_or_path}/${dataset}.txt \
      --pyserini_dataset ${dataset}-passage \
      --pyserini_index msmarco-v1-passage \
      --hits 100 \
      --query_length 32 \
      --passage_length 512 \
      --scoring generation \
      listwise --window_size 20 \
               --step_size 10 \
               --num_repeat 1
            
    python -m pyserini.eval.trec_eval -c -l 2 -m ndcg_cut.10 ${dataset}-passage runs/${model_name_or_path}/${dataset}.txt > runs/${model_name_or_path}/${dataset}.eval
done
```
</details>

---
### BRIGHT examples:
First go the `bright` folder and follow the instructions to obtain the BM25 run files.

Then similar to TREC DL exmaples, you can run the following commands to evaluate Rank-R1 with BM25 run files:
```bash
for size in 3 7 14; do
    model_name_or_path=Qwen/Qwen2.5-${size}B-Instruct
    lora_path=ielabgroup/Rank-R1-${size}B-v0.1
    
    for dataset in biology earth_science economics psychology robotics stackoverflow sustainable_living pony leetcode aops theoremqa_theorems theoremqa_questions; do
        CUDA_VISIBLE_DEVICES=0 python3 run_setwise.py \
            run --model_name_or_path ${model_name_or_path} \
              --lora_path ${lora_path} \
              --prompt_file prompts/prompt_setwise-R1.toml \
              --run_path bright/runs_bm25/bm25.${dataset}.filtered.trec \
              --save_path results/setwise/runs_bright/${lora_path}/${prompt_file}/run_${dataset}.txt \
              --query_file bright/data/pyserini_queries/${dataset}.tsv \
              --pyserini_index bright/data/pyserini_indexes/${dataset} \
              --hits 100 \
              --query_length 1024 \
              --passage_length 1024 \
              --scoring generation \
            setwise --num_child 19 \
                   --method heapsort \
                   --k 10

        python -m pyserini.eval.trec_eval -c -l 2 -m ndcg_cut.10 bright/data/pyserini_qrels/${dataset}.tsv \
          results/setwise/runs_bright/${lora_path}/${prompt_file}/run_${dataset}.txt > results/setwise/runs_bright/${lora_path}/${prompt_file}/run_${dataset}.eval
    done
done
```
## Training 
<details>
<summary>Train Rank-R1 GRPO</summary>

Step 1: Create training dataset
```bash
python create_dataset.py
```
Step 2: Train
```bash
deepspeed --include localhost:0,1,2,3 --master_port 60000 train_grpo.py
```
</details>

<details>
<summary>Train Setwise SFT</summary>

Step 1: Create training dataset
```bash
python create_dataset_sft.py
```
Step 2: Train
```bash
deepspeed --include localhost:0,1,2,3 --master_port 60000 train_sft.py
```
</details>
