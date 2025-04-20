# Setwise ranking with LLMs

## TREC DL2019 example
Here is an example of using pyserini command lines to generate BM25 run files on TREC DL 2019:
```bash
python -m pyserini.search.lucene \
  --threads 16 --batch-size 128 \
  --index msmarco-v1-passage \
  --topics dl19-passage \
  --output run.msmarco-v1-passage.bm25-default.dl19.txt \
  --bm25 --k1 0.9 --b 0.4
```
To evaluate NDCG@10 scores of BM25:

```bash
python -m pyserini.eval.trec_eval -c -l 2 -m ndcg_cut.10 dl19-passage \
  run.msmarco-v1-passage.bm25-default.dl19.txt
  
Results:
ndcg_cut_10           	all	0.5058
```


### LLM Setwise ranking
#### transformers inference
```bash
python3 -m llmrankers.experiment.run \
    --run_path run.msmarco-v1-passage.bm25-default.dl19.txt \
    --save_path run.msmarco-v1-passage.bm25-default.dl19.setwise.Setwise-SFT-3B-v0.1.txt \
    --pyserini_index msmarco-v1-passage \
    --pyserini_topic dl19-passage \
    --hits 100 \
    --seed 929 \
    --prompt_file prompts/prompt-setwise-llm.toml \
    --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
    --tokenizer_name_or_path Qwen/Qwen2.5-3B-Instruct \
    --lora_name_or_path ielabgroup/Setwise-SFT-3B-v0.1 \
    --max_query_length 32 \
    --max_doc_length 512 \
    --dtype bfloat16 \
    --cache_dir cache \
    --verbose False \
    --scoring generation \
    --method setwise \
    --num_child 19 \
    --sort heapsort \
    --k 10

```

Eval result:
```bash
python -m pyserini.eval.trec_eval -c -l 2 -m ndcg_cut.10 dl19-passage \
  run.msmarco-v1-passage.bm25-default.dl19.setwise.Setwise-SFT-3B-v0.1.txt

Results:
ndcg_cut_10             all     0.7342
```

#### vLLM inference
```bash
python3 -m llmrankers.experiment.run \
    --run_path run.msmarco-v1-passage.bm25-default.dl19.txt \
    --save_path run.msmarco-v1-passage.bm25-default.dl19.setwise.Setwise-SFT-3B-v0.1.vllm.txt \
    --pyserini_index msmarco-v1-passage \
    --pyserini_topic dl19-passage \
    --hits 100 \
    --seed 929 \
    --use_vllm \
    --prompt_file prompts/prompt-setwise-llm.toml \
    --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
    --tokenizer_name_or_path Qwen/Qwen2.5-3B-Instruct \
    --lora_name_or_path ielabgroup/Setwise-SFT-3B-v0.1 \
    --max_query_length 32 \
    --max_doc_length 512 \
    --dtype float16 \
    --cache_dir cache \
    --verbose False \
    --scoring generation \
    --method setwise \ 
    --num_child 19 \
    --sort heapsort \
    --k 10
    
```

Eval result:
```bash
python -m pyserini.eval.trec_eval -c -l 2 -m ndcg_cut.10 dl19-passage \
  run.msmarco-v1-passage.bm25-default.dl19.setwise.Setwise-SFT-3B-v0.1.vllm.txt

Results:
ndcg_cut_10             all     0.7340
```
> Note that the nDCG@10 score of vLLM inference might be slightly different due to the randomness introduced by vllm. 
> When using dtype `bfloat16`, then randomness will be even larger (see vllm [FAQ](https://docs.vllm.ai/en/latest/getting_started/faq.html#mitigation-strategies))


### FlanT5 Setwise ranking
```bash
python3 -m llmrankers.experiment.run \
    --run_path run.msmarco-v1-passage.bm25-default.dl19.txt \
    --save_path run.msmarco-v1-passage.bm25-default.dl19.setwise.flan-t5-large.txt \
    --pyserini_index msmarco-v1-passage \
    --pyserini_topic dl19-passage \
    --hits 100 \
    --prompt_file prompts/prompt-setwise-flant5.toml \
    --model_name_or_path google/flan-t5-large \
    --tokenizer_name_or_path google/flan-t5-large \
    --max_query_length 32 \
    --max_doc_length 128 \
    --dtype bfloat16 \
    --cache_dir cache \
    --verbose True \
    --scoring generation \
    --method setwise \
    --sort heapsort \
    --num_child 2 \
    --k 10

```
Eval result:
```bash
python -m pyserini.eval.trec_eval -c -l 2 -m ndcg_cut.10 dl19-passage \
  run.msmarco-v1-passage.bm25-default.dl19.setwise.flan-t5-large.txt
  
Results:
ndcg_cut_10           	all	0.6645
```

Set scoring method to `--sort likelihood` you will get the following results, and you may notice the reranking is faster.
```bash
Results:
ndcg_cut_10             all     0.6667
```
