# Listwise ranking with LLMs

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


### LLM Listwise ranking
In this example, we use RankZephyr listwise LLM ranker introduced by Ronak et al, [[paper](https://arxiv.org/abs/2312.02724)].
#### transformers inference
```bash
python3 -m llmrankers.experiment.run \
    --run_path run.msmarco-v1-passage.bm25-default.dl19.txt \
    --save_path run.msmarco-v1-passage.bm25-default.dl19.listwise.rank_zephyr_7b_v1_full.txt \
    --pyserini_index msmarco-v1-passage \
    --pyserini_topic dl19-passage \
    --hits 100 \
    --seed 929 \
    --prompt_file prompts/prompt-listwise-rankzephyr.toml \
    --model_name_or_path castorini/rank_zephyr_7b_v1_full \
    --tokenizer_name_or_path castorini/rank_zephyr_7b_v1_full \
    --max_query_length 32 \
    --max_doc_length 512 \
    --dtype bfloat16 \
    --cache_dir cache \
    --verbose False \
    --method listwise \
    --window_size 20 \
    --step_size 10 \
    --num_repeat 1

```

Eval result:
```bash
python -m pyserini.eval.trec_eval -c -l 2 -m ndcg_cut.10 dl19-passage \
  run.msmarco-v1-passage.bm25-default.dl19.listwise.rank_zephyr_7b_v1_full.txt

Results:
ndcg_cut_10             all     0.7345
```

#### vLLM inference
```bash
python3 -m llmrankers.experiment.run \
    --run_path run.msmarco-v1-passage.bm25-default.dl19.txt \
    --save_path run.msmarco-v1-passage.bm25-default.dl19.listwise.rank_zephyr_7b_v1_full.vllm.txt \
    --pyserini_index msmarco-v1-passage \
    --pyserini_topic dl19-passage \
    --hits 100 \
    --seed 929 \
    --use_vllm \
    --prompt_file prompts/prompt-listwise-rankzephyr.toml \
    --model_name_or_path castorini/rank_zephyr_7b_v1_full \
    --tokenizer_name_or_path castorini/rank_zephyr_7b_v1_full \
    --max_query_length 32 \
    --max_doc_length 512 \
    --dtype float16 \
    --cache_dir cache \
    --verbose False \
    --method listwise \
    --window_size 20 \
    --step_size 10 \
    --num_repeat 1
    
```

Eval result:
```bash
python -m pyserini.eval.trec_eval -c -l 2 -m ndcg_cut.10 dl19-passage \
  run.msmarco-v1-passage.bm25-default.dl19.listwise.rank_zephyr_7b_v1_full.vllm.txt

Results:
ndcg_cut_10             all     0.7350
```
> Note that the nDCG@10 score of vLLM inference might be slightly different due to the randomness introduced by vllm. 
> When using dtype `bfloat16`, then randomness will be even larger (see vllm [FAQ](https://docs.vllm.ai/en/latest/getting_started/faq.html#mitigation-strategies)).

