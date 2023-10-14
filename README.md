# llm-rankers
Document Ranking with Large Language Models.

## Examples
Pointwise
```bash
YEAR=2019
MODEL=flan-t5-large
CUDA_VISIBLE_DEVICES=0 python3 run.py \
  run --run_path runs/dl${YEAR}/run.bm25.txt \
      --save_path runs/dl${YEAR}/run.bm25.${MODEL}.pointwise.yes_no.txt \
      --model_name_or_path google/${MODEL} \
      --tokenizer_name_or_path google/${MODEL} \
      --ir_dataset_name msmarco-passage/trec-dl-${YEAR} \
      --hits 100 \
      --query_length 32 \
      --passage_length 128 \
      --device cuda \
      --cache_dir ./cache \
  pointwise --method yes_no \
            --batch_size 16
```

Pairwise
```bash
YEAR=2019
MODEL=flan-t5-large
K=10
CUDA_VISIBLE_DEVICES=0 python3 run.py \
  run --run_path runs/dl${YEAR}/run.bm25.txt \
      --save_path runs/dl${YEAR}/run.bm25.${MODEL}.pairwise.heapsort.k=${K}.txt \
      --model_name_or_path google/${MODEL} \
      --tokenizer_name_or_path google/${MODEL} \
      --ir_dataset_name msmarco-passage/trec-dl-${YEAR} \
      --hits 100 \
      --k ${K} \
      --query_length 32 \
      --passage_length 128 \
      --device cuda \
      --cache_dir ./cache \
  pairwise --method heapsort # or bubblesort or allpair
```

Listwise
```bash
YEAR=2019
MODEL=flan-t5-large
K=10
CUDA_VISIBLE_DEVICES=0 python3 run.py \
  run --run_path runs/dl${YEAR}/run.bm25.txt \
      --save_path runs/dl${YEAR}/run.bm25.${MODEL}.listwise.txt \
      --model_name_or_path google/${MODEL} \
      --tokenizer_name_or_path google/${MODEL} \
      --ir_dataset_name msmarco-passage/trec-dl-${YEAR} \
      --hits 100 \
      --k ${K} \
      --query_length 32 \
      --passage_length 128 \
      --device cuda \
      --cache_dir ./cache \
  listwise --window_size 3 \
           --step_size 1
```

Setwise
```bash
YEAR=2019
MODEL=flan-t5-large
K=10
METHOD=heapsort # or bubblesort
CUDA_VISIBLE_DEVICES=0 python3 run.py \
  run --run_path runs/dl${YEAR}/run.bm25.txt \
      --save_path runs/dl${YEAR}/run.bm25.${MODEL}.setwise.${METHOD}.n=2.k=10.txt \
      --model_name_or_path ${MODEL} \
      --tokenizer_name_or_path ${MODEL} \
      --ir_dataset_name msmarco-passage/trec-dl-${YEAR} \
      --hits 100 \
      --k 10 \
      --query_length 32 \
      --passage_length 128 \
      --device cuda \
      --cache_dir ./cache \
  setwise --num_child 2 \ # num_child=2 means 2 child node + 1 parent node = 3 docs to compare
          --method ${METHOD}
```

