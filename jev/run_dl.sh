#!/usr/bin/env bash
# Re-rank the pyserini BM25 top-100 of TREC DL 2019 or 2020 with Jev, all methods of the README table, then evaluate.
#   bash jev/run_dl.sh dl19
#   bash jev/run_dl.sh dl20
#   EXTRA="--max_queries 3" bash jev/run_dl.sh dl19   # cheap smoke test
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

DS=${1:-dl19}
case "$DS" in
  dl19) DATASET=msmarco-passage/trec-dl-2019/judged ;;
  dl20) DATASET=msmarco-passage/trec-dl-2020/judged ;;
  *) echo "usage: $0 dl19|dl20" >&2; exit 1 ;;
esac
PY=${PYTHON:-.venv/bin/python}
PROVIDER=${PROVIDER:-vercel}
EXTRA=${EXTRA:-}
BM25=jev/runs/bm25/run.rank_llm.bm25.$DS.top100.txt   # from prepare_data.sh
DOCS=jev/runs/bm25/docs.$DS.top100.tsv
OUT=jev/runs/jev

common="run --provider $PROVIDER --run_path $BM25 --docs_file $DOCS --ir_dataset_name $DATASET --hits 100 --query_length 32 --passage_length 128 --num_workers 4 --query_workers 12 --max_rps 20 $EXTRA"

$PY jev/run_jev.py $common --save_path $OUT/$DS.setwise.heapsort.c10.txt    setwise  --num_child 10 --k 10
$PY jev/run_jev.py $common --save_path $OUT/$DS.listwise.choice.w20s10.txt  listwise --window_size 20 --step_size 10 --mode choice
$PY jev/run_jev.py $common --save_path $OUT/$DS.listwise.score.w20s10.txt   listwise --window_size 20 --step_size 10 --mode score
$PY jev/run_jev.py $common --save_path $OUT/$DS.listwise.choice.w100.txt    listwise --window_size 100 --step_size 100 --mode choice   # all 100 candidates in ONE request
$PY jev/run_jev.py $common --save_path $OUT/$DS.listwise.score.w100.txt     listwise --window_size 100 --step_size 100 --mode score
$PY jev/run_jev.py $common --save_path $OUT/$DS.pointwise.noul.txt          pointwise --method noul
$PY jev/run_jev.py $common --save_path $OUT/$DS.pointwise.score.txt         pointwise --method score
$PY jev/run_jev.py $common --save_path $OUT/$DS.pairwise.heapsort.txt       pairwise --k 10

$PY jev/eval_run.py --dataset $DATASET "$BM25" "$OUT"/$DS.*.txt
