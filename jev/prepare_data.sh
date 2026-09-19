#!/usr/bin/env bash
# Fetch the first-stage BM25 candidates (pyserini, msmarco-v1-passage, k1=0.9 b=0.4, top-100, with passage texts)
# for TREC DL 2019 / 2020 from castorini/rank_llm_data and convert them into TREC run files + docid<TAB>text TSVs.
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root
PY=${PYTHON:-.venv/bin/python}
mkdir -p jev/runs/bm25
for ds in dl19 dl20; do
  f=jev/runs/bm25/rank_llm.retrieve_results_${ds}_top100.jsonl
  [ -s "$f" ] || curl -L --fail -o "$f" \
    "https://huggingface.co/datasets/castorini/rank_llm_data/resolve/main/retrieve_results/BM25/retrieve_results_${ds}_top100.jsonl"
  $PY jev/convert_rank_llm.py --input "$f" \
    --run_out jev/runs/bm25/run.rank_llm.bm25.${ds}.top100.txt --docs_out jev/runs/bm25/docs.${ds}.top100.tsv
done
