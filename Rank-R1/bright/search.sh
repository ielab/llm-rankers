for dataset in biology earth_science economics psychology robotics stackoverflow sustainable_living pony leetcode aops theoremqa_theorems theoremqa_questions
do
mkdir -p runs_bm25
python -m pyserini.search.lucene \
  --index data/pyserini_indexes/$dataset \
  --topics data/pyserini_queries/$dataset.tsv \
  --output runs_bm25/bm25.$dataset.trec \
  --bm25 \
  --hits 100

python filter_run.py --run runs_bm25/bm25.$dataset.trec --split $dataset
done