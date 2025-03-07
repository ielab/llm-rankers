for dataset in biology earth_science economics psychology robotics stackoverflow sustainable_living pony leetcode aops theoremqa_theorems theoremqa_questions
do
python -m pyserini.eval.trec_eval -c -m ndcg_cut.10 data/pyserini_qrels/$dataset.tsv \
  runs_bm25/bm25.$dataset.filtered.trec > runs_bm25/bm25.$dataset.filtered.eval
echo "Done with $dataset"
done