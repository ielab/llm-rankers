# BRIGHT BM25

Run the following command to generate Pyserini Bright indices and BM25 run files:
```bash
python3 write_pyserini_qrels.py
python3 write_pyserini_queries.py
python3 write_pyserini_corpus.py

bash index_corpus.sh
bash search.sh
bash eval.sh
```