"""Convert castorini/rank_llm_data BM25 retrieve-results JSONL into a TREC run file plus a docid<TAB>text TSV.

The JSONL files (https://huggingface.co/datasets/castorini/rank_llm_data, retrieve_results/BM25/) hold the pyserini
BM25 (msmarco-v1-passage, k1=0.9, b=0.4) top-100/top-1000 candidates *with passage text*, so the re-rankers can run
without downloading the full MS MARCO collection through ir_datasets.

    python jev/convert_rank_llm.py --input jev/runs/bm25/rank_llm.retrieve_results_dl19_top100.jsonl \
        --run_out jev/runs/bm25/run.rank_llm.bm25.dl19.top100.txt --docs_out jev/runs/bm25/docs.dl19.top100.tsv
"""
import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--run_out', required=True)
    parser.add_argument('--docs_out', required=True)
    parser.add_argument('--tag', default='rank_llm-bm25')
    args = parser.parse_args()

    for p in (args.run_out, args.docs_out):
        os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)

    n_q, n_d = 0, 0
    texts = {}
    with open(args.input) as f, open(args.run_out, 'w') as run_f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            qid = str(item['query']['qid'])
            n_q += 1
            for rank, cand in enumerate(item['candidates'], 1):
                docid = str(cand['docid'])
                doc = cand.get('doc') or {}
                text = doc.get('contents') or doc.get('text') or doc.get('segment') or ''
                if 'title' in doc and doc['title']:
                    text = f"{doc['title']} {text}"
                texts[docid] = ' '.join(text.split())
                run_f.write(f'{qid} Q0 {docid} {rank} {cand["score"]} {args.tag}\n')
                n_d += 1
    with open(args.docs_out, 'w') as docs_f:
        for docid, text in texts.items():
            docs_f.write(f'{docid}\t{text}\n')
    print(f'{n_q} queries, {n_d} candidates -> {args.run_out}; {len(texts)} unique passages -> {args.docs_out}')


if __name__ == '__main__':
    main()
