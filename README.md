# llm-rankers
Document Ranking with Large Language Models.
> Note: The current code base only supports T5-style open-source LLMs, and OpenAI APIs for several methods. We are in the process of implementing support for more LLMs.

---
## Installation
Git clone this repository, then pip install the following libraries:
```bash
torch==2.0.1
transformers==4.31.0
pyserini==0.21.0
ir-datasets==0.5.5
openai==0.27.10
tiktoken==0.4.0
accelerate==0.22.0 
```
> Note the code base is tested with python=3.9 conda environment. You may also need to install some pyserini dependencies such as faiss. We refer to pyserini installation doc [link](https://github.com/castorini/pyserini/blob/master/docs/installation.md#development-installation)

---
## First-stage runs
We use LLMs to re-rank top documents retrieved by a first-stage retriever. In this repo we take BM25 as the retriever.

We rely on [pyserini](https://github.com/castorini/pyserini) IR toolkit to get BM25 ranking. 

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

You can find the command line examples for full TREC DL datasets [here](https://castorini.github.io/pyserini/2cr/msmarco-v1-passage.html).

Similarly, you can find command lines for obtaining BM25 results on BEIR datasets [here](https://castorini.github.io/pyserini/2cr/beir.html).

In this repository, we use DL 2019 as an example. That is, we always re-rank `run.msmarco-v1-passage.bm25-default.dl19.txt` with LLMs.

--- 

## Prompting Methods for zero-shot document ranking with LLMs

<details>
<summary><h3>Pointwise</h3></summary>
We have two pointwise methods implemented so far:

`yes_no`: LLMs are prompted to generate whether the provided candidate document is relevant to the query. Candidate documents are re-ranked based on the normalized likelihood of generating a "yes" response.

`qlm`: Query Likelihood Modelling (QLM), LLMs are prompted to produce a relevant query for each candidate document. The documents are then re-ranked based on the likelihood of generating the given query. [1]

These methods rely on access to the model output logits to compute relevance scores.

Command line example:
```bash
CUDA_VISIBLE_DEVICES=0 python3 run.py \
  run --model_name_or_path google/flan-t5-large \
      --tokenizer_name_or_path google/flan-t5-large \
      --run_path run.msmarco-v1-passage.bm25-default.dl19.txt \
      --save_path run.pointwise.yes_no.txt \
      --ir_dataset_name msmarco-passage/trec-dl-2019 \
      --hits 100 \
      --query_length 32 \
      --passage_length 128 \
      --device cuda \
  pointwise --method yes_no \
            --batch_size 32
```   
```bash     
# evaluation
python -m pyserini.eval.trec_eval -c -l 2 -m ndcg_cut.10 dl19-passage \
  run.pointwise.qlm.txt
 
Results:
ndcg_cut_10             all     0.6544
```

Change `--method yes_no` to `--method qlm` for QLM pointwise ranking. You can also set larger `--batch_size` that you gpu can afford for faster inference.

We also have implemented supervised [monoT5](https://github.com/castorini/pygaggle) pointwise re-ranker. Simply set `--model_name_or_path` and `--tokenizer_name_or_path` to `castorini/monot5-3b-msmarco`, or other monoT5 models listed in [here](https://huggingface.co/castorini).

</details>


<details>
<summary><h3>Listwise</h3></summary>

Our implementation of listwise approach is following [RankGPT](https://github.com/sunnweiwei/RankGPT) [2]. It uses a sliding window sorting algorithm to re-rank documents.
```bash
CUDA_VISIBLE_DEVICES=0 python3 run.py \
  run --model_name_or_path google/flan-t5-large \
      --tokenizer_name_or_path google/flan-t5-large \
      --run_path run.msmarco-v1-passage.bm25-default.dl19.txt \
      --save_path run.liswise.generation.txt \
      --ir_dataset_name msmarco-passage/trec-dl-2019 \
      --hits 100 \
      --query_length 32 \
      --passage_length 100 \
      --scoring generation \
      --device cuda \
  listwise --window_size 4 \
           --step_size 2 \
           --num_repeat 5
```

```bash     
# evaluation
python -m pyserini.eval.trec_eval -c -l 2 -m ndcg_cut.10 dl19-passage \
  run.liswise.generation.txt
 
Results:
ndcg_cut_10             all     0.5612
```

Use `--window_size`, `--step_size` and `--num_repeat` to configure sliding window process. 

We also provide Openai API implementation, simply do:

```bash
python3 run.py \
  run --model_name_or_path gpt-3.5-turbo \
      --openai_key [your key] \
      --run_path run.msmarco-v1-passage.bm25-default.dl19.txt \
      --save_path run.iswise.generation.openai.txt \
      --ir_dataset_name msmarco-passage/trec-dl-2019 \
      --hits 100 \
      --query_length 32 \
      --passage_length 100 \
      --scoring generation \
  listwise --window_size 4 \
           --step_size 2 \
           --num_repeat 5
```

The above two listwise runs are relying on LLM generated tokens to do the sliding window. 
However, if we have local model, for example flan-t5, we can use Setwise prompting proposed in our [paper](https://arxiv.org/abs/2310.09497) [3] to estimate the likehood of document rankings to do the sliding window:

```bash
CUDA_VISIBLE_DEVICES=0 python3 run.py \
  run --model_name_or_path google/flan-t5-large \
      --tokenizer_name_or_path google/flan-t5-large \
      --run_path run.msmarco-v1-passage.bm25-default.dl19.txt \
      --save_path run.liswise.likelihood.txt \
      --ir_dataset_name msmarco-passage/trec-dl-2019 \
      --hits 100 \
      --query_length 32 \
      --passage_length 100 \
      --scoring likelihood \
      --device cuda \
  listwise --window_size 4 \
           --step_size 2 \
           --num_repeat 5
```

```bash     
# evaluation
python -m pyserini.eval.trec_eval -c -l 2 -m ndcg_cut.10 dl19-passage \
  run.liswise.likelihood.txt
 
Results:
ndcg_cut_10             all     0.6691
```

</details>

<details>
<summary><h3>Pairwise</h3></summary>
We implement Pairwise prompting method proposed in [4].

```bash
CUDA_VISIBLE_DEVICES=0 python3 run.py \
  run --model_name_or_path google/flan-t5-large \
      --tokenizer_name_or_path google/flan-t5-large \
      --run_path run.msmarco-v1-passage.bm25-default.dl19.txt \
      --save_path run.pairwise.heapsort.txt \
      --ir_dataset_name msmarco-passage/trec-dl-2019 \
      --hits 100 \
      --query_length 32 \
      --passage_length 128 \
      --scoring generation \
      --device cuda \
  pairwise --method heapsort \
           --k 10
```

```bash     
# evaluation
python -m pyserini.eval.trec_eval -c -l 2 -m ndcg_cut.10 dl19-passage \
  run.pairwise.heapsort.txt
 
Results:
ndcg_cut_10             all     0.6571
```

`--method heapsort` does pairwise inferences with heap sort algorithm. Change to `--method bubblesort` for bubble sort algorithm. 
You can set `--method allpair` for comparing all possible pairs. In this case you can set `--batch_size` for batching inference. But `allpair` is very expensive.

We also have supervised [duoT5](https://github.com/castorini/pygaggle) pairwise ranking model implemented.
Simply set `--model_name_or_path` and `--tokenizer_name_or_path` to `castorini/duot5-3b-msmarco`, or other duoT5 models listed in [here](https://huggingface.co/castorini).

</details>

<details>
<summary><h3>Setwise</h3></summary>

Our proposed Setwise prompting can considerably speed up the sorting-based Pairwise methods. Check our paper [here](https://arxiv.org/abs/2310.09497) for more details.  

```bash
CUDA_VISIBLE_DEVICES=0 python3 run.py \
  run --model_name_or_path google/flan-t5-large \
      --tokenizer_name_or_path google/flan-t5-large \
      --run_path run.msmarco-v1-passage.bm25-default.dl19.txt \
      --save_path run.setwise.heapsort.txt \
      --ir_dataset_name msmarco-passage/trec-dl-2019 \
      --hits 100 \
      --query_length 32 \
      --passage_length 128 \
      --scoring generation \
      --device cuda \
  setwise --num_child 2 \
          --method heapsort \
          --k 10
```

```bash     
# evaluation
python -m pyserini.eval.trec_eval -c -l 2 -m ndcg_cut.10 dl19-passage \
  run.setwise.heapsort.txt
 
Results:
ndcg_cut_10             all     0.6697
```

`--num_child 2` means comparing two child node documents + one parent node document = 3 documents in total to compare in the prompt.
increasing `--num_child` will give more efficiency gain, but you may need to truncate documents more by setting a small `--passage_length`, otherwise prompt may exceed input limitation.
You can also set `--scoring likelihood` for faster inference.

We also have Openai API implementation for Setwise method:

```bash
python3 run.py \
  run --model_name_or_path gpt-3.5-turbo \
      --openai_key [your key] \
      --run_path run.msmarco-v1-passage.bm25-default.dl19.txt \
      --save_path run.setwise.heapsort.openai.txt \
      --ir_dataset_name msmarco-passage/trec-dl-2019 \
      --hits 100 \
      --query_length 32 \
      --passage_length 128 \
      --scoring generation \
  setwise --num_child 2 \
          --method heapsort \
          --k 10
```

</details>

<details>
<summary><h3>BEIR experiments</h3></summary>

For BEIR datasets experiments, change `--ir_dataset_name` to `--pyserini_index` with pyserini pre-build index.

For example:

```bash
DATASET=trec-covid # change to: trec-covid robust04 webis-touche2020 scifact signal1m trec-news dbpedia-entity nfcorpus for other experiments in the paper.

# Get BM25 first stage results
python -m pyserini.search.lucene \
  --index beir-v1.0.0-${DATASET}.flat \
  --topics beir-v1.0.0-${DATASET}-test \
  --output run.bm25.${DATASET}.txt \
  --output-format trec \
  --batch 36 --threads 12 \
  --hits 1000 --bm25 --remove-query

python -m pyserini.eval.trec_eval \
  -c -m ndcg_cut.10 beir-v1.0.0-${DATASET}-test \
  run.bm25.${DATASET}.txt

Results:
ndcg_cut_10             all     0.5947

# Setwise with heapsort
CUDA_VISIBLE_DEVICES=0 python3 run.py \
  run --model_name_or_path google/flan-t5-large \
      --tokenizer_name_or_path google/flan-t5-large \
      --run_path run.bm25.${DATASET}.txt \
      --save_path run.setwise.heapsort.${DATASET}.txt \
      --pyserini_index beir-v1.0.0-${DATASET} \
      --hits 100 \
      --query_length 32 \
      --passage_length 128 \
      --scoring generation \
      --device cuda \
  setwise --num_child 2 \
          --method heapsort \
          --k 10

python -m pyserini.eval.trec_eval \
  -c -m ndcg_cut.10 beir-v1.0.0-${DATASET}-test \
  run.setwise.heapsort.${DATASET}.txt

Results:
ndcg_cut_10             all     0.7675
```
</details>

> Note: If you remove CUDA_VISIBLE_DEVICES=0, our code should automatically perform multi-GPU inference, but we may observe slight changes in the NDCG@10 scores

---

## References
[1] Devendra Sachan, Mike Lewis, Mandar Joshi, Armen Aghajanyan, Wen-tau Yih, Joelle Pineau, and Luke Zettlemoyer. 2022. Improving Passage Retrieval with Zero-Shot Question Generation

[2] Weiwei Sun,Lingyong Yan,Xinyu Ma,Pengjie Ren,Dawei Yin,and Zhaochun Ren. 2023. Is ChatGPT Good at Search? 

[3] Shengyao Zhuang, Honglei Zhuang, Bevan Koopman, and Guido Zuccon. 2023. A Setwise Approach for Effective and Highly Efficient Zero-shot Ranking with Large Language Models

[4] Zhen Qin, Rolf Jagerman, Kai Hui, Honglei Zhuang, Junru Wu, Jiaming Shen, Tianqi Liu, Jialu Liu, Donald Metzler, Xuanhui Wang, and Michael Bendersky. 2023. Large language models are effective text rankers with pairwise ranking prompting



---
If you used our code for your research, please consider to cite our paper:

```text
@article{zhuang2021fast,
  title={A Setwise Approach for Effective and Highly Efficient Zero-shot Ranking with Large Language Models},
  author={Zhuang, Shengyao and Zhuang, Honglei and Koopman, Bevan and Zuccon, Guido},
  journal={arXiv preprint arXiv:2310.09497},
  year={2023}
}
```