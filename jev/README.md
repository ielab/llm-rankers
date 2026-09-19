# Jev (TypeSafe AI) as a pointwise / pairwise / setwise / listwise re-ranker

[Jev](https://typesafe.ai/) is TypeSafe AI's "System One" model. It does **not** generate text: a request is a JSON
`state` plus a dict of *typed questions*, and every answer is a calibrated probability distribution.

| question type | what it returns |
|---|---|
| `noul` | P(yes) for a yes/no statement |
| `choice` | a distribution over up to 255 named options (sums to 1) plus the argmax |
| `score` | a distribution over 2–10 ordered rubric levels plus its expected value |

All questions of one request are answered in a single parallel pass, so a request can carry 100 passages and 100
questions. Limits (jev-1.13): 64k tokens per request, 32k for state + longest question, text only. List price $0.042
per million input tokens, output free.

## Results

TREC DL 2019 (43 queries) / 2020 (54 queries) passage ranking, re-ranking the pyserini BM25 top-100
(k1=0.9, b=0.4, the same first stage as the main README), nDCG@10. Every Jev run is zero-shot, one run each, no prompt
tuning. Baseline numbers are copied from the RankGPT paper [1] (Sun et al., EMNLP 2023, Table 1) and the RankZephyr
paper [2] (Pradeep et al., 2023, Table 5, BM25 first stage), both window 20 / step 10 over the same BM25 top-100.
API latency is the mean HTTP round trip per request during the runs (12 queries in parallel, at most 20 requests/s);
cost is the list price of the input tokens per query.

| Method | DL19 | DL20 | Calls / query | API latency (s) | Cost / query |
|---|---|---|---|---|---|
| **Baselines** | | | | | |
| BM25 (first stage) | 0.506 | 0.480 | – | – | – |
| monoBERT-340M [1] | 0.705 | 0.673 | 100 | – | – |
| RankGPT gpt-3.5 [1] | 0.658 | 0.629 | 9 | – | – |
| RankGPT gpt-4 [1] | 0.756 | 0.706 | 9 | – | – |
| RankZephyr-7B [2] | 0.742 | 0.709 | 9 | – | – |
| **Jev (typesafe-ai/jev)** | | | | | |
| Jev pointwise, noul | 0.693 | 0.674 | 100 | 0.42 | $0.0018 |
| Jev pointwise, score | 0.728 | 0.691 | 100 | 0.43 | $0.0020 |
| Jev pairwise, heapsort | 0.733 | 0.720 | 512 | 0.44 | $0.011 |
| Jev setwise, heapsort c=10 | 0.719 | 0.714 | 28 | 0.57 | $0.0017 |
| Jev listwise w20/s10, choice | 0.663 | 0.635 | 9 | 0.57 | $0.0009 |
| Jev listwise w20/s10, score | 0.737 | 0.709 | 9 | 0.81 | $0.0017 |
| Jev listwise 100-in-1, choice | 0.656 | 0.643 | 1 | 0.82 | $0.0004 |
| Jev listwise 100-in-1, score | 0.728 | 0.711 | 1 | 1.13 | $0.0009 |

### What the numbers say

* **Graded `score` questions beat `choice` questions everywhere** (pointwise 0.728 vs 0.693, listwise 0.737 vs 0.663
  on DL19). A `choice` returns one distribution that sums to 1 over the window and the API rounds probabilities to
  two decimals, so most of a 20- or 100-passage window ties at 0.00 and keeps its BM25 order; a `score` question
  gives every passage its own fine-grained expected relevance level.
* **All 100 candidates in one request works.** Listwise `score` with 100 passages and 100 questions in a single
  request (about 19k input tokens, 1.1 s, $0.0009 per query) is within 0.01 of the 9-request sliding window on DL19
  and slightly better on DL20. This is only possible because Jev answers all questions of a request in one pass and
  charges nothing for output.
* **Pointwise `score` (100 requests) is close to listwise `score` (9 requests)**: seeing the other candidates in the
  same request adds little; the graded rubric (instead of a yes/no question) is what matters.
* **Pairwise heapsort** is as effective as the best listwise variant but needs about 512 requests per query.
* **Against the baselines**, Jev listwise `score` (0.737 / 0.709) matches RankZephyr-7B (0.742 / 0.709) and is just
  below RankGPT gpt-4 on DL19 (0.756) and above it on DL20 (0.706), at 9 requests of 0.6–0.8 s and $0.0017 per query,
  with no GPU and no generated tokens.

### The probabilities are calibrated (roughly)

The pointwise runs store one number per query-passage pair, so they can be checked against the NIST graded
judgments (5,116 judged pairs over DL19 + DL20). P(yes) is monotone in the human grade and roughly calibrated against
"grade >= 2", over-confident in the 0.6–0.9 range; the expected graded level is monotone too but about half a level
optimistic. For ranking the ordering is what matters; for thresholding, the numbers can be used almost as they are.

Pointwise `noul`, P("the passage answers the query") vs. judgment:

| P(yes) | passages | share with grade >= 2 | mean grade (0-3) |
|---|---|---|---|
| 0.0 – 0.1 | 2250 | 0.04 | 0.28 |
| 0.1 – 0.2 | 558 | 0.17 | 0.70 |
| 0.2 – 0.3 | 296 | 0.22 | 0.84 |
| 0.3 – 0.4 | 140 | 0.27 | 0.95 |
| 0.4 – 0.5 | 155 | 0.30 | 1.08 |
| 0.5 – 0.6 | 170 | 0.41 | 1.21 |
| 0.6 – 0.7 | 171 | 0.44 | 1.33 |
| 0.7 – 0.8 | 177 | 0.50 | 1.44 |
| 0.8 – 0.9 | 290 | 0.56 | 1.60 |
| 0.9 – 1.0 | 909 | 0.80 | 2.17 |

Pointwise `score`, expected level (sum of level x probability over the 4 TREC-DL levels) vs. judgment:

| expected level | passages | share with grade >= 2 | mean grade (0-3) |
|---|---|---|---|
| 0.0 – 0.5 | 427 | 0.01 | 0.04 |
| 0.5 – 1.0 | 1095 | 0.02 | 0.16 |
| 1.0 – 1.5 | 1283 | 0.12 | 0.61 |
| 1.5 – 2.0 | 632 | 0.25 | 0.94 |
| 2.0 – 2.5 | 615 | 0.52 | 1.47 |
| 2.5 – 3.0 | 993 | 0.76 | 2.04 |
| 3.0 – 3.5 | 71 | 0.89 | 2.59 |

### Jev is not deterministic

Re-running DL19 pointwise `noul` gave the identical P(yes) for only 61% of the 4,300 query-passage pairs
(mean |diff| 0.008, max 0.17; nDCG@10 0.6927 vs 0.6884); for pointwise `score` only 24% of the expected levels were
identical (mean |diff| 0.024, max 0.53; nDCG@10 0.7279 vs 0.7221). Single-run numbers therefore carry about ±0.005
of noise (`compare_runs.py`).

## How the four prompting methods become Jev questions

| method | state | question | ranking signal |
|---|---|---|---|
| pointwise `--method noul` | `{query, passage}` | `noul` "The passage answers the query." | P(yes) (analogue of the repo's `yes_no`) |
| pointwise `--method score` | `{query, passage}` | `score` with the 4 TREC-DL graded levels | expected level (0–3) |
| pairwise | `{query, passage_A, passage_B}`, both orders | `choice` {A, B} "Which passage is more relevant?" | A wins if the mean of P(A first) and 1 − P(B first) is >= 0.5; heap sort for the top-k |
| setwise | `{query, passages: {P1..P11}}` | `choice` {P1..P11} "Which passage is the most relevant one?" | argmax; heap sort with 10 children per node |
| listwise `--mode choice` | window of 20 passages | one `choice` over the window | sort by P(most relevant) (analogue of the repo's `--scoring likelihood`) |
| listwise `--mode score` | window | one `score` question per passage, all in one request | sort by expected level |

`--window_size 100 --step_size 100` turns listwise into a single request per query (all 100 candidates, 100 questions).

A real listwise `score` request (window of 5 for readability) and its answer:

```jsonc
// request body
{"state": {"query": "how long is life cycle of flea",
           "passages": {"P1": "5. Cancel. A flea can live up to a year, ...", "P2": "The life cycle of a flea can last anywhere from 20 days to an entire year. ...", ...}},
 "questions": {"P1": {"type": "score", "instructions": "How relevant is passage P1 to the query?",
                      "criteria": ["Irrelevant: the passage has nothing to do with the query.",
                                   "Related: the passage seems related to the query but does not answer it.",
                                   "Highly relevant: the passage has some answer for the query, but the answer may be a bit unclear, or hidden amongst extraneous information.",
                                   "Perfectly relevant: the passage is dedicated to the query and contains the exact answer."]},
               "P2": {...}, ...}}
// answers (score = expected level = sum of level x probability)
{"P1": {"type": "score", "score": 2.02, "probabilities": {"0": 0, "1": 0.06, "2": 0.86, "3": 0.08}},
 "P2": {"type": "score", "score": 2.93, "probabilities": {"0": 0, "1": 0, "2": 0.07, "3": 0.93}}, ...}
```

Pointwise sends its 100 requests concurrently (`--num_workers`); heap sort and the sliding window are sequential
within a query (pairwise asks the two orders of a pair in parallel). Independent queries run concurrently
(`--query_workers`, default 12) and `--max_rps` caps the total request rate (default 20/s, TypeSafe allows ~1200/min).

## Reproduce

### Access

The official sign-up at typesafe.ai is an early-access waitlist (`console.typesafe.ai`, `TYPESAFE_API_KEY`). Two
resellers expose the same model without a waitlist and are supported by `--provider`:

| `--provider` | endpoint | model id | key |
|---|---|---|---|
| `vercel` (default, used here) | `POST https://ai-gateway.vercel.sh/v4/ai/evaluation-model` | `typesafe-ai/jev` | `AI_GATEWAY_API_KEY` |
| `openrouter` | `POST https://openrouter.ai/api/alpha/decisions` | `typesafe/jev-1.13` | `OPENROUTER_API_KEY` |
| `typesafe` | `POST https://api.typesafe.ai/v1/systemone` | `jev-latest` | `TYPESAFE_API_KEY` |

Vercel AI Gateway, step by step:

1. Vercel dashboard → *AI Gateway* → *API Keys* → *Create key*; save it as `AI_GATEWAY_API_KEY` (a `.env` file in the
   repo root is read automatically and is git-ignored).
2. Add a credit card to the team (until then every request fails with `403 customer_verification_required`; the
   error carries the link `https://vercel.com/<team>/~/ai?modal=add-credit-card`).
3. Buy a few dollars of AI Gateway Credits (`https://vercel.com/<team>/~/ai?modal=top-up`). With a card but no
   credits the account is on the free tier and Jev is not a free-tier model: the first request works, the rest get
   `429 Free tier requests on this model are rate-limited`. The paid tier has no gateway rate limit and no markup.
4. `python jev/check_access.py --provider vercel` sends one tiny request and prints the typed answers.

Vercel documents evaluation only through the TypeScript AI SDK (`experimental_evaluate`); `jev_rankers.py` calls the
REST endpoint the SDK calls (`@ai-sdk/gateway`, `gateway-evaluation-model.ts`): body `{state, questions}`, headers
`ai-gateway-protocol-version: 0.0.1`, `ai-gateway-auth-method: api-key`,
`ai-evaluation-model-specification-version: 4`, `ai-model-id: typesafe-ai/jev`; the yes/no type is called `boolean`
there and returns `probability`. OpenRouter and TypeSafe use TypeSafe's own schema (`noul`).

### Setup and data

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r jev/requirements.txt
# or: pip install -r jev/requirements.txt   (requests, tiktoken, ir_datasets, ir_measures, tqdm; no torch, no Java)
echo 'AI_GATEWAY_API_KEY=vck_...' > .env && chmod 600 .env
bash jev/prepare_data.sh                        # BM25 top-100 with passage texts for DL19 / DL20
.venv/bin/python jev/check_access.py --provider vercel
```

`prepare_data.sh` downloads the pyserini BM25 (`msmarco-v1-passage`, k1=0.9, b=0.4) top-100 candidates *with passage
texts* from [castorini/rank_llm_data](https://huggingface.co/datasets/castorini/rank_llm_data) and converts them
(`convert_rank_llm.py`) into `jev/runs/bm25/run.rank_llm.bm25.dl19.top100.txt` + `docs.dl19.top100.tsv` (same for
dl20). The DL19 run is identical, position by position, to the `run.msmarco-v1-passage.bm25-default.dl19.txt` of the
main README (nDCG@10 0.5058). The 3 GB MS MARCO collection is never downloaded; queries and qrels come from
`ir_datasets`.

### Run and evaluate

```bash
bash jev/run_dl.sh dl19          # every method of the table on DL19, then nDCG@10 (about 30 minutes)
bash jev/run_dl.sh dl20
python jev/make_table.py         # the Jev rows of the table from jev/runs/jev/*.stats.json
python jev/calibration.py        # the calibration tables from the pointwise run files
python jev/test_sorters.py       # oracle test of the heap sort / sliding window code
```

One method at a time (same two-level CLI as `../run.py`):

```bash
.venv/bin/python jev/run_jev.py \
  run --provider vercel \
      --run_path jev/runs/bm25/run.rank_llm.bm25.dl19.top100.txt \
      --docs_file jev/runs/bm25/docs.dl19.top100.tsv \
      --ir_dataset_name msmarco-passage/trec-dl-2019/judged \
      --save_path jev/runs/jev/dl19.setwise.heapsort.c10.txt \
      --hits 100 --query_length 32 --passage_length 128 --num_workers 4 --query_workers 12 --max_rps 20 \
  setwise --num_child 10 --k 10
#   pointwise --method noul|score
#   pairwise  --k 10
#   listwise  --window_size 20 --step_size 10 --mode choice|score
#   listwise  --window_size 100 --step_size 100 --mode score      (all 100 candidates in one request)

.venv/bin/python jev/eval_run.py --dataset msmarco-passage/trec-dl-2019/judged jev/runs/bm25/run.rank_llm.bm25.dl19.top100.txt jev/runs/jev/dl19.*.txt
```

Every run writes `<save_path>.stats.json` with the number of comparisons, API calls, input tokens, estimated cost,
retries and the per-request HTTP latency (mean / p50 / p95 / max). Truncation to `--query_length` /
`--passage_length` uses tiktoken `cl100k_base` as an approximation; Jev's tokenizer is not public.

## Notes

* Jev reads instructions literally and gets distracted by unrelated state
  (TypeSafe's [jaggedness page](https://docs.typesafe.ai/model-jaggedness/jev-1.13)); position bias of `choice`
  over many passages was not measured.
* The API returns probabilities rounded to two decimals.
* Two bugs in the original sorting code were found while porting it (perfect-oracle simulation, `test_sorters.py`)
  and fixed in `llmrankers/`: the setwise bubble-sort `last_start` optimisation could skip comparisons (wrong top-k for
  large `num_child` or `k` close to `n`), and the listwise sliding window never reached the top of the list when
  `(n - window_size) % step_size != 0` or `n < window_size`. The default paper settings were not affected.
