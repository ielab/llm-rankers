"""Jev (TypeSafe AI "System One" model) as a pointwise / pairwise / setwise / listwise re-ranker.

Jev does not generate text. A request is a JSON ``state`` plus a dict of typed questions; every answer is a
probability distribution:

* ``noul``   - yes/no, returns P(yes)
* ``choice`` - one option out of up to 255, returns a distribution over the options (sums to 1)
* ``score``  - 2-10 ordered rubric levels, returns a distribution over the levels and its expected value

Ranking prompts become questions (see README.md for the exact mapping). The sorting code (heapsort, sliding
window) is ported from ``llmrankers/`` and checked against a perfect oracle in ``test_sorters.py``.

Providers (same ``state`` / ``questions`` schema; Vercel names the yes/no type ``boolean``):

=========== ================================================================ ==============================
provider    endpoint                                                          key (env or repo-root .env)
=========== ================================================================ ==============================
vercel      POST https://ai-gateway.vercel.sh/v4/ai/evaluation-model          AI_GATEWAY_API_KEY
openrouter  POST https://openrouter.ai/api/alpha/decisions                    OPENROUTER_API_KEY
typesafe    POST https://api.typesafe.ai/v1/systemone                         TYPESAFE_API_KEY
=========== ================================================================ ==============================
"""
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List

import requests

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from llmrankers.rankers import SearchResult  # noqa: E402

JEV_INPUT_USD_PER_MTOK = 0.042  # list price; output tokens are free


# ------------------------------------------------------------------------------------------------ questions
def noul(instructions, true=None, false=None):
    q = {"type": "noul", "instructions": instructions}
    if true is not None or false is not None:
        q["criteria"] = {"true": true, "false": false}
    return q


def choice(instructions, criteria: Dict[str, object]):
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


def score(instructions, levels: List[object]):
    return {"type": "score", "instructions": instructions, "criteria": list(levels)}


# TREC DL passage relevance scale, lowest level first.
TREC_DL_GRADED_LEVELS = [
    "Irrelevant: the passage has nothing to do with the query.",
    "Related: the passage seems related to the query but does not answer it.",
    "Highly relevant: the passage has some answer for the query, but the answer may be a bit unclear, "
    "or hidden amongst extraneous information.",
    "Perfectly relevant: the passage is dedicated to the query and contains the exact answer.",
]


class JevApiError(RuntimeError):
    pass


# ------------------------------------------------------------------------------------------------ client
def load_dotenv(paths=None):
    """Minimal .env loader (KEY=VALUE lines); never overrides variables that are already set."""
    if paths is None:
        paths = [os.path.join(os.getcwd(), '.env'), os.path.join(_REPO_ROOT, '.env')]
    for path in paths:
        if not os.path.isfile(path):
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                key, value = line.split('=', 1)
                key, value = key.strip(), value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value


class JevClient:
    PROVIDERS = ('vercel', 'openrouter', 'typesafe')
    DEFAULT_MODELS = {'vercel': 'typesafe-ai/jev', 'openrouter': 'typesafe/jev-1.13', 'typesafe': 'jev-latest'}
    ENV_KEYS = {'vercel': 'AI_GATEWAY_API_KEY', 'openrouter': 'OPENROUTER_API_KEY', 'typesafe': 'TYPESAFE_API_KEY'}
    URLS = {'vercel': 'https://ai-gateway.vercel.sh/v4/ai/evaluation-model',
            'openrouter': 'https://openrouter.ai/api/alpha/decisions',
            'typesafe': 'https://api.typesafe.ai/v1/systemone'}
    RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}

    def __init__(self, provider='vercel', api_key=None, model=None, timeout=60, max_retries=12, max_rps=None,
                 verbose=False, on_call=None):
        """``max_rps`` meters all threads to at most that many requests per second (TypeSafe allows ~1200/min);
        ``on_call(elapsed_seconds)`` runs after every completed request (progress bars)."""
        if provider not in self.PROVIDERS:
            raise ValueError(f'Unknown provider {provider!r}, choose from {self.PROVIDERS}')
        load_dotenv()
        self.provider = provider
        self.model = model or self.DEFAULT_MODELS[provider]
        self.url = self.URLS[provider]
        self.api_key = api_key or os.environ.get(self.ENV_KEYS[provider])
        if not self.api_key:
            raise ValueError(f'No API key: pass api_key or set {self.ENV_KEYS[provider]} (env or .env).')
        self.timeout, self.max_retries, self.max_rps, self.verbose, self.on_call = \
            timeout, max_retries, max_rps, verbose, on_call

        self.session = requests.Session()
        self._lock = threading.Lock()
        self._rate_lock = threading.Lock()
        self._next_slot = 0.0
        self.total_calls = self.total_retries = self.total_input_tokens = self.total_output_tokens = 0
        self.latencies = []  # HTTP round-trip seconds of each successful request

    def _headers(self):
        h = {'Content-Type': 'application/json', 'Authorization': f'Bearer {self.api_key}'}
        if self.provider == 'vercel':  # what @ai-sdk/gateway sends for `experimental_evaluate`
            h.update({'ai-gateway-protocol-version': '0.0.1', 'ai-gateway-auth-method': 'api-key',
                      'ai-evaluation-model-specification-version': '4', 'ai-model-id': self.model})
        return h

    def _body(self, state, questions):
        if self.provider == 'vercel':
            qs = {name: dict(q, type='boolean') if q['type'] == 'noul' else q for name, q in questions.items()}
            return {'state': state, 'questions': qs}
        return {'model': self.model, 'state': state, 'questions': questions}

    @staticmethod
    def _parse(data):
        """Normalise to TypeSafe's answer format: {'answers': {name: {...}}, 'usage': {...}}."""
        answers = data.get('answers') if isinstance(data, dict) else None
        if not isinstance(answers, dict):
            raise JevApiError(f'Malformed response: {str(data)[:500]}')
        norm = {}
        for name, a in answers.items():
            norm[name] = {'type': 'noul', 'noul': a['probability']} if a.get('type') == 'boolean' else dict(a)
        usage = data.get('usage') or {}
        return {'answers': norm,
                'usage': {'input_tokens': int(usage.get('input_tokens', usage.get('inputTokens')) or 0),
                          'output_tokens': int(usage.get('output_tokens', usage.get('outputTokens')) or 0)}}

    def system_one(self, state, questions: Dict[str, dict]) -> dict:
        result, elapsed = self._request(state, questions)
        with self._lock:
            self.total_calls += 1
            self.total_input_tokens += result['usage']['input_tokens']
            self.total_output_tokens += result['usage']['output_tokens']
            self.latencies.append(elapsed)
        if self.on_call is not None:
            self.on_call(elapsed)
        return result

    def estimated_cost_usd(self):
        return self.total_input_tokens / 1e6 * JEV_INPUT_USD_PER_MTOK

    def latency_summary(self):
        if not self.latencies:
            return {}
        xs = sorted(self.latencies)
        n = len(xs)
        pct = lambda p: xs[min(n - 1, int(round(p * (n - 1))))]  # noqa: E731
        return {'calls': n, 'mean_s': sum(xs) / n, 'p50_s': pct(0.5), 'p95_s': pct(0.95), 'max_s': xs[-1]}

    def _throttle(self):
        if not self.max_rps:
            return
        with self._rate_lock:
            now = time.monotonic()
            wait = max(0.0, self._next_slot - now)
            self._next_slot = max(now, self._next_slot) + 1.0 / self.max_rps
        if wait > 0:
            time.sleep(wait)

    def _request(self, state, questions):
        body, headers, last_err = self._body(state, questions), self._headers(), None
        for attempt in range(self.max_retries + 1):
            self._throttle()
            try:
                t0 = time.time()
                r = self.session.post(self.url, json=body, headers=headers, timeout=self.timeout)
                elapsed = time.time() - t0
            except requests.RequestException as e:
                last_err = f'connection error: {e}'
                self._sleep(attempt)
                continue
            if r.status_code == 429 and 'free tier' in r.text.lower():
                raise JevApiError(self._explain(r))  # not transient: the account needs paid credits
            if r.status_code in self.RETRY_STATUS:
                last_err = f'HTTP {r.status_code}: {r.text[:300]}'
                with self._lock:
                    self.total_retries += 1
                if self.verbose:
                    print(f'[jev] retrying after {last_err}')
                self._sleep(attempt, r.headers.get('Retry-After'), cap=5.0 if r.status_code == 429 else 30.0)
                continue
            if r.status_code >= 400:
                raise JevApiError(self._explain(r))
            return self._parse(r.json()), elapsed
        raise JevApiError(f'Giving up after {self.max_retries + 1} attempts. Last error: {last_err}')

    def _explain(self, r):
        msg = f'HTTP {r.status_code} from {self.provider}: {r.text[:800]}'
        if self.provider == 'vercel':
            if 'customer_verification_required' in r.text:
                msg += '\n  -> Vercel AI Gateway needs a credit card on file (dashboard > AI Gateway).'
            elif r.status_code == 402 or 'free tier' in r.text.lower():
                msg += '\n  -> typesafe-ai/jev is not a free-tier model: buy AI Gateway credits.'
        return msg

    @staticmethod
    def _sleep(attempt, retry_after=None, cap=30.0):
        try:
            delay = float(retry_after) if retry_after else None
        except ValueError:
            delay = None
        if delay is None:
            delay = min(cap, 0.5 * (2 ** attempt)) + random.uniform(0, 0.25)
        time.sleep(delay)


# ------------------------------------------------------------------------------------------------ sorting
def _finish(original: List[SearchResult], ordered: List[SearchResult], k: int) -> List[SearchResult]:
    """Top-k of ``ordered`` first (score = -rank), the remaining docs in their original order."""
    results, top, rank = [], set(), 1
    for doc in ordered[:k]:
        top.add(doc.docid)
        results.append(SearchResult(docid=doc.docid, score=-rank, text=None))
        rank += 1
    for doc in original:
        if doc.docid not in top:
            results.append(SearchResult(docid=doc.docid, score=-rank, text=None))
            rank += 1
    return results


def heapsort_pairwise(docs: List, k: int, first_wins: Callable) -> List:
    """Top-k by heap sort with a pairwise comparison ``first_wins(a, b) -> bool`` (port of llmrankers/pairwise.py)."""
    arr, n = list(docs), len(docs)

    def heapify(size, i):
        largest, l, r = i, 2 * i + 1, 2 * i + 2
        if l < size and first_wins(arr[l], arr[i]):
            largest = l
        if r < size and first_wins(arr[r], arr[largest]):
            largest = r
        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]
            heapify(size, largest)

    for i in range(n // 2, -1, -1):
        heapify(n, i)
    for ranked, i in enumerate(range(n - 1, 0, -1), 1):
        arr[i], arr[0] = arr[0], arr[i]
        if ranked == k:
            break
        heapify(i, 0)
    return _finish(docs, list(reversed(arr)), k)


def heapsort_setwise(docs: List, k: int, num_child: int, best_index: Callable) -> List:
    """Top-k by a ``num_child``-ary heap sort; ``best_index(window) -> int`` picks the most relevant doc of a
    parent + children window (port of llmrankers/setwise.py)."""
    arr, n, c = list(docs), len(docs), num_child

    def heapify(size, i):
        if c * i + 1 < size:
            inds = [i] + list(range(c * i + 1, min(c * (i + 1) + 1, size)))
            best = best_index([arr[j] for j in inds])
            largest = inds[best] if 0 <= best < len(inds) else i
            if largest != i:
                arr[i], arr[largest] = arr[largest], arr[i]
                heapify(size, largest)

    for i in range(n // c, -1, -1):
        heapify(n, i)
    for ranked, i in enumerate(range(n - 1, 0, -1), 1):
        arr[i], arr[0] = arr[0], arr[i]
        if ranked == k:
            break
        heapify(i, 0)
    return _finish(docs, list(reversed(arr)), k)


def sliding_window(docs: List, window_size: int, step_size: int, num_repeat: int, order_fn: Callable) -> List:
    """RankGPT-style sliding window from the bottom to the top; ``order_fn(window) -> indices, best first``.
    Unlike the original loop the top window is always processed, whatever ``len(docs)`` and ``step_size``."""
    ranking, n = list(docs), len(docs)
    for _ in range(num_repeat):
        end, start = n, max(0, n - window_size)
        while True:
            window = ranking[start:end]
            order = []
            for j in order_fn(window):
                if 0 <= j < len(window) and j not in order:
                    order.append(j)
            order += [j for j in range(len(window)) if j not in order]
            ranking[start:end] = [window[j] for j in order]
            if start == 0:
                break
            start, end = start - step_size, end - step_size
            if start < 0:
                start, end = 0, min(n, window_size)
    return ranking


# ------------------------------------------------------------------------------------------------ rankers
class _Tokenizer:
    """tiktoken cl100k_base as an approximation for --query_length / --passage_length (Jev's tokenizer is not public)."""

    def __init__(self):
        import tiktoken
        self._enc = tiktoken.get_encoding('cl100k_base')

    def truncate(self, text, length):
        return self._enc.decode(self._enc.encode(text)[:length])


class JevRanker:
    def __init__(self, client: JevClient, num_workers=4):
        self.client = client
        self.tokenizer = _Tokenizer()
        self._pool = ThreadPoolExecutor(max_workers=max(2, num_workers))
        self.total_compare = self.total_prompt_tokens = self.total_completion_tokens = 0

    def _begin(self):
        self.total_compare = 0
        self._tok0 = (self.client.total_input_tokens, self.client.total_output_tokens)

    def _end(self):
        self.total_prompt_tokens = self.client.total_input_tokens - self._tok0[0]
        self.total_completion_tokens = self.client.total_output_tokens - self._tok0[1]

    def truncate(self, text, length):
        return self.tokenizer.truncate(text, length)


def _labels(n):
    return [f'P{i + 1}' for i in range(n)]


def _passages_state(query, docs, labels):
    return {'query': query, 'passages': {label: doc.text for label, doc in zip(labels, docs)}}


class JevPointwiseLlmRanker(JevRanker):
    """One request per (query, passage), sent concurrently.
    method='noul':     P("the passage answers the query")
    method='score':    expected TREC graded level (0-3)
    method='cookbook': the noul of TypeSafe's re-ranking cookbook (docs.typesafe.ai/cookbooks/rerank_typesafe),
                       adapted from legal citations to web search: state keys `query` / `candidate_passage`, the
                       question phrased with its context, criteria contrasting a specific answer with a similar topic."""

    def __init__(self, client, method='noul', num_workers=4):
        super().__init__(client, num_workers)
        if method not in ('noul', 'score', 'cookbook'):
            raise ValueError("method must be 'noul', 'score' or 'cookbook'")
        self.method = method

    def _score_one(self, query, doc):
        if self.method == 'cookbook':
            state = {'query': query, 'candidate_passage': doc.text}
            qs = {'relevant': noul(
                'The query is a web search query typed by a user looking for a specific piece of information. '
                'Could the candidate passage be the passage a search engine should return for it - does it provide '
                'the specific information the query asks for?',
                true='The candidate passage states or explains the specific fact, answer or procedure the query asks for.',
                false='The candidate passage is merely on a similar topic; it does not supply the specific '
                      'information the query asks for.')}
            return self.client.system_one(state, qs)['answers']['relevant']['noul']
        state = {'query': query, 'passage': doc.text}
        if self.method == 'noul':
            qs = {'relevant': noul('The passage answers the query.',
                                   true='The passage contains the information the query is asking for.',
                                   false='The passage does not answer the query, even if it is on a related topic.')}
            return self.client.system_one(state, qs)['answers']['relevant']['noul']
        qs = {'relevance': score('How relevant is the passage to the query?', TREC_DL_GRADED_LEVELS)}
        return self.client.system_one(state, qs)['answers']['relevance']['score']

    def rerank(self, query: str, ranking: List[SearchResult]) -> List[SearchResult]:
        self._begin()
        for doc, s in zip(ranking, self._pool.map(lambda d: self._score_one(query, d), ranking)):
            doc.score = float(s)
        self.total_compare = len(ranking)
        self._end()
        return sorted(ranking, key=lambda x: x.score, reverse=True)


class JevPairwiseLlmRanker(JevRanker):
    """Heap sort with a `choice` {A, B} question asked in both orders; A wins if the mean of P(A first) and
    1 - P(B first) is >= 0.5."""

    def __init__(self, client, k=10, num_workers=2):
        super().__init__(client, num_workers)
        self.k = k

    def _p_first(self, query, text_a, text_b):
        state = {'query': query, 'passage_A': text_a, 'passage_B': text_b}
        qs = {'more_relevant': choice('Which of the two passages is more relevant to the query?',
                                      {'A': 'Passage A is more relevant to the query.',
                                       'B': 'Passage B is more relevant to the query.'})}
        return float(self.client.system_one(state, qs)['answers']['more_relevant']['probabilities'].get('A', 0.0))

    def _first_wins(self, a, b):
        self.total_compare += 1
        f1 = self._pool.submit(self._p_first, self._query, a.text, b.text)
        f2 = self._pool.submit(self._p_first, self._query, b.text, a.text)
        return (f1.result() + 1.0 - f2.result()) / 2.0 >= 0.5

    def rerank(self, query: str, ranking: List[SearchResult]) -> List[SearchResult]:
        self._begin()
        self._query = query
        results = heapsort_pairwise(ranking, self.k, self._first_wins)
        self._end()
        return results


class JevSetwiseLlmRanker(JevRanker):
    """Heap sort where each parent + num_child window is one `choice` question over its passages."""

    def __init__(self, client, num_child=10, k=10, num_workers=2):
        super().__init__(client, num_workers)
        self.num_child, self.k = num_child, k

    def _best_index(self, docs):
        if len(docs) <= 1:
            return 0
        self.total_compare += 1
        labels = _labels(len(docs))
        qs = {'most_relevant': choice('Which passage is the most relevant one to the query?', {l: None for l in labels})}
        probs = self.client.system_one(_passages_state(self._query, docs, labels), qs)['answers']['most_relevant']['probabilities']
        return max(range(len(docs)), key=lambda i: float(probs.get(labels[i], 0.0)))

    def rerank(self, query: str, ranking: List[SearchResult]) -> List[SearchResult]:
        self._begin()
        self._query = query
        results = heapsort_setwise(ranking, self.k, self.num_child, self._best_index)
        self._end()
        return results


class JevListwiseLlmRanker(JevRanker):
    """Sliding window (RankGPT style). mode='choice': one `choice` over the window, sorted by probability;
    mode='score': one `score` question per passage in the same request, sorted by expected level.
    window_size=100, step_size=100 evaluates all 100 candidates in a single request."""

    def __init__(self, client, window_size=20, step_size=10, num_repeat=1, mode='choice', num_workers=2):
        super().__init__(client, num_workers)
        if mode not in ('choice', 'score'):
            raise ValueError("mode must be 'choice' or 'score'")
        self.window_size, self.step_size, self.num_repeat, self.mode = window_size, step_size, num_repeat, mode

    def _order(self, docs):
        self.total_compare += 1
        labels = _labels(len(docs))
        state = _passages_state(self._query, docs, labels)
        if self.mode == 'choice':
            qs = {'most_relevant': choice('Which passage is the most relevant one to the query?', {l: None for l in labels})}
            probs = self.client.system_one(state, qs)['answers']['most_relevant']['probabilities']
            values = [float(probs.get(l, 0.0)) for l in labels]
        else:
            qs = {l: score(f'How relevant is passage {l} to the query?', TREC_DL_GRADED_LEVELS) for l in labels}
            answers = self.client.system_one(state, qs)['answers']
            values = [float(answers[l]['score']) for l in labels]
        return sorted(range(len(docs)), key=lambda i: (-values[i], i))

    def rerank(self, query: str, ranking: List[SearchResult]) -> List[SearchResult]:
        self._begin()
        self._query = query
        ordered = sliding_window(ranking, self.window_size, self.step_size, self.num_repeat, self._order)
        self._end()
        return [SearchResult(docid=doc.docid, score=-i, text=None) for i, doc in enumerate(ordered)]
