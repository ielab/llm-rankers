"""Send one tiny request to Jev to check that a provider/key works, and show what the model returns.

    python jev/check_access.py --provider vercel      # AI_GATEWAY_API_KEY from the environment or repo-root .env
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jev_rankers import JevApiError, JevClient, choice, noul, score, TREC_DL_GRADED_LEVELS  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--provider', default='vercel', choices=list(JevClient.PROVIDERS))
    parser.add_argument('--model', default=None)
    parser.add_argument('--api_key', default=None)
    args = parser.parse_args()

    client = JevClient(provider=args.provider, api_key=args.api_key, model=args.model, max_retries=1)
    state = {'query': 'what is the capital of france',
             'passages': {'P1': 'Paris is the capital and most populous city of France.',
                          'P2': 'Berlin is the capital of Germany and its largest city.',
                          'P3': 'The Eiffel Tower is a wrought-iron lattice tower in Paris.'}}
    questions = {'most_relevant': choice('Which passage is the most relevant one to the query?',
                                         {'P1': None, 'P2': None, 'P3': None}),
                 'p1_answers': noul('Passage P1 answers the query.'),
                 'p2_grade': score('How relevant is passage P2 to the query?', TREC_DL_GRADED_LEVELS)}
    print(f'provider={client.provider} model={client.model} url={client.url}')
    tic = time.time()
    try:
        result = client.system_one(state, questions)
    except JevApiError as e:
        print('FAILED:', e)
        sys.exit(1)
    print(f'ok in {time.time() - tic:.2f}s, usage={result["usage"]}')
    print(json.dumps(result['answers'], indent=2))


if __name__ == '__main__':
    main()
