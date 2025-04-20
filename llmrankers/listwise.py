from typing import List
import random
from .arguments import ListwiseArguments, RankerArguments

from .rankers import LlmRanker, SearchResult
import re
from transformers import AutoModelForCausalLM
import torch
import copy
import logging
logger = logging.getLogger(__name__)

def clean_response(response: str):
    new_response = ''
    for c in response:
        if not c.isdigit():
            new_response += ' '
        else:
            new_response += c
    new_response = new_response.strip()
    return new_response


def remove_duplicate(response):
    new_response = []
    for c in response:
        if c not in new_response:
            new_response.append(c)
    return new_response

def receive_permutation(ranking, permutation, rank_start=0, rank_end=100):
    response = clean_response(permutation)
    response = [int(x) - 1 for x in response.split()]
    response = remove_duplicate(response)
    cut_range = copy.deepcopy(ranking[rank_start: rank_end])
    original_rank = [tt for tt in range(len(cut_range))]
    response = [ss for ss in response if ss in original_rank]
    response = response + [tt for tt in original_rank if tt not in response]

    for j, x in enumerate(response):
        ranking[j + rank_start] = cut_range[x]

    return ranking

class ListwiseLlmRanker(LlmRanker):
    TRANSFORMER_CLS = AutoModelForCausalLM
    def __init__(self,
                 ranker_args: RankerArguments,
                 listwise_args: ListwiseArguments):

        super().__init__(ranker_args)

        self.window_size = listwise_args.window_size
        self.step_size = listwise_args.step_size
        self.num_repeat = listwise_args.num_repeat

    def compare(self, query: str, docs: List[SearchResult]):
        self.total_compare += 1
        input_text = self.format_input_text(query=query, docs=docs)

        if self.scoring == 'generation':
            if self.use_vllm:
                input_ids = self.tokenizer.encode(input_text, add_special_tokens=False)
                self.total_prompt_tokens += len(input_ids)
                output = self.model.generate(
                    prompt_token_ids=input_ids,
                    sampling_params=self.sampling_params,
                    use_tqdm=False,
                    lora_request=self.lora_request if self.lora_name_or_path is not None else None,
                )
                self.total_completion_tokens += len(output[0].outputs[0].token_ids)
                output = output[0].outputs[0].text

            else:
                input_ids = self.tokenizer(input_text,
                                           return_tensors="pt",
                                           add_special_tokens=False).input_ids.to(self.model.device)
                self.total_prompt_tokens += input_ids.shape[1]
                output_ids = self.model.generate(input_ids,
                                                 max_new_tokens=2048,
                                                 do_sample=False,
                                                 )[0]

                output_ids = output_ids[input_ids.shape[1]:]
                self.total_completion_tokens += output_ids.shape[0]

                output = self.tokenizer.decode(output_ids,
                                               skip_special_tokens=True).strip()

            pattern = rf'{self.prompt["pattern"]}'
            match = re.search(pattern, output, re.DOTALL)
            if match:
                result = match.group(1).strip()
            else:
                logger.warning(f"Pattern '{pattern}' not found in output '{output}'.")
                result = 'None'

        elif self.scoring == 'likelihood':
            raise NotImplementedError(f"Scoring method {self.scoring} is not implemented yet for setwise ranking.")

        else:
            raise NotImplementedError(f"Scoring method {self.scoring} is not implemented yet for setwise ranking.")
        if self.verbose:
            print('--------------------------------------')
            print(f'query:\n"{query}"')
            print(f'input_text:\n"{input_text}"')
            print(f'completion:\n"{output}"')
            print(f'match:\n"{result}"')
            print('--------------------------------------')


        return result


    def rerank(self,  query: str, ranking: List[SearchResult]) -> List[SearchResult]:
        self.total_compare = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

        ranking = [SearchResult(docid=doc.docid, text=self.truncate(doc.text, self.max_doc_length), score=doc.score)
                   for doc in ranking]

        for _ in range(self.num_repeat):
            ranking = copy.deepcopy(ranking)
            end_pos = len(ranking)
            start_pos = end_pos - self.window_size
            while start_pos >= 0:
                start_pos = max(start_pos, 0)
                result = self.compare(query, ranking[start_pos: end_pos])
                ranking = receive_permutation(ranking, result, start_pos, end_pos)
                end_pos = end_pos - self.step_size
                start_pos = start_pos - self.step_size

        for i, doc in enumerate(ranking):
            doc.score = -i
        return ranking

    def truncate(self, text, length):
        return self.tokenizer.convert_tokens_to_string(self.tokenizer.tokenize(text)[:length])
