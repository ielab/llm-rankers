from typing import List
import random
from .arguments import SetwiseArguments, RankerArguments

from .rankers import LlmRanker, SearchResult
import re
from transformers import T5ForConditionalGeneration, AutoModelForCausalLM
import torch
import copy
import logging
logger = logging.getLogger(__name__)


class SetwiseLlmRanker(LlmRanker):
    TRANSFORMER_CLS = AutoModelForCausalLM
    def __init__(self,
                 ranker_args: RankerArguments,
                 setwise_args: SetwiseArguments):

        super().__init__(ranker_args)

        self.num_child = setwise_args.num_child
        self.k = setwise_args.k
        self.sort = setwise_args.sort
        if self.use_vllm:
            from vllm import SamplingParams
            from vllm.lora.request import LoRARequest
            self.sampling_params = SamplingParams(temperature=0.0,
                                                  max_tokens=2048)
            if self.lora_path is not None:
                self.lora_request = LoRARequest("R1adapter", 1, self.lora_path)

    def compare(self, query: str, docs: List[SearchResult]):
        self.total_compare += 1

        id_passage = [(i, p) for i, p in zip(self.labels, docs)]
        id_passage = random.sample(id_passage, len(id_passage)) # shuffle the label-passage pairs
        ids = [i for i, p in id_passage]
        input_text = self.format_input_text(query=query, docs=[p for i, p in id_passage])

        if self.scoring == 'generation':
            if self.use_vllm:
                input_ids = self.tokenizer.encode(input_text, add_special_tokens=False)
                self.total_prompt_tokens += len(input_ids)
                output = self.model.generate(
                    prompt_token_ids=input_ids,
                    sampling_params=self.sampling_params,
                    use_tqdm=False,
                    lora_request=self.lora_request if self.lora_path is not None else None,
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
                result = self.labels[0]

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

        result = ids[self.labels.index(result)] # return the real label (before shuffle)
        return result


    def heapify(self, arr, n, i, query):
        # Find largest among root and children
        if self.num_child * i + 1 < n:  # if there are children
            docs = [arr[i]] + arr[self.num_child * i + 1: min((self.num_child * (i + 1) + 1), n)]
            inds = [i] + list(range(self.num_child * i + 1, min((self.num_child * (i + 1) + 1), n)))
            output = self.compare(query, docs)
            try:
                best_ind = self.labels.index(output)
            except ValueError:
                best_ind = 0
            try:
                largest = inds[best_ind]
            except IndexError:
                largest = i
            # If root is not largest, swap with largest and continue heapifying
            if largest != i:
                arr[i], arr[largest] = arr[largest], arr[i]
                self.heapify(arr, n, largest, query)

    def heapSort(self, arr, query, k):
        n = len(arr)
        ranked = 0
        # Build max heap
        for i in range(n // self.num_child, -1, -1):
            self.heapify(arr, n, i, query)
        for i in range(n - 1, 0, -1):
            # Swap
            arr[i], arr[0] = arr[0], arr[i]
            ranked += 1
            if ranked == k:
                break
            # Heapify root element
            self.heapify(arr, i, 0, query)

    def rerank(self,  query: str, ranking: List[SearchResult]) -> List[SearchResult]:
        original_ranking = copy.deepcopy(ranking)

        ranking = [SearchResult(docid=doc.docid, text=self.truncate(doc.text, self.max_doc_length), score=doc.score)
                   for doc in ranking]

        query = self.truncate(query, self.max_query_length)

        self.total_compare = 0
        self.total_completion_tokens = 0
        self.total_prompt_tokens = 0

        if self.sort == "heapsort":
            self.heapSort(ranking, query, self.k)
            ranking = list(reversed(ranking))
        elif self.sort == "bubblesort":
            last_start = len(ranking) - (self.num_child + 1)

            for i in range(self.k):
                start_ind = last_start
                end_ind = last_start + (self.num_child + 1)
                is_change = False
                while True:
                    if start_ind < i:
                        start_ind = i
                    output = self.compare(query, ranking[start_ind:end_ind])
                    try:
                        best_ind = self.labels.index(output)
                    except ValueError:
                        best_ind = 0
                    if best_ind != 0:
                        ranking[start_ind], ranking[start_ind + best_ind] = ranking[start_ind + best_ind], ranking[start_ind]
                        if not is_change:
                            is_change = True
                            if last_start != len(ranking) - (self.num_child + 1) \
                                    and best_ind == len(ranking[start_ind:end_ind])-1:
                                last_start += len(ranking[start_ind:end_ind])-1

                    if start_ind == i:
                        break

                    if not is_change:
                        last_start -= self.num_child

                    start_ind -= self.num_child
                    end_ind -= self.num_child
                    
        ##  this is a bit slower but standard bobblesort implementation, keep here FYI
        # elif self.sort == "bubblesort":
        #     for i in range(k):
        #         start_ind = len(ranking) - (self.num_child + 1)
        #         end_ind = len(ranking)
        #         while True:
        #             if start_ind < i:
        #                 start_ind = i
        #             output = self.compare(query, ranking[start_ind:end_ind])
        #             try:
        #                 best_ind = self.CHARACTERS.index(output)
        #             except ValueError:
        #                 best_ind = 0
        #             if best_ind != 0:
        #                 ranking[start_ind], ranking[start_ind + best_ind] = ranking[start_ind + best_ind], ranking[start_ind]
        #
        #             if start_ind == i:
        #                 break
        #
        #             start_ind -= self.num_child
        #             end_ind -= self.num_child

        else:
            raise NotImplementedError(f'Method {self.sort} is not implemented.')

        results = []
        top_doc_ids = set()
        rank = 1

        for i, doc in enumerate(ranking[:self.k]):
            top_doc_ids.add(doc.docid)
            results.append(SearchResult(docid=doc.docid, score=-rank, text=None))
            rank += 1
        for doc in original_ranking:
            if doc.docid not in top_doc_ids:
                results.append(SearchResult(docid=doc.docid, score=-rank, text=None))
                rank += 1

        return results

    def truncate(self, text, length):
        return self.tokenizer.convert_tokens_to_string(self.tokenizer.tokenize(text)[:length])


class SetwiseT5Ranker(SetwiseLlmRanker):
    TRANSFORMER_CLS = T5ForConditionalGeneration

    def __init__(self,
                 ranker_args: RankerArguments,
                 setwise_args: SetwiseArguments):
        super().__init__(ranker_args, setwise_args)

        self.decoder_input_ids = self.tokenizer.encode(self.prompt['assistant_prefix'],
                                                       return_tensors="pt",
                                                       add_special_tokens=False).to(self.model.device)

        self.target_token_ids = self.tokenizer.batch_encode_plus([f'{self.prompt["assistant_prefix"]}{self.labels[i]}'
                                                                  for i in range(len(self.labels))],
                                                                 return_tensors="pt",
                                                                 add_special_tokens=False,
                                                                 padding=True).input_ids[:, -1]


    def format_input_text(self, query: str, docs: List[SearchResult]) -> str:
        docs = self.prompt['doc_separator'].join(
            [f'{self.prompt["doc_prefix"].format(label=self.labels[i])}{docs[i].text}' for i in range(len(docs))]
        )
        input_text = self.prompt['user'].format(query=query,
                                                docs=docs)
        return input_text


    def compare(self, query: str, docs: List[SearchResult]):
        self.total_compare += 1

        input_text = self.format_input_text(query=query, docs=docs)
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.model.device)
        self.total_prompt_tokens += input_ids.shape[1]

        if self.scoring == 'generation':
            output_ids = self.model.generate(input_ids,
                                             decoder_input_ids=self.decoder_input_ids if self.decoder_input_ids is not None else None,
                                             max_new_tokens=5,
                                             do_sample=False)[0]


            self.total_completion_tokens += output_ids.shape[0]

            output = self.tokenizer.decode(output_ids,
                                           skip_special_tokens=False).strip()

            pattern = rf'{self.prompt["pattern"]}'
            match = re.search(pattern, output, re.DOTALL)
            if match:
                result = match.group(1).strip()
            else:
                logger.warning(f"Pattern '{pattern}' not found in output '{output}'.")
                result = self.labels[0]


        elif self.scoring == 'likelihood':
            with torch.no_grad():
                logits = self.model(input_ids=input_ids, decoder_input_ids=self.decoder_input_ids).logits[0][-1]
                scores = logits[self.target_token_ids[:len(docs)]]
                ranked = sorted(zip(self.labels[:len(docs)], scores), key=lambda x: x[1], reverse=True)
                output = ranked[0][0]
                result = output

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