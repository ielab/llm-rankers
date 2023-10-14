from typing import List, Tuple
from .rankers import LlmRanker, SearchResult
import openai
import time
import re
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoConfig, AutoModelForCausalLM, AutoTokenizer
import torch
import copy
import tiktoken


class SetwiseLlmRanker(LlmRanker):
    CHARACTERS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L",
                  "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W"]  # "Passage X" and "Passage Y" will be tokenized into 3 tokens, so we dont use for now

    def __init__(self, model_name_or_path, tokenizer_name_or_path, device, cache_dir, num_child=3,
                 scoring='generation', method="heapsort"):

        self.device = device
        self.num_child = num_child
        self.config = AutoConfig.from_pretrained(model_name_or_path, cache_dir=cache_dir)
        if self.config.model_type == 't5':
            self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_name_or_path
                                                         if tokenizer_name_or_path is not None else
                                                         model_name_or_path,
                                                         cache_dir=cache_dir)
            self.llm = T5ForConditionalGeneration.from_pretrained(model_name_or_path,
                                                                  device_map='auto',
                                                                  torch_dtype=torch.float16 if device == 'cuda'
                                                                  else torch.float32,
                                                                  cache_dir=cache_dir)
            self.decoder_input_ids = self.tokenizer.encode("<pad> Passage",
                                                           return_tensors="pt",
                                                           add_special_tokens=False).to(self.device) if self.tokenizer else None

            test = []
            for i in range(len(self.CHARACTERS)):
                test.append(f'<pad> Passage {self.CHARACTERS[i]}')

            self.target_token_ids = self.tokenizer.batch_encode_plus([f'<pad> Passage {self.CHARACTERS[i]}'
                                                                      for i in range(len(self.CHARACTERS))],
                                                                     return_tensors="pt",
                                                                     add_special_tokens=False,
                                                                     padding=True).input_ids[:, -1]
        elif self.config.model_type == 'llama':
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)
            self.llm = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                            device_map='auto',
                                                            torch_dtype=torch.float16 if device == 'cuda'
                                                            else torch.float32,
                                                            cache_dir=cache_dir)
            self.system_prompt = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\
"""
        self.scoring = scoring
        self.method = method
        self.total_compare = 0
        self.total_completion_tokens = 0
        self.total_prompt_tokens = 0

    def compare(self, query: str, docs: List):
        self.total_compare += 1
        passages = "\n\n".join([f'Passage {self.CHARACTERS[i]}: "{doc.text}"' for i, doc in enumerate(docs)])

        if self.scoring == 'generation':
            input_text = f'Given a query "{query}", which of the following passages is the most relevant one to the query?\n\n' \
                         + passages + '\n\nOutput only the passage label of the most relevant passage:'
 
            input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
            self.total_prompt_tokens += input_ids.shape[1]

            if self.config.model_type == 't5':
                output_ids = self.llm.generate(input_ids,
                                               decoder_input_ids=self.decoder_input_ids,
                                               max_new_tokens=2)[0]

                self.total_completion_tokens += output_ids.shape[0]

                output = self.tokenizer.decode(output_ids,
                                               skip_special_tokens=True).strip()
                output = output[-1]

            elif self.config.model_type == 'llama':
                input_text = f'Given a query "{query}", which of the following passages is the most relevant one to the query?\n\n' \
                             + passages + '\n\nOutput only the passage label of the most relevant passage:'

                input_text = f'<s>[INST] <<SYS>>\n{self.system_prompt}\n<</SYS>>\n\n{input_text} [/INST] ' \
                             f'Based on the query "{query}", the most relevant passage is:\nPassage'
                input_ids = self.tokenizer(input_text, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
                self.total_prompt_tokens += input_ids.shape[1]

                output_ids = self.llm.generate(input_ids,
                                               do_sample=False,
                                               temperature=0.0,
                                               top_p=None,
                                               max_new_tokens=1)[0]
                output = self.tokenizer.decode(output_ids[input_ids.shape[1]:],
                                               skip_special_tokens=True).strip()
            else:
                raise NotImplementedError

        elif self.scoring == 'likelihood':
            input_text = f'Given a query "{query}", which of the following passages is the most relevant one to the query?\n\n' \
                         + passages + '\n\nOutput only the passage label of the most relevant passage:'
            if self.config.model_type == 't5':
                input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
                self.total_prompt_tokens += input_ids.shape[1]
                with torch.no_grad():
                    logits = self.llm(input_ids=input_ids, decoder_input_ids=self.decoder_input_ids).logits[0][-1]
                    distributions = torch.softmax(logits, dim=0)
                    scores = distributions[self.target_token_ids[:len(docs)]]
                    ranked = sorted(zip(self.CHARACTERS[:len(docs)], scores), key=lambda x: x[1], reverse=True)
                    output = ranked[0][0]

            else:
                raise NotImplementedError

        if len(output) == 1 and output in self.CHARACTERS:
            pass
        else:
            print(f"Unexpected output: {output}")
            output = self.CHARACTERS[0]

        return output

    def heapify(self, arr, n, i, query):
        # Find largest among root and children
        if self.num_child * i + 1 < n:  # if there are children
            docs = [arr[i]] + arr[self.num_child * i + 1: min((self.num_child * (i + 1) + 1), n)]
            inds = [i] + list(range(self.num_child * i + 1, min((self.num_child * (i + 1) + 1), n)))
            output = self.compare(query, docs)
            try:
                best_ind = self.CHARACTERS.index(output)
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

    def rerank(self,  query: str, ranking: List[SearchResult], k=10) -> List[SearchResult]:
        original_ranking = copy.deepcopy(ranking)
        self.total_compare = 0
        self.total_completion_tokens = 0
        self.total_prompt_tokens = 0
        if self.method == "heapsort":
            self.heapSort(ranking, query, k)
            ranking = list(reversed(ranking))

        # elif self.method == "bubblesort":
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
        elif self.method == "bubblesort":
            last_start = len(ranking) - (self.num_child + 1)

            for i in range(k):
                start_ind = last_start
                end_ind = last_start + (self.num_child + 1)
                is_change = False
                while True:
                    if start_ind < i:
                        start_ind = i
                    output = self.compare(query, ranking[start_ind:end_ind])
                    try:
                        best_ind = self.CHARACTERS.index(output)
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

        else:
            raise NotImplementedError(f'Method {self.method} is not implemented.')

        results = []
        top_doc_ids = set()
        rank = 1

        for i, doc in enumerate(ranking[:k]):
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


class OpenAiSetwiseLlmRanker(SetwiseLlmRanker):
    def __init__(self, model_name_or_path, api_key, num_child=3, method='heapsort'):
        self.llm = model_name_or_path
        self.tokenizer = tiktoken.encoding_for_model(model_name_or_path)
        self.num_child = num_child
        self.method = method
        self.total_compare = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        openai.api_key = api_key

    def compare(self, query: str, docs: List):
        self.total_compare += 1
        passages = "\n\n".join([f'Passage {self.CHARACTERS[i]}: "{doc.text}"' for i, doc in enumerate(docs)])
        input_text = f'Given a query "{query}", which of the following passages is the most relevant one to the query?\n\n' \
                     + passages + '\n\nOutput only the passage label of the most relevant passage:'

        while True:
            try:
                response = openai.ChatCompletion.create(
                    model=self.llm,
                    messages=[
                        {"role": "user",
                         "content": input_text},
                    ],
                    temperature=0.0,
                    max_tokens=100,
                    request_timeout=15
                )
                self.total_completion_tokens += int(response['usage']['completion_tokens'])
                self.total_prompt_tokens += int(response['usage']['prompt_tokens'])

                output = response['choices'][0]['message']['content']
                matches = re.findall(r"(Passage [A-Z])", output, re.MULTILINE)
                if matches:
                    output = matches[0][8]
                elif output.strip() in self.CHARACTERS:
                    pass
                else:
                    print(f"Unexpected output: {output}")
                    output = "A"
                return output

            except openai.error.APIError as e:
                # Handle API error here, e.g. retry or log
                print(f"OpenAI API returned an API Error: {e}")
                time.sleep(5)
                continue
            except openai.error.APIConnectionError as e:
                # Handle connection error here
                print(f"Failed to connect to OpenAI API: {e}")
                time.sleep(5)
                continue
            except openai.error.RateLimitError as e:
                # Handle rate limit error (we recommend using exponential backoff)
                print(f"OpenAI API request exceeded rate limit: {e}")
                time.sleep(5)
                continue
            except openai.error.InvalidRequestError as e:
                # Handle invalid request error
                print(f"OpenAI API request was invalid: {e}")
                raise e
            except openai.error.AuthenticationError as e:
                # Handle authentication error
                print(f"OpenAI API request failed authentication: {e}")
                raise e
            except openai.error.Timeout as e:
                # Handle timeout error
                print(f"OpenAI API request timed out: {e}")
                time.sleep(5)
                continue
            except Exception as e:
                print(f"Unknown error: {e}")
                raise e

    def truncate(self, text, length):
        return self.tokenizer.decode(self.tokenizer.encode(text)[:length])
