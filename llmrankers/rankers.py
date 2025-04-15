from dataclasses import dataclass
from typing import List, Tuple
import os
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import toml

@dataclass
class SearchResult:
    docid: str
    score: float
    text: str


class LlmRanker:
    TRANSFORMER_CLS = AutoModelForCausalLM

    def __init__(self,
                 prompt_file,
                 model_name_or_path,
                 lora_name_or_path=None,
                 tokenizer_name_or_path=None,
                 apply_chat_template=True,
                 max_query_length=512,
                 max_doc_length=512,
                 dtype='float16',
                 use_vllm=False,
                 cache_dir=None,
                 verbose=False
                 ):

        self.prompt = toml.load(prompt_file)
        self.max_query_length = max_query_length
        self.max_doc_length = max_doc_length
        self.labels = self.prompt['labels']

        self.lora_name_or_path = lora_name_or_path
        if tokenizer_name_or_path is None:
            tokenizer_name_or_path = model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, cache_dir=cache_dir)

        if use_vllm:
            from vllm import LLM, SamplingParams
            from vllm.lora.request import LoRARequest
            from huggingface_hub import snapshot_download
            if lora_name_or_path is not None:
                # check if the path exists
                if not os.path.exists(lora_name_or_path):
                    # download the model
                    lora_path = snapshot_download(lora_name_or_path)
                else:
                    lora_path = lora_name_or_path
            else:
                lora_path = None

            self.lora_path = lora_path
            self.model = LLM(model=model_name_or_path,
                             tokenizer=tokenizer_name_or_path,
                             enable_lora=True if lora_name_or_path is not None else False,
                             max_lora_rank=32,
                             dtype=dtype,
                           )
        else:
            from peft import PeftModel

            self.config = AutoConfig.from_pretrained(model_name_or_path, cache_dir=cache_dir)
            self.model = self.TRANSFORMER_CLS.from_pretrained(model_name_or_path,
                                                   device_map="auto",
                                                   torch_dtype=dtype,
                                                   config=self.config,
                                                   cache_dir=cache_dir)
            if lora_name_or_path is not None:
                self.model = PeftModel.from_pretrained(self.model, lora_name_or_path)
                self.model = self.model.merge_and_unload()
            self.model = self.model.eval()

        self.apply_chat_template = apply_chat_template
        self.verbose = verbose
        self.total_compare = 0
        self.total_completion_tokens = 0
        self.total_prompt_tokens = 0

    def format_input_text(self, query: str, docs: List[str]) -> str:
        def format_input_text(self, query: str, docs: List[str]) -> str:
            docs = self.prompt['doc_separator'].join(
                [f'{self.prompt["doc_prefix"].format(label=self.labels[i])}{docs[i]}' for i in range(len(docs))]
            )
            user_message = self.prompt['user'].format(query=query,
                                                      docs=docs)
            if self.apply_chat_template:
                message = []
                if 'system' in self.prompt:
                    message.append({
                        'role': 'system',
                        'content': self.prompt['system']
                    })
                message.append({
                    'role': 'user',
                    'content': user_message
                })

                input_text = self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
                if 'assistant_prefix' in self.prompt:
                    input_text += self.prompt['assistant_prefix']
            else:
                input_text = ''
                if 'system' in self.prompt:
                    input_text += self.prompt['system']
                input_text += self.prompt['user'].format(query=query,
                                                         docs=docs)
                if 'assistant_prefix' in self.prompt:
                    input_text += self.prompt['assistant_prefix']

            return input_text

    def rerank(self,  query: str, ranking: List[SearchResult]) -> Tuple[str, List[SearchResult]]:
        raise NotImplementedError

    def truncate(self, text, length):
        raise NotImplementedError