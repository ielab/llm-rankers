import os
from dataclasses import dataclass, field
from typing import Optional, Literal


@dataclass
class ExperimentArguments:
    run_path: str = field(
        default=None,
        metadata={"help": "Path to the run file."}
    )
    save_path: str = field(
        default=None,
        metadata={"help": "Path to save the results."}
    )
    pyserini_index: str = field(
        default=None,
        metadata={"help": "Pyserini pre-built index."}
    )
    pyserini_topic: str = field(
        default=None,
        metadata={"help": "Pyserini topic name."}
    )

    hits: int = field(
        default=100,
        metadata={"help": "Number of hits."}
    )

    method: Literal['pointwise', 'pairwise', 'listwise', 'setwise'] = field(
        default='setwise',
        metadata={"help": "Ranking method."}
    )

    shuffle_ranking: Optional[Literal['random', 'inverse']] = field(
        default=None,
        metadata={"help": "Shuffle the ranking."}
    )

    seed: int = field(
        default=42,
        metadata={"help": "Random seed."}
    )



@dataclass
class RankerArguments:
    prompt_file: str = field(
        default=None,
        metadata={"help": "Path to the prompt file."}
    )
    model_name_or_path: str = field(
        default=None,
        metadata={"help": "Path to the model."}
    )
    lora_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the lora model."}
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the tokenizer."}
    )
    apply_chat_template: bool = field(
        default=True,
        metadata={"help": "Apply chat template."}
    )
    max_query_length: int = field(
        default=512,
        metadata={"help": "Maximum query length."}
    )
    max_doc_length: int = field(
        default=512,
        metadata={"help": "Maximum document length."}
    )
    dtype: str = field(
        default='float16',
        metadata={"help": "Data type."}
    )
    use_vllm: bool = field(
        default=False,
        metadata={"help": "Use vllm."}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Cache directory."}
    )
    verbose: bool = field(
        default=False,
        metadata={"help": "Verbose mode."}
    )


@dataclass
class SetwiseArguments:
    num_child: int = field(
        default=3,
        metadata={"help": "Number of child models."}
    )
    scoring: Literal['likelihood', 'generation'] = field(
        default='generation',
        metadata={"help": "Scoring method."}
    )

    sort: Literal['heapsort', 'bubblesort'] = field(
        default='heapsort',
        metadata={"help": "Sorting method."}
    )

    k: int = field(
        default=10,
        metadata={"help": "Top k results."}
    )