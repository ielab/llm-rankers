from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class SearchResult:
    docid: str
    score: float
    text: str


class LlmRanker:
    def rerank(self,  query: str, ranking: List[SearchResult]) -> Tuple[str, List[SearchResult]]:
        raise NotImplementedError

    def truncate(self, text, length):
        raise NotImplementedError