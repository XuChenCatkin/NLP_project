from __future__ import annotations
from dataclasses import dataclass
from typing import List

@dataclass
class DenseConfig:
    top_k: int = 5

class DenseRetriever:
    def __init__(self, cfg: DenseConfig):
        self.cfg = cfg
        # TODO: load index lazily (path from config)

    def search(self, query: str) -> List[str]:
        # TODO: actual search
        return []
