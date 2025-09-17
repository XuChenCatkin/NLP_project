from __future__ import annotations
from dataclasses import dataclass
from typing import List, Any

@dataclass
class KGConfig:
    top_k: int = 5

class KGRetriever:
    def __init__(self, cfg: KGConfig):
        self.cfg = cfg

    def search(self, query: str) -> List[Any]:
        # TODO: implement actual KG retrieval
        return []
