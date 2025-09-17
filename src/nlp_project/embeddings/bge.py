from __future__ import annotations
from typing import List
try:
    from transformers import AutoTokenizer, AutoModel
    import torch
except Exception:
    AutoTokenizer = AutoModel = torch = None  # type: ignore

class BGEEmbedder:
    """Tiny wrapper; replace with your code during migration."""
    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5") -> None:
        self.model_name = model_name
        if AutoTokenizer is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
        else:
            self.tokenizer = self.model = None

    def encode(self, texts: List[str]):  # type: ignore
        assert self.tokenizer and self.model, "Transformers not installed"
        with torch.no_grad():  # type: ignore
            tokens = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
            out = self.model(**tokens)
            return out.last_hidden_state.mean(dim=1)  # simple pooling
