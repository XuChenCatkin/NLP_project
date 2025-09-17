from __future__ import annotations
import logging

_DEF_FMT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format=_DEF_FMT)
