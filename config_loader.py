from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

_DEFAULT_PATH = Path(__file__).resolve().parent / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path or os.environ.get("RAG_CONFIG_PATH", _DEFAULT_PATH))
    if not p.is_file():
        return _defaults()
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    base = _defaults()
    base.update({k: v for k, v in data.items() if v is not None})
    return base


def _defaults() -> dict[str, Any]:
    return {
        "embedding_model": "text-embedding-3-small",
        "chat_model": "gpt-4o-mini",
        "rewrite_model": "gpt-4o-mini",
        "chunk_words": 300,
        "chunk_overlap": 50,
        "tabular_rows_per_chunk": 45,
        "retrieval_pool": 80,
        "final_k": 8,
        "min_cosine_similarity": 0.2,
        "use_hybrid": True,
        "rrf_k": 60,
        "bm25_top_n": 80,
        "use_mmr": True,
        "mmr_lambda": 0.55,
        "query_rewrite": False,
        "max_context_chars": 7000,
        "embedding_batch_size": 256,
        "temperature": 0.12,
        "weak_retrieval_threshold": 0.32,
    }


def merge_config(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for k, v in overrides.items():
        if v is not None:
            out[k] = v
    return out
