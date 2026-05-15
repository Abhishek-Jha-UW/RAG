"""
Lightweight offline check: loads sample_data, runs retrieval for golden queries,
and prints whether retrieved chunk texts contain expected hints.

Usage (from project root):
  set OPENAI_API_KEY=...   # Windows: $env:OPENAI_API_KEY='...'
  python eval/eval_retrieval.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_loader import load_config
from ingestion import load_demo_chunks
from retrieval import rebuild_store_from_chunks, get_embeddings


def main() -> None:
    cfg = load_config(ROOT / "config.yaml")
    chunks, errs = load_demo_chunks(
        ROOT / "sample_data",
        int(cfg["tabular_rows_per_chunk"]),
        chunk_words=int(cfg["chunk_words"]),
        chunk_overlap=int(cfg["chunk_overlap"]),
    )
    if errs:
        print("warnings:", errs)
    vs = rebuild_store_from_chunks(chunks, cfg)
    golden = json.loads((ROOT / "eval" / "golden.json").read_text(encoding="utf-8"))
    for row in golden:
        q = row["query"]
        hints: list[str] = row.get("relevant_sources_substring", [])
        emb = get_embeddings([q], str(cfg["embedding_model"]), int(cfg["embedding_batch_size"]))
        hits = vs.search(q, emb, cfg=cfg)
        text_blob = " ".join(str(h.get("source", "")) + " " + h.get("text", "") for h in hits).lower()
        ok = all(h.lower() in text_blob for h in hints)
        print("---")
        print("Q:", q)
        print("hit_ok:", ok, "n_hits:", len(hits))
        for i, h in enumerate(hits[:3], 1):
            print(f"  [{i}] sim={h.get('dense_similarity')} src={h.get('source')} :: {h.get('text','')[:160]}...")


if __name__ == "__main__":
    main()
