"""Public façade for the RAG app (Streamlit + scripts)."""

from __future__ import annotations

from config_loader import load_config, merge_config
from generation import answer_query
from ingestion import (
    assign_chunk_ids,
    chunk_narrative,
    corpus_manifest,
    dedupe_chunks,
    ingest_uploaded_file,
    load_demo_chunks,
)
from retrieval import VectorStore, get_embeddings, rebuild_store_from_chunks

__all__ = [
    "load_config",
    "merge_config",
    "answer_query",
    "ingest_uploaded_file",
    "load_demo_chunks",
    "corpus_manifest",
    "dedupe_chunks",
    "assign_chunk_ids",
    "chunk_narrative",
    "VectorStore",
    "get_embeddings",
    "rebuild_store_from_chunks",
]
