from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from llm_client import get_openai_client

logger = logging.getLogger(__name__)


def get_embeddings(texts: list[str], model: str, batch_size: int = 256) -> np.ndarray:
    if not texts:
        return np.zeros((0, 0), dtype="float32")
    client = get_openai_client()
    rows: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        res = client.embeddings.create(model=model, input=batch)
        batch_emb: list[list[float] | None] = [None] * len(batch)
        for d in res.data:
            batch_emb[d.index] = d.embedding
        rows.extend([e for e in batch_emb if e is not None])
    arr = np.array(rows, dtype="float32")
    faiss.normalize_L2(arr)
    return arr


def _bm25_tokenize(text: str) -> list[str]:
    return text.lower().split()


def rrf_fuse(rank_lists: list[list[int]], k_rrf: float = 60.0) -> list[int]:
    scores: dict[int, float] = {}
    for ranks in rank_lists:
        for rank, idx in enumerate(ranks):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k_rrf + rank + 1)
    return sorted(scores.keys(), key=lambda i: -scores[i])


def mmr_select(
    query_emb: np.ndarray,
    candidate_order: list[int],
    embeddings: np.ndarray,
    k: int,
    lambda_: float,
) -> list[int]:
    q = np.asarray(query_emb, dtype="float32").reshape(-1)
    cand = [i for i in candidate_order if 0 <= i < len(embeddings)]
    selected: list[int] = []
    remaining = list(cand)
    sim_q_all = embeddings @ q

    while len(selected) < k and remaining:
        best_i: int | None = None
        best_val = -1e9
        for i in remaining:
            rel = float(sim_q_all[i])
            if not selected:
                score = rel
            else:
                sub = embeddings[selected]
                div = float(np.max(sub @ embeddings[i]))
                score = lambda_ * rel - (1.0 - lambda_) * div
            if score > best_val:
                best_val = score
                best_i = i
        if best_i is None:
            break
        selected.append(best_i)
        remaining.remove(best_i)
    return selected


class VectorStore:
    """FAISS inner-product (cosine on L2-normalized rows) + optional BM25 hybrid + MMR."""

    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.chunks: list[dict[str, Any]] = []
        self.embeddings = np.zeros((0, dim), dtype="float32")
        self._bm25: BM25Okapi | None = None

    def _sync_bm25(self) -> None:
        tokenized = [_bm25_tokenize(c.get("text", "")) for c in self.chunks]
        if tokenized:
            self._bm25 = BM25Okapi(tokenized)
        else:
            self._bm25 = None

    def add(self, embeddings: np.ndarray, chunks: list[dict[str, Any]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")
        em = embeddings.astype("float32", copy=False)
        faiss.normalize_L2(em)
        self.index.add(em)
        self.chunks.extend(chunks)
        self.embeddings = em if self.embeddings.size == 0 else np.vstack([self.embeddings, em])
        self._sync_bm25()

    def _dense_search(self, query_emb: np.ndarray, topn: int) -> tuple[list[int], dict[int, float]]:
        q = query_emb.astype("float32").reshape(1, -1)
        faiss.normalize_L2(q)
        ntotal = int(self.index.ntotal)
        if ntotal == 0:
            return [], {}
        k = min(max(1, topn), ntotal)
        sims, idxs = self.index.search(q, k)
        order: list[int] = []
        sim_map: dict[int, float] = {}
        for sim, ix in zip(sims[0], idxs[0]):
            ix = int(ix)
            if ix < 0 or ix >= len(self.chunks):
                continue
            order.append(ix)
            sim_map[ix] = float(sim)
        return order, sim_map

    def _bm25_rank(self, query_text: str, topn: int) -> list[int]:
        if not self._bm25:
            return []
        toks = _bm25_tokenize(query_text)
        scores = self._bm25.get_scores(toks)
        idx_sorted = np.argsort(-scores)[:topn]
        return [int(i) for i in idx_sorted if scores[int(i)] > 0]

    def dense_similarity(self, query_emb: np.ndarray, chunk_index: int) -> float:
        q = np.asarray(query_emb, dtype="float32").reshape(-1)
        faiss.normalize_L2(q.reshape(1, -1))
        if chunk_index < 0 or chunk_index >= len(self.embeddings):
            return 0.0
        row = self.embeddings[chunk_index].astype("float32", copy=False)
        return float(np.dot(row, q))

    def search(
        self,
        query_text: str,
        query_embedding: np.ndarray,
        *,
        cfg: dict[str, Any],
    ) -> list[dict[str, Any]]:
        if self.index.ntotal == 0:
            return []

        pool = int(cfg.get("retrieval_pool", 60))
        final_k = int(cfg.get("final_k", 8))
        min_sim = float(cfg.get("min_cosine_similarity", 0.2))
        use_hybrid = bool(cfg.get("use_hybrid", True))
        rrf_k = float(cfg.get("rrf_k", 60))
        bm25_top = int(cfg.get("bm25_top_n", 80))
        use_mmr = bool(cfg.get("use_mmr", True))
        mmr_lambda = float(cfg.get("mmr_lambda", 0.55))

        dense_order, sim_map = self._dense_search(query_embedding, pool)
        dense_filtered = [i for i in dense_order if sim_map.get(i, 0.0) >= min_sim]

        if use_hybrid and self._bm25 and len(self.chunks) > 0:
            bm25_order = self._bm25_rank(query_text, bm25_top)
            list_a = dense_filtered if dense_filtered else dense_order
            fused = rrf_fuse([list_a, bm25_order], rrf_k)
            candidates_all = [i for i in fused if 0 <= i < len(self.chunks)]
        else:
            candidates_all = dense_filtered if dense_filtered else dense_order

        if not candidates_all:
            candidates_all = dense_order[: max(final_k, 8)]

        seen: set[int] = set()
        candidates: list[int] = []
        for i in candidates_all:
            if i in seen:
                continue
            seen.add(i)
            candidates.append(i)
            if len(candidates) >= pool:
                break

        if use_mmr and len(candidates) > 1:
            picked = mmr_select(
                query_embedding,
                candidates,
                self.embeddings,
                min(final_k, len(candidates)),
                mmr_lambda,
            )
        else:
            picked = candidates[:final_k]

        results: list[dict[str, Any]] = []
        for i in picked:
            row = deepcopy(self.chunks[i])
            row["_chunk_index"] = i
            row["dense_similarity"] = round(self.dense_similarity(query_embedding, i), 4)
            results.append(row)

        best_dense = max((r["dense_similarity"] for r in results), default=0.0)
        logger.info(
            "retrieval ntotal=%s hybrid=%s candidates=%s picked=%s best_dense=%.4f",
            self.index.ntotal,
            use_hybrid,
            len(candidates),
            len(results),
            best_dense,
        )
        return results


def rebuild_store_from_chunks(
    chunks: list[dict[str, Any]],
    cfg: dict[str, Any],
) -> VectorStore:
    texts = [c["text"] for c in chunks]
    emb = get_embeddings(
        texts,
        str(cfg.get("embedding_model", "text-embedding-3-small")),
        int(cfg.get("embedding_batch_size", 256)),
    )
    if emb.size == 0:
        raise ValueError("No embeddings produced")
    vs = VectorStore(emb.shape[1])
    vs.add(emb, chunks)
    return vs
