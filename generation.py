from __future__ import annotations

import logging
from typing import Any

from llm_client import get_openai_client

from retrieval import VectorStore, get_embeddings

logger = logging.getLogger(__name__)


def rewrite_query_for_retrieval(query: str, model: str) -> str:
    client = get_openai_client()
    res = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": (
                    "Rewrite the user's question into a short keyword-rich search query "
                    "for retrieving document passages (8–18 words). "
                    "Reply with the query only, no quotes.\n\n"
                    f"Question:\n{query}"
                ),
            }
        ],
    )
    out = (res.choices[0].message.content or "").strip()
    return out or query


def _format_context_block(i: int, chunk: dict[str, Any]) -> str:
    parts = [f"[{i}]"]
    src = chunk.get("source", "?")
    parts.append(f"File: {src}")
    if chunk.get("sheet"):
        parts.append(f"Sheet: {chunk['sheet']}")
    if chunk.get("row_range"):
        parts.append(f"Rows: {chunk['row_range']}")
    if chunk.get("page"):
        parts.append(f"Page: {chunk['page']}")
    parts.append(f"Kind: {chunk.get('kind', 'unknown')}")
    parts.append(f"Cosine≈{chunk.get('dense_similarity', 0)}")
    parts.append("")
    parts.append(chunk.get("text", "").strip())
    return "\n".join(parts)


def answer_query(
    query: str,
    vector_store: VectorStore,
    cfg: dict[str, Any],
    *,
    strict_grounding: bool,
    use_query_rewrite: bool,
    temperature: float | None = None,
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    client = get_openai_client()
    temp = float(temperature if temperature is not None else cfg.get("temperature", 0.15))
    emb_model = str(cfg.get("embedding_model", "text-embedding-3-small"))
    batch = int(cfg.get("embedding_batch_size", 256))
    chat_model = str(cfg.get("chat_model", "gpt-4o-mini"))
    rewrite_model = str(cfg.get("rewrite_model", chat_model))
    max_chars = int(cfg.get("max_context_chars", 7000))
    weak_thr = float(cfg.get("weak_retrieval_threshold", 0.32))

    retrieval_query = query
    rewritten: str | None = None
    if use_query_rewrite or bool(cfg.get("query_rewrite")):
        try:
            rewritten = rewrite_query_for_retrieval(query, rewrite_model)
            retrieval_query = rewritten
            logger.info("query_rewrite: %r -> %r", query, retrieval_query)
        except Exception as e:  # noqa: BLE001
            logger.warning("query rewrite failed: %s", e)
            retrieval_query = query

    q_emb = get_embeddings([retrieval_query], emb_model, batch)
    results = vector_store.search(retrieval_query, q_emb, cfg=cfg)

    meta: dict[str, Any] = {
        "retrieval_query": retrieval_query,
        "original_query": query,
        "rewritten": rewritten,
        "strict_grounding": strict_grounding,
        "weak_retrieval": False,
        "best_dense_similarity": 0.0,
    }

    if not results:
        meta["weak_retrieval"] = True
        return (
            "No passages met the similarity threshold. Try lowering the minimum cosine in settings, "
            "rephrasing your question, or uploading richer documents.",
            [],
            meta,
        )

    meta["best_dense_similarity"] = max(r.get("dense_similarity", 0.0) for r in results)
    if meta["best_dense_similarity"] < weak_thr:
        meta["weak_retrieval"] = True

    context_parts: list[str] = []
    total = 0
    for i, r in enumerate(results, start=1):
        block = _format_context_block(i, r)
        if total + len(block) > max_chars:
            break
        context_parts.append(block)
        total += len(block)

    context = "\n\n---\n\n".join(context_parts)

    if strict_grounding:
        system = (
            "You are a careful analyst. Answer ONLY using the numbered context passages. "
            "Every factual claim must include inline citations like [1] or [2] pointing to those numbers. "
            "If the context is insufficient, say exactly: "
            "\"The uploaded documents do not contain enough information to answer.\" "
            "Do not use outside knowledge. Do not invent numbers or names not present in the context."
        )
        user = f"Context passages:\n\n{context}\n\nQuestion:\n{query}\n\nAnswer with citations:"
    else:
        system = (
            "You are an analyst. Prioritize the numbered context passages and cite them inline as [1], [2], etc. "
            "when you use them. If you add general knowledge not in the context, add a short line starting with "
            "\"General knowledge:\" and keep it clearly separate. If support is weak, say so explicitly."
        )
        user = f"Context passages:\n\n{context}\n\nQuestion:\n{query}\n\nAnswer:"

    res = client.chat.completions.create(
        model=chat_model,
        temperature=temp,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    answer = (res.choices[0].message.content or "").strip()

    if meta["weak_retrieval"]:
        answer += (
            "\n\n_Retrieval confidence: moderate/low — the best matching passages were only weakly "
            "similar to your question; verify against the sources below._"
        )

    return answer, results, meta
