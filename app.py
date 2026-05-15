from __future__ import annotations

import logging
from pathlib import Path

import streamlit as st

from config_loader import load_config, merge_config
from generation import answer_query
from ingestion import assign_chunk_ids, corpus_manifest, dedupe_chunks, ingest_uploaded_file
from llm_client import get_openai_api_key
from retrieval import rebuild_store_from_chunks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_app")

ROOT = Path(__file__).resolve().parent

st.set_page_config(page_title="Analyst RAG Copilot", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "all_chunks" not in st.session_state:
    st.session_state.all_chunks = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "pending_user" not in st.session_state:
    st.session_state.pending_user = None
if "cfg_sidebar" not in st.session_state:
    st.session_state.cfg_sidebar = {}


def _ensure_adv_defaults(cfg0: dict) -> None:
    """One-time defaults for advanced widgets (session_state keys)."""
    defaults = {
        "adv_final_k": int(cfg0["final_k"]),
        "adv_retrieval_pool": int(cfg0["retrieval_pool"]),
        "adv_min_cos": float(cfg0["min_cosine_similarity"]),
        "adv_use_hybrid": bool(cfg0["use_hybrid"]),
        "adv_use_mmr": bool(cfg0["use_mmr"]),
        "adv_mmr_lambda": float(cfg0["mmr_lambda"]),
        "adv_chunk_w": int(cfg0["chunk_words"]),
        "adv_chunk_o": int(cfg0["chunk_overlap"]),
        "adv_rows_per": int(cfg0["tabular_rows_per_chunk"]),
        "adv_strict": True,
        "adv_rewrite": False,
        "adv_temp": float(cfg0["temperature"]),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def effective_config() -> dict:
    base = load_config(ROOT / "config.yaml")
    return merge_config(base, st.session_state.cfg_sidebar)


def rebuild_index_from_chunks(chunks: list, cfg: dict) -> None:
    if not chunks:
        st.session_state.vector_store = None
        st.session_state.all_chunks = []
        return
    with st.spinner("Embedding corpus and building FAISS + BM25 index…"):
        st.session_state.vector_store = rebuild_store_from_chunks(chunks, cfg)
    st.session_state.all_chunks = chunks


def _render_sources(sources: list) -> None:
    with st.expander("Sources & scores", expanded=False):
        for i, r in enumerate(sources, start=1):
            line = (
                f"[{i}] **{r.get('source')}** · `{r.get('kind', '?')}` · "
                f"cosine **{r.get('dense_similarity')}**"
            )
            if r.get("sheet"):
                line += f" · sheet `{r['sheet']}`"
            if r.get("row_range"):
                line += f" · rows `{r['row_range']}`"
            st.markdown(line)
            txt = r.get("text") or ""
            st.caption(txt[:900] + ("…" if len(txt) > 900 else ""))


def _render_message(m: dict) -> None:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant" and m.get("sources"):
            _render_sources(m["sources"])
        if m["role"] == "assistant" and m.get("meta"):
            meta = m["meta"]
            rq = meta.get("retrieval_query")
            if rq and rq != meta.get("original_query"):
                st.caption(f"Retrieval query used: {rq}")
            if meta.get("weak_retrieval"):
                st.caption("Weak retrieval: best dense similarity was below the configured threshold.")


# --- API key ---
try:
    get_openai_api_key()
except Exception as e:  # noqa: BLE001
    st.error(
        "OpenAI API key is not configured. Add `OPENAI_API_KEY` in Streamlit Cloud **Secrets** "
        "(Settings), or set the `OPENAI_API_KEY` environment variable for local runs."
    )
    st.caption(str(e))
    st.stop()

cfg0 = load_config(ROOT / "config.yaml")
_ensure_adv_defaults(cfg0)

with st.sidebar:
    st.markdown("### Analyst RAG Copilot")
    st.caption("Use **your own files** so you can judge answers against familiar data.")
    if st.button("Reset session", type="secondary", help="Clears chat, corpus, and advanced controls"):
        st.session_state.messages = []
        st.session_state.all_chunks = []
        st.session_state.vector_store = None
        st.session_state.pending_user = None
        st.session_state.cfg_sidebar = {}
        for adv_key in (
            "adv_final_k",
            "adv_retrieval_pool",
            "adv_min_cos",
            "adv_use_hybrid",
            "adv_use_mmr",
            "adv_mmr_lambda",
            "adv_chunk_w",
            "adv_chunk_o",
            "adv_rows_per",
            "adv_strict",
            "adv_rewrite",
            "adv_temp",
        ):
            st.session_state.pop(adv_key, None)
        _ensure_adv_defaults(cfg0)
        st.success("Session cleared.")
        st.rerun()

st.title("Analyst RAG Copilot")
st.caption(
    "Ask questions in plain English — answers are grounded in **your** uploads with cited sources."
)

tab_chat, tab_about = st.tabs(["Ask questions", "How this RAG works"])

with tab_about:
    emb_m = str(cfg0.get("embedding_model", "text-embedding-3-small"))
    chat_m = str(cfg0.get("chat_model", "gpt-4o-mini"))
    rw_m = str(cfg0.get("rewrite_model", chat_m))

    st.markdown(
        """
### What is RAG?

**Retrieval-Augmented Generation (RAG)** combines search with a language model: the app **finds relevant excerpts**
from **your** documents, passes them to the model as **context**, and asks it to answer from that evidence.
You get **source citations** so you can verify claims against the original material.

### Pipeline in this app

1. **Ingest** — CSV / Excel / PDF / Word / text files are read into structured text (tables keep columns and row ranges).
2. **Chunk** — Text is split into overlapping segments with metadata (file, sheet, rows, etc.).
3. **Embed** — Each chunk is encoded as a vector using an OpenAI embedding model.
4. **Index** — Vectors are stored in **FAISS** for similarity search; **BM25** adds keyword matching when hybrid mode is on.
5. **Retrieve** — Your question is embedded; top passages are selected (with optional diversity).
6. **Generate** — A chat model reads those passages and responds, citing `[1]`, `[2]`, … when strict grounding is enabled.
"""
    )
    st.markdown(
        f"""
### Stack details

| Component | Implementation |
| --------- | -------------- |
| **Embeddings** | `{emb_m}` |
| **Vector index** | **FAISS** `IndexFlatIP` on normalized vectors (cosine-style similarity) |
| **Hybrid retrieval** | Dense embeddings + **BM25** merged with **RRF** |
| **Diversity** | **MMR** (Maximal Marginal Relevance) |
| **Chat model** | `{chat_m}` |
| **Optional query rewrite** | `{rw_m}` (off by default — see Advanced settings) |

### Grounding

- **Strict**: answer only from retrieved text; say when context is insufficient.
- **Relaxed**: may add a short, labeled general-knowledge note if helpful.

### Privacy note

Uploaded content is processed in session memory. Embeddings and answers use the **OpenAI API**; retrieved **snippets**
(not necessarily entire files) are sent as context. Review OpenAI’s policies for your use case.
"""
    )

with tab_chat:
    with st.expander("What should I upload?", expanded=False):
        st.markdown(
            """
| Format | When to use it |
| ------ | -------------- |
| **CSV / Excel** | Structured metrics, sales, surveys — includes automatic **column profiles** and **row-range chunks**. |
| **PDF / DOCX / TXT** | Narratives: policies, methodology, definitions. |

**Why use your own data?** When you know the ground truth, you can validate answers using the **Sources** expander under each reply.
Pairing a spreadsheet with a short methodology note (TXT/DOCX) often improves definition-style questions.
"""
        )

    st.markdown("### Step 1 — Upload and index")

    files = st.file_uploader(
        "Drop files here, then click **Process uploads**",
        type=["csv", "xlsx", "xlsm", "pdf", "docx", "txt"],
        accept_multiple_files=True,
    )

    b1, b2 = st.columns(2)
    with b1:
        process_clicked = st.button("Process uploads", type="primary")
    with b2:
        rebuild_clicked = st.button(
            "Rebuild index",
            help="Re-embed the corpus in memory after changing chunking or retrieval-related advanced settings.",
        )

    with st.expander("Advanced settings (optional)", expanded=False):
        st.caption("Defaults are fine for exploring RAG. Change these only if you want to experiment.")
        st.markdown("##### Retrieval")
        st.session_state.adv_final_k = st.slider(
            "Chunks passed to the model (final_k)",
            3,
            16,
            value=int(st.session_state.adv_final_k),
        )
        st.session_state.adv_retrieval_pool = st.slider(
            "Candidate pool size",
            20,
            200,
            value=int(st.session_state.adv_retrieval_pool),
            step=10,
        )
        st.session_state.adv_min_cos = st.slider(
            "Minimum cosine similarity",
            0.05,
            0.55,
            value=float(st.session_state.adv_min_cos),
            step=0.01,
        )
        st.session_state.adv_use_hybrid = st.toggle(
            "Hybrid BM25 + dense (RRF)",
            value=bool(st.session_state.adv_use_hybrid),
        )
        st.session_state.adv_use_mmr = st.toggle(
            "MMR diversify results",
            value=bool(st.session_state.adv_use_mmr),
        )
        st.session_state.adv_mmr_lambda = st.slider(
            "MMR λ (relevance vs diversity)",
            0.1,
            0.9,
            value=float(st.session_state.adv_mmr_lambda),
            step=0.05,
        )

        st.markdown("##### Chunking (next Process uploads)")
        st.session_state.adv_chunk_w = st.slider(
            "Target chunk size (words)",
            120,
            500,
            value=int(st.session_state.adv_chunk_w),
            step=10,
        )
        st.session_state.adv_chunk_o = st.slider(
            "Chunk overlap (words)",
            0,
            120,
            value=int(st.session_state.adv_chunk_o),
            step=5,
        )
        st.session_state.adv_rows_per = st.slider(
            "Tabular rows per chunk",
            10,
            120,
            value=int(st.session_state.adv_rows_per),
            step=5,
        )

        st.markdown("##### Generation")
        st.session_state.adv_strict = st.toggle(
            "Strict grounding (no outside knowledge)",
            value=bool(st.session_state.adv_strict),
            help="Requires citations; refuses unsupported guesses.",
        )
        st.session_state.adv_rewrite = st.toggle(
            "LLM query rewrite (extra API call)",
            value=bool(st.session_state.adv_rewrite),
        )
        st.session_state.adv_temp = st.slider(
            "Answer temperature",
            0.0,
            0.7,
            value=float(st.session_state.adv_temp),
            step=0.01,
        )

    # Sync cfg_sidebar after expander widgets run (rerun uses updated session values).
    st.session_state.cfg_sidebar["final_k"] = int(st.session_state.adv_final_k)
    st.session_state.cfg_sidebar["retrieval_pool"] = int(st.session_state.adv_retrieval_pool)
    st.session_state.cfg_sidebar["min_cosine_similarity"] = float(st.session_state.adv_min_cos)
    st.session_state.cfg_sidebar["use_hybrid"] = bool(st.session_state.adv_use_hybrid)
    st.session_state.cfg_sidebar["use_mmr"] = bool(st.session_state.adv_use_mmr)
    st.session_state.cfg_sidebar["mmr_lambda"] = float(st.session_state.adv_mmr_lambda)

    chunk_w = int(st.session_state.adv_chunk_w)
    chunk_o = int(st.session_state.adv_chunk_o)
    rows_per = int(st.session_state.adv_rows_per)
    strict = bool(st.session_state.adv_strict)
    use_rewrite = bool(st.session_state.adv_rewrite)
    temp = float(st.session_state.adv_temp)
    cfg = effective_config()

    if process_clicked:
        if not files:
            st.warning("Upload at least one supported file.")
        else:
            combined: list = []
            errs: list[str] = []
            for f in files:
                ch, err = ingest_uploaded_file(
                    f,
                    rows_per,
                    chunk_words=chunk_w,
                    chunk_overlap=chunk_o,
                )
                if err:
                    errs.append(f"{f.name}: {err}")
                combined.extend(ch)
            combined = dedupe_chunks(combined)
            assign_chunk_ids(combined)
            if not combined:
                st.error("No usable chunks produced.")
                for e in errs:
                    st.warning(e)
            else:
                for e in errs:
                    if e:
                        st.warning(e)
                rebuild_index_from_chunks(combined, cfg)
                st.success(f"Indexed **{len(combined)}** chunks from **{len(files)}** file(s). You can chat below.")

    if rebuild_clicked:
        if not st.session_state.all_chunks:
            st.warning("No corpus in memory — process uploads first.")
        else:
            rebuild_index_from_chunks(st.session_state.all_chunks, cfg)
            st.success("Index rebuilt with current settings.")

    if st.session_state.all_chunks:
        st.markdown("**Corpus summary**")
        st.dataframe(corpus_manifest(st.session_state.all_chunks), use_container_width=True, hide_index=True)

        sources_set = sorted({c["source"] for c in st.session_state.all_chunks})
        drop = st.selectbox("Remove one file from the corpus", options=["(none)"] + sources_set)
        if st.button("Remove selected file") and drop != "(none)":
            filt = [c for c in st.session_state.all_chunks if c.get("source") != drop]
            assign_chunk_ids(filt)
            if not filt:
                st.session_state.all_chunks = []
                st.session_state.vector_store = None
                st.info("Corpus empty after removal.")
            else:
                rebuild_index_from_chunks(filt, cfg)
            st.rerun()
    else:
        st.info("Upload documents above, then ask questions in Step 2.")

    st.divider()

    st.markdown("### Step 2 — Chat with your data")

    vs = st.session_state.vector_store

    for m in st.session_state.messages:
        _render_message(m)

    if not vs or vs.index.ntotal == 0:
        st.info("Finish Step 1 (**Process uploads**) to enable chat.")

    ctrl1, ctrl2 = st.columns(2)
    with ctrl1:
        if st.button(
            "Regenerate last answer",
            disabled=len(st.session_state.messages) < 2,
        ):
            msgs = st.session_state.messages
            if len(msgs) >= 2 and msgs[-1]["role"] == "assistant" and msgs[-2]["role"] == "user":
                user_q = msgs[-2]["content"]
                msgs.pop()
                msgs.pop()
                st.session_state.pending_user = user_q
                st.rerun()
    with ctrl2:
        last = next((m for m in reversed(st.session_state.messages) if m["role"] == "assistant"), None)
        if last:
            st.download_button(
                "Download last answer (.txt)",
                data=last.get("content", ""),
                file_name="last_answer.txt",
                mime="text/plain",
            )

    prompt = st.chat_input("Ask a question about your uploaded documents…")

    if st.session_state.pending_user:
        prompt = st.session_state.pending_user
        st.session_state.pending_user = None

    if prompt and vs is not None and vs.index.ntotal > 0:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        cfg_run = effective_config()
        with st.chat_message("assistant"):
            with st.spinner("Retrieving + generating…"):
                try:
                    ans, sources, meta = answer_query(
                        prompt,
                        vs,
                        cfg_run,
                        strict_grounding=strict,
                        use_query_rewrite=use_rewrite,
                        temperature=temp,
                    )
                except Exception as e:  # noqa: BLE001
                    logger.exception("answer_query failed")
                    ans = f"**Error:** {e}"
                    sources = []
                    meta = {}
            st.markdown(ans)
            if sources:
                _render_sources(sources)
            if meta.get("retrieval_query") and meta.get("retrieval_query") != meta.get("original_query"):
                st.caption(f"Retrieval query used: {meta.get('retrieval_query')}")
            if meta.get("weak_retrieval"):
                st.caption("Weak retrieval: verify sources carefully.")

        st.session_state.messages.append(
            {"role": "assistant", "content": ans, "sources": sources, "meta": meta}
        )
        st.rerun()

st.divider()
st.caption(
    "Tip: open **How this RAG works** for embeddings, FAISS, and grounding details. "
    "Use **Advanced settings** only if you want to tune retrieval."
)
