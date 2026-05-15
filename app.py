from __future__ import annotations

import logging
from pathlib import Path

import streamlit as st

from config_loader import load_config, merge_config
from generation import answer_query
from ingestion import assign_chunk_ids, corpus_manifest, dedupe_chunks, ingest_uploaded_file, load_demo_chunks
from llm_client import get_openai_api_key
from retrieval import rebuild_store_from_chunks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_app")

ROOT = Path(__file__).resolve().parent
SAMPLE_DIR = ROOT / "sample_data"

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

st.title("Analyst RAG Copilot")
st.caption(
    "Hybrid retrieval (dense + BM25), MMR diversification, strict grounding mode, "
    "and transparent source cards — tuned for tabular + narrative documents."
)

cfg0 = load_config(ROOT / "config.yaml")
sb = st.sidebar
sb.header("Retrieval")
st.session_state.cfg_sidebar["final_k"] = sb.slider(
    "Chunks in context (final_k)", 3, 16, int(cfg0["final_k"])
)
st.session_state.cfg_sidebar["retrieval_pool"] = sb.slider(
    "Candidate pool size", 20, 200, int(cfg0["retrieval_pool"]), step=10
)
st.session_state.cfg_sidebar["min_cosine_similarity"] = sb.slider(
    "Min cosine similarity", 0.05, 0.55, float(cfg0["min_cosine_similarity"]), 0.01
)
st.session_state.cfg_sidebar["use_hybrid"] = sb.toggle(
    "Hybrid BM25 + dense (RRF)", value=bool(cfg0["use_hybrid"])
)
st.session_state.cfg_sidebar["use_mmr"] = sb.toggle("MMR diversify results", value=bool(cfg0["use_mmr"]))
st.session_state.cfg_sidebar["mmr_lambda"] = sb.slider(
    "MMR λ (relevance vs diversity)", 0.1, 0.9, float(cfg0["mmr_lambda"]), 0.05
)

sb.header("Chunking (next ingest)")
chunk_w = sb.slider("Target chunk size (words)", 120, 500, int(cfg0["chunk_words"]), 10)
chunk_o = sb.slider("Chunk overlap (words)", 0, 120, int(cfg0["chunk_overlap"]), 5)
rows_per = sb.slider("Tabular rows per chunk", 10, 120, int(cfg0["tabular_rows_per_chunk"]), 5)

sb.header("Generation")
strict = sb.toggle("Strict grounding (no outside knowledge)", value=True)
use_rewrite = sb.toggle("LLM query rewrite (extra API call)", value=False)
temp = sb.slider("Answer temperature", 0.0, 0.7, float(cfg0["temperature"]), 0.01)

if sb.button("Reset app state", type="secondary"):
    st.session_state.messages = []
    st.session_state.all_chunks = []
    st.session_state.vector_store = None
    st.session_state.pending_user = None
    st.session_state.cfg_sidebar = {}
    st.success("State cleared.")
    st.rerun()

cfg = effective_config()

left, right = st.columns([0.44, 0.56], gap="large")

with left:
    st.subheader("Corpus")
    files = st.file_uploader(
        "Upload CSV, XLSX, PDF, DOCX, or TXT",
        type=["csv", "xlsx", "xlsm", "pdf", "docx", "txt"],
        accept_multiple_files=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Process uploads", type="primary"):
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
                    st.success(f"Indexed {len(combined)} chunks from {len(files)} file(s).")

    with c2:
        if st.button("Load demo corpus"):
            demo_chunks, demo_errs = load_demo_chunks(
                SAMPLE_DIR,
                rows_per,
                chunk_words=chunk_w,
                chunk_overlap=chunk_o,
            )
            for e in demo_errs:
                if e:
                    st.warning(e)
            if not demo_chunks:
                st.error("Demo folder is empty or unreadable.")
            else:
                rebuild_index_from_chunks(demo_chunks, cfg)
                st.success(f"Loaded demo: {len(demo_chunks)} chunks.")

    with c3:
        if st.button("Rebuild index", help="Re-embed in-memory corpus with current sidebar settings"):
            if not st.session_state.all_chunks:
                st.warning("No corpus in memory — process uploads or load demo first.")
            else:
                rebuild_index_from_chunks(st.session_state.all_chunks, cfg)
                st.success("Index rebuilt.")

    if st.session_state.all_chunks:
        st.markdown("**Corpus summary**")
        st.dataframe(corpus_manifest(st.session_state.all_chunks), use_container_width=True, hide_index=True)

        sources = sorted({c["source"] for c in st.session_state.all_chunks})
        drop = st.selectbox("Remove a source from corpus", options=["(none)"] + sources)
        if st.button("Remove selected source") and drop != "(none)":
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
        st.info("No corpus loaded yet — upload files or load the demo.")

with right:
    st.subheader("Conversation")
    vs = st.session_state.vector_store

    for m in st.session_state.messages:
        _render_message(m)

    if not vs or vs.index.ntotal == 0:
        st.info("Process documents or load the demo to enable Q&A.")

    prompt = st.chat_input("Ask a grounded question about your corpus…")

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

    col_a, col_b = st.columns(2)
    with col_a:
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
    with col_b:
        last = next((m for m in reversed(st.session_state.messages) if m["role"] == "assistant"), None)
        if last:
            st.download_button(
                "Download last answer (.txt)",
                data=last.get("content", ""),
                file_name="last_answer.txt",
                mime="text/plain",
            )

st.divider()
st.caption(
    "Tip: combine a CSV with a short methodology note (TXT/DOCX). "
    "Strict grounding reduces hallucinations but may refuse more when context is thin."
)
