import streamlit as st
from model import extract_text, chunk_text, get_embeddings, VectorStore, answer_query

st.set_page_config(page_title="Document Q&A", layout="wide")

# -----------------------------
# Title
# -----------------------------
st.title("📄 Document Q&A Assistant")
st.markdown("Analyze your documents and ask intelligent questions.")

# -----------------------------
# Session State
# -----------------------------
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Controls")

    if st.button("Reset"):
        st.session_state.vector_store = None
        st.session_state.chat_history = []
        st.success("App reset complete.")

# -----------------------------
# Upload Section
# -----------------------------
st.subheader("Upload Documents")

files = st.file_uploader(
    "Upload CSV, Excel, PDF, or Word files",
    accept_multiple_files=True
)

# -----------------------------
# Process Files
# -----------------------------
if st.button("Process Files"):
    if not files:
        st.warning("Please upload at least one file.")
    else:
        all_chunks = []

        with st.spinner("Processing files..."):
            for file in files:
                text = extract_text(file)

                st.write(f"{file.name} — {len(text)} characters")

                if text.strip() == "":
                    st.warning(f"No readable text in {file.name}")
                    continue

                chunks = chunk_text(text, file.name)
                st.write(f"Chunks created: {len(chunks)}")

                all_chunks.extend(chunks)

        if len(all_chunks) == 0:
            st.error("No valid content found in uploaded files.")
        else:
            with st.spinner("Creating embeddings..."):
                texts = [c["text"] for c in all_chunks]
                embeddings = get_embeddings(texts)

                vs = VectorStore(embeddings.shape[1])
                vs.add(embeddings, all_chunks)

                st.session_state.vector_store = vs

            st.success("Documents processed successfully. You can now ask questions.")

# -----------------------------
# Question Section
# -----------------------------
st.subheader("Ask Questions")

query = st.text_input("Enter your question")

if st.button("Ask"):
    if not st.session_state.vector_store:
        st.warning("Please process documents first.")
    elif not query.strip():
        st.warning("Enter a valid question.")
    else:
        with st.spinner("Thinking..."):
            answer, sources = answer_query(query, st.session_state.vector_store)

        st.session_state.chat_history.append((query, answer, sources))

# -----------------------------
# Conversation Section
# -----------------------------
if st.session_state.chat_history:
    st.subheader("Conversation")

    for q, a, s in reversed(st.session_state.chat_history):
        st.markdown("**You**")
        st.write(q)

        st.markdown("**Assistant**")
        st.write(a)

        if s:
            with st.expander("View Sources"):
                for r in s:
                    source_info = f"{r['source']} (Chunk {r.get('chunk_id', '-')})"
                    st.markdown(f"**{source_info}**")
                    st.write(r["text"][:300] + "...")
        else:
            st.info("No supporting document chunks found.")
