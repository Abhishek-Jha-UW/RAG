import streamlit as st
from model import extract_text, chunk_text, get_embeddings, VectorStore, answer_query

st.title("📄 Document Q&A Assistant")

# -----------------------------
# Session State
# -----------------------------
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -----------------------------
# Upload
# -----------------------------
files = st.file_uploader("Upload files", accept_multiple_files=True)

# -----------------------------
# Process Files
# -----------------------------
if st.button("Process Files"):
    all_chunks = []

    with st.spinner("Processing..."):
        for file in files:
            text = extract_text(file)

            st.write(f"📄 {file.name} → {len(text)} characters")

            if text.strip() == "":
                st.warning(f"No text found in {file.name}")
                continue

            chunks = chunk_text(text, file.name)
            st.write(f"Chunks created: {len(chunks)}")

            all_chunks.extend(chunks)

        if len(all_chunks) == 0:
            st.error("No valid content found.")
        else:
            texts = [c["text"] for c in all_chunks]
            embeddings = get_embeddings(texts)

            vs = VectorStore(embeddings.shape[1])
            vs.add(embeddings, all_chunks)

            st.session_state.vector_store = vs
            st.success("Processing complete!")

# -----------------------------
# Ask Question
# -----------------------------
query = st.text_input("Ask a question")

if st.button("Ask"):
    if not st.session_state.vector_store:
        st.warning("Please process files first.")
    elif not query:
        st.warning("Enter a question.")
    else:
        with st.spinner("Thinking..."):
            answer, sources = answer_query(query, st.session_state.vector_store)

        st.session_state.chat_history.append((query, answer, sources))

# -----------------------------
# Show Chat
# -----------------------------
for q, a, s in reversed(st.session_state.chat_history):
    st.markdown(f"**You:** {q}")
    st.markdown(f"**Answer:** {a}")

    with st.expander("Sources"):
        for r in s:
            st.write(f"📄 {r['source']}: {r['text'][:200]}...")
