import streamlit as st
import os
from model import (
    extract_text_from_csv,
    extract_text_from_excel,
    extract_text_from_pdf,
    chunk_text,
    get_embeddings,
    VectorStore,
    answer_query
)

st.set_page_config(page_title="Document Q&A Assistant")

st.title("📄 Document Q&A Assistant")
st.write("Ask conceptual questions about your uploaded files.")

# Session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# File upload
uploaded_files = st.file_uploader(
    "Upload CSV, Excel, or PDF files",
    type=["csv", "xlsx", "pdf"],
    accept_multiple_files=True
)

if st.button("Process Files"):
    all_text = ""

    for file in uploaded_files:
        if file.name.endswith(".csv"):
            all_text += extract_text_from_csv(file)
        elif file.name.endswith(".xlsx"):
            all_text += extract_text_from_excel(file)
        elif file.name.endswith(".pdf"):
            all_text += extract_text_from_pdf(file)

    chunks = chunk_text(all_text)
    embeddings = get_embeddings(chunks)

    vector_store = VectorStore(embeddings.shape[1])
    vector_store.add(embeddings, chunks)

    st.session_state.vector_store = vector_store
    st.success("Files processed successfully!")

# Chat
query = st.text_input("Ask a question:")

if query and st.session_state.vector_store:
    answer, sources = answer_query(query, st.session_state.vector_store)

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Sources")
    for i, src in enumerate(sources):
        st.write(f"{i+1}. {src[:300]}...")
