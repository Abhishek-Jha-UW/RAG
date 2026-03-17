import streamlit as st
from model import (
    extract_text,
    chunk_text,
    get_embeddings,
    VectorStore,
    answer_query
)

st.set_page_config(page_title="Document Q&A Assistant")

# -----------------------------
# Title
# -----------------------------
st.title("📄 Document Q&A Assistant")
st.markdown("Ask conceptual questions about your uploaded files.")
st.markdown("⚠️ *Best for text-based insights, not heavy calculations.*")

# -----------------------------
# Session State
# -----------------------------
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -----------------------------
# Sample Dataset
# -----------------------------
st.subheader("📥 Try Sample Dataset")

sample_csv = """Product,Category,Rating,Price
Phone,Electronics,4.5,699
Laptop,Electronics,4.7,1200
Shoes,Fashion,4.2,80
Watch,Accessories,4.0,150
Tablet,Electronics,4.3,400
"""

st.download_button(
    label="Download Sample Product Dataset",
    data=sample_csv,
    file_name="sample_products.csv",
    mime="text/csv"
)

# -----------------------------
# File Upload
# -----------------------------
uploaded_files = st.file_uploader(
    "Upload CSV, Excel, PDF, or Word files",
    type=["csv", "xlsx", "pdf", "docx"],
    accept_multiple_files=True
)

# -----------------------------
# Process Files
# -----------------------------
if st.button("Process Files"):
    if not uploaded_files:
        st.warning("Please upload at least one file.")
    else:
        with st.spinner("Processing files... ⏳"):
            all_text = ""

            for file in uploaded_files:
                all_text += extract_text(file)

            chunks = chunk_text(all_text)

            if len(chunks) == 0:
                st.error("No readable content found in files.")
            else:
                embeddings = get_embeddings(chunks)

                vector_store = VectorStore(embeddings.shape[1])
                vector_store.add(embeddings, chunks)

                st.session_state.vector_store = vector_store

        st.success("Files processed successfully!")

# -----------------------------
# Chat Section
# -----------------------------
st.subheader("💬 Ask Questions")

query = st.text_input("Enter your question:")

if query and st.session_state.vector_store:
    with st.spinner("Thinking... 🤖"):
        answer, sources = answer_query(query, st.session_state.vector_store)

    st.session_state.chat_history.append((query, answer, sources))

# -----------------------------
# Clear Chat Button
# -----------------------------
if st.button("🗑️ Clear Chat"):
    st.session_state.chat_history = []

# -----------------------------
# Display Chat
# -----------------------------
for q, a, s in reversed(st.session_state.chat_history):
    st.markdown(f"### 🧑 You: {q}")
    st.markdown(f"**🤖 Answer:**\n{a}")

    with st.expander("📄 Sources"):
        for i, src in enumerate(s):
            st.write(f"{i+1}. {src[:300]}...")
