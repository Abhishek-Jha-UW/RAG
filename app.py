import streamlit as st
from model import (
    extract_text,
    chunk_text,
    get_embeddings,
    VectorStore,
    answer_query,
    get_client
)

st.set_page_config(page_title="Document Q&A Assistant")

st.title("📄 Document Q&A Assistant")
st.markdown("Ask conceptual questions about your uploaded files.")

# -----------------------------
# API Client
# -----------------------------
client = get_client(st.secrets["OPENAI_API_KEY"])

# -----------------------------
# Session State
# -----------------------------
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -----------------------------
# Sample Dataset Download
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
            embeddings = get_embeddings(client, chunks)

            vector_store = VectorStore(embeddings.shape[1])
            vector_store.add(embeddings, chunks)

            st.session_state.vector_store = vector_store

        st.success("Files processed successfully!")

# -----------------------------
# Chat Interface
# -----------------------------
st.subheader("💬 Ask Questions")

query = st.text_input("Enter your question:")

if query and st.session_state.vector_store:
    with st.spinner("Thinking... 🤖"):
        answer, sources = answer_query(client, query, st.session_state.vector_store)

    # Save history
    st.session_state.chat_history.append((query, answer, sources))

# -----------------------------
# Display Chat History
# -----------------------------
for i, (q, a, s) in enumerate(st.session_state.chat_history[::-1]):
    st.markdown(f"### 🧑 You: {q}")
    st.markdown(f"**🤖 Answer:** {a}")

    with st.expander("📄 Sources"):
        for j, src in enumerate(s):
            st.write(f"{j+1}. {src[:300]}...")
