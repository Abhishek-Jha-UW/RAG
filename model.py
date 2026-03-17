import streamlit as st
from openai import OpenAI
import faiss
import numpy as np
import pandas as pd
from PyPDF2 import PdfReader
import docx

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# -----------------------------
# Extract Text (with filename)
# -----------------------------
def extract_text(file):
    name = file.name.lower()
    text = ""

    try:
        if name.endswith(".csv"):
            df = pd.read_csv(file)
            text = df.to_string(index=False)

        elif name.endswith(".xlsx"):
            df = pd.read_excel(file)
            text = df.to_string(index=False)

        elif name.endswith(".pdf"):
            reader = PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() or ""

        elif name.endswith(".docx"):
            doc = docx.Document(file)
            text = "\n".join([p.text for p in doc.paragraphs])

    except Exception as e:
        st.error(f"Error reading {file.name}: {e}")

    return text

# -----------------------------
# Chunking with metadata
# -----------------------------
def chunk_text(text, source, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append({
            "text": chunk,
            "source": source
        })

    return chunks

# -----------------------------
# Embeddings
# -----------------------------
def get_embeddings(texts):
    embeddings = []

    for t in texts:
        res = client.embeddings.create(
            model="text-embedding-3-small",
            input=t
        )
        embeddings.append(res.data[0].embedding)

    return np.array(embeddings).astype("float32")

# -----------------------------
# Vector Store with metadata
# -----------------------------
class VectorStore:
    def __init__(self, dim):
        self.index = faiss.IndexFlatL2(dim)
        self.data = []

    def add(self, embeddings, chunks):
        self.index.add(embeddings)
        self.data.extend(chunks)

    def search(self, query_embedding, k=5):
        D, I = self.index.search(query_embedding, k)
        return [self.data[i] for i in I[0]]

# -----------------------------
# Answer Query
# -----------------------------
def answer_query(query, vector_store):
    query_embedding = get_embeddings([query])
    results = vector_store.search(query_embedding)

    context = "\n\n".join([r["text"] for r in results])

    prompt = f"""
Answer ONLY from context.
If not found → say "Not found in documents"

Context:
{context}

Question:
{query}

Answer:
"""

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return res.choices[0].message.content, results
