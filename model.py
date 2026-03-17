import streamlit as st
from openai import OpenAI
import faiss
import numpy as np
import pandas as pd
from PyPDF2 import PdfReader
import docx

# -----------------------------
# OpenAI Client
# -----------------------------
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


# -----------------------------
# Extract Text
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
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                text += f"\n[Page {i+1}]\n{page_text}"

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
        chunk = " ".join(words[i:i + chunk_size])

        chunks.append({
            "text": chunk,
            "source": source,
            "chunk_id": len(chunks)
        })

    return chunks


# -----------------------------
# Batch Embeddings (FAST)
# -----------------------------
def get_embeddings(texts):
    res = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )

    embeddings = np.array([d.embedding for d in res.data]).astype("float32")

    # Normalize for better similarity search
    faiss.normalize_L2(embeddings)

    return embeddings


# -----------------------------
# Vector Store
# -----------------------------
class VectorStore:
    def __init__(self, dim):
        self.index = faiss.IndexFlatL2(dim)
        self.data = []

    def add(self, embeddings, chunks):
        self.index.add(embeddings)
        self.data.extend(chunks)

    def search(self, query_embedding, k=5, threshold=1.5):
        faiss.normalize_L2(query_embedding)

        D, I = self.index.search(query_embedding, k)

        results = []
        for dist, idx in zip(D[0], I[0]):
            if idx < len(self.data) and dist < threshold:
                results.append(self.data[idx])

        return results


# -----------------------------
# Answer Query
# -----------------------------
def answer_query(query, vector_store):
    query_embedding = get_embeddings([query])

    results = vector_store.search(query_embedding)

    # Handle no results
    if len(results) == 0:
        return "No relevant information found in uploaded documents.", []

    # Limit context size
    MAX_CONTEXT_CHARS = 4000
    context = ""

    for r in results:
        if len(context) + len(r["text"]) < MAX_CONTEXT_CHARS:
            context += r["text"] + "\n\n"
        else:
            break

    prompt = f"""
You are an intelligent assistant designed to answer questions using uploaded documents and your own knowledge.

Guidelines:
1. Prioritize information from the provided context.
2. If the context contains relevant information, base your answer primarily on it.
3. If the context is incomplete or insufficient, you may use your general knowledge to provide a helpful answer.
4. Clearly distinguish when you are using information beyond the provided context.
5. Keep answers clear, concise, and well-structured.
6. If the answer is not clearly supported, say so instead of guessing.

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
