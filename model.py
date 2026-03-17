import faiss
import numpy as np
import streamlit as st
from openai import OpenAI
from typing import List
import pandas as pd
from PyPDF2 import PdfReader
import docx

# -----------------------------
# Client
# -----------------------------
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# -----------------------------
# Text Extraction
# -----------------------------
def extract_text(file):
    name = file.name.lower()

    if name.endswith(".csv"):
        df = pd.read_csv(file)
        return df.to_string(index=False)

    elif name.endswith(".xlsx"):
        df = pd.read_excel(file)
        return df.to_string(index=False)

    elif name.endswith(".pdf"):
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text

    elif name.endswith(".docx"):
        doc = docx.Document(file)
        return "\n".join([p.text for p in doc.paragraphs])

    else:
        return ""

# -----------------------------
# Chunking
# -----------------------------
def chunk_text(text, chunk_size=500, overlap=100):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunks.append(" ".join(words[i:i+chunk_size]))

    return chunks

# -----------------------------
# Embeddings
# -----------------------------
def get_embeddings(client, texts: List[str]):
    embeddings = []
    for text in texts:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        embeddings.append(response.data[0].embedding)

    return np.array(embeddings).astype("float32")

# -----------------------------
# Vector Store
# -----------------------------
class VectorStore:
    def __init__(self, dim):
        self.index = faiss.IndexFlatL2(dim)
        self.text_chunks = []

    def add(self, embeddings, chunks):
        self.index.add(embeddings)
        self.text_chunks.extend(chunks)

    def search(self, query_embedding, k=5):
        D, I = self.index.search(query_embedding, k)
        return [self.text_chunks[i] for i in I[0]]

# -----------------------------
# Answer Query
# -----------------------------
def answer_query(client, query, vector_store):
    query_embedding = get_embeddings(client, [query])
    relevant_chunks = vector_store.search(query_embedding)

    context = "\n\n".join(relevant_chunks)

    prompt = f"""
You are a helpful assistant.

Answer ONLY using the provided context.
- If answer not found → say "Not found in documents"
- Keep answer concise and structured (bullet points if needed)

Context:
{context}

Question:
{query}

Answer:
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return response.choices[0].message.content, relevant_chunks
