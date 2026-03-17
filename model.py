import os
import streamlit as st
import faiss
import numpy as np
from openai import OpenAI
from typing import List
import pandas as pd
from PyPDF2 import PdfReader

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# -----------------------------
# Text Extraction
# -----------------------------
def extract_text_from_csv(file):
    df = pd.read_csv(file)
    return df.to_string(index=False)

def extract_text_from_excel(file):
    df = pd.read_excel(file)
    return df.to_string(index=False)

def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

# -----------------------------
# Chunking
# -----------------------------
def chunk_text(text, chunk_size=500):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i+chunk_size]))
    return chunks

# -----------------------------
# Embeddings
# -----------------------------
def get_embeddings(texts: List[str]):
    embeddings = []
    for text in texts:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        embeddings.append(response.data[0].embedding)
    return np.array(embeddings).astype("float32")

# -----------------------------
# FAISS Index
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
# Query Answering
# -----------------------------
def answer_query(query, vector_store):
    query_embedding = get_embeddings([query])
    relevant_chunks = vector_store.search(query_embedding)

    context = "\n\n".join(relevant_chunks)

    prompt = f"""
You are a helpful assistant.

Answer the question using ONLY the context below.
If the answer is not present, say "I could not find this in the documents."

Context:
{context}

Question:
{query}

Answer:
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response.choices[0].message.content, relevant_chunks
