# 📄 Document Q&A Assistant (RAG System)

A Retrieval-Augmented Generation (RAG) application that allows users to upload documents (CSV, Excel, PDF) and ask conceptual questions using natural language.
It enables you to upload multiple and confidential data files and use them to get insights wihtout revealing them to LLMs. This preserves confidentiality.

---

##  Features

- Upload multiple files (CSV, Excel, PDF, Word)
- Ask natural language questions
- Context-aware answers using LLMs
- Source attribution for transparency
- Designed for conceptual and text-based queries

---

##  How It Works

1. Files are uploaded and converted into text
2. Text is split into chunks
3. Chunks are converted into embeddings
4. Stored in a FAISS vector database
5. User query is embedded and matched with relevant chunks
6. LLM generates answer using retrieved context

---

## ⚠️ Limitations

- Not optimized for heavy numerical or mathematical queries
- Best suited for conceptual and text-based analysis
