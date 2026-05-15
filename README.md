# 📄 Document Q&A Assistant (RAG System)

A Retrieval-Augmented Generation (RAG) application that allows users to upload documents (CSV, Excel, PDF, Word) and ask conceptual questions using natural language.
It enables you to upload multiple and confidential data files and use them to get insights without revealing full files to third parties: only retrieved passages are sent to the model API along with your question. Tune strict grounding in the app when you need citation-only answers.

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

---

## Run locally

- Install: `pip install -r requirements.txt`
- Secrets: set `OPENAI_API_KEY` in `.streamlit/secrets.toml` or the environment (Streamlit Cloud: **Settings → Secrets**).
- App: `streamlit run app.py`
- Demo corpus: `sample_data/` (use **Load demo corpus** in the sidebar).

## Project layout (modules)

- `app.py` — Streamlit UI
- `ingestion.py`, `retrieval.py`, `generation.py`, `config_loader.py`, `llm_client.py`
- `model.py` — re-exports for scripts/notebooks
- `eval/eval_retrieval.py` — optional retrieval smoke check (`OPENAI_API_KEY` required)
- Tests: `pytest` from this folder (`pytest.ini` sets `pythonpath`)
