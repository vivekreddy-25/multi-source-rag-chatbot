# Multi-Source RAG Chatbot

A Retrieval-Augmented Generation (RAG) customer care chatbot for telecom support. It answers questions about mobile connectivity, billing, SIM issues, and roaming by retrieving relevant context from three knowledge sources and generating responses with Llama 3.1 8B, running fully locally via Ollama.

## About This Project

This project started from a YouTube RAG tutorial and was substantially modified to run entirely on local infrastructure:

- Replaced the Groq-hosted Qwen3-32B LLM with a fully local **Llama 3.1 8B** served via **Ollama**
- Replaced **ChromaDB** with **FAISS** as the vector store (ChromaDB's gRPC/OpenTelemetry dependency chain hit native-DLL loading issues on Windows; FAISS has clean prebuilt wheels)
- Replaced HuggingFace `sentence-transformers` embeddings with Ollama's local **nomic-embed-text** model (avoided a similar native-dependency chain through scipy/scikit-learn)
- The result runs **entirely locally** — no API keys, no external services, no per-token costs

## Architecture

```
User question
     │
     ▼
Merged Retriever (top-k from each store)
  ├── FAISS · faq        (FAQ entries from CSV)
  ├── FAISS · tickets    (resolved support tickets from SQLite)
  └── FAISS · guides     (PDF guide chunks)
     │
     ▼
ChatPromptTemplate → Llama 3.1 8B (Ollama) → Answer
```

**Embedding model:** `nomic-embed-text` (served locally via [Ollama](https://ollama.com))
**LLM:** `llama3.1:8b` (served locally via [Ollama](https://ollama.com))

## Project Structure

```
rag-telecom-chatbot/
├── app.py              # Streamlit web UI
├── main.py             # CLI entry point
├── rag_chain.py        # Builds the LangChain RAG chain
├── retriever.py        # Merges the three FAISS retrievers
├── ingest_faq.py       # Loads data/faq.csv → FAISS 'faq' index
├── ingest_tickets.py   # Loads data/tickets.db → FAISS 'tickets' index
├── ingest_pdf.py       # Loads data/telecom_guide.pdf → FAISS 'guides' index
├── data/
│   ├── faq.csv             # FAQ question/answer pairs
│   ├── tickets.db          # SQLite database of resolved support tickets
│   ├── telecom_guide.pdf   # Telecom user guide (chunked at ingest)
│   ├── seed_tickets.py     # Script to seed the tickets database
│   └── generate_pdf.py     # Script to generate the telecom guide PDF
├── faiss_store/         # Persisted FAISS indexes (created at ingest)
├── pyproject.toml
└── .env.example
```

## Prerequisites

- Python 3.11+
- pip
- [Ollama](https://ollama.com) installed and running locally
- The following models pulled in Ollama:
  ```bash
  ollama pull llama3.1:8b
  ollama pull nomic-embed-text
  ```

## Setup

**1. Clone and install dependencies**

```bash
git clone <repo-url>
cd rag-telecom-chatbot
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # macOS/Linux
python -m pip install langchain langchain-core faiss-cpu langchain-ollama pandas python-dotenv streamlit fpdf2 pypdf langchain-community langchain-text-splitters
```

**2. Make sure Ollama is running**

```bash
ollama serve
```

(No API keys required — everything runs locally.)

**3. Ingest data into FAISS**

Run the three ingestion scripts once to build the vector stores:

```bash
python ingest_faq.py
python ingest_tickets.py
python ingest_pdf.py
```

Each script embeds the source data and persists it under `faiss_store/`. Re-run a script only when its source data changes.

## Running the App

**Streamlit web UI**

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. The sidebar has one-click sample questions and a button to clear the conversation history.

**CLI**

```bash
python main.py
```

Interactive prompt — type a question and press Enter. Type `quit` to exit.

## Data Sources

| Index | Source file | Granularity |
|---|---|---|
| `faq` | `data/faq.csv` | 1 document per FAQ row |
| `tickets` | `data/tickets.db` | 1 document per resolved ticket |
| `guides` | `data/telecom_guide.pdf` | Chunks of 600 chars with 100-char overlap |

The retriever fetches the top 3 results from each index (9 context documents total) for every query.

## Regenerating Seed Data

```bash
# Seed the SQLite ticket database
python data/seed_tickets.py

# Regenerate the PDF guide
python data/generate_pdf.py
```

After regenerating, re-run the corresponding ingest script.