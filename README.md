# Multi-Source RAG Chatbot

This project is a local telecom support chatbot built with a simple RAG pipeline. It answers customer questions by retrieving relevant context from multiple sources, then generating a response with a local LLM served by Ollama.

## About This Project

I started from a tutorial-style RAG setup that used Groq for generation and ChromaDB for the vector store. That was a good learning setup, but it was not a fit for my machine. I wanted the whole stack to run locally, without API keys and without native-library issues.

The main problem was compatibility. ChromaDB and some of the ML dependencies pulled in packages that triggered application-control restrictions on my Windows machine. Rather than weaken system security, I switched to a simpler local stack:

- FAISS instead of ChromaDB
- Ollama embeddings instead of sentence-transformers
- Llama 3.1 8B through Ollama instead of a hosted model

That kept the project practical and fully local. It is still a learning project, but it is honest about why the stack changed.

## Why FAISS over Chroma

I chose FAISS for this project because it is lighter and easier to run in a local setup. It also avoids the native dependency chain that kept getting blocked on my machine. For a small, offline project, FAISS gives a clean and reliable vector store without bringing in a bigger ML stack than I needed.

## How the retrieval works

The chatbot does not rely on a single knowledge source. It queries three separate FAISS indexes and combines the results:

```text
User question
     |
     v
Merged retriever
  -> FAQ index
  -> Ticket index
  -> PDF guide index
     |
     v
Prompt + Llama 3.1 8B (Ollama)
     |
     v
Answer
```

I intentionally use `top-k = 3` from each index, so each query gets 9 context chunks total:

- 3 FAQ matches
- 3 resolved-ticket matches
- 3 guide chunks

That gives a useful spread of answers without overloading the prompt with too much irrelevant text. The model still has to choose what matters, but the retrieval set is broad enough to capture both policy answers and real-world ticket resolutions.

## How the merged retriever works

The merged retriever is defined in `retriever.py`. It loads each FAISS store separately and builds a retriever for each one:

- `faq_store.as_retriever(search_kwargs={"k": 3})`
- `tickets_store.as_retriever(search_kwargs={"k": 3})`
- `guides_store.as_retriever(search_kwargs={"k": 3})`

Then it concatenates the results in one function:

```python
return faq_retriever.invoke(query) + tickets_retriever.invoke(query) + guides_retriever.invoke(query)
```

This is a simple multi-source retrieval pattern. It is easy to understand, easy to debug, and good for a first version. The tradeoff is that it does not do any reranking or source-aware deduplication yet.

## File-by-file overview

- `app.py` — Streamlit UI. It renders the chat interface, loads sample telecom questions, and streams the model response.
- `main.py` — CLI version for running the chatbot from the terminal. This is useful when you want a no-UI workflow.
- `retriever.py` — Builds the merged retriever across FAQ, tickets, and guide chunks.
- `rag_chain.py` — Connects the retriever to the prompt and Llama 3.1 8B via Ollama, then parses the final text output.
- `ingest_faq.py` — Loads the FAQ CSV and stores each row as an embedded document in a FAISS index.
- `ingest_tickets.py` — Loads resolved tickets from SQLite and embeds each ticket as a document.
- `ingest_pdf.py` — Loads the PDF guide, splits it into chunks, and stores those chunks in FAISS.
- `data/seed_tickets.py` — Creates the sample resolved-ticket database seed data.
- `data/generate_pdf.py` — Rebuilds the telecom guide PDF used as a knowledge source.
- `faiss_store/` — Persisted vector indexes generated after ingestion.

## System prompt and model behavior

The system prompt in `rag_chain.py` is intentionally strict. It tells the model to answer only from the retrieved context and to say when the answer is missing. That is important in telecom support work, because a confident answer based on weak context is worse than a cautious one.

The prompt should match the real system design. This project uses three sources of context, not two, so the prompt was corrected to say: FAQ entries, resolved tickets, and PDF guide chunks.

## Quick retrieval sanity check

This is a simple human check, not a formal benchmark. It helps verify that the retrieval mix is sensible for common telecom questions.

| Question | Likely best source | Why |
|---|---|---|
| How do I activate international roaming? | `guides` / `faq` | Usually a setup or policy instruction |
| My SIM is not detected after restart | `tickets` / `faq` | Real user issues often show up in tickets |
| Why is my bill higher than usual? | `faq` / `tickets` | Billing problems often repeat across support data |
| How do I enable Wi‑Fi calling? | `guides` | Device setup instructions are often in the guide |
| I was charged for roaming but had a bundle active | `tickets` | Past complaint patterns are especially useful here |

This is the kind of quick evaluation I would expand into a real benchmark next: record which source produced the best answer for each question, measure hit rate per source, and compare retrieval quality across prompts.

## Next improvements I would make

The current version is intentionally simple, but these are the next upgrades I would add:

- Reranking: run a lightweight reranker over the 9 retrieved chunks before sending them to the model.
- Deduplication across sources: remove near-duplicate chunks from FAQ and tickets when they overlap heavily.
- Source weighting: give tickets or guides more importance for certain query types.
- Retrieval evaluation: compute source hit rate, answer correctness, and retrieval recall on a labeled question set.
- Query rewriting: improve search quality for vague telecom questions like “my data is broken” or “can’t connect after update”.

## Prerequisites

Before running the project, make sure you have:

- Python 3.11+
- pip
- [Ollama](https://ollama.com) installed and running locally
- These models pulled into Ollama:

```bash
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

## Setup

### 1) Clone the project and install dependencies

```bash
git clone <repo-url>
cd rag-telecom-chatbot
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # macOS/Linux
python -m pip install langchain langchain-core faiss-cpu langchain-ollama streamlit fpdf2 pypdf langchain-community langchain-text-splitters
```

### 2) Start Ollama

```bash
ollama serve
```

No API keys are needed. Everything runs locally.

### 3) Build the FAISS indexes

Run the ingestion scripts once to load the knowledge sources into the vector stores:

```bash
python ingest_faq.py
python ingest_tickets.py
python ingest_pdf.py
```

Each script embeds its source data and saves the index under `faiss_store/`. Re-run only the script for the source that changed.

## Running the app

### Streamlit UI

```bash
streamlit run app.py
```

This opens the app at `http://localhost:8501` in your browser. The sidebar includes example questions and a button to clear the chat history.

### CLI mode

```bash
python main.py
```

This starts an interactive prompt. Type a question and press Enter. Type `quit` to exit.

## Data sources

The retriever pulls the top 3 results from each source, giving a total of 9 context chunks per query.

- `faq` → `data/faq.csv` → one document per FAQ row
- `tickets` → `data/tickets.db` → one document per resolved ticket
- `guides` → `data/telecom_guide.pdf` → chunks of 600 characters with 100-character overlap

## Regenerating the seed data

If you want to rebuild the sample data:

```bash
# Seed the SQLite ticket database
python data/seed_tickets.py

# Regenerate the telecom guide PDF
python data/generate_pdf.py
```

After that, run the matching ingestion script again to refresh the FAISS indexes.

## Final note

This is a practical local RAG project for telecom support scenarios. It is intentionally simple, fully offline, and easy to run on a personal machine without a managed AI backend.
