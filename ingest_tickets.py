"""
Ingests the resolved-ticket database into a FAISS index.
This makes real past support cases searchable so the assistant can answer from lived examples.
Run once (or after adding new tickets): python ingest_tickets.py
"""
import os
# Keep local indexing output minimal while embeddings are created.
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import sqlite3
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

# Index folder for ticket-based retrieval.
FAISS_DIR   = os.path.join("faiss_store", "tickets")
# SQLite database containing resolved telecom support cases.
DB_PATH     = os.path.join("data", "tickets.db")
# The embedding model served locally by Ollama.
EMBED_MODEL = "nomic-embed-text"


def load_ticket_documents(db_path: str) -> list[Document]:
    # Only resolved tickets are indexed because they are the most useful historical evidence.
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM tickets WHERE status = 'resolved'"
    ).fetchall()
    conn.close()

    docs = []
    for row in rows:
        content = (
            f"Issue: {row['issue_type']}\n"
            f"Description: {row['description']}\n"
            f"Resolution: {row['resolution']}"
        )
        docs.append(Document(
            page_content=content,
            metadata={
                "source":    "ticket",
                "ticket_id": row["ticket_id"],
                "category":  row["category"],
                "status":    row["status"],
            },
        ))
    return docs


def main():
    print("Loading ticket documents from SQLite...")
    docs = load_ticket_documents(DB_PATH)
    print(f"  {len(docs)} resolved tickets loaded.")

    print("Initialising embedding model...")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    print(f"Embedding and storing FAISS index at '{FAISS_DIR}'...")
    vectorstore = FAISS.from_documents(docs, embeddings)
    os.makedirs(FAISS_DIR, exist_ok=True)
    vectorstore.save_local(FAISS_DIR)
    print(f"  Done. {len(docs)} vectors stored.")


if __name__ == "__main__":
    main()