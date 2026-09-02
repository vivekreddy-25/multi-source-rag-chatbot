"""
Ingests resolved tickets from data/tickets.db into a FAISS index at faiss_store/tickets.
Run once (or after adding new tickets): python ingest_tickets.py
"""
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import sqlite3
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

FAISS_DIR   = os.path.join("faiss_store", "tickets")
DB_PATH     = os.path.join("data", "tickets.db")
EMBED_MODEL = "nomic-embed-text"


def load_ticket_documents(db_path: str) -> list[Document]:
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