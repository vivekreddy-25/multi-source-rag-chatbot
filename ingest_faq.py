"""
Ingests the FAQ CSV into a FAISS index.
This creates the searchable knowledge base for general telecom policy and how-to answers.
Run once (or whenever the CSV changes): python ingest_faq.py
"""
import csv
import os
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

FAISS_DIR   = os.path.join("faiss_store", "faq")
CSV_PATH    = os.path.join("data", "faq.csv")
EMBED_MODEL = "nomic-embed-text"


def load_faq_documents(csv_path: str) -> list[Document]:
    # Each FAQ row is stored as a single retrievable document.
    docs = []
    with open(csv_path, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            content = f"Q: {row['question']}\nA: {row['answer']}"
            docs.append(Document(
                page_content=content,
                metadata={"source": "faq", "category": row["category"], "faq_id": str(row["id"])},
            ))
    return docs


def main():
    print("Loading FAQ documents...")
    docs = load_faq_documents(CSV_PATH)
    print(f"  {len(docs)} FAQ entries loaded.")

    print("Initialising embedding model...")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    print(f"Embedding and storing FAISS index at '{FAISS_DIR}'...")
    vectorstore = FAISS.from_documents(docs, embeddings)
    os.makedirs(FAISS_DIR, exist_ok=True)
    vectorstore.save_local(FAISS_DIR)
    print(f"  Done. {len(docs)} vectors stored.")


if __name__ == "__main__":
    main()