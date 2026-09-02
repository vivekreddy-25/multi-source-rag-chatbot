"""
Ingests data/faq.csv into a FAISS index at faiss_store/faq.
Run once (or whenever the CSV changes): python ingest_faq.py
"""
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import pandas as pd
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

FAISS_DIR   = os.path.join("faiss_store", "faq")
CSV_PATH    = os.path.join("data", "faq.csv")
EMBED_MODEL = "nomic-embed-text"


def load_faq_documents(csv_path: str) -> list[Document]:
    df = pd.read_csv(csv_path)
    docs = []
    for _, row in df.iterrows():
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