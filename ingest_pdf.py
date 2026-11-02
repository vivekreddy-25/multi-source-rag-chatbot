"""
Ingests the telecom guide PDF into a FAISS index.
This file splits the guide into chunks so the retriever can match more precise pieces of documentation.
Run once (or after regenerating the PDF): python ingest_pdf.py
"""
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

# Storage path for the PDF guide index. We keep this separate from FAQ and tickets so each source can be queried independently.
FAISS_DIR  = os.path.join("faiss_store", "guides")
# Source PDF used as reference material for product/service questions.
PDF_PATH   = os.path.join("data", "telecom_guide.pdf")
# Local embedding model served by Ollama.
EMBED_MODEL = "nomic-embed-text"

# Chunking settings: enough context for a clear answer, but small enough to stay focused and retrievable.
CHUNK_SIZE    = 600
CHUNK_OVERLAP = 100


def main():
    # Load the PDF pages as LangChain documents so the content can be chunked and embedded.
    print("Loading PDF...")
    loader = PyPDFLoader(PDF_PATH)
    pages = loader.load()
    print(f"  {len(pages)} pages loaded.")

    # Split long policy/reference documents into smaller chunks so retrieval stays precise.
    # A chunk size of 600 with 100-character overlap keeps related ideas together without making context too noisy.
    print(f"Chunking (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(pages)

    # Add metadata so later retrieval can tell these chunks apart from FAQ and ticket evidence.
    for i, chunk in enumerate(chunks):
        chunk.metadata["source"] = "guide"
        chunk.metadata["chunk_index"] = i

    print(f"  {len(chunks)} chunks produced.")

    # The embedding model must be initialized before creating the vector index.
    print("Initialising embedding model...")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    # FAISS stores vectors and metadata on disk so the app can load them without re-embedding the PDF every time.
    print(f"Embedding and storing FAISS index at '{FAISS_DIR}'...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    os.makedirs(FAISS_DIR, exist_ok=True)
    vectorstore.save_local(FAISS_DIR)
    print(f"  Done. {len(chunks)} vectors stored.")


if __name__ == "__main__":
    main()