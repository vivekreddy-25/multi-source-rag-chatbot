"""
Ingests data/telecom_guide.pdf into a FAISS index at faiss_store/guides.
Applies RecursiveCharacterTextSplitter to break the long document into chunks.
Run once (or after regenerating the PDF): python ingest_pdf.py
"""
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

FAISS_DIR  = os.path.join("faiss_store", "guides")
PDF_PATH   = os.path.join("data", "telecom_guide.pdf")
EMBED_MODEL = "nomic-embed-text"

CHUNK_SIZE    = 600
CHUNK_OVERLAP = 100


def main():
    print("Loading PDF...")
    loader = PyPDFLoader(PDF_PATH)
    pages = loader.load()
    print(f"  {len(pages)} pages loaded.")

    print(f"Chunking (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(pages)

    for i, chunk in enumerate(chunks):
        chunk.metadata["source"] = "guide"
        chunk.metadata["chunk_index"] = i

    print(f"  {len(chunks)} chunks produced.")

    print("Initialising embedding model...")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    print(f"Embedding and storing FAISS index at '{FAISS_DIR}'...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    os.makedirs(FAISS_DIR, exist_ok=True)
    vectorstore.save_local(FAISS_DIR)
    print(f"  Done. {len(chunks)} vectors stored.")


if __name__ == "__main__":
    main()