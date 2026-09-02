"""
Builds a merged retriever across all three FAISS indexes:
  - faq     : FAQ entries (no chunking — 1 row = 1 doc)
  - tickets : resolved support tickets (no chunking — 1 ticket = 1 doc)
  - guides  : PDF guide chunks (RecursiveCharacterTextSplitter applied at ingest)
"""
import os
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document

FAISS_ROOT  = "faiss_store"
EMBED_MODEL = "nomic-embed-text"


def build_retriever(
    k_faq: int = 3,
    k_tickets: int = 3,
    k_guides: int = 3,
) -> RunnableLambda:
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    faq_store = FAISS.load_local(
        os.path.join(FAISS_ROOT, "faq"), embeddings,
        allow_dangerous_deserialization=True,
    )
    tickets_store = FAISS.load_local(
        os.path.join(FAISS_ROOT, "tickets"), embeddings,
        allow_dangerous_deserialization=True,
    )
    guides_store = FAISS.load_local(
        os.path.join(FAISS_ROOT, "guides"), embeddings,
        allow_dangerous_deserialization=True,
    )

    faq_retriever     = faq_store.as_retriever(search_kwargs={"k": k_faq})
    tickets_retriever = tickets_store.as_retriever(search_kwargs={"k": k_tickets})
    guides_retriever  = guides_store.as_retriever(search_kwargs={"k": k_guides})

    def retrieve(query: str) -> list[Document]:
        return (
            faq_retriever.invoke(query)
            + tickets_retriever.invoke(query)
            + guides_retriever.invoke(query)
        )

    return RunnableLambda(retrieve)