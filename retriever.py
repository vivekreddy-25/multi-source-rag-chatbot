"""
Builds the merged retriever for the project.
It loads the three FAISS indexes and combines their top matches into a single list of documents.
This is the retrieval layer used by the RAG chain before the model answers a question.
"""
import os
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document

# FAISS indexes are stored on disk so we can reuse them without re-embedding every query.
FAISS_ROOT  = "faiss_store"
# This matches the local embedding model served by Ollama.
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
        # Each source contributes a small, focused slice of evidence.
        # Using top-3 from each index keeps the final prompt informative without becoming noisy.
        return (
            faq_retriever.invoke(query)
            + tickets_retriever.invoke(query)
            + guides_retriever.invoke(query)
        )

    return RunnableLambda(retrieve)