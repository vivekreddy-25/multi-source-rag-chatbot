"""
Builds the RAG chain for the telecom support chatbot.
The chain retrieves relevant documents, formats them into context, and passes them to the local
Llama 3.1 8B model served by Ollama.
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_ollama import ChatOllama

from retriever import build_retriever

SYSTEM_PROMPT = """You are a helpful and professional telecom customer care assistant.
Your job is to help customers resolve technical issues with their mobile service.

Use ONLY the context below to answer the customer's question.
The context comes from three sources:
- FAQ entries (general policy and how-to information)
- Past support tickets (real resolved cases with step-by-step resolutions)
- Telecom guide chunks (reference material and setup instructions)

If the context does not contain enough information to answer confidently, say so clearly \
and suggest the customer call 611 or use the MyTelecom app.

Context:
{context}
"""


def _format_docs(docs: list[Document]) -> str:
    # Keep the retrieved evidence readable for the model by labeling each document with its source.
    sections = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown").upper()
        sections.append(f"[{source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(sections)


def build_chain():
    # The retriever is the retrieval layer; the chain combines it with a system prompt and local LLM.
    retriever = build_retriever()

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    llm = ChatOllama(
        model="llama3.1:8b",
        temperature=0,
        max_tokens=None,
        # Keep local reasoning output predictable for this small support assistant.
        reasoning_format="parsed",
        timeout=None,
        max_retries=2,
    )

    chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain
