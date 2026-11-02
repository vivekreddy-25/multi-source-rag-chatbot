"""
CLI entry point for the telecom RAG chatbot.
This script takes a user question, runs it through the local RAG chain,
and streams the answer in the terminal.
Usage: python main.py
"""
from rag_chain import build_chain


def main():
    print("=== Telecom Customer Care Chatbot (RAG) ===")
    print("Type your question and press Enter. Type 'quit' to exit.\n")

    # Build the RAG chain once and reuse it for the CLI session.
    chain = build_chain()

    while True:
        question = input("Customer: ").strip()
        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break

        print("\nAssistant: ", end="", flush=True)
        for chunk in chain.stream(question):
            print(chunk, end="", flush=True)
        print("\n")


if __name__ == "__main__":
    main()
