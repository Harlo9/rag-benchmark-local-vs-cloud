"""
STEP 8 / 10: Ask a question end to end.

Usage: python -m scripts.ask "does vitamin d reduce mortality"
"""

import sys
from pathlib import Path

from dotenv import load_dotenv

from src.generation.answer import answer_question
from src.generation.llm import get_llm
from src.retrieval import build_retriever

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def main() -> None:
    question = " ".join(sys.argv[1:]) or "does vitamin d reduce mortality"

    retriever = build_retriever(strategy="hybrid", diversify=True)
    result = answer_question(question, retriever, get_llm(), k=5)

    print(f"Q: {result.question}\n")
    print(f"A: {result.text}\n")

    print("Sources:")
    for i, hit in enumerate(result.sources, 1):
        mark = "*" if i in result.cited else " "   # * marks passages actually used
        print(f" {mark}[{i}] {hit.doc_id}  score={hit.score:.3f}")

    print(f"\nretrieval {result.retrieval_ms} ms | generation {result.generation_ms} ms")
    if result.abstained:
        print("model abstained: context judged insufficient")


if __name__ == "__main__":
    main()