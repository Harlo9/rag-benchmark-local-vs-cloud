"""
STEP 5 / 10: Manual search, now going through the Retriever component.

Usage: python -m scripts.search "why is broccoli good for you"
"""

import sys
from pathlib import Path

from dotenv import load_dotenv

from src.retrieval import build_retriever

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def main() -> None:
    query = " ".join(sys.argv[1:]) or "vitamin d and mortality"
    retriever = build_retriever()

    print(f"query: {query}\n")
    for hit in retriever.retrieve(query, k=5):
        preview = hit.text[:200].replace("\n", " ")
        print(f"{hit.rank}. [{hit.score:.3f}] {hit.doc_id}\n   {preview}...\n")


if __name__ == "__main__":
    main()