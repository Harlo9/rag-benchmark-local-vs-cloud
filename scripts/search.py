"""
STEP 4 / 10: Manual search, to eyeball retrieval quality before measuring it.

Usage: python -m scripts.search "why is broccoli good for you"
"""

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.embeddings import get_embedder
from src.retrieval.index import VectorIndex

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


def main() -> None:
    query = " ".join(sys.argv[1:]) or "vitamin d and mortality"

    embedder = get_embedder()
    index = VectorIndex(ROOT / "data" / "index" / embedder.name, embedder)

    # Chunk texts live in the chunks file, not in the index, so we load them
    # only for display.
    texts = {
        json.loads(line)["chunk_id"]: json.loads(line)["text"]
        for line in (ROOT / "data" / "nfcorpus" / "chunks.jsonl").open()
    }

    print(f"query: {query}\n")
    for rank, hit in enumerate(index.search(query, k=5), 1):
        preview = texts[hit["chunk_id"]][:200].replace("\n", " ")
        print(f"{rank}. [{hit['score']:.3f}] {hit['doc_id']}\n   {preview}...\n")


if __name__ == "__main__":
    main()