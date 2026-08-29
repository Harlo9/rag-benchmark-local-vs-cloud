"""
STEP 2 / 10: Loading and chunking.

Goal: read data/nfcorpus/corpus.jsonl and turn it into chunks, the retrieval
unit of the RAG pipeline.

Output: data/nfcorpus/chunks.jsonl, one chunk per line, each keeping its
parent document id (required for evaluation in step 6).

Usage: python src/chunking.py
"""

import json
from pathlib import Path

import tiktoken

DATA = Path(__file__).resolve().parents[1] / "data" / "nfcorpus"
CHUNK_SIZE = 300      # in tokens, not characters: tokens are what the model counts
CHUNK_OVERLAP = 50    # overlap, so an idea is never split across two chunks

# Splitting on tokens rather than words, because tokens are the unit seen by
# both the embedding model and the LLM.
enc = tiktoken.get_encoding("cl100k_base")


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Split text into overlapping windows of `size` tokens."""
    tokens = enc.encode(text)
    chunks, start = [], 0
    while start < len(tokens):
        end = start + size
        chunks.append(enc.decode(tokens[start:end]))
        if end >= len(tokens):
            break
        start = end - overlap   # step back to create the overlap
    return chunks


def main() -> None:
    n_docs = n_chunks = 0
    with (DATA / "corpus.jsonl").open() as src, (DATA / "chunks.jsonl").open("w") as out:
        for line in src:
            doc = json.loads(line)
            # The title is prepended to the body: it usually carries the topic,
            # and without it an isolated chunk loses its anchor.
            full = f"{doc['title']}\n\n{doc['text']}".strip()
            if not full:
                continue
            n_docs += 1
            for i, piece in enumerate(chunk_text(full, CHUNK_SIZE, CHUNK_OVERLAP)):
                out.write(json.dumps({
                    "chunk_id": f"{doc['id']}::{i}",  # unique chunk identifier
                    "doc_id": doc["id"],              # parent, needed for evaluation (step 6)
                    "position": i,
                    "text": piece,
                }) + "\n")
                n_chunks += 1

    print(f"documents {n_docs}\nchunks    {n_chunks}\nratio     {n_chunks / n_docs:.2f}")


if __name__ == "__main__":
    main()