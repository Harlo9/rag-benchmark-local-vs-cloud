"""
STEP 3 / 10: Embedding the corpus.

Goal: turn every chunk into a vector and persist the result, so indexing and
retrieval never have to recompute embeddings.

Outputs, in data/index/<backend-model>/:
    embeddings.npy   float32 matrix, one row per chunk
    chunk_ids.json   row order, mapping each row back to its chunk and document

Vectors are stored apart from ids on purpose: numpy loads the matrix in a single
read, which stays fast as the corpus grows. The model name is part of the output
path, so two models can be embedded side by side and compared in step 6.

Usage: python scripts/embed_corpus.py
"""

import json
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from src.embeddings import get_embedder

ROOT = Path(__file__).resolve().parents[1]
CHUNKS = ROOT / "data" / "nfcorpus" / "chunks.jsonl"
BATCH_SIZE = 32

load_dotenv(ROOT / ".env")


def main() -> None:
    embedder = get_embedder()
    out_dir = ROOT / "data" / "index" / embedder.name
    out_dir.mkdir(parents=True, exist_ok=True)

    chunks = [json.loads(line) for line in CHUNKS.open()]
    print(f"embedding {len(chunks)} chunks with {embedder.name}")

    vectors, started = [], time.time()
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        vectors.extend(embedder.embed([c["text"] for c in batch]))
        done = min(i + BATCH_SIZE, len(chunks))
        rate = done / (time.time() - started)
        print(f"  {done}/{len(chunks)}  {rate:.0f} chunks/s", end="\r")

    matrix = np.asarray(vectors, dtype=np.float32)

    # Sanity check: a mismatch here means the DIMS table is wrong for this model.
    if matrix.shape[1] != embedder.dim:
        raise ValueError(f"expected dim {embedder.dim}, got {matrix.shape[1]}")

    np.save(out_dir / "embeddings.npy", matrix)

    # Row order is the contract between matrix and metadata: row i of the matrix
    # is chunk_ids[i]. Losing this file makes the matrix meaningless.
    (out_dir / "chunk_ids.json").write_text(json.dumps(
        [{"chunk_id": c["chunk_id"], "doc_id": c["doc_id"]} for c in chunks]
    ))

    print(f"\nshape {matrix.shape}  elapsed {time.time() - started:.0f}s  -> {out_dir}")


if __name__ == "__main__":
    main()