"""
STEP 4 / 10: Vector index and search.

Goal: given a query, return the k most semantically similar chunks.

No vector database here, on purpose. With ~6k vectors of 1024 dims the whole
matrix is ~25 MB and an exhaustive dot product runs in milliseconds, so a
dependency like FAISS or Qdrant would add operational cost with no measured
benefit. Step 7 revisits this once the corpus grows (TREC-COVID, 171k docs).

Vectors are L2-normalised at load time, which turns cosine similarity into a
plain dot product.
"""

import json
from pathlib import Path

import numpy as np

from src.embeddings import Embedder


class VectorIndex:
    def __init__(self, index_dir: Path, embedder: Embedder):
        self.embedder = embedder
        self.matrix = np.load(index_dir / "embeddings.npy")
        self.meta = json.loads((index_dir / "chunk_ids.json").read_text())

        # Row order is the contract between matrix and metadata: row i is meta[i].
        if len(self.meta) != self.matrix.shape[0]:
            raise ValueError("embeddings and chunk_ids are out of sync")

        self.matrix = self._normalise(self.matrix)

    @staticmethod
    def _normalise(m: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        norms[norms == 0] = 1.0   # guard against zero vectors
        return m / norms

    def search(self, query: str, k: int = 10) -> list[dict]:
        """Return the k chunks closest to the query, best first."""
        vector = np.asarray(self.embedder.embed([query])[0], dtype=np.float32)
        vector /= np.linalg.norm(vector) or 1.0

        scores = self.matrix @ vector   # cosine similarity for every chunk at once

        # argpartition finds the top k without sorting the whole array, then we
        # sort only those k. Cheaper than a full sort over the corpus.
        top = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
        top = top[np.argsort(-scores[top])]

        return [
            {
                "chunk_id": self.meta[i]["chunk_id"],
                "doc_id": self.meta[i]["doc_id"],
                "score": float(scores[i]),
            }
            for i in top
        ]