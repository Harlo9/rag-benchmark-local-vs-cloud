"""
STEP 5 / 10: Retriever.

Goal: expose one stable entry point, retrieve(query, k) -> list[Hit], used by
evaluation (step 6), generation (step 8) and the API (step 10).

The output shape is the contract. Search strategy will change in step 7
(hybrid, reranking) but this signature must not, so nothing downstream breaks.
"""

from dataclasses import dataclass
from pathlib import Path

from src.embeddings import Embedder
from src.retrieval.index import VectorIndex
from src.retrieval.store import ChunkStore


@dataclass
class Hit:
    """One retrieved chunk, with everything downstream needs."""

    chunk_id: str
    doc_id: str      # parent document, required to score against qrels in step 6
    text: str        # the passage itself, fed to the LLM in step 8
    score: float
    rank: int        # 1-based position, kept so reranking effects stay visible


class Retriever:
    def __init__(self, index_dir: Path, chunks_path: Path, embedder: Embedder):
        self.index = VectorIndex(index_dir, embedder)
        self.store = ChunkStore(chunks_path)

    def retrieve(self, query: str, k: int = 10) -> list[Hit]:
        raw = self.index.search(query, k=k)
        return [
            Hit(
                chunk_id=h["chunk_id"],
                doc_id=h["doc_id"],
                text=self.store.text(h["chunk_id"]),
                score=h["score"],
                rank=rank,
            )
            for rank, h in enumerate(raw, 1)
        ]