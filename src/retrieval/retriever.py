"""
STEP 7 / 10: Retriever with pluggable strategy.

The Hit contract from step 5 is unchanged, so evaluation, generation and the API
keep working while the retrieval strategy varies. That stability is what makes
the ablation table possible: same harness, one variable at a time.
"""

from dataclasses import dataclass
from pathlib import Path

from src.embeddings import Embedder
from src.retrieval.bm25 import BM25Index
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.index import VectorIndex
from src.retrieval.store import ChunkStore


@dataclass
class Hit:
    chunk_id: str
    doc_id: str
    text: str
    score: float
    rank: int


class Retriever:
    def __init__(self, index_dir: Path, chunks_path: Path, embedder: Embedder,
                 strategy: str = "dense"):
        self.strategy = strategy
        self.store = ChunkStore(chunks_path)
        self.dense = VectorIndex(index_dir, embedder) if strategy in ("dense", "hybrid") else None
        self.sparse = BM25Index(chunks_path) if strategy in ("bm25", "hybrid") else None

    def retrieve(self, query: str, k: int = 10) -> list[Hit]:
        if self.strategy == "dense":
            raw = self.dense.search(query, k=k)
        elif self.strategy == "bm25":
            raw = self.sparse.search(query, k=k)
        elif self.strategy == "hybrid":
            # Each retriever contributes k candidates; fusion picks the best k overall.
            raw = reciprocal_rank_fusion(
                [self.dense.search(query, k=k), self.sparse.search(query, k=k)]
            )[:k]
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        return [
            Hit(chunk_id=h["chunk_id"], doc_id=h["doc_id"], text=self.store.text(h["chunk_id"]),
                score=h["score"], rank=rank)
            for rank, h in enumerate(raw, 1)
        ]