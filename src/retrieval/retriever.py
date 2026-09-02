"""
STEP 7 / 10: Retriever with pluggable strategy.

The Hit contract from step 5 is unchanged, so evaluation, generation and the API
keep working while the retrieval strategy varies. That stability is what makes
the ablation table possible: same harness, one variable at a time.
"""

from dataclasses import dataclass
from pathlib import Path

from src.retrieval.bm25 import BM25Index
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.index import VectorIndex
from src.retrieval.store import ChunkStore
import numpy as np 

@dataclass
class Hit:
    chunk_id: str
    doc_id: str
    text: str
    score: float
    rank: int


class Retriever:
    def __init__(self, index_dir: Path, chunks_path: Path, embedder: "Embedder",
                 strategy: str = "dense", rerank: bool = False,
                 candidates: int = 50, diversify: bool = False):

        self.strategy = strategy
        self.candidates = candidates
        self.diversify = diversify
        self.store = ChunkStore(chunks_path)
        self.dense = VectorIndex(index_dir, embedder) if strategy in ("dense", "hybrid") else None
        self.sparse = BM25Index(chunks_path) if strategy in ("bm25", "hybrid") else None

        # Imported lazily: loading the cross-encoder costs seconds and memory,
        # so runs without reranking should not pay for it.
        self.reranker = None
        if rerank:
            from src.retrieval.rerank import Reranker
            self.reranker = Reranker()

    def retrieve(self, query: str, k: int = 10) -> list[Hit]:
        # Reranking only reorders what first-stage retrieval found, so its depth
        # caps the final quality: nothing outside the candidate pool can be recovered.
        depth = max(k, self.candidates) if (self.reranker or self.diversify) else k

        if self.strategy == "dense":
            raw = self.dense.search(query, k=depth)
        elif self.strategy == "bm25":
            raw = self.sparse.search(query, k=depth)
        elif self.strategy == "hybrid":
            raw = reciprocal_rank_fusion(
                [self.dense.search(query, k=depth), self.sparse.search(query, k=depth)]
            )[:depth]
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        for h in raw:
            h["text"] = self.store.text(h["chunk_id"])

        if self.reranker:
            # Return more than k when diversification follows, otherwise MMR has
            # nothing left to choose from.
            keep = self.candidates if self.diversify else k
            raw = self.reranker.rerank(query, raw, top_k=keep)

        if self.diversify and self.dense is not None and len(raw) > k:
            from src.retrieval.mmr import mmr_select
            qv = self.dense.embed_query(query)
            cv = self.dense.vectors_for([h["chunk_id"] for h in raw])

            # After reranking, relevance comes from the cross-encoder, not from
            # cosine similarity: otherwise MMR discards the reranking entirely.
            rel = None
            if self.reranker:
                scores = np.array([h["score"] for h in raw], dtype=np.float32)
                # Cross-encoder scores are unbounded logits; MMR mixes relevance
                # and redundancy additively, so both must share a scale.
                rel = (scores - scores.min()) / (np.ptp(scores) or 1.0)

            raw = [raw[i] for i in mmr_select(qv, cv, k=k, relevance=rel)]
        elif not self.diversify:
            raw = raw[:k]
        
        return [
            Hit(chunk_id=h["chunk_id"], doc_id=h["doc_id"], text=h["text"],
                score=h["score"], rank=rank)
            for rank, h in enumerate(raw, 1)
        ]