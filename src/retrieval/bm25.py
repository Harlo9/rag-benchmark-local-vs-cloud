"""
STEP 7 / 10: Sparse retrieval (BM25).

Goal: add a lexical retriever alongside the dense one.

Dense embeddings capture meaning but blur exact terms: drug names, dosages,
rare medical vocabulary. BM25 scores on term overlap weighted by inverse
document frequency, so it excels exactly where dense retrieval is weakest.
The two are complementary, which is what makes fusion worth doing.

Output shape matches VectorIndex.search, so both can be fused blindly.
"""

import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokenizer. Deliberately simple: BM25 needs terms, not linguistics."""
    return TOKEN_RE.findall(text.lower())


class BM25Index:
    def __init__(self, chunks_path: Path):
        self.meta: list[dict] = []
        corpus: list[list[str]] = []

        with chunks_path.open() as f:
            for line in f:
                chunk = json.loads(line)
                self.meta.append({"chunk_id": chunk["chunk_id"], "doc_id": chunk["doc_id"]})
                corpus.append(tokenize(chunk["text"]))

        # Same row-order contract as the dense index: row i is meta[i].
        self.bm25 = BM25Okapi(corpus)

    def search(self, query: str, k: int = 10) -> list[dict]:
        scores = self.bm25.get_scores(tokenize(query))
        top = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [
            {"chunk_id": self.meta[i]["chunk_id"], "doc_id": self.meta[i]["doc_id"],
             "score": float(scores[i])}
            for i in top
        ]