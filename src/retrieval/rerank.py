"""
STEP 7 / 10: Cross-encoder reranking.

Goal: reorder the candidates returned by first-stage retrieval, using a model
that reads query and passage together.

Bi-encoders (what we use for the index) embed query and document separately,
so they never see the interaction between them: that is what makes the index
precomputable, and also what limits its precision. A cross-encoder scores the
pair jointly, which is far more accurate but costs one forward pass per
candidate. Hence the two-stage design: cheap retrieval narrows to 50, expensive
reranking reorders those.

Known risk on this corpus: rerankers trained on web search distributions often
transfer poorly to scientific text, and can degrade ranking while adding
latency. Whether it helps here is an empirical question, which is why it sits
behind a flag and gets measured like everything else.
"""

import os

from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(self, model: str | None = None):
        self.model_name = model or os.getenv(
            "RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        # Runs locally on Apple Silicon through MPS when available.
        self.model = CrossEncoder(self.model_name, max_length=512)

    def rerank(self, query: str, hits: list[dict], top_k: int) -> list[dict]:
        """Rescore candidates against the query and return the best top_k."""
        if not hits:
            return []

        pairs = [(query, h["text"]) for h in hits]
        scores = self.model.predict(pairs, show_progress_bar=False)

        scored = [{**h, "score": float(s)} for h, s in zip(hits, scores)]
        scored.sort(key=lambda h: -h["score"])
        return scored[:top_k]