"""
STEP 7 / 10: Reciprocal Rank Fusion.

Goal: merge several ranked lists into one, without comparing their scores.

Cosine similarity and BM25 scores live on incompatible scales, so normalising
them is fragile. RRF sidesteps this entirely by using rank only: a document at
rank r contributes 1/(k + r). A document ranked well by both retrievers
accumulates from both lists and rises to the top.

k dampens the weight of top ranks. k=60 is the value from the original paper
and the common default; lower values weight the head of each list more heavily.
"""

from collections import defaultdict


def reciprocal_rank_fusion(rankings: list[list[dict]], k: int = 60) -> list[dict]:
    """Fuse ranked hit lists into one, best first."""
    fused: dict[str, float] = defaultdict(float)
    seen: dict[str, dict] = {}

    for ranking in rankings:
        for rank, hit in enumerate(ranking, 1):
            fused[hit["chunk_id"]] += 1.0 / (k + rank)
            seen.setdefault(hit["chunk_id"], hit)

    ordered = sorted(fused.items(), key=lambda kv: -kv[1])
    return [{**seen[cid], "score": score} for cid, score in ordered]