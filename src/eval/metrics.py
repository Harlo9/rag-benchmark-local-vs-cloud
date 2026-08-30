"""
STEP 6 / 10: Retrieval metrics.

Goal: turn a ranked list of documents into a number, so pipeline changes can be
compared instead of eyeballed.

Two metrics, answering two different questions:
    recall@k   did we find the relevant documents at all?
    nDCG@k     did we rank them well? (graded relevance, position-weighted)

nDCG is the BEIR standard, which makes these numbers comparable to published
results rather than only to our own runs.
"""

import math


def recall_at_k(retrieved: list[str], relevant: dict[str, int], k: int) -> float:
    """Share of relevant documents present in the top k."""
    # Only positively judged documents count as relevant; score 0 means judged
    # and not relevant, which is different from unjudged.
    gold = {doc for doc, score in relevant.items() if score > 0}
    if not gold:
        return 0.0
    return len(gold & set(retrieved[:k])) / len(gold)


def ndcg_at_k(retrieved: list[str], relevant: dict[str, int], k: int) -> float:
    """Normalised discounted cumulative gain: rewards relevance high in the list."""
    # DCG: each hit contributes its grade, discounted by how far down it sits.
    dcg = sum(
        relevant.get(doc, 0) / math.log2(rank + 1)
        for rank, doc in enumerate(retrieved[:k], 1)
    )

    # IDCG: the score a perfect ranking would obtain, used to normalise to [0, 1].
    ideal = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum(grade / math.log2(rank + 1) for rank, grade in enumerate(ideal, 1))

    return dcg / idcg if idcg else 0.0