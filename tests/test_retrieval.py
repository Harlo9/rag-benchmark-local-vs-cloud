"""
Tests for fusion and diversification.

The MMR test guards a real bug: when a reranker ran first, MMR recomputed
relevance from dense cosine similarity and silently discarded the cross-encoder
ordering. Retrieval overlap was identical to four decimal places across runs,
which is what exposed it.
"""

import numpy as np

from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.mmr import mmr_select


def hit(chunk_id: str) -> dict:
    return {"chunk_id": chunk_id, "doc_id": chunk_id.split("::")[0], "score": 0.0}


def test_fusion_rewards_agreement_between_rankers():
    # "b" is second in both lists, "a" and "c" are first in one list only.
    dense = [hit("a"), hit("b")]
    sparse = [hit("c"), hit("b")]
    fused = reciprocal_rank_fusion([dense, sparse])
    assert fused[0]["chunk_id"] == "b"


def test_fusion_uses_rank_not_score():
    # Scores are deliberately absurd: RRF must ignore them entirely, which is
    # what makes it safe to mix cosine similarity with BM25.
    dense = [{**hit("a"), "score": 999.0}, {**hit("b"), "score": 0.001}]
    sparse = [{**hit("b"), "score": 0.001}, {**hit("a"), "score": 999.0}]
    fused = reciprocal_rank_fusion([dense, sparse])
    assert {h["chunk_id"] for h in fused} == {"a", "b"}


def test_fusion_deduplicates():
    same = [hit("a"), hit("b")]
    fused = reciprocal_rank_fusion([same, same])
    assert len(fused) == 2


def test_mmr_picks_the_most_relevant_first():
    query = np.array([1.0, 0.0], dtype=np.float32)
    cands = np.array([[0.5, 0.87], [1.0, 0.0]], dtype=np.float32)
    assert mmr_select(query, cands, k=1)[0] == 1


def test_mmr_prefers_a_novel_candidate_over_a_near_duplicate():
    query = np.array([1.0, 0.0], dtype=np.float32)
    # Row 1 is a near-duplicate of row 0 and slightly less relevant.
    # Row 2 is orthogonal: much less relevant, but it adds something new.
    #   lambda=0.3 -> candidate 1: 0.3*0.99 - 0.7*0.99 = -0.396
    #                 candidate 2: 0.3*0.00 - 0.7*0.00 =  0.000
    cands = np.array([[1.0, 0.0], [0.99, 0.14], [0.0, 1.0]], dtype=np.float32)
    selected = mmr_select(query, cands, k=2, lambda_=0.3)
    assert selected[0] == 0
    assert selected[1] == 2

def test_mmr_keeps_the_near_duplicate_when_relevance_dominates():
    """lambda=1.0 disables diversification entirely: pure relevance ranking."""
    query = np.array([1.0, 0.0], dtype=np.float32)
    cands = np.array([[1.0, 0.0], [0.99, 0.14], [0.0, 1.0]], dtype=np.float32)
    assert mmr_select(query, cands, k=2, lambda_=1.0) == [0, 1]
    
def test_mmr_uses_supplied_relevance_over_cosine():
    """Guards the reranking bug: an external relevance signal must win."""
    query = np.array([1.0, 0.0], dtype=np.float32)
    cands = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    # By cosine, candidate 0 wins. The reranker says otherwise.
    assert mmr_select(query, cands, k=1)[0] == 0
    reranked = np.array([0.0, 1.0], dtype=np.float32)
    assert mmr_select(query, cands, k=1, relevance=reranked)[0] == 1