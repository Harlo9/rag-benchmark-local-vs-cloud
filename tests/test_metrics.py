"""
Tests for retrieval metrics.

These are the most load-bearing functions in the project: every number in the
README comes out of them. Nothing verified them until now, so a refactor could
have silently shifted every score.

Expected values are computed by hand in the comments, not copied from a run.
A test that asserts what the code currently does proves nothing.
"""

import math

from src.eval.metrics import ndcg_at_k, recall_at_k


def test_recall_counts_only_positive_judgments():
    # Score 0 means "judged, not relevant", which is different from unjudged.
    relevant = {"a": 2, "b": 1, "c": 0}
    assert recall_at_k(["a", "b"], relevant, k=10) == 1.0
    assert recall_at_k(["a"], relevant, k=10) == 0.5
    assert recall_at_k(["c"], relevant, k=10) == 0.0


def test_recall_respects_the_cutoff():
    relevant = {"a": 1, "b": 1}
    assert recall_at_k(["x", "a", "b"], relevant, k=1) == 0.0
    assert recall_at_k(["x", "a", "b"], relevant, k=3) == 1.0


def test_recall_is_zero_when_nothing_is_relevant():
    assert recall_at_k(["a"], {}, k=10) == 0.0
    assert recall_at_k(["a"], {"a": 0}, k=10) == 0.0


def test_ndcg_is_one_for_a_perfect_ranking():
    relevant = {"a": 2, "b": 1}
    assert ndcg_at_k(["a", "b"], relevant, k=10) == 1.0


def test_ndcg_penalises_a_reversed_ranking():
    # DCG  = 1/log2(2) + 2/log2(3) = 1.0 + 1.2619 = 2.2619
    # IDCG = 2/log2(2) + 1/log2(3) = 2.0 + 0.6309 = 2.6309
    relevant = {"a": 2, "b": 1}
    expected = (1 + 2 / math.log2(3)) / (2 + 1 / math.log2(3))
    assert math.isclose(ndcg_at_k(["b", "a"], relevant, k=10), expected, rel_tol=1e-9)


def test_ndcg_ignores_documents_beyond_k():
    relevant = {"a": 2}
    assert ndcg_at_k(["x", "y", "a"], relevant, k=2) == 0.0
    assert ndcg_at_k(["x", "y", "a"], relevant, k=3) > 0.0


def test_ndcg_is_zero_when_no_judgments_exist():
    assert ndcg_at_k(["a", "b"], {}, k=10) == 0.0