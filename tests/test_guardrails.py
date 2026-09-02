"""
Tests for deterministic guardrails.

Every case here comes from a real failure. The grouped-citation tests exist
because an earlier regex matched only [2], which reported a hosted model as
hallucinating 20% of the time when in fact every claim was cited. The bug was
invisible with the local model, which never used those styles.
"""

from src.eval.guardrails import check, cited_numbers, split_sentences


def test_single_citations_are_extracted():
    assert cited_numbers("A claim [2].") == [2]


def test_adjacent_citations_are_extracted():
    # Real output style: "... risk. [1][3][4][2]"
    assert cited_numbers("A claim [1][3][4].") == [1, 3, 4]


def test_grouped_citations_are_extracted():
    # Real output style: "... prostate cancer risk [1,5,3]."
    assert cited_numbers("A claim [1,5,3].") == [1, 5, 3]
    assert cited_numbers("A claim [1, 5, 3].") == [1, 5, 3]


def test_trailing_citation_stays_with_its_sentence():
    # Splitting on the period alone would strip the citation off the claim,
    # which is what made cited answers look uncited.
    sentences = split_sentences("First claim. [1] Second claim. [2]")
    assert all("[" in s for s in sentences)


def test_uncited_claim_is_reported():
    report = check("Fasting improves insulin sensitivity in healthy adults.", 5, abstained=False)
    assert report.uncited_claims
    assert not report.passed


def test_short_connectors_are_not_treated_as_claims():
    report = check("Yes. Fiber lowers cholesterol in these trials [1].", 5, abstained=False)
    assert report.uncited_claims == []
    assert report.passed


def test_citation_outside_range_is_invalid():
    report = check("A claim [9].", 5, abstained=False)
    assert not report.citations_valid
    assert not report.passed


def test_abstention_passes_without_citations():
    # Declining is a valid outcome: there is nothing to cite.
    report = check("INSUFFICIENT_CONTEXT", 5, abstained=True)
    assert report.passed