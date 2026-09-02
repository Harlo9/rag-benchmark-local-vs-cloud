"""
Retrieval confidence triage.

Goal: decide, before calling the LLM, whether retrieval found enough to answer.

Why this exists: the robustness evaluation showed that 20% of badly phrased
questions got an answer containing uncited claims. Retrieval had returned
tangential passages, weak enough to be useless and plausible enough that the
model answered anyway. The signal was already there, nothing was reading it.

Why the fusion score cannot be used: RRF scores were almost always 1/(60+1) or
2/(60+1). RRF measures agreement between rankers, not relevance, which is what
makes it scale-independent and what makes it useless as confidence.

Two raw signals, measured separately, because they degrade differently:

    dense cosine   captures topic. Survives vague phrasing, since meaning is
                   preserved even when precise terms are not.
    BM25 score     captures precision. Collapses on vague phrasing, since it
                   has no rare terms left to match.

That asymmetry is what makes three outcomes possible instead of two:

    ANSWER      both signals hold
    CLARIFY     topic is right, wording is too vague to retrieve precisely
    OUT_OF_SCOPE no topical signal at all

Thresholds are calibrated on 25 questions (10 in-domain, 5 out-of-domain,
10 vague variants) for this corpus and this embedding model. They are not
universal constants: see scripts/calibrate_threshold.py to recalibrate.

Observed distributions (top-1 score):
    dense   in-domain 0.589 to 0.774 | vague 0.492 to 0.623 | out 0.263 to 0.358
    bm25    in-domain 16.4 to 27.4   | vague 8.3 to 17.3    | out 12.6 to 15.8
"""

import os
from dataclasses import dataclass
from enum import Enum


class Verdict(str, Enum):
    ANSWER = "answer"
    CLARIFY = "clarify"
    OUT_OF_SCOPE = "out_of_scope"


# Midway between the highest out-of-domain score (0.358) and the lowest vague
# in-domain score (0.492). A 13-point margin, not a value tuned to the decimal.
DENSE_FLOOR = float(os.getenv("TRIAGE_DENSE_FLOOR", "0.45"))

# Just below the lowest canonical in-domain score (16.37) and above the highest
# out-of-domain one (15.79).
BM25_FLOOR = float(os.getenv("TRIAGE_BM25_FLOOR", "16.0"))


@dataclass
class TriageResult:
    verdict: Verdict
    dense_score: float
    bm25_score: float

    @property
    def should_answer(self) -> bool:
        return self.verdict is Verdict.ANSWER

    def message(self) -> str:
        """User-facing text for the two refusal cases."""
        if self.verdict is Verdict.CLARIFY:
            return ("The question seems to be about this corpus, but it is phrased too "
                    "broadly to retrieve precise evidence. Try naming the specific "
                    "substance, condition or outcome you have in mind.")
        return ("Nothing in this corpus addresses that question. The corpus covers "
                "nutrition and health research only.")


def triage(query: str, retriever) -> TriageResult:
    """Read raw first-stage scores and decide whether to call the LLM at all."""
    dense_top = retriever.dense.search(query, k=1)[0]["score"] if retriever.dense else 0.0
    bm25_top = retriever.sparse.search(query, k=1)[0]["score"] if retriever.sparse else 0.0

    if dense_top < DENSE_FLOOR:
        # No topical signal: the question is not about this corpus at all.
        verdict = Verdict.OUT_OF_SCOPE
    elif bm25_top < BM25_FLOOR:
        # Topic is right, but no rare terms to anchor retrieval on. Asking the
        # user to be specific beats answering from tangential passages.
        verdict = Verdict.CLARIFY
    else:
        verdict = Verdict.ANSWER

    return TriageResult(verdict=verdict, dense_score=dense_top, bm25_score=bm25_top)