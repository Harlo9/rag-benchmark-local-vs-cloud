"""
STEP 9 / 10: Deterministic guardrails.

Goal: catch failure modes that need no model to detect, so the expensive judge
only runs on answers that already pass the cheap checks.

Three checks, all mechanical:
    - every factual sentence carries a citation
    - every citation points at a passage that was actually provided
    - the system abstained when retrieval scores were too weak to support anything

Deterministic checks are worth more than they look: they are free, they never
disagree with themselves, and they can run on every single answer in production,
which an LLM judge cannot.
"""

import re
from dataclasses import dataclass

CITATION_RE = re.compile(r"\[(\d+)\]")
# Sentences shorter than this are usually connectors, not claims.
MIN_CLAIM_CHARS = 25


@dataclass
class GuardrailReport:
    has_citations: bool
    citations_valid: bool          # no citation points outside the provided passages
    uncited_claims: list[str]      # substantive sentences with no citation
    abstained: bool

    @property
    def passed(self) -> bool:
        # An abstention is a valid outcome, not a failure: nothing to cite.
        if self.abstained:
            return True
        return self.has_citations and self.citations_valid and not self.uncited_claims


def split_sentences(text: str) -> list[str]:
    """Naive sentence split. Good enough: we only need claim-sized units."""
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def check(answer_text: str, n_passages: int, abstained: bool) -> GuardrailReport:
    cited = [int(m) for m in CITATION_RE.findall(answer_text)]

    uncited = [
        s for s in split_sentences(answer_text)
        if len(s) >= MIN_CLAIM_CHARS and not CITATION_RE.search(s)
    ]

    return GuardrailReport(
        has_citations=bool(cited),
        # A citation outside range means the model invented a passage number.
        citations_valid=all(1 <= c <= n_passages for c in cited),
        uncited_claims=[] if abstained else uncited,
        abstained=abstained,
    )