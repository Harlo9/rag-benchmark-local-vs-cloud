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

Note on citation styles: different models cite differently. A pattern matching
only [2] silently reports [1,5,3] as uncited, which turns a working answer into
a false hallucination report. The measurement must not depend on the model's
formatting habits.
"""

import re
from dataclasses import dataclass

# Matches [2], [1][3], and [1,5,3].
CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")

# Sentences shorter than this are usually connectors, not claims.
MIN_CLAIM_CHARS = 25


def cited_numbers(text: str) -> list[int]:
    """Every passage number cited in the text, whatever the grouping style."""
    numbers = []
    for group in CITATION_RE.findall(text):
        numbers.extend(int(n) for n in group.split(","))
    return numbers


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
    """Split into claim-sized units, keeping trailing citations with their claim.

    Models often write "... risk. [1][3]", so splitting on the period alone
    strips the citation off the sentence it belongs to.
    """
    # Only break when the next character is not an opening bracket.
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?!\[)", text) if s.strip()]


def check(answer_text: str, n_passages: int, abstained: bool) -> GuardrailReport:
    cited = cited_numbers(answer_text)

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