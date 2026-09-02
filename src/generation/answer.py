"""
STEP 8 / 10: The RAG pipeline, end to end.

Goal: question in, grounded answer out, with the sources that produced it.

The returned object carries the retrieved passages alongside the answer, not
just the text. Step 9 needs them to verify grounding, and step 10 needs them to
make citations clickable. An answer without its sources is not auditable.
"""

import re
import time
from dataclasses import dataclass, field

from src.generation.llm import LLM
from src.generation.prompts import SYSTEM_PROMPT, build_user_prompt
from src.retrieval import Hit, Retriever
from src.retrieval.triage import TriageResult, Verdict, triage

INSUFFICIENT = "INSUFFICIENT_CONTEXT"


@dataclass
class Answer:
    question: str
    text: str
    sources: list[Hit]
    cited: list[int] = field(default_factory=list)  # passage numbers actually cited
    abstained: bool = False
    retrieval_ms: int = 0
    generation_ms: int = 0
    triage: TriageResult | None = None

def cited_passages(text: str, n_passages: int) -> list[int]:
    """Extract cited passage numbers, keeping only those pointing at a real passage."""
    from src.eval.guardrails import cited_numbers
    found = set(cited_numbers(text))
    return sorted(i for i in found if 1 <= i <= n_passages)


def answer_question(question: str, retriever: Retriever, llm: LLM, k: int = 5,
                    use_triage: bool = True) -> Answer:
    t0 = time.time()

    # Confidence triage runs before generation, not after: an answer that should
    # not have been produced is cheaper to prevent than to detect.
    verdict = triage(question, retriever) if use_triage else None
    hits = retriever.retrieve(question, k=k)
    retrieval_ms = int((time.time() - t0) * 1000)

    if verdict and not verdict.should_answer:
        # No LLM call at all: the refusal is deterministic and costs nothing.
        return Answer(
            question=question,
            text=f"{INSUFFICIENT}: {verdict.message()}",
            sources=hits,
            abstained=True,
            triage=verdict,
            retrieval_ms=retrieval_ms,
        )

    t1 = time.time()
    raw = llm.complete(SYSTEM_PROMPT, build_user_prompt(question, [h.text for h in hits]))
    generation_ms = int((time.time() - t1) * 1000)

    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    return Answer(
        question=question,
        text=text,
        sources=hits,
        cited=cited_passages(text, len(hits)),
        abstained=INSUFFICIENT in text,
        triage=verdict,
        retrieval_ms=retrieval_ms,
        generation_ms=generation_ms,
    )