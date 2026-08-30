"""
STEP 9 / 10: LLM-as-a-judge faithfulness scoring.

Goal: check that each claim in an answer is actually supported by the passages
it cites, rather than plausible-sounding text the model produced on its own.

Method: one judgment per sentence, with only the cited passages in view. Judging
sentence by sentence rather than whole answers keeps the task narrow, which is
where small local models stay reliable.

Known limitation, stated rather than hidden: the judge is itself an LLM, and a
small local one. Its verdicts correlate with human judgment but do not replace
it. A stronger judge model can be set via JUDGE_MODEL without touching the code.
"""

import os
import re
from src.eval.guardrails import CITATION_RE, MIN_CLAIM_CHARS, cited_numbers, split_sentences

from src.generation.llm import LLM, OllamaLLM

JUDGE_SYSTEM = """You verify whether a claim is supported by evidence.

Reply with exactly one word:
SUPPORTED    if the evidence states or directly implies the claim
PARTIAL      if the evidence is related but does not fully support the claim
UNSUPPORTED  if the evidence does not support the claim

Judge only what the evidence says. Your own knowledge is irrelevant."""

SCORES = {"SUPPORTED": 1.0, "PARTIAL": 0.5, "UNSUPPORTED": 0.0}


def get_judge() -> LLM:
    return OllamaLLM(model=os.getenv("JUDGE_MODEL", "qwen2.5:3b-instruct"))


def faithfulness(answer_text: str, passages: list[str], judge: LLM) -> float | None:
    """Share of cited claims supported by their evidence, in [0, 1]. None if no claims."""
    verdicts = []

    for sentence in split_sentences(answer_text):
        if len(sentence) < MIN_CLAIM_CHARS:
            continue

        
        refs = [r for r in cited_numbers(sentence) if 1 <= r <= len(passages)]
        if not refs:
            continue

        # Only the cited passages are shown, so the judge cannot rescue a claim
        # using evidence the answer never pointed to.
        evidence = "\n\n".join(passages[r - 1] for r in refs)
        claim = CITATION_RE.sub("", sentence).strip()

        raw = judge.complete(JUDGE_SYSTEM, f"Evidence:\n{evidence}\n\nClaim: {claim}\n\nVerdict:")
        verdict = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip().upper()

        for label, score in SCORES.items():
            if label in verdict:
                verdicts.append(score)
                break

    return sum(verdicts) / len(verdicts) if verdicts else None