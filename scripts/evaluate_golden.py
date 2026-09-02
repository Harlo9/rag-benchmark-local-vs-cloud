"""
Completeness evaluation against the golden dataset.

Goal: measure what faithfulness cannot. Faithfulness asks whether what was said
is supported. This asks whether what should have been said was said at all.

Three metrics:
    coverage        share of expected points present in the answer
    absence_respect share of expected absences the answer did not claim
    correct_refusal whether unanswerable questions were declined

Judging is per point, not per answer: the judge sees one expected point and the
answer, and decides only whether that point is present. Narrow questions are
where a small local judge stays reliable.

Usage: python -m scripts.evaluate_golden [tag]
"""

import json
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.eval.golden import GOLDEN, UNANSWERABLE
from src.eval.judge import get_judge
from src.generation.answer import answer_question
from src.generation.llm import get_llm
from src.retrieval import build_retriever

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
K = 5
TAG = sys.argv[1] if len(sys.argv) > 1 else "local"

load_dotenv(ROOT / ".env")

COVERAGE_SYSTEM = """You check whether an answer contains a specific piece of information.

Reply with exactly one word:
PRESENT  if the answer states this information, in any wording
ABSENT   if the answer does not state it

Paraphrase counts as present. Do not judge whether the information is true."""


def covers(answer: str, point: str, judge) -> bool:
    raw = judge.complete(COVERAGE_SYSTEM, f"Answer:\n{answer}\n\nInformation: {point}\n\nVerdict:")
    verdict = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip().upper()
    return "PRESENT" in verdict


def main() -> None:
    retriever = build_retriever(strategy="hybrid", diversify=True, rerank=True)
    llm, judge = get_llm(), get_judge()

    records, covered, expected, absences_ok, absences_total = [], 0, 0, 0, 0

    for question, golden in GOLDEN.items():
        answer = answer_question(question, retriever, llm, k=K)
        print(f"\n{question}")

        if answer.abstained:
            # An abstention covers nothing. Counted as zero rather than skipped,
            # since refusing an answerable question is a completeness failure.
            print("  declined -> coverage 0")
            expected += len(golden["points"])
            records.append({"question": question, "abstained": True, "coverage": 0.0})
            continue

        hits = [covers(answer.text, p, judge) for p in golden["points"]]
        covered += sum(hits)
        expected += len(hits)
        for point, hit in zip(golden["points"], hits):
            print(f"  {'ok  ' if hit else 'MISS'} {point[:70]}")

        # An absence is respected when the answer does NOT claim it.
        misses = [not covers(answer.text, a, judge) for a in golden["absences"]]
        absences_ok += sum(misses)
        absences_total += len(misses)
        for absence, ok in zip(golden["absences"], misses):
            print(f"  {'ok  ' if ok else 'CLAIMED'} (absence) {absence[:60]}")

        records.append({
            "question": question,
            "abstained": False,
            "coverage": sum(hits) / len(hits) if hits else None,
            "missed": [p for p, h in zip(golden["points"], hits) if not h],
            "answer": answer.text,
        })

    # Unanswerable questions: the only correct behaviour is to decline.
    refusals = []
    for question in UNANSWERABLE:
        answer = answer_question(question, retriever, llm, k=K)
        refusals.append(answer.abstained)
        print(f"\n{question}\n  {'ok   declined' if answer.abstained else 'FAIL answered'}")

    summary = {
        "coverage": covered / expected if expected else None,
        "absence_respect": absences_ok / absences_total if absences_total else None,
        "correct_refusal": sum(refusals) / len(refusals) if refusals else None,
        "questions": len(GOLDEN),
        "points_expected": expected,
        "points_covered": covered,
    }

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"golden-{TAG}.json").write_text(json.dumps(
        {"tag": TAG, "summary": summary, "records": records}, indent=2
    ))
    print("\n" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()