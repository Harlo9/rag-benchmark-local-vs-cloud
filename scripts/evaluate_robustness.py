"""
Robustness to phrasing.

Goal: measure what happens to answerable questions when they are asked badly.

Each canonical question is its own control: a variant is only counted as a
false abstention if the canonical form was answered. That turns a vague worry
("does it break on bad phrasing?") into a delta against a known baseline.

Reported metrics:
    false_abstention   variant declined, canonical answered. The failure that
                       kills adoption: users reformulate twice, then stop.
    guardrail_break    variant answered but with uncited or invalid claims.
    stable             variant behaved like its canonical form.

Usage: python -m scripts.evaluate_robustness [tag]
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

from src.eval import guardrails
from src.eval.questions import PARAPHRASED
from src.generation.answer import answer_question
from src.generation.llm import get_llm
from src.retrieval import build_retriever

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
K = 5
TAG = sys.argv[1] if len(sys.argv) > 1 else "local"

load_dotenv(ROOT / ".env")


def ask(question: str, retriever, llm) -> dict:
    answer = answer_question(question, retriever, llm, k=K)
    report = guardrails.check(answer.text, len(answer.sources), answer.abstained)
    return {
        "question": question,
        "abstained": answer.abstained,
        "guardrail_passed": report.passed,
        # Which documents were retrieved: lets us see whether a bad phrasing
        # broke retrieval or only the generation step.
        "doc_ids": sorted({h.doc_id for h in answer.sources}),
        "answer": answer.text,
    }


def main() -> None:
    retriever = build_retriever(strategy="hybrid", diversify=True, rerank=True)
    llm = get_llm()

    records, by_style = [], defaultdict(lambda: {"false_abstention": 0, "guardrail_break": 0, "stable": 0})

    for canonical, variants in PARAPHRASED.items():
        base = ask(canonical, retriever, llm)
        print(f"\n{canonical}  ->  {'declined' if base['abstained'] else 'answered'}")

        for style, variant in variants.items():
            got = ask(variant, retriever, llm)

            # Overlap of retrieved documents tells us whether retrieval drifted.
            overlap = len(set(base["doc_ids"]) & set(got["doc_ids"]))

            if got["abstained"] and not base["abstained"]:
                verdict = "false_abstention"
            elif not got["guardrail_passed"]:
                verdict = "guardrail_break"
            else:
                verdict = "stable"

            by_style[style][verdict] += 1
            records.append({
                "canonical": canonical, "style": style, "variant": variant,
                "verdict": verdict, "doc_overlap": overlap,
                "base_docs": len(base["doc_ids"]), **got,
            })
            print(f"  {style:<11} {verdict:<17} docs shared with canonical: {overlap}/{len(base['doc_ids'])}")

    total = len(records)
    summary = {
        "variants_tested": total,
        "false_abstention_rate": sum(r["verdict"] == "false_abstention" for r in records) / total,
        "guardrail_break_rate": sum(r["verdict"] == "guardrail_break" for r in records) / total,
        "stable_rate": sum(r["verdict"] == "stable" for r in records) / total,
        # Average share of retrieved documents preserved under rephrasing.
        "retrieval_overlap": sum(r["doc_overlap"] / max(r["base_docs"], 1) for r in records) / total,
        "by_style": {s: dict(v) for s, v in by_style.items()},
    }

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"robustness-{TAG}.json").write_text(json.dumps(
        {"tag": TAG, "summary": summary, "records": records}, indent=2
    ))

    print("\n" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()