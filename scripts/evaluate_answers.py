"""
STEP 9 / 10: End-to-end answer evaluation.

Goal: measure the generation stage the way step 6 measured retrieval, so the
whole pipeline is backed by numbers instead of impressions.

Reported metrics:
    guardrail_pass     share of answers with valid, complete citations
    faithfulness       share of claims supported by their cited evidence
    abstention_rate    share of out-of-domain questions correctly refused
    latency            median end-to-end time per answer

Usage: python -m scripts.evaluate_answers
"""

import json
import statistics
import time
from pathlib import Path

from dotenv import load_dotenv

from src.eval import guardrails
from src.eval.judge import faithfulness, get_judge
from src.eval.questions import IN_DOMAIN, OUT_OF_DOMAIN
from src.generation.answer import answer_question
from src.generation.llm import get_llm
from src.retrieval import build_retriever

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
K = 5

load_dotenv(ROOT / ".env")


def main() -> None:
    retriever = build_retriever(strategy="hybrid", diversify=True)
    llm, judge = get_llm(), get_judge()

    records, latencies = [], []

    for question in IN_DOMAIN + OUT_OF_DOMAIN:
        in_domain = question in IN_DOMAIN
        started = time.time()
        answer = answer_question(question, retriever, llm, k=K)
        latencies.append(time.time() - started)

        report = guardrails.check(answer.text, len(answer.sources), answer.abstained)

        # Faithfulness is only meaningful when the model actually made claims.
        score = None if answer.abstained else faithfulness(
            answer.text, [h.text for h in answer.sources], judge
        )

        records.append({
            "question": question,
            "in_domain": in_domain,
            "abstained": answer.abstained,
            "guardrail_passed": report.passed,
            "uncited_claims": len(report.uncited_claims),
            "faithfulness": score,
            "answer": answer.text,
        })
        print(f"  {'ok ' if report.passed else 'FAIL'} {question[:50]}")

    scored = [r["faithfulness"] for r in records if r["faithfulness"] is not None]
    ood = [r for r in records if not r["in_domain"]]

    summary = {
        "guardrail_pass": sum(r["guardrail_passed"] for r in records) / len(records),
        "faithfulness": sum(scored) / len(scored) if scored else None,
        # The key safety metric: did the system refuse what it could not answer?
        "abstention_rate_ood": sum(r["abstained"] for r in ood) / len(ood),
        # Abstaining on answerable questions is a failure too, in the other direction.
        "abstention_rate_in_domain": sum(
            r["abstained"] for r in records if r["in_domain"]
        ) / len(IN_DOMAIN),
        "median_latency_s": round(statistics.median(latencies), 1),
    }

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "answers.json").write_text(json.dumps(
        {"summary": summary, "records": records}, indent=2
    ))

    print("\n" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()