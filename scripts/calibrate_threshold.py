"""
Threshold calibration for retrieval-based abstention.

Goal: find the score below which retrieval is not good enough to answer from.

The robustness run showed the real failure mode: a badly phrased question
retrieves tangential passages, and the model answers anyway with uncited claims.
Retrieval scores already carry that signal, nothing was reading them.

Method: compare the top fusion score on questions we know are answerable
(canonical in-domain) against questions we know are not (out-of-domain). If the
two distributions separate, a threshold exists. If they overlap, this approach
cannot work and we say so instead of picking a number that looks nice.

Usage: python -m scripts.calibrate_threshold
"""

import statistics
from pathlib import Path

from dotenv import load_dotenv

from src.eval.questions import IN_DOMAIN, OUT_OF_DOMAIN, PARAPHRASED
from src.retrieval import build_retriever

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def top_scores(questions: list[str], retriever, k: int = 5) -> dict[str, list[float]]:
    """Raw first-stage scores, before fusion.

    RRF scores turned out to be useless here: 1/(60+1) = 0.0164 and 2/(60+1) =
    0.0328 were nearly the only values observed. RRF measures agreement between
    rankers, not relevance, which is exactly what makes it scale-independent and
    exactly what makes it unusable as a confidence signal. Cosine similarity is
    bounded in [0, 1] and does carry that signal.
    """
    dense, sparse = [], []
    for q in questions:
        dense.append(retriever.dense.search(q, k=k)[0]["score"])
        sparse.append(retriever.sparse.search(q, k=k)[0]["score"])
    return {"dense": dense, "bm25": sparse}


def describe(name: str, scores: list[float]) -> None:
    scores = sorted(scores)
    print(f"{name:<24} n={len(scores):<3} min={scores[0]:.4f} "
          f"p25={scores[len(scores) // 4]:.4f} median={statistics.median(scores):.4f} "
          f"max={scores[-1]:.4f}")


def main() -> None:
    retriever = build_retriever(strategy="hybrid", diversify=True)

    sets = {
        "in-domain canonical": IN_DOMAIN,
        "out-of-domain": OUT_OF_DOMAIN,
        "in-domain, vague": [v["vague"] for v in PARAPHRASED.values()],
    }
    scores = {name: top_scores(qs, retriever) for name, qs in sets.items()}

    for signal in ("dense", "bm25"):
        print(f"\n--- {signal} top-1 score ---")
        for name, values in scores.items():
            describe(name, values[signal])

        answerable = scores["in-domain canonical"][signal]
        unanswerable = scores["out-of-domain"][signal]
        print(f"highest out-of-domain : {max(unanswerable):.4f}")
        print(f"lowest in-domain      : {min(answerable):.4f}")
        print(f"separable             : {max(unanswerable) < min(answerable)}")


if __name__ == "__main__":
    main()

