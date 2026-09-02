"""
Golden dataset preparation.

Goal: dump the passages retrieved for each in-domain question, so expected
answer points can be written FROM the corpus rather than from memory.

This matters more than it looks. A golden dataset written from general knowledge
measures whether the model knows medicine. Written from the corpus, it measures
what this project actually claims: whether the system reports what its sources
say. Those are different questions, and only the second one is answerable here.

Output: docs/golden_draft.md, to be edited by hand.

Usage: python -m scripts.prepare_golden
"""

from pathlib import Path

from dotenv import load_dotenv

from src.eval.questions import IN_DOMAIN
from src.retrieval import build_retriever

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "golden_draft.md"

load_dotenv(ROOT / ".env")


def main() -> None:
    # Same configuration as the answer path, so the passages listed here are the
    # ones the system actually sees.
    retriever = build_retriever(strategy="hybrid", diversify=True, rerank=True)

    lines = ["# Golden dataset draft\n",
             "For each question, write 2 to 4 expected points based ONLY on the\n"
             "passages below. One short factual claim per line.\n"]

    for question in IN_DOMAIN:
        lines.append(f"\n---\n\n## {question}\n")
        lines.append("**Expected points** (fill this in):\n")
        lines.append("- \n- \n- \n")
        lines.append("<details><summary>Retrieved passages</summary>\n")

        for hit in retriever.retrieve(question, k=5):
            lines.append(f"\n**[{hit.rank}] {hit.doc_id}** (score {hit.score:.3f})\n")
            lines.append(f"> {hit.text}\n")

        lines.append("\n</details>\n")

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join(lines))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()