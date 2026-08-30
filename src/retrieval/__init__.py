"""
STEP 5 / 10: Retriever assembly.

Every script calls build_retriever(): paths and backend selection live here,
nowhere else.
"""

from pathlib import Path

from src.embeddings import get_embedder
from src.retrieval.retriever import Hit, Retriever

ROOT = Path(__file__).resolve().parents[2]


def build_retriever(strategy: str = "dense") -> Retriever:
    embedder = get_embedder()
    return Retriever(
        index_dir=ROOT / "data" / "index" / embedder.name,
        chunks_path=ROOT / "data" / "nfcorpus" / "chunks.jsonl",
        embedder=embedder,
        strategy=strategy,
    )


__all__ = ["Hit", "Retriever", "build_retriever"]