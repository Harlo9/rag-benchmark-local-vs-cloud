"""
STEP 5 / 10: Retriever assembly.

Every script calls build_retriever(): paths and backend selection live here.

The embedder import is deferred inside the function on purpose. Importing this
package should not pull in an inference client: fusion, MMR and BM25 are pure
computation and must stay importable, and testable, without Ollama installed.
"""

from pathlib import Path

from src.retrieval.retriever import Hit, Retriever

ROOT = Path(__file__).resolve().parents[2]


def build_retriever(strategy: str = "dense", rerank: bool = False,
                    diversify: bool = False) -> Retriever:
    from src.embeddings import get_embedder

    embedder = get_embedder()
    return Retriever(
        index_dir=ROOT / "data" / "index" / embedder.name,
        chunks_path=ROOT / "data" / "nfcorpus" / "chunks.jsonl",
        embedder=embedder,
        strategy=strategy,
        rerank=rerank,
        diversify=diversify,
    )


__all__ = ["Hit", "Retriever", "build_retriever"]