"""
STEP 3 / 10: Backend selection.

The whole pipeline calls get_embedder(). Which provider answers is decided by
the EMBEDDING_BACKEND environment variable, nowhere else.
"""

import os

from .base import Embedder
from .ollama import OllamaEmbedder


def get_embedder() -> Embedder:
    backend = os.getenv("EMBEDDING_BACKEND", "ollama").lower()
    if backend == "ollama":
        return OllamaEmbedder()
    # The Azure implementation lands here later, behind the same interface.
    raise ValueError(f"Unknown embedding backend: {backend}")


__all__ = ["Embedder", "get_embedder"]