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
    if backend == "azure":
        # Imported lazily so the local path never needs the openai package.
        from .azure import AzureEmbedder
        return AzureEmbedder()
    raise ValueError(f"Unknown embedding backend: {backend}")


__all__ = ["Embedder", "get_embedder"]