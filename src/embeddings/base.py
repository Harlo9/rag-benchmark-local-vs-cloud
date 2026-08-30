"""
STEP 3 / 10: Embedding interface.

Goal: define the contract every embedding backend must satisfy, so the rest of
the pipeline never knows which provider is in use. Swapping local Ollama for
Azure OpenAI becomes a config change, not a code change.

This also makes the benchmark in step 6 possible: two backends, same interface,
same test set, comparable numbers.
"""

from abc import ABC, abstractmethod


class Embedder(ABC):
    """Turns text into vectors. One implementation per provider."""

    name: str   # identifies backend and model in output paths and eval reports
    dim: int    # vector dimensionality, needed to size the index

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts, preserving input order."""
        raise NotImplementedError