"""
STEP 3 / 10: Local embedding backend (Ollama).

Runs entirely on the machine: no text leaves the host, which is the point of
this project. Default model is qwen3-embedding:0.6b, small enough for an M2
laptop while ranking near the top of the MTEB leaderboard.
"""

import os

import ollama

from .base import Embedder

# Output dimensionality per model. Wrong values here silently corrupt the index,
# so every supported model must be listed explicitly.
DIMS = {
    "qwen3-embedding:0.6b": 1024,
    "qwen3-embedding:4b": 2560,
    "nomic-embed-text": 768,
    "mxbai-embed-large": 1024,
}


class OllamaEmbedder(Embedder):
    def __init__(self, model: str | None = None, host: str | None = None):
        self.model = model or os.getenv("EMBEDDING_MODEL", "qwen3-embedding:0.6b")
        self.client = ollama.Client(
            host=host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        )
        # Slashes and colons would break filesystem paths downstream.
        self.name = f"ollama-{self.model}".replace(":", "-").replace("/", "-")
        self.dim = DIMS.get(self.model, 1024)

    def embed(self, texts: list[str]) -> list[list[float]]:
        # Ollama accepts a list, so one HTTP round trip per batch instead of one per text.
        return self.client.embed(model=self.model, input=texts)["embeddings"]