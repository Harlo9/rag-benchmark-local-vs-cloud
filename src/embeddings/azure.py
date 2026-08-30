"""
Azure OpenAI embedding backend.

Goal: run the exact same pipeline against a hosted provider, so local and cloud
can be compared on the same benchmark rather than argued about.

Trade-off this exists to measure: text leaves the machine, in exchange for
throughput and no local hardware. The numbers, not the intuition, decide.
"""

import os

from openai import AzureOpenAI

from .base import Embedder

# Output dimensionality per model, as with the Ollama backend: a wrong value
# here silently corrupts the index.
DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}


class AzureEmbedder(Embedder):
    def __init__(self, deployment: str | None = None):
        # Azure addresses a *deployment*, not a model: the deployment name is
        # chosen at creation time and is what the API expects.
        self.deployment = deployment or os.environ["AZURE_EMBEDDING_DEPLOYMENT"]
        self.client = AzureOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=os.getenv("AZURE_API_VERSION", "2024-10-21"),
        )
        self.name = f"azure-{self.deployment}"
        self.dim = DIMS.get(self.deployment, 1536)

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(model=self.deployment, input=texts)
        # Results come back with an index; sorting guarantees input order is kept,
        # which is the contract the whole index depends on.
        return [item.embedding for item in sorted(response.data, key=lambda d: d.index)]