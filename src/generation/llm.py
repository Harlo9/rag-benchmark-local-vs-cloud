"""
STEP 8 / 10: LLM backend.

Same pattern as the embedding layer: one interface, swappable implementations.
Local Ollama by default, so no text leaves the machine; an Azure OpenAI backend
can be added later behind the same contract.
"""

import os
from abc import ABC, abstractmethod

import ollama



class LLM(ABC):
    name: str

    @abstractmethod
    def complete(self, system: str, user: str) -> str:
        """Return the model's answer to a single-turn prompt."""
        raise NotImplementedError


class OllamaLLM(LLM):
    def __init__(self, model: str | None = None, host: str | None = None):
        self.model = model or os.getenv("LLM_MODEL", "qwen3:4b")
        self.client = ollama.Client(
            host=host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        )
        self.name = f"ollama-{self.model}".replace(":", "-")

    def complete(self, system: str, user: str) -> str:
        response = self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            # Temperature 0: grounded answering is not a creative task, and
            # determinism makes the evaluation in step 9 reproducible.
            options={"temperature": 0},
            keep_alive=-1
        )
        return response["message"]["content"]

class AzureLLM(LLM):
    def __init__(self, deployment: str | None = None):
        from openai import AzureOpenAI

        self.deployment = deployment or os.environ["AZURE_LLM_DEPLOYMENT"]
        self.client = AzureOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=os.getenv("AZURE_API_VERSION", "2024-10-21"),
        )
        self.name = f"azure-{self.deployment}"

    def complete(self, system: str, user: str) -> str:
        response = self.client.chat.completions.create(
            model=self.deployment,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content
    
def get_llm() -> LLM:
    backend = os.getenv("LLM_BACKEND", "ollama").lower()
    if backend == "ollama":
        return OllamaLLM()
    if backend == "azure":
        return AzureLLM()
    raise ValueError(f"Unknown LLM backend: {backend}")