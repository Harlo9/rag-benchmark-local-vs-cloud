"""
STEP 10 / 10: API request and response models.

Goal: describe the API contract with Pydantic, which validates inputs, serialises
outputs and generates the OpenAPI documentation from the same declaration.

The response deliberately carries the retrieved passages and per-stage timings
alongside the answer text. An answer without its sources cannot be audited, and
without timings the latency trade-offs measured in steps 7 and 8 become invisible
to whoever uses the API.
"""

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    k: int = Field(default=5, ge=1, le=20)          # passages given to the model
    strategy: str = Field(default="hybrid")          # dense | bm25 | hybrid


class Source(BaseModel):
    n: int              # passage number as cited in the answer, e.g. [2]
    doc_id: str
    text: str
    score: float
    cited: bool         # whether the model actually used this passage


class AskResponse(BaseModel):
    question: str
    answer: str
    abstained: bool     # true when the corpus could not answer
    sources: list[Source]
    retrieval_ms: int
    generation_ms: int