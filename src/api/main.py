"""
STEP 10 / 10: HTTP API.

Goal: expose the pipeline over HTTP so any client (demo UI, notebook, curl) can
query it without importing the Python code.

Key design point: the retriever and the LLM client are built once at startup and
held in application state. Building them per request would reload the vector
matrix and rebuild the BM25 index every time, which is what makes the CLI script
slow. Loading once turns a multi-second cost into a one-off.

Run: uvicorn src.api.main:app --reload
Docs: http://localhost:8000/docs
"""

from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from src.api.schemas import AskRequest, AskResponse, Source
from src.generation.answer import answer_question
from src.generation.llm import get_llm
from src.retrieval import build_retriever





load_dotenv(Path(__file__).resolve().parents[2] / ".env")

# One retriever per strategy, built lazily and then reused.
_retrievers: dict[str, object] = {}

STATIC = Path(__file__).parent / "static"



@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm the default strategy at startup so the first request is not the one
    # paying for index construction.
    _retrievers["hybrid"] = build_retriever(strategy="hybrid", diversify=True)
    app.state.llm = get_llm()

    # Warm-up call: pins both models in memory so the first real request does not
    # pay the model load. Measured 1507 ms cold vs 130 ms warm.
    app.state.llm.complete("Reply with OK.", "ping")
    _retrievers["hybrid"].retrieve("warmup", k=1)

    yield
    _retrievers.clear()


app = FastAPI(
    title="Local RAG API",
    description="Grounded question answering over a biomedical corpus, running fully offline.",
    version="0.1.0",
    lifespan=lifespan,
)


def retriever_for(strategy: str):
    if strategy not in ("dense", "bm25", "hybrid"):
        raise HTTPException(status_code=400, detail=f"unknown strategy: {strategy}")
    if strategy not in _retrievers:
        _retrievers[strategy] = build_retriever(strategy=strategy, diversify=True)
    return _retrievers[strategy]


@app.get("/health")
def health() -> dict:
    """Liveness probe, also reporting which strategies are already warm."""
    return {"status": "ok", "loaded_strategies": sorted(_retrievers)}

@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """Serve the demo page. Kept as a single static file: no build step, no CDN,
    so the whole project still runs with no network access."""
    return FileResponse(STATIC / "index.html")

@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    result = answer_question(
        request.question,
        retriever_for(request.strategy),
        app.state.llm,
        k=request.k,
    )

    return AskResponse(
        question=result.question,
        answer=result.text,
        abstained=result.abstained,
        sources=[
            Source(
                n=i,
                doc_id=hit.doc_id,
                text=hit.text,
                score=hit.score,
                cited=i in result.cited,
            )
            for i, hit in enumerate(result.sources, 1)
        ],
        retrieval_ms=result.retrieval_ms,
        generation_ms=result.generation_ms,
    )