"""
STEP 6 / 10: Evaluating retrieval on NFCorpus.

Goal: run every test query through the pipeline and report averaged metrics,
so any later change (chunk size, embedding model, hybrid search) can be judged
on evidence.

Chunks are mapped back to their parent document and deduplicated, because qrels
judge documents, not chunks. We therefore retrieve CHUNK_K chunks to end up with
roughly DOC_K distinct documents.

Results are written to results/<backend-model>.json, one file per configuration,
which is what builds the comparison table in the README.

Usage: python -m scripts.evaluate [dense|bm25|hybrid]
"""
import sys
import json
import time
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

from src.embeddings import get_embedder
from src.eval.metrics import ndcg_at_k, recall_at_k
from src.retrieval import build_retriever

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "nfcorpus"
RESULTS = ROOT / "results"

DOC_K = 10      # documents evaluated per query
CHUNK_K = 50    # chunks retrieved, before collapsing to documents
STRATEGY = sys.argv[1] if len(sys.argv) > 1 else "dense"   # dense | bm25 | hybrid
RERANK = len(sys.argv) > 2 and sys.argv[2] == "rerank"

load_dotenv(ROOT / ".env")


def load_qrels() -> dict[str, dict[str, int]]:
    """query_id -> {doc_id: relevance grade}."""
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    with (DATA / "qrels_test.tsv").open() as f:
        next(f)  # header
        for line in f:
            query_id, doc_id, score = line.rstrip("\n").split("\t")
            qrels[query_id][doc_id] = int(score)
    return qrels


def load_queries() -> dict[str, str]:
    return {
        json.loads(line)["id"]: json.loads(line)["text"]
        for line in (DATA / "queries.jsonl").open()
    }


def to_documents(hits, limit: int) -> list[str]:
    """Collapse ranked chunks into ranked distinct documents, best rank wins."""
    seen, docs = set(), []
    for hit in hits:
        if hit.doc_id not in seen:
            seen.add(hit.doc_id)
            docs.append(hit.doc_id)
        if len(docs) == limit:
            break
    return docs


def main() -> None:
    embedder = get_embedder()
    retriever = build_retriever(strategy=STRATEGY, rerank=RERANK)
    qrels, queries = load_qrels(), load_queries()

    # Only queries that have judgments can be scored.
    query_ids = [qid for qid in qrels if qid in queries]
    print(f"evaluating {len(query_ids)} queries with {embedder.name}")

    scores = {"ndcg@10": [], "recall@10": [], "recall@50": []}
    started = time.time()

    for i, qid in enumerate(query_ids, 1):
        hits = retriever.retrieve(queries[qid], k=CHUNK_K)
        docs = to_documents(hits, limit=50)

        scores["ndcg@10"].append(ndcg_at_k(docs, qrels[qid], DOC_K))
        scores["recall@10"].append(recall_at_k(docs, qrels[qid], DOC_K))
        scores["recall@50"].append(recall_at_k(docs, qrels[qid], 50))
        print(f"  {i}/{len(query_ids)}", end="\r")

    report = {
        "config": {
            "strategy": STRATEGY,
            "embedder": embedder.name,
            "dim": embedder.dim,
            "rerank": RERANK,
            "chunk_k": CHUNK_K,
            "doc_k": DOC_K,
            "queries": len(query_ids),
        },
        "metrics": {name: sum(v) / len(v) for name, v in scores.items()},
        "elapsed_s": round(time.time() - started, 1),
    }

    RESULTS.mkdir(exist_ok=True)
   # The rerank flag is part of the configuration, so it must be part of the
    # filename: otherwise a reranked run silently overwrites its own baseline.
    suffix = f"{STRATEGY}-rerank" if RERANK else STRATEGY
    (RESULTS / f"{embedder.name}-{suffix}.json").write_text(json.dumps(report, indent=2))
    
    print("\n" + json.dumps(report["metrics"], indent=2))
    print(f"elapsed {report['elapsed_s']}s")


if __name__ == "__main__":
    main()