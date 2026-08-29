"""
STEP 1 / 10: Data acquisition.

Goal: download NFCorpus (BEIR version) and write it to disk in a neutral
format, so the rest of the pipeline no longer depends on HuggingFace.

Outputs, in data/nfcorpus/:
    corpus.jsonl     documents to index (~3,600 PubMed abstracts)
    queries.jsonl    evaluation queries
    qrels_test.tsv   ground truth: which document is relevant to which query

data/ is git-ignored: we version this script, never the data itself.
Usage: python scripts/download_data.py
"""

import json
from pathlib import Path

from datasets import load_dataset

# Output directory resolved from the script location, so the command works
# from any working directory.
OUT = Path(__file__).resolve().parents[1] / "data" / "nfcorpus"
OUT.mkdir(parents=True, exist_ok=True)

# NFCorpus ships as three separate pieces on HuggingFace.
corpus = load_dataset("BeIR/nfcorpus", "corpus", split="corpus")      # documents
queries = load_dataset("BeIR/nfcorpus", "queries", split="queries")   # queries
qrels = load_dataset("BeIR/nfcorpus-qrels", split="test")             # relevance judgments

# JSONL: one JSON object per line. Readable as a stream, so a larger corpus
# (TREC-COVID) can be processed without loading everything into memory.
with (OUT / "corpus.jsonl").open("w") as f:
    for row in corpus:
        f.write(json.dumps({"id": row["_id"], "title": row["title"], "text": row["text"]}) + "\n")

with (OUT / "queries.jsonl").open("w") as f:
    for row in queries:
        f.write(json.dumps({"id": row["_id"], "text": row["text"]}) + "\n")

# TSV with a header: each line maps a query to a relevant document, scored 0 to 2.
# This file is what makes evaluation possible in step 6 (nDCG@10, recall@k).
with (OUT / "qrels_test.tsv").open("w") as f:
    f.write("query_id\tdoc_id\tscore\n")
    for row in qrels:
        f.write(f"{row['query-id']}\t{row['corpus-id']}\t{row['score']}\n")

# Sanity counters: they tell you immediately whether the download is complete.
print(f"corpus  {len(corpus)}\nqueries {len(queries)}\nqrels   {len(qrels)}")