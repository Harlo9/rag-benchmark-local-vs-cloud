"""
STEP 5 / 10: Chunk store.

Goal: give retrieval access to chunk text, which the index does not hold.

The index stores vectors and ids only, so it stays compact and reloads fast.
Text lives here, loaded once into a dict. At ~6k chunks this is a few MB in
memory; a larger corpus would call for SQLite instead, same interface.
"""

import json
from pathlib import Path

class ChunkStore:
    def __init__(self, chunks_path: Path):
        self.by_id: dict[str, dict] = {}
        with chunks_path.open() as f:
            for line in f:
                chunk = json.loads(line)
                self.by_id[chunk["chunk_id"]] = chunk

    def text(self, chunk_id: str) -> str:
        return self.by_id[chunk_id]["text"]

    def __len__(self) -> int:
        return len(self.by_id)