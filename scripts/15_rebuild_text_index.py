"""Rebuild pharma_text from cached raw Docling blocks, skipping the ~70 min Docling pass.

data/embeddings/text_blocks.jsonl is written by scripts/14_index_text_all.py (one raw text/table block per line).
This script re-chunks those blocks with the current retriever/chunking.py rules, re-embeds with BGE-M3 and
replaces the collection. Use it after changing chunking rules or after losing the Qdrant volume.

History: on 2026-09-19 every pharma_text vector came back zero-norm after an unclean host shutdown
(Windows bind mount; compose now uses a named volume). The final assert guards against that state.

Usage:
    PYTHONIOENCODING=utf-8 python scripts/15_rebuild_text_index.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env")

BLOCKS = ROOT / "data" / "embeddings" / "text_blocks.jsonl"


def main() -> None:
    from pharma_vision_rag.retriever.docling_text import DoclingTextRetriever
    if not BLOCKS.exists():
        raise SystemExit(f"{BLOCKS} missing: run scripts/14_index_text_all.py first")
    blocks = [json.loads(l) for l in open(BLOCKS, encoding="utf-8") if l.strip()]
    r = DoclingTextRetriever(qdrant_url=os.environ.get("QDRANT_URL", "http://localhost:6335"))
    if r.client.collection_exists(r.collection):
        r.client.delete_collection(r.collection)  # chunk ids change with the rules; no orphans
    stats = r.index_blocks(blocks)
    print(f"{len(blocks)} blocks -> {stats['chunks']} chunks {stats['by_type']}")

    n = zero = 0
    off = None
    while True:
        pts, off = r.client.scroll(r.collection, limit=500, offset=off, with_vectors=True, with_payload=False)
        n += len(pts)
        zero += sum(float(np.linalg.norm(p.vector)) < 1e-6 for p in pts)
        if off is None:
            break
    print(f"collection {r.collection}: {n} points, zero-norm vectors {zero}")
    assert n == stats["chunks"] and zero == 0, "rebuild incomplete"


if __name__ == "__main__":
    main()
