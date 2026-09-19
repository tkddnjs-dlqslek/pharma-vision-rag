"""Dump / restore the pharma_text collection via a chunk cache, skipping the ~70 min Docling pass.

Why: on 2026-09-19 every vector in pharma_text came back zero-norm after an unclean host shutdown
(payloads intact). Qdrant storage was a Windows bind mount, which Qdrant documents as unsafe; compose now
uses a named volume. Payloads hold the chunk text, so vectors can be rebuilt with BGE-M3 alone (minutes).

    dump : scroll pharma_text -> data/embeddings/text_chunks.jsonl  ({id, payload} per line)
    load : re-embed every cached chunk, upsert with the same ids, then assert no zero-norm vectors

Usage:
    PYTHONIOENCODING=utf-8 python scripts/15_text_chunk_cache.py dump
    PYTHONIOENCODING=utf-8 python scripts/15_text_chunk_cache.py load
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

CACHE = ROOT / "data" / "embeddings" / "text_chunks.jsonl"
COLLECTION = "pharma_text"


def client():
    from urllib.parse import urlparse

    from qdrant_client import QdrantClient
    host = urlparse(os.environ.get("QDRANT_URL", "http://localhost:6335")).hostname or "localhost"
    return QdrantClient(host=host, grpc_port=6336, prefer_grpc=True, check_compatibility=False, timeout=60)


def scroll_all(c, with_vectors: bool):
    off = None
    while True:
        pts, off = c.scroll(COLLECTION, limit=500, offset=off, with_payload=True, with_vectors=with_vectors)
        yield from pts
        if off is None:
            return


def dump() -> None:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(CACHE, "w", encoding="utf-8") as f:
        for p in scroll_all(client(), with_vectors=False):
            f.write(json.dumps({"id": p.id, "payload": p.payload}, ensure_ascii=False) + "\n")
            n += 1
    print(f"dumped {n} chunks -> {CACHE}")


def zero_norm_count(c) -> tuple[int, int]:
    n = z = 0
    for p in scroll_all(c, with_vectors=True):
        n += 1
        z += float(np.linalg.norm(p.vector)) < 1e-6
    return n, z


def load() -> None:
    from qdrant_client.models import PointStruct

    from pharma_vision_rag.retriever.docling_text import DoclingTextRetriever
    rows = [json.loads(l) for l in open(CACHE, encoding="utf-8") if l.strip()]
    r = DoclingTextRetriever(qdrant_url=os.environ.get("QDRANT_URL", "http://localhost:6335"))
    r.ensure_collection()
    for i in range(0, len(rows), 256):
        part = rows[i:i + 256]
        vecs = r.embedder.encode([x["payload"]["text"] for x in part], batch_size=32,
                                 normalize_embeddings=True, show_progress_bar=False)
        r.client.upsert(COLLECTION, points=[PointStruct(id=x["id"], vector=v.tolist(), payload=x["payload"])
                                            for x, v in zip(part, vecs)])
        print(f"  upserted {min(i + 256, len(rows))}/{len(rows)}", flush=True)
    n, z = zero_norm_count(r.client)
    print(f"collection {COLLECTION}: {n} points, zero-norm vectors {z}")
    assert n == len(rows) and z == 0, "restore incomplete"


if __name__ == "__main__":
    {"dump": dump, "load": load}.get(sys.argv[1] if len(sys.argv) > 1 else "", lambda: sys.exit(__doc__))()
