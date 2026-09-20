"""Rebuild pharma_text from cached raw Docling blocks, skipping the ~70 min Docling pass.

data/embeddings/text_blocks.jsonl is written by scripts/14_index_text_all.py (one raw text/table block per line).
This script re-chunks those blocks with the current retriever/chunking.py rules, re-embeds with BGE-M3 and
replaces the collection. Use it after changing chunking rules or after losing the Qdrant volume.

History: on 2026-09-19 every pharma_text vector came back zero-norm after an unclean host shutdown
(Windows bind mount; compose now uses a named volume). The final assert guards against that state.

With `--precomputed DIR` (text_chunks.jsonl + text_vectors.npy from scripts/text_retrieval_gpu.py) no model
is loaded at all: the vectors were computed on the GPU box and are only upserted here.

Usage:
    PYTHONIOENCODING=utf-8 python scripts/15_rebuild_text_index.py
    PYTHONIOENCODING=utf-8 python scripts/15_rebuild_text_index.py --precomputed data/embeddings/v2/text
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


def upsert_precomputed(src: Path) -> None:
    import importlib.util
    from urllib.parse import urlparse

    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams
    spec = importlib.util.spec_from_file_location("chunking", ROOT / "src/pharma_vision_rag/retriever/chunking.py")
    chunking = importlib.util.module_from_spec(spec)   # loaded by path: the package __init__ imports torch/docling
    spec.loader.exec_module(chunking)

    chunks = [json.loads(l) for l in open(src / "text_chunks.jsonl", encoding="utf-8") if l.strip()]
    vecs = np.load(src / "text_vectors.npy").astype(np.float32)
    assert len(chunks) == len(vecs) and vecs.shape[1] == 1024, (len(chunks), vecs.shape)
    assert (np.linalg.norm(vecs, axis=1) > 0.99).all(), "zero or unnormalised vectors in the precomputed file"
    host = urlparse(os.environ.get("QDRANT_URL", "http://localhost:6335")).hostname or "localhost"
    client = QdrantClient(host=host, grpc_port=6336, prefer_grpc=True, check_compatibility=False, timeout=120)
    name = "pharma_text"
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(name, vectors_config=VectorParams(size=1024, distance=Distance.COSINE))
    for i in range(0, len(chunks), 512):
        client.upsert(name, points=[PointStruct(id=chunking.chunk_id(c), vector=v.tolist(), payload=c)
                                    for c, v in zip(chunks[i:i + 512], vecs[i:i + 512])])
    n = client.count(name, exact=True).count
    print(f"collection {name}: {n} points from precomputed vectors")
    assert n == len(chunks), "duplicate chunk ids or failed upserts"


def main() -> None:
    if "--precomputed" in sys.argv:
        upsert_precomputed(Path(sys.argv[sys.argv.index("--precomputed") + 1]))
        return
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
