"""Upload the BGE-M3 chunk vectors (data/embeddings/v2/text/) to Qdrant as collection pharma_text_v2.

Target: QDRANT_CLOUD_URL + QDRANT_CLOUD_API_KEY from .env, else QDRANT_URL (local docker). REST only.
Deterministic ids -> re-running overwrites the same points (idempotent).

    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/29_upload_text_vectors.py
    ... --limit 300 --collection smoke_text        # smoke test
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pharma_vision_rag.retriever import text_cloud as tc  # noqa: E402

TEXT_DIR = ROOT / "data" / "embeddings" / "v2" / "text"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", default=tc.COLLECTION)
    ap.add_argument("--limit", type=int, default=0, help="upload only the first N chunks (smoke test)")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    chunks = [json.loads(l) for l in (TEXT_DIR / "text_chunks.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    vectors = np.load(TEXT_DIR / "text_vectors.npy", mmap_mode="r")
    assert vectors.shape == (len(chunks), tc.DIM), f"vectors {vectors.shape} vs {len(chunks)} chunks"
    if args.limit:
        chunks, vectors = chunks[:args.limit], vectors[:args.limit]

    target = tc.qdrant_target()
    if target is None:
        sys.exit("no Qdrant target: set QDRANT_CLOUD_URL + QDRANT_CLOUD_API_KEY or QDRANT_URL in .env")
    print(f"target {target[0]} ({'cloud' if target[1] else 'local'}), collection {args.collection}, {len(chunks)} points")
    client = tc.client_from_env()
    print("created collection" if tc.ensure_collection(client, args.collection) else "collection exists")

    t0 = time.time()
    tc.upsert_chunks(client, chunks, vectors, collection=args.collection, batch=args.batch, log=print)
    count = client.count(args.collection, exact=True).count
    print(f"done in {time.time() - t0:.0f}s, collection count {count} (expected {len(chunks)})")
    if count != len(chunks):
        sys.exit(f"count mismatch: {count} != {len(chunks)}")


if __name__ == "__main__":
    main()
