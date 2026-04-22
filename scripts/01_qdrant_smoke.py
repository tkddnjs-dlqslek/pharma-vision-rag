"""Phase 0 smoke test — Qdrant connection and basic ops.

Usage:
    python scripts/01_qdrant_smoke.py

Expected:
    - Connects to QDRANT_URL from .env (should be http://localhost:6335)
    - Creates a dummy collection, upserts one vector, searches it, deletes
    - Confirms multi-vector collection support (needed for Nemotron patches)
"""
import os
import uuid

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, MultiVectorConfig, MultiVectorComparator

load_dotenv()

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6335")


def main() -> None:
    client = QdrantClient(url=QDRANT_URL)
    print(f"Connected: {QDRANT_URL}")

    # List existing collections (should not affect other projects' collections)
    cols = client.get_collections().collections
    print(f"Existing collections: {[c.name for c in cols]}")

    # Single-vector test collection
    single_name = "pharma_smoke_single"
    if client.collection_exists(single_name):
        client.delete_collection(single_name)
    client.create_collection(
        collection_name=single_name,
        vectors_config=VectorParams(size=8, distance=Distance.COSINE),
    )
    client.upsert(
        collection_name=single_name,
        points=[PointStruct(id=str(uuid.uuid4()), vector=[0.1] * 8, payload={"hello": "world"})],
    )
    hits = client.query_points(
        collection_name=single_name,
        query=[0.1] * 8,
        limit=1,
    ).points
    print(f"\n[single-vector] upsert + search: {len(hits)} hit, score={hits[0].score:.3f}")
    client.delete_collection(single_name)

    # Multi-vector test collection (required for Nemotron late interaction)
    multi_name = "pharma_smoke_multi"
    if client.collection_exists(multi_name):
        client.delete_collection(multi_name)
    client.create_collection(
        collection_name=multi_name,
        vectors_config=VectorParams(
            size=8,
            distance=Distance.COSINE,
            multivector_config=MultiVectorConfig(comparator=MultiVectorComparator.MAX_SIM),
        ),
    )
    client.upsert(
        collection_name=multi_name,
        points=[PointStruct(
            id=str(uuid.uuid4()),
            vector=[[0.1] * 8, [0.2] * 8, [0.3] * 8],
            payload={"page": 1},
        )],
    )
    hits = client.query_points(
        collection_name=multi_name,
        query=[[0.1] * 8, [0.2] * 8],
        limit=1,
    ).points
    print(f"[multi-vector] MaxSim upsert + search: {len(hits)} hit, score={hits[0].score:.3f}")
    client.delete_collection(multi_name)

    print("\n=== Acceptance ===")
    print("  OK  Qdrant reachable")
    print("  OK  Single-vector collection works")
    print("  OK  Multi-vector (MaxSim) collection works  ← required for Nemotron")


if __name__ == "__main__":
    main()
