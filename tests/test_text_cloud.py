"""retriever/text_cloud.py with an in-memory Qdrant, a fake embedder and reranker. No network, no models.

Run: PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m pytest tests/test_text_cloud.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from qdrant_client import QdrantClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pharma_vision_rag.retriever import text_cloud as tc  # noqa: E402

# 4 synthetic chunks whose vectors are axis-aligned so cosine ranking is obvious.
CHUNKS = [
    {"source": "a.pdf", "page": 1, "block_index": 1, "block_type": "text", "context": "A | p1", "text": "dupixent sales"},
    {"source": "a.pdf", "page": 2, "block_index": 1, "block_type": "table", "context": "A | p2", "text": "beyfortus sales"},
    {"source": "b.pdf", "page": 1, "block_index": 1, "block_type": "text", "context": "B | p1", "text": "dupixent growth"},
    {"source": "b.pdf", "page": 3, "block_index": 2, "block_type": "text", "context": "B | p3", "text": "other"},
]


def _vec(i: int, n: int = 4, dim: int = tc.DIM) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float16)
    v[i] = 1.0
    return v


VECTORS = np.stack([_vec(i) for i in range(len(CHUNKS))])


def _query_vec(weights: dict[int, float]) -> list[float]:
    v = np.zeros(tc.DIM)
    for i, w in weights.items():
        v[i] = w
    return (v / np.linalg.norm(v)).tolist()


@pytest.fixture()
def client():
    c = QdrantClient(":memory:")
    assert tc.ensure_collection(c) is True
    assert tc.ensure_collection(c) is False  # second call is a no-op
    n = tc.upsert_chunks(c, CHUNKS, VECTORS, batch=3)
    assert n == 4 and c.count(tc.COLLECTION, exact=True).count == 4
    tc.upsert_chunks(c, CHUNKS, VECTORS, batch=3)  # idempotent re-run
    assert c.count(tc.COLLECTION, exact=True).count == 4
    return c


def _searcher(client, qvec, rerank_scores=None):
    embed = lambda texts: [qvec for _ in texts]  # noqa: E731
    rerank = lambda q, texts: rerank_scores or [0.0] * len(texts)  # noqa: E731
    return tc.CloudTextSearch(embedder=embed, reranker=rerank, client=client)


def test_point_id_deterministic():
    assert tc.point_id(CHUNKS[0], 0) == tc.point_id(CHUNKS[0], 0) != tc.point_id(CHUNKS[0], 1)


def test_dense_order_and_shape(client):
    s = _searcher(client, _query_vec({0: 1.0, 2: 0.5}))
    hits = s.search("q", k=3, rerank=False)
    assert [(h["source"], h["page"]) for h in hits] == [("a.pdf", 1), ("b.pdf", 1)] + [(hits[2]["source"], hits[2]["page"])]
    assert {"source", "page", "block_type", "block_index", "text", "context", "score"} <= hits[0].keys()
    assert hits[0]["text"] == "dupixent sales" and hits[0]["score"] > hits[1]["score"]


def test_document_filter(client):
    s = _searcher(client, _query_vec({0: 1.0, 2: 0.5}))
    hits = s.search("q", k=5, document_ids=["b.pdf"], rerank=False)
    assert hits and all(h["source"] == "b.pdf" for h in hits)
    assert hits[0]["page"] == 1


def test_rerank_reorders_and_cuts(client):
    # dense order: row0, row2, row1, row3; reranker prefers the 3rd dense hit (row1, beyfortus).
    s = _searcher(client, _query_vec({0: 1.0, 2: 0.6, 1: 0.3}), rerank_scores=[0.1, 0.2, 0.9, 0.0])
    hits = s.search("q", k=2, pool=4)
    assert [h["text"] for h in hits] == ["beyfortus sales", "dupixent growth"]
    assert hits[0]["score"] == 0.9


def test_rrf_merge():
    a = [{"source": "x", "page": 1, "block_index": 1, "text": "t1", "score": 9.0},
         {"source": "x", "page": 2, "block_index": 1, "text": "t2", "score": 5.0}]
    b = [{"source": "x", "page": 2, "block_index": 1, "text": "t2", "score": 0.8},
         {"source": "y", "page": 1, "block_index": 1, "text": "t3", "score": 0.7}]
    merged = tc.rrf_merge(a, b)
    assert [h["text"] for h in merged] == ["t2", "t1", "t3"]  # t2 appears in both lists
    assert merged[0]["score"] == pytest.approx(1 / 61 + 1 / 62, abs=1e-6)
    assert tc.rrf_merge([], []) == []


def test_available_env(monkeypatch, tmp_path):
    monkeypatch.setattr(tc, "_load_env", lambda: None)
    for var in ("QDRANT_CLOUD_URL", "QDRANT_CLOUD_API_KEY", "QDRANT_URL", "MODAL_TOKEN_ID"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))  # no ~/.modal.toml
    assert tc.qdrant_target() is None and not tc.available()

    monkeypatch.setenv("QDRANT_URL", "http://localhost:6335")
    assert tc.qdrant_target() == ("http://localhost:6335", None) and not tc.available()

    monkeypatch.setenv("MODAL_TOKEN_ID", "ak-x")
    assert tc.available()

    monkeypatch.setenv("QDRANT_CLOUD_URL", "https://c.cloud.qdrant.io")
    assert tc.qdrant_target() == ("http://localhost:6335", None)  # key missing -> still local
    monkeypatch.setenv("QDRANT_CLOUD_API_KEY", "k")
    assert tc.qdrant_target() == ("https://c.cloud.qdrant.io", "k") and tc.available()


def test_modal_wrapper_lookup_injection():
    class Fake:
        class embed:
            remote = staticmethod(lambda texts: [[1.0] * 4 for _ in texts])

        class rerank:
            remote = staticmethod(lambda q, texts: [float(len(t)) for t in texts])

    embed, rerank = tc.modal_text_models(lookup=lambda app, cls: Fake())
    assert embed(["a", "b"]) == [[1.0] * 4] * 2 and rerank("q", ["ab", "a"]) == [2.0, 1.0]

    def boom(app, cls):
        raise ConnectionError("offline")
    embed, _ = tc.modal_text_models(lookup=boom)
    with pytest.raises(tc.CloudTextError, match="offline"):
        embed(["a"])
