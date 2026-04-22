"""Nemotron ColEmbed V2 visual retriever.

Architecture
------------
- ``NemotronEmbeddingClient``: thin HTTP client to a Colab tunnel served by
  ``notebooks/02_nemotron_tunnel.ipynb``. Endpoints:
    GET  /health
    POST /embed_image    (multipart image -> [N_patches, 3072] float16 b64)
    POST /embed_query    (json query -> [N_tokens, 3072] float16 b64)
    POST /embed_queries  (batch)

- ``NemotronVisionRetriever``: indexes per-page multi-vector embeddings into a
  Qdrant **multi-vector collection** (``MAX_SIM`` comparator), then serves
  ``search(query, k)`` returning top-k pages by ColBERT-style late interaction.
  Qdrant handles MaxSim server-side — we just upsert and query.

Embedding wire format: float16 numpy bytes, base64-encoded. Decoded back to
float32 client-side before sending to Qdrant (Qdrant stores fp32).
"""
from __future__ import annotations

import base64
import io
import logging
import uuid
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    MultiVectorComparator,
    MultiVectorConfig,
    PointStruct,
    VectorParams,
)

log = logging.getLogger(__name__)

DEFAULT_COLLECTION = "pharma_vision"
EMBED_DIM = 3072  # Nemotron ColEmbed V2 hidden size (all variants)

# Stable UUID namespace for per-page point IDs (idempotent re-indexing).
_NS = uuid.UUID("18a6e0d2-4c18-4f0e-b0a9-9a8f9f7d4c11")


def _page_id(source: str, page: int) -> str:
    return str(uuid.uuid5(_NS, f"{source}:page:{page}"))


def _decode_payload(payload: dict[str, Any]) -> np.ndarray:
    """Server returns {shape, dtype, embedding_b64} -> np.ndarray (fp32)."""
    arr = np.frombuffer(base64.b64decode(payload["embedding_b64"]), dtype=np.float16)
    arr = arr.reshape(payload["shape"])
    return arr.astype(np.float32)  # Qdrant stores fp32


# ─── HTTP client ──────────────────────────────────────────────────────────


class NemotronEmbeddingClient:
    """Tiny HTTP wrapper for the Colab FastAPI tunnel."""

    def __init__(self, base_url: str, timeout: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(base_url=self.base_url, timeout=timeout)

    def health(self) -> dict[str, Any]:
        return self.client.get("/health").raise_for_status().json()

    def embed_image(self, image: Image.Image) -> np.ndarray:
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="PNG")
        files = {"file": ("page.png", buf.getvalue(), "image/png")}
        r = self.client.post("/embed_image", files=files)
        r.raise_for_status()
        return _decode_payload(r.json())

    def embed_query(self, query: str) -> np.ndarray:
        r = self.client.post("/embed_query", json={"query": query})
        r.raise_for_status()
        return _decode_payload(r.json())

    def embed_queries(self, queries: list[str]) -> list[np.ndarray]:
        r = self.client.post("/embed_queries", json={"queries": queries})
        r.raise_for_status()
        return [_decode_payload(p) for p in r.json()["embeddings"]]

    def close(self) -> None:
        self.client.close()


# ─── Retriever (Qdrant-backed) ────────────────────────────────────────────


class NemotronVisionRetriever:
    """Indexes per-page multi-vector embeddings into Qdrant; searches with MaxSim."""

    variant_name = "nemotron_vision"

    def __init__(
        self,
        embedding_client: NemotronEmbeddingClient,
        qdrant_url: str,
        qdrant_api_key: str | None = None,
        collection: str = DEFAULT_COLLECTION,
        grpc_port: int = 6336,
    ) -> None:
        self.client = embedding_client
        self.collection = collection
        # Use gRPC for multi-vector traffic — JSON over REST balloons a single
        # page from ~14 MB binary to ~100 MB and exceeds Qdrant's default cap.
        # gRPC is binary, no inflation, and our docker-compose maps host 6336
        # -> container 6334.
        from urllib.parse import urlparse
        host = urlparse(qdrant_url).hostname or "localhost"
        self.qdrant = QdrantClient(
            host=host,
            grpc_port=grpc_port,
            prefer_grpc=True,
            api_key=qdrant_api_key or None,
            check_compatibility=False,
            timeout=120,
        )

    def ensure_collection(self) -> None:
        if not self.qdrant.collection_exists(self.collection):
            self.qdrant.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=EMBED_DIM,
                    distance=Distance.COSINE,
                    multivector_config=MultiVectorConfig(comparator=MultiVectorComparator.MAX_SIM),
                ),
            )
            log.info("Created multi-vector collection %s (dim=%d, MAX_SIM)", self.collection, EMBED_DIM)

    # ─── Index ────────────────────────────────────────────────────────────

    def index_page(self, image: Image.Image, source: str, page: int) -> dict[str, Any]:
        emb = self.client.embed_image(image)  # [N_patches, 3072] fp32
        self.qdrant.upsert(
            collection_name=self.collection,
            points=[PointStruct(
                id=_page_id(source, page),
                vector=emb.tolist(),
                payload={"source": source, "page": page, "n_patches": int(emb.shape[0])},
            )],
        )
        return {"source": source, "page": page, "n_patches": int(emb.shape[0]), "dim": int(emb.shape[1])}

    def index_pdf(
        self,
        pdf_path: str | Path,
        source: str | None = None,
        render_scale: float = 0.85,
    ) -> dict[str, Any]:
        """Render every page of ``pdf_path`` and index each as a multi-vector point.

        ``render_scale`` controls patch count quadratically. Lower scales reduce
        response size over the Colab tunnel: at scale 1.5 a single page can
        produce a ~100 MB JSON response that Cloudflare Quick Tunnels reject
        with 502. 0.85 keeps it under ~30 MB while preserving most retrieval
        signal.
        """
        from pharma_vision_rag.utils.pdf import iter_pages

        pdf_path = Path(pdf_path)
        source = source or pdf_path.name
        self.ensure_collection()

        per_page = []
        for page_num, img in iter_pages(pdf_path, scale=render_scale):
            stats = self.index_page(img, source=source, page=page_num)
            per_page.append(stats)
            log.info("Indexed %s p%d (n_patches=%d)", source, page_num, stats["n_patches"])

        return {
            "source": source,
            "pages": len(per_page),
            "total_patches": sum(p["n_patches"] for p in per_page),
            "collection": self.collection,
        }

    # ─── Search ───────────────────────────────────────────────────────────

    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        q_emb = self.client.embed_query(query)  # [N_tokens, 3072]
        result = self.qdrant.query_points(
            collection_name=self.collection,
            query=q_emb.tolist(),
            limit=k,
            with_payload=True,
        )
        hits = []
        for point in result.points:
            payload = point.payload or {}
            hits.append({
                "score": point.score,
                "source": payload.get("source"),
                "page": payload.get("page"),
                "n_patches": payload.get("n_patches"),
                "id": point.id,
                "variant": self.variant_name,
            })
        return hits
