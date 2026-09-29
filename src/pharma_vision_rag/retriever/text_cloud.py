"""Dense text retrieval for the MCP server: BGE-M3 chunk vectors in Qdrant, query embedding and
bge-reranker-v2-m3 on a remote service (services/encoders over HTTP, or Modal via serverless/modal_text.py).
Same recipe as the benchmark's best text arm (text_rerank, R@5 0.72) instead of BM25 alone (0.53).
No model is loaded locally; no torch import.

    QDRANT_CLOUD_URL, QDRANT_CLOUD_API_KEY   Qdrant Cloud (free tier). Absent -> QDRANT_URL (local docker).
    ENCODER_URL, ENCODER_TOKEN               HTTP query-side models (POST /text/embed, /text/rerank); else
    ~/.modal.toml                            Modal token for the query-side models.

    scripts/29_upload_text_vectors.py        fills collection pharma_text_v2 from data/embeddings/v2/text/
    CloudTextSearch().search("Dupixent Q2 2025 sales", k=5, document_ids=[...])
"""
from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any, Callable, Iterable

ROOT = Path(__file__).resolve().parent.parent.parent.parent
COLLECTION = "pharma_text_v2"
DIM = 1024
MODAL_APP, MODAL_CLASS = "pharma-text-models", "TextModels"


class CloudTextError(RuntimeError):
    pass


# ─── env ────────────────────────────────────────────────────────────────────

def _load_env() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass


def qdrant_target() -> tuple[str, str | None] | None:
    """(url, api_key) for Qdrant Cloud when both cloud vars are set, else (QDRANT_URL, None), else None."""
    _load_env()
    url, key = os.environ.get("QDRANT_CLOUD_URL", "").strip(), os.environ.get("QDRANT_CLOUD_API_KEY", "").strip()
    if url and key:
        return url, key
    local = os.environ.get("QDRANT_URL", "").strip()
    return (local, None) if local else None


from pharma_vision_rag.retriever.vision_remote import (  # noqa: E402  (stdlib + numpy only)
    HTTP_HINTS, HTTP_TIMEOUT_S, RemoteEncoderError, _call, http_encoder_target, modal_token_present)


def available() -> bool:
    """A Qdrant target and query-side models (HTTP service or Modal token) exist. Does not check that the
    collection was uploaded."""
    return qdrant_target() is not None and (http_encoder_target() is not None or modal_token_present())


def client_from_env():
    target = qdrant_target()
    if target is None:
        raise CloudTextError("set QDRANT_CLOUD_URL + QDRANT_CLOUD_API_KEY (or QDRANT_URL) in .env")
    from qdrant_client import QdrantClient
    url, key = target
    return QdrantClient(url=url, api_key=key, timeout=300, check_compatibility=False)


# ─── query-side models: HTTP service (services/encoders) or Modal ───────────

def http_text_models(base_url: str, token: str, timeout: float = HTTP_TIMEOUT_S):
    """(embed, rerank) callables over services/encoders: POST /text/embed {"queries"} -> {"vectors"},
    POST /text/rerank {"query", "texts"} -> {"scores"}."""
    base = base_url.rstrip("/")

    def post(path: str, body: dict, key: str):
        try:
            res = _call(base + path, token, body, timeout, service="encoder service", hints=HTTP_HINTS)
        except RemoteEncoderError as e:
            raise CloudTextError(str(e)) from None
        if not isinstance(res, dict) or key not in res:
            raise CloudTextError(f"encoder service {path} returned no {key!r}: {str(res)[:200]}")
        return res[key]

    def embed(texts: list[str]) -> list[list[float]]:
        return post("/text/embed", {"queries": list(texts)}, "vectors")

    def rerank(query: str, texts: list[str]) -> list[float]:
        return post("/text/rerank", {"query": query, "texts": list(texts)}, "scores")

    return embed, rerank


def text_models_from_env():
    """HTTP service when ENCODER_URL/ENCODER_TOKEN are set, else Modal (looked up on first use)."""
    http = http_encoder_target()
    return http_text_models(*http) if http else modal_text_models()


def modal_text_models(app: str = MODAL_APP, cls: str = MODAL_CLASS, lookup=None):
    """(embed, rerank) callables backed by the deployed Modal class; the class is looked up on first use.

    `lookup` (tests) replaces modal.Cls.from_name; it must return an object with `.embed.remote` / `.rerank.remote`.
    """
    holder: dict[str, Any] = {}

    def models():
        if "cls" not in holder:
            try:
                if lookup is not None:
                    holder["cls"] = lookup(app, cls)
                else:
                    import modal
                    holder["cls"] = modal.Cls.from_name(app, cls)()
            except Exception as e:  # noqa: BLE001 - one clear message for the tool caller
                raise CloudTextError(f"Modal text models unavailable ({type(e).__name__}: {e}). Deploy with "
                                     "`modal deploy serverless/modal_text.py` after `modal token new`.") from None
        return holder["cls"]

    def embed(texts: list[str]) -> list[list[float]]:
        try:
            return models().embed.remote(list(texts))
        except CloudTextError:
            raise
        except Exception as e:  # noqa: BLE001
            raise CloudTextError(f"Modal embed failed ({type(e).__name__}: {e})") from None

    def rerank(query: str, texts: list[str]) -> list[float]:
        try:
            return models().rerank.remote(query, list(texts))
        except CloudTextError:
            raise
        except Exception as e:  # noqa: BLE001
            raise CloudTextError(f"Modal rerank failed ({type(e).__name__}: {e})") from None

    return embed, rerank


# ─── upload (shared by scripts/29 and the tests) ────────────────────────────

def point_id(chunk: dict[str, Any], row: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{chunk['source']}:{chunk['page']}:{chunk.get('block_index')}:{row}"))


def ensure_collection(client, collection: str = COLLECTION) -> bool:
    """Create the cosine 1024-d collection (+ keyword index on source) if missing. Returns True when created."""
    from qdrant_client.models import Distance, PayloadSchemaType, VectorParams
    if client.collection_exists(collection):
        return False
    client.create_collection(collection, vectors_config=VectorParams(size=DIM, distance=Distance.COSINE))
    client.create_payload_index(collection, "source", PayloadSchemaType.KEYWORD)
    return True


def upsert_chunks(client, chunks: list[dict[str, Any]], vectors, collection: str = COLLECTION,
                  batch: int = 64, start_row: int = 0, log: Callable[[str], None] | None = None,
                  retries: int = 3) -> int:
    """Upsert chunk i with vectors[i] under a deterministic id (re-runs overwrite in place). Returns points sent.

    REST JSON to a distant free-tier cluster: 64 points is ~0.8 MB and each batch is retried on timeout.
    Floats are rounded to 6 decimals, lossless for the float16 source vectors."""
    import time

    from qdrant_client.models import PointStruct
    n = len(chunks)
    if len(vectors) != n:
        raise CloudTextError(f"{n} chunks but {len(vectors)} vectors")
    for lo in range(0, n, batch):
        pts = []
        for i in range(lo, min(lo + batch, n)):
            c, row = chunks[i], start_row + i
            payload = {k: c.get(k) for k in ("source", "page", "context", "text", "block_type", "block_index")}
            payload["row"] = row
            pts.append(PointStruct(id=point_id(c, row), vector=[round(float(x), 6) for x in vectors[i]], payload=payload))
        for attempt in range(retries):
            try:
                client.upsert(collection, points=pts, wait=True)
                break
            except Exception as e:  # noqa: BLE001 - timeouts and 5xx from the cloud tier
                if attempt == retries - 1:
                    raise CloudTextError(f"upsert of rows {start_row + lo}..{start_row + lo + len(pts) - 1} failed "
                                         f"after {retries} tries: {type(e).__name__}: {e}") from None
                if log:
                    log(f"  retry {attempt + 1}/{retries - 1} at {start_row + lo}: {type(e).__name__}")
                time.sleep(2.0 * (attempt + 1))
        if log:
            log(f"  upserted {min(lo + batch, n)}/{n}")
    return n


# ─── search ─────────────────────────────────────────────────────────────────

def _hit(payload: dict[str, Any], score: float) -> dict[str, Any]:
    """Same keys agentic._tool_search_text emits for BM25, plus context and block_index."""
    return {"source": payload.get("source"), "page": payload.get("page"), "block_type": payload.get("block_type"),
            "block_index": payload.get("block_index"), "text": (payload.get("text") or "")[:400],
            "context": payload.get("context"), "score": round(float(score), 4)}


def rrf_merge(*hit_lists: Iterable[dict[str, Any]], k: int = 60) -> list[dict[str, Any]]:
    """Reciprocal rank fusion over ranked hit lists keyed on (source, page, block_index). Pure."""
    fused: dict[tuple, float] = {}
    first: dict[tuple, dict[str, Any]] = {}
    for hits in hit_lists:
        for rank, h in enumerate(hits):
            key = (h.get("source"), h.get("page"), h.get("block_index", h.get("text")))
            fused[key] = fused.get(key, 0.0) + 1.0 / (k + rank + 1)
            first.setdefault(key, h)
    return [{**first[key], "score": round(s, 6)} for key, s in sorted(fused.items(), key=lambda kv: -kv[1])]


class CloudTextSearch:
    """Dense top-`pool` from Qdrant (payload filter on source), then bge-reranker on the chunk text -> top-k."""

    def __init__(self, embedder=None, reranker=None, client=None, collection: str = COLLECTION):
        if embedder is None or reranker is None:
            embed, rerank = text_models_from_env()
            embedder, reranker = embedder or embed, reranker or rerank
        self.embed, self.rerank_fn, self.collection = embedder, reranker, collection
        self._client = client

    @property
    def client(self):
        if self._client is None:
            self._client = client_from_env()
        return self._client

    def dense(self, query: str, pool: int = 30, document_ids: list[str] | None = None) -> list[dict[str, Any]]:
        from qdrant_client.models import FieldCondition, Filter, MatchAny
        vec = self.embed([query])[0]
        flt = Filter(must=[FieldCondition(key="source", match=MatchAny(any=list(document_ids)))]) if document_ids else None
        res = self.client.query_points(self.collection, query=vec, limit=pool, query_filter=flt, with_payload=True)
        return [_hit(p.payload or {}, p.score) for p in res.points]

    def rerank_hits(self, query: str, hits: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
        if not hits:
            return []
        scores = self.rerank_fn(query, [h["text"] for h in hits])
        return [{**h, "score": round(float(s), 4)} for h, s in sorted(zip(hits, scores), key=lambda hs: -hs[1])][:k]

    def search(self, query: str, k: int = 5, document_ids: list[str] | None = None, pool: int = 30,
               rerank: bool = True) -> list[dict[str, Any]]:
        hits = self.dense(query, pool=pool, document_ids=document_ids)
        return self.rerank_hits(query, hits, k) if rerank else hits[:k]
