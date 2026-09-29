"""Query-side models of the pharma corpus MCP server as one HTTP service (CPU-only Docker container, any host).

    POST /vision/encode {"queries": [...]}          -> [{"shape": [tokens, 3072], "dtype": "float16", "data": base64}, ...]
    POST /text/embed    {"queries": [...]}          -> {"vectors": [[1024 floats], ...]}   (BGE-M3, normalised)
    POST /text/rerank   {"query": ..., "texts": [...]} -> {"scores": [...]}                (bge-reranker-v2-m3)
    GET  /health                                    -> {"status": "ok" | "loading", ...}

Same models, loading code and settings as serverless/modal_app.py (Nemotron, CPU path) and serverless/modal_text.py
(BGE-M3 max_seq 1024, reranker max_length 512), without Modal. Every POST needs ``Authorization: Bearer $ENCODER_TOKEN``.
Clients: retriever/vision_remote.http_query_encoder and retriever/text_cloud.http_text_models.

Memory: Nemotron 3B in bf16 is about 7 GB, BGE-M3 and the reranker about 2.3 GB each, so all three resident is about
12 GB: a 16 GB box is enough, 8 GB is not. The models load in a background thread after the port opens, so the
container is reachable at once and POSTs answer 503 until /health says ok. Weights are cached under $HF_HOME
(/data/hf in the Dockerfile; mount a volume there so a restart does not download 12 GB again).
"""
from __future__ import annotations

import base64
import hmac
import os
import threading
import time

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

import numpy as np  # noqa: E402
from fastapi import Depends, FastAPI, HTTPException, Request  # noqa: E402
from pydantic import BaseModel  # noqa: E402

VISION_MODEL = "nvidia/llama-nemotron-colembed-vl-3b-v2"
EMBED_MODEL, RERANK_MODEL = "BAAI/bge-m3", "BAAI/bge-reranker-v2-m3"
EMBED_MAX_SEQ, RERANK_MAX_LEN = 1024, 512     # = scripts/text_retrieval_gpu.py, serverless/modal_text.py
MAX_QUERIES, MAX_CHARS, MAX_TEXTS = 16, 2_000, 64

MODELS: dict[str, object] = {}     # "vision", "embed", "rerank"; filled by load_models (or by tests with fakes)
LOAD_ERROR: list[str] = []
STARTED = time.time()
_infer = threading.Lock()   # ponytail: one inference at a time on a 2~4 core box; parallel encodes would only thrash


def load_models() -> None:
    import torch
    from sentence_transformers import CrossEncoder, SentenceTransformer
    from transformers import AutoModel
    try:
        torch.set_default_device("cpu")  # transformers 4.57 resolved the device to cuda on a CPU box and died warming its allocator
        model = AutoModel.from_pretrained(VISION_MODEL, trust_remote_code=True, torch_dtype=torch.bfloat16,
                                          device_map={"": "cpu"}).eval()
        floats = (torch.float32, torch.float16, torch.bfloat16, torch.float64)
        for t in list(model.parameters()) + list(model.buffers()):
            if t.dtype in floats:  # internal ViT is fp32-pinned
                t.data = t.data.to(torch.bfloat16)
        MODELS["vision"] = model
        embedder = SentenceTransformer(EMBED_MODEL, device="cpu")
        embedder.max_seq_length = EMBED_MAX_SEQ
        MODELS["embed"] = embedder
        MODELS["rerank"] = CrossEncoder(RERANK_MODEL, max_length=RERANK_MAX_LEN, device="cpu")
        print(f"models loaded in {time.time() - STARTED:.0f}s", flush=True)
    except Exception as e:  # noqa: BLE001 - reported by /health instead of killing the process silently
        LOAD_ERROR.append(f"{type(e).__name__}: {e}")
        print(f"model loading failed: {LOAD_ERROR[-1]}", flush=True)


def parse_queries(qs, limit: int = MAX_QUERIES) -> list[str]:
    """= serverless/modal_app.parse_queries, as a 422."""
    if isinstance(qs, str):
        qs = [qs]
    if not isinstance(qs, list) or not qs:
        raise HTTPException(422, "queries must be a non-empty list of strings")
    if len(qs) > limit:
        raise HTTPException(422, f"at most {limit} items per call, got {len(qs)}")
    for i, q in enumerate(qs):
        if not isinstance(q, str) or not q.strip():
            raise HTTPException(422, f"item {i} must be a non-empty string")
        if len(q) > MAX_CHARS:
            raise HTTPException(422, f"item {i} has {len(q)} chars, max {MAX_CHARS}")
    return qs


def require_token(request: Request) -> None:
    expected = os.environ.get("ENCODER_TOKEN", "")
    auth = request.headers.get("authorization", "")
    got = auth[7:] if auth[:7].lower() == "bearer " else ""
    if not expected or not hmac.compare_digest(got.encode(), expected.encode()):  # no token configured -> everything 401
        raise HTTPException(401, "missing or wrong bearer token")


def model(name: str):
    if name not in MODELS:
        raise HTTPException(503, f"models still loading ({time.time() - STARTED:.0f}s since start)"
                            + (f"; last error: {LOAD_ERROR[-1]}" if LOAD_ERROR else ""))
    return MODELS[name]


class Queries(BaseModel):
    queries: list[str] | str


class RerankRequest(BaseModel):
    query: str
    texts: list[str]


app = FastAPI(title="pharma-encoders", docs_url=None, redoc_url=None)


@app.on_event("startup")
def start_loading() -> None:
    if not MODELS:  # tests pre-fill MODELS with fakes and skip the download
        threading.Thread(target=load_models, name="load_models", daemon=True).start()


@app.get("/health")
def health() -> dict:
    return {"status": "ok" if len(MODELS) == 3 else "loading", "loaded": sorted(MODELS), "errors": LOAD_ERROR,
            "uptime_s": round(time.time() - STARTED)}


@app.post("/vision/encode", dependencies=[Depends(require_token)])
def vision_encode(body: Queries) -> list[dict]:
    """[{"shape": [tokens, 3072], "dtype": "float16", "data": base64}, ...] in input order (= modal_app.Encoder.encode)."""
    import torch
    m = model("vision")
    out = []
    with _infer:
        for q in parse_queries(body.queries):
            with torch.no_grad():
                emb = m.forward_queries([q], batch_size=1)
            t = emb[0] if isinstance(emb, list) or emb.dim() == 3 else emb
            a = np.ascontiguousarray(t.detach().float().cpu().numpy(), dtype=np.float16)
            out.append({"shape": list(a.shape), "dtype": "float16", "data": base64.b64encode(a.tobytes()).decode("ascii")})
    return out


@app.post("/text/embed", dependencies=[Depends(require_token)])
def text_embed(body: Queries) -> dict:
    qs = parse_queries(body.queries)
    with _infer:
        vecs = model("embed").encode(qs, batch_size=16, normalize_embeddings=True)
    return {"vectors": np.asarray(vecs, dtype="float32").tolist()}


@app.post("/text/rerank", dependencies=[Depends(require_token)])
def text_rerank(body: RerankRequest) -> dict:
    parse_queries(body.query)
    texts = parse_queries(body.texts, limit=MAX_TEXTS)
    with _infer:
        scores = model("rerank").predict([[body.query, t] for t in texts], show_progress_bar=False)
    return {"scores": [float(s) for s in scores]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "7860")))
