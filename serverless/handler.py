"""RunPod Serverless worker: query text -> Nemotron ColEmbed multi-vector embedding (fp16, base64).

Standalone inside the image (no repo imports). Model loading mirrors scripts/embed_pages_gpu.py:
bf16 on CUDA, then every float param/buffer cast to bf16 (the internal ViT is fp32-pinned), and the
dedicated forward_queries API (model(**inputs) returns None).

Input   {"query": str} or {"queries": [str, ...]}   (max 16 queries, max 2,000 chars each)
Output  {"model_id": ..., "embeddings": [{"shape": [tokens, 3072], "dtype": "float16", "data": base64}, ...]}
        in input order; decode with np.frombuffer(b64decode(data), float16).reshape(shape).
Error   {"error": "..."} (RunPod marks the job FAILED)
"""
from __future__ import annotations

import base64
import contextlib
import os

import numpy as np

MODEL_ID = os.environ.get("MODEL_ID", "nvidia/llama-nemotron-colembed-vl-3b-v2")
MAX_QUERIES = 16
MAX_CHARS = 2_000

if os.environ.get("BAKED_MODEL") == "1":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")        # weights are in the image: no hub calls at cold start
elif os.path.isdir("/runpod-volume"):
    os.environ["HF_HOME"] = "/runpod-volume/hf"  # network volume: download once, reuse across workers

MODEL = None  # set in __main__ (tests inject a fake)
NO_GRAD = contextlib.nullcontext  # torch.no_grad once the real model is loaded


def load_model():
    import torch
    from transformers import AutoModel
    assert torch.cuda.is_available(), "no CUDA device"
    model = AutoModel.from_pretrained(MODEL_ID, device_map="cuda", trust_remote_code=True,
                                      torch_dtype=torch.bfloat16).eval()
    floats = (torch.float32, torch.float16, torch.bfloat16, torch.float64)
    for t in list(model.parameters()) + list(model.buffers()):
        if t.dtype in floats:
            t.data = t.data.to(torch.bfloat16)
    global NO_GRAD
    NO_GRAD = torch.no_grad  # per call: RunPod may run the handler off the main thread (grad mode is thread-local)
    return model


def parse_queries(inp) -> list[str]:
    """Validate the job input; raises ValueError with a message meant for the caller."""
    if not isinstance(inp, dict):
        raise ValueError('input must be an object: {"query": str} or {"queries": [str]}')
    if ("query" in inp) == ("queries" in inp):
        raise ValueError('give exactly one of "query" or "queries"')
    qs = [inp["query"]] if "query" in inp else inp["queries"]
    if not isinstance(qs, list) or not qs:
        raise ValueError('"queries" must be a non-empty list of strings')
    if len(qs) > MAX_QUERIES:
        raise ValueError(f"at most {MAX_QUERIES} queries per job, got {len(qs)}")
    for i, q in enumerate(qs):
        if not isinstance(q, str) or not q.strip():
            raise ValueError(f"query {i} must be a non-empty string")
        if len(q) > MAX_CHARS:
            raise ValueError(f"query {i} has {len(q)} chars, max {MAX_CHARS}")
    return qs


def encode_array(a: np.ndarray) -> dict:
    a = np.ascontiguousarray(a, dtype=np.float16)
    return {"shape": list(a.shape), "dtype": "float16", "data": base64.b64encode(a.tobytes()).decode("ascii")}


def embed(model, query: str) -> np.ndarray:
    """One query per call, as the local CPU encoder does: no padding rows from batching."""
    with NO_GRAD():
        out = model.forward_queries([query], batch_size=1)
    t = out[0] if isinstance(out, list) or out.dim() == 3 else out
    return t.detach().float().cpu().numpy()


def handler(job: dict) -> dict:
    try:
        qs = parse_queries(job.get("input"))
    except ValueError as e:
        return {"error": str(e)}
    return {"model_id": MODEL_ID, "embeddings": [encode_array(embed(MODEL, q)) for q in qs]}


if __name__ == "__main__":
    import runpod
    MODEL = load_model()  # once per worker, before the first job
    runpod.serverless.start({"handler": handler})
