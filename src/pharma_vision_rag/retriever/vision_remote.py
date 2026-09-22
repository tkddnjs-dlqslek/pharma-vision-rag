"""Query encoder backed by the RunPod Serverless worker in serverless/handler.py (no local 3B model).

    RUNPOD_API_KEY, RUNPOD_ENDPOINT_ID   in the environment or .env -> encoder_from_env() returns an encoder
    LocalVisionIndex(dir, encoder=encoder_from_env())                -> same search, remote query embeddings

/runsync waits a limited time; a cold start (model download or load) can outlast it, so an unfinished job is
polled on /status/{id} until TIMEOUT_S.
"""
from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent.parent
API = "https://api.runpod.ai/v2"
TIMEOUT_S = 120.0
POLL_S = 2.0


class RemoteEncoderError(RuntimeError):
    pass


def decode_embedding(item: dict) -> np.ndarray:
    """handler.encode_array output -> float32 [tokens, dim], the layout LocalVisionIndex.encode returns."""
    if item.get("dtype") != "float16":
        raise RemoteEncoderError(f"unexpected dtype from endpoint: {item.get('dtype')!r}")
    a = np.frombuffer(base64.b64decode(item["data"]), dtype="<f2").reshape(item["shape"])
    if a.ndim != 2:
        raise RemoteEncoderError(f"expected a [tokens, dim] embedding, got shape {a.shape}")
    return a.astype(np.float32)


def _call(url: str, api_key: str, body: dict | None, timeout: float) -> dict:
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode("utf-8"),
                                 headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                                 method="GET" if body is None else "POST")
    try:
        with urllib.request.urlopen(req, timeout=max(timeout, 1.0)) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        hint = " (check RUNPOD_API_KEY)" if e.code in (401, 403) else " (check RUNPOD_ENDPOINT_ID)" if e.code == 404 else ""
        raise RemoteEncoderError(f"RunPod endpoint returned HTTP {e.code}{hint}") from None
    except (urllib.error.URLError, TimeoutError) as e:
        raise RemoteEncoderError(f"RunPod endpoint unreachable or timed out: {e}") from None


def runpod_query_encoder(endpoint_id: str, api_key: str, timeout: float = TIMEOUT_S):
    base = f"{API}/{endpoint_id}"

    def encode(query: str) -> np.ndarray:
        deadline = time.monotonic() + timeout
        res = _call(f"{base}/runsync", api_key, {"input": {"query": query}}, timeout)
        while res.get("status") in ("IN_QUEUE", "IN_PROGRESS") and res.get("id"):
            if time.monotonic() >= deadline:
                raise RemoteEncoderError(f"RunPod job {res['id']} not done after {timeout:.0f} s "
                                         "(cold start? retry in a minute)")
            time.sleep(POLL_S)
            res = _call(f"{base}/status/{res['id']}", api_key, None, deadline - time.monotonic())
        out = res.get("output")
        if res.get("status") != "COMPLETED" or not isinstance(out, dict) or "embeddings" not in out:
            err = res.get("error") or (out.get("error") if isinstance(out, dict) else out)
            raise RemoteEncoderError(f"RunPod job {res.get('status', 'no status')}: {err or res}")
        return decode_embedding(out["embeddings"][0])
    return encode


def encoder_from_env():
    """Remote encoder when RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID are set (env wins over .env), else None."""
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    key, endpoint = os.environ.get("RUNPOD_API_KEY", "").strip(), os.environ.get("RUNPOD_ENDPOINT_ID", "").strip()
    return runpod_query_encoder(endpoint, key) if key and endpoint else None
