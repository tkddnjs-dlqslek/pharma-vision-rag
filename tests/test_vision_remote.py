"""serverless/handler.py (fake model, no torch/GPU/runpod) and retriever/vision_remote.py (mocked HTTP, no network).

Run: PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m pytest tests/test_vision_remote.py -q
"""
from __future__ import annotations

import importlib.util
import io
import json
import sys
import urllib.error
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pharma_vision_rag.retriever import vision_remote as vr  # noqa: E402

_spec = importlib.util.spec_from_file_location("_test_handler", ROOT / "serverless" / "handler.py")
h = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(h)  # type: ignore[union-attr]


class FakeTensor:
    def __init__(self, a):
        self.a = a

    def detach(self):
        return self

    float = cpu = detach

    def numpy(self):
        return self.a


class FakeModel:
    """forward_queries -> list of [tokens, dim] tensors; token count depends on the text."""
    def forward_queries(self, texts, batch_size):
        return [FakeTensor(np.random.default_rng(len(t)).standard_normal((len(t.split()) + 2, 8)).astype(np.float32))
                for t in texts]


def test_handler_validation():
    for bad in [None, [], {}, {"query": ""}, {"query": "a", "queries": ["b"]}, {"queries": []},
                {"queries": "not a list"}, {"queries": ["ok", 3]}, {"query": "x" * (h.MAX_CHARS + 1)},
                {"queries": ["q"] * (h.MAX_QUERIES + 1)}]:
        out = h.handler({"input": bad})
        assert set(out) == {"error"}, (bad, out)
    with mock.patch.object(h, "MODEL", FakeModel()):
        assert len(h.handler({"input": {"queries": ["q"] * h.MAX_QUERIES}})["embeddings"]) == h.MAX_QUERIES


def test_handler_round_trip():
    with mock.patch.object(h, "MODEL", FakeModel()):
        out = h.handler({"input": {"queries": ["Dupixent sales chart", "net sales Q2 2025 by region"]}})
        single = h.handler({"input": {"query": "Dupixent sales chart"}})
    assert len(out["embeddings"]) == 2 and out["model_id"] == h.MODEL_ID
    for text, item in zip(["Dupixent sales chart", "net sales Q2 2025 by region"], out["embeddings"]):
        want = h.embed(FakeModel(), text)
        got = vr.decode_embedding(json.loads(json.dumps(item)))  # through JSON, as over HTTP
        assert got.dtype == np.float32 and got.shape == want.shape
        assert np.allclose(got, want, atol=2e-3)                  # fp16 rounding only
    assert single["embeddings"][0] == out["embeddings"][0]


def _response(body: dict):
    return io.BytesIO(json.dumps(body).encode())  # BytesIO is a context manager with .read()


def test_remote_encoder_runsync_completed():
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    body = {"id": "j1", "status": "COMPLETED", "output": {"embeddings": [h.encode_array(a)]}}
    with mock.patch.object(vr.urllib.request, "urlopen", return_value=_response(body)) as m:
        got = vr.runpod_query_encoder("ep123", "key")("some query")
    assert np.array_equal(got, a)
    req = m.call_args.args[0]
    assert req.full_url == "https://api.runpod.ai/v2/ep123/runsync"
    assert req.get_header("Authorization") == "Bearer key"
    assert json.loads(req.data) == {"input": {"query": "some query"}}


def test_remote_encoder_polls_cold_start():
    a = np.ones((2, 4), dtype=np.float32)
    replies = [_response({"id": "j2", "status": "IN_QUEUE"}), _response({"id": "j2", "status": "IN_PROGRESS"}),
               _response({"id": "j2", "status": "COMPLETED", "output": {"embeddings": [h.encode_array(a)]}})]
    with mock.patch.object(vr.urllib.request, "urlopen", side_effect=replies) as m, \
            mock.patch.object(vr, "POLL_S", 0):
        got = vr.runpod_query_encoder("ep", "key")("q")
    assert np.array_equal(got, a)
    assert [c.args[0].full_url.rsplit("/", 2)[-2:] for c in m.call_args_list[1:]] == [["status", "j2"]] * 2


def test_remote_encoder_errors():
    enc = vr.runpod_query_encoder("ep", "key")
    with mock.patch.object(vr.urllib.request, "urlopen", return_value=_response({"status": "FAILED", "error": "boom"})):
        with pytest.raises(vr.RemoteEncoderError, match="boom"):
            enc("q")
    with mock.patch.object(vr.urllib.request, "urlopen", return_value=_response(
            {"status": "COMPLETED", "output": {"error": "query 0 has 3000 chars, max 2000"}})):
        with pytest.raises(vr.RemoteEncoderError, match="3000 chars"):
            enc("q")
    err = urllib.error.HTTPError("u", 401, "Unauthorized", {}, None)
    with mock.patch.object(vr.urllib.request, "urlopen", side_effect=err):
        with pytest.raises(vr.RemoteEncoderError, match="RUNPOD_API_KEY"):
            enc("q")
    with mock.patch.object(vr.urllib.request, "urlopen", side_effect=lambda *a, **k: _response({"id": "j", "status": "IN_QUEUE"})), \
            mock.patch.object(vr, "POLL_S", 0):
        with pytest.raises(vr.RemoteEncoderError, match="not done"):
            vr.runpod_query_encoder("ep", "key", timeout=0)("q")


def test_encoder_from_env():
    with mock.patch.object(vr, "ROOT", ROOT / "does-not-exist"):  # keep the real .env out of it
        with mock.patch.dict(vr.os.environ, {"RUNPOD_API_KEY": "", "RUNPOD_ENDPOINT_ID": "ep"}):
            assert vr.encoder_from_env() is None
        with mock.patch.dict(vr.os.environ, {"RUNPOD_API_KEY": "k", "RUNPOD_ENDPOINT_ID": "ep"}):
            assert callable(vr.encoder_from_env())
