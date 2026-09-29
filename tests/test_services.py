"""services/encoders (FastAPI app with fake models, TestClient), pharma_vision_rag.mcp_auth.JsonStore under the OAuth
provider, and services/mcp's app build (MCP_ROOT = this checkout, no vision preload, no download). No network, no models.

Run: PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m pytest tests/test_services.py -q
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from unittest import mock
from urllib.parse import parse_qs, urlparse

import numpy as np
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pharma_vision_rag import mcp_auth  # noqa: E402
from pharma_vision_rag.retriever import text_cloud as tc  # noqa: E402
from pharma_vision_rag.retriever import vision_remote as vr  # noqa: E402


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# ─── services/encoders ──────────────────────────────────────────────────────

class FakeTensor:
    def __init__(self, a):
        self.a = a

    def detach(self):
        return self

    float = cpu = detach

    def numpy(self):
        return self.a


class FakeVision:
    def forward_queries(self, texts, batch_size):
        return [FakeTensor(np.random.default_rng(len(t)).standard_normal((len(t.split()) + 2, 8)).astype(np.float32))
                for t in texts]


class FakeEmbedder:
    def encode(self, texts, batch_size, normalize_embeddings):
        return np.array([[1.0, 0.0, 0.0, 0.0]] * len(texts), dtype=np.float32)


class FakeReranker:
    def predict(self, pairs, show_progress_bar):
        return np.array([float(len(t)) for _, t in pairs])


@pytest.fixture()
def enc():
    mod = _load("_services_encoders", ROOT / "services" / "encoders" / "app.py")
    mod.MODELS.update(vision=FakeVision(), embed=FakeEmbedder(), rerank=FakeReranker())
    with mock.patch.dict(os.environ, {"ENCODER_TOKEN": "tok"}), TestClient(mod.app) as c:
        yield mod, c


AUTH = {"Authorization": "Bearer tok"}


def test_encoders_health_and_auth(enc):
    mod, c = enc
    assert c.get("/health").json()["status"] == "ok"
    assert c.post("/vision/encode", json={"queries": ["q"]}).status_code == 401
    assert c.post("/text/embed", json={"queries": ["q"]}, headers={"Authorization": "Bearer tik"}).status_code == 401
    with mock.patch.dict(os.environ, {"ENCODER_TOKEN": ""}):  # unset token: fail closed
        assert c.post("/text/embed", json={"queries": ["q"]}, headers=AUTH).status_code == 401


def test_encoders_round_trip_through_clients(enc):
    """The service's JSON goes through the real client code in vision_remote / text_cloud."""
    mod, c = enc
    r = c.post("/vision/encode", json={"queries": ["Dupixent sales chart", "net sales"]}, headers=AUTH)
    assert r.status_code == 200 and len(r.json()) == 2
    got = vr.decode_embedding(r.json()[0])
    want = FakeVision().forward_queries(["Dupixent sales chart"], 1)[0].a
    assert got.shape == want.shape == (5, 8) and np.allclose(got, want, atol=2e-3)

    def urlopen(req, timeout):  # route the stdlib clients into the TestClient
        path = urlparse(req.full_url).path
        assert req.get_header("Authorization") == "Bearer tok"
        res = c.post(path, content=req.data, headers={"Content-Type": "application/json", **AUTH})
        assert res.status_code == 200, res.text
        import io
        return io.BytesIO(res.content)

    with mock.patch.object(vr.urllib.request, "urlopen", side_effect=urlopen):
        emb = vr.http_query_encoder("https://enc.example.com/", "tok")("Dupixent sales chart")
        assert emb.shape == (5, 8) and emb.dtype == np.float32
        embed, rerank = tc.http_text_models("https://enc.example.com", "tok")
        assert embed(["a", "b"]) == [[1.0, 0.0, 0.0, 0.0]] * 2
        assert rerank("q", ["abc", "a"]) == [3.0, 1.0]


def test_encoders_validation_and_loading(enc):
    mod, c = enc
    assert c.post("/vision/encode", json={"queries": []}, headers=AUTH).status_code == 422
    assert c.post("/vision/encode", json={"queries": ["x" * (mod.MAX_CHARS + 1)]}, headers=AUTH).status_code == 422
    assert c.post("/text/rerank", json={"query": "q", "texts": ["t"] * (mod.MAX_TEXTS + 1)}, headers=AUTH).status_code == 422
    assert c.post("/text/embed", json={"queries": "one string"}, headers=AUTH).json()["vectors"] == [[1.0, 0.0, 0.0, 0.0]]
    vision = mod.MODELS.pop("vision")
    try:
        r = c.post("/vision/encode", json={"queries": ["q"]}, headers=AUTH)
        assert r.status_code == 503 and c.get("/health").json()["status"] == "loading"
    finally:
        mod.MODELS["vision"] = vision


# ─── http clients: error paths (mocked urlopen) ─────────────────────────────

def test_http_clients_errors():
    import urllib.error
    enc = vr.http_query_encoder("https://enc.example.com", "tok")
    for code, hint in [(401, "ENCODER_TOKEN"), (404, "ENCODER_URL"), (503, "still loading")]:
        err = urllib.error.HTTPError("u", code, "x", {}, None)
        with mock.patch.object(vr.urllib.request, "urlopen", side_effect=err):
            with pytest.raises(vr.RemoteEncoderError, match=hint):
                enc("q")
    import io
    with mock.patch.object(vr.urllib.request, "urlopen", return_value=io.BytesIO(b"[]")):
        with pytest.raises(vr.RemoteEncoderError, match="no embedding"):
            enc("q")
    embed, _ = tc.http_text_models("https://enc.example.com", "tok")
    with mock.patch.object(vr.urllib.request, "urlopen", side_effect=urllib.error.URLError("down")):
        with pytest.raises(tc.CloudTextError, match="unreachable"):
            embed(["q"])
    with mock.patch.object(vr.urllib.request, "urlopen", return_value=io.BytesIO(b'{"oops": 1}')):
        with pytest.raises(tc.CloudTextError, match="vectors"):
            embed(["q"])


def test_env_selection(monkeypatch, tmp_path):
    monkeypatch.setattr(vr, "ROOT", tmp_path)          # no .env
    monkeypatch.setattr(tc, "_load_env", lambda: None)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))   # no ~/.modal.toml
    for var in ("RUNPOD_API_KEY", "RUNPOD_ENDPOINT_ID", "MODAL_TOKEN_ID", "MODAL_IS_REMOTE", "ENCODER_URL", "ENCODER_TOKEN",
                "QDRANT_CLOUD_URL", "QDRANT_CLOUD_API_KEY", "QDRANT_URL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("VISION_ENCODER", "auto")
    assert vr.encoder_from_env() is None and vr.http_encoder_target() is None
    monkeypatch.setenv("ENCODER_URL", "https://enc.example.com")
    assert vr.encoder_from_env() is None                # token missing
    monkeypatch.setenv("ENCODER_TOKEN", "t")
    assert vr.http_encoder_target() == ("https://enc.example.com", "t") and callable(vr.encoder_from_env())
    monkeypatch.setenv("RUNPOD_API_KEY", "k")
    monkeypatch.setenv("RUNPOD_ENDPOINT_ID", "e")
    assert vr.encoder_from_env().__qualname__.startswith("runpod_query_encoder")   # runpod keys win in auto
    monkeypatch.setenv("VISION_ENCODER", "http")
    assert vr.encoder_from_env().__qualname__.startswith("http_query_encoder")
    monkeypatch.delenv("ENCODER_TOKEN")
    with pytest.raises(vr.RemoteEncoderError, match="ENCODER_TOKEN"):
        vr.encoder_from_env()
    # text_cloud.available: qdrant target + (http service or modal token)
    monkeypatch.setenv("QDRANT_URL", "http://localhost:6335")
    assert not tc.available()
    monkeypatch.setenv("ENCODER_TOKEN", "t")
    assert tc.available()
    embed, rerank = tc.text_models_from_env()
    assert embed.__qualname__.startswith("http_text_models")


# ─── JsonStore under the OAuth provider ─────────────────────────────────────

run = asyncio.run


def test_json_store_persists_and_survives_bad_file(tmp_path):
    path = tmp_path / "auth" / "auth.json"
    s = mcp_auth.JsonStore(path)
    assert len(s) == 0 and s.get("x") is None
    s["a"] = {"v": 1}
    s["b"] = {"v": 2}
    assert json.loads(path.read_text())["a"] == {"v": 1}
    assert s.pop("a") == {"v": 1} and "a" not in json.loads(path.read_text())
    with pytest.raises(KeyError):
        s.pop("a")
    assert mcp_auth.JsonStore(path)["b"] == {"v": 2}       # reloaded from disk
    path.write_text("{not json")
    assert len(mcp_auth.JsonStore(path)) == 0              # corrupt file -> empty store, not a crash


def test_provider_over_json_store(tmp_path):
    """The Modal provider tests' code -> token -> refresh chain, with the state read back by a fresh store."""
    from mcp.server.auth.provider import AuthorizationParams
    from mcp.shared.auth import OAuthClientInformationFull
    path = tmp_path / "auth.json"
    client = OAuthClientInformationFull(client_id="c1", client_secret="s", redirect_uris=["https://claude.ai/cb"],
                                        token_endpoint_auth_method="client_secret_post")
    pv = mcp_auth.DictAuthProvider(mcp_auth.JsonStore(path), "static-hex", "u", "p", public_url="https://mcp.example.com")
    assert run(pv.load_access_token("static-hex")).client_id == "static-token"
    run(pv.register_client(client))
    params = AuthorizationParams(state="st", scopes=None, code_challenge="chal", redirect_uri="https://claude.ai/cb",
                                 redirect_uri_provided_explicitly=True, resource=None)
    login_url = run(pv.authorize(client, params))
    assert login_url.startswith("https://mcp.example.com/login?req=")
    req = parse_qs(urlparse(login_url).query)["req"][0]
    assert pv.login(req, "u", "wrong") is None
    code = parse_qs(urlparse(pv.login(req, "u", "p")).query)["code"][0]
    ac = run(pv.load_authorization_code(client, code))
    tok = run(pv.exchange_authorization_code(client, ac))

    # a "restarted" provider over the same file still knows the client and the tokens
    pv2 = mcp_auth.DictAuthProvider(mcp_auth.JsonStore(path), "static-hex", "u", "p")
    assert run(pv2.get_client("c1")).client_secret == "s"
    at = run(pv2.load_access_token(tok.access_token))
    assert at.client_id == "c1" and at.expires_at > time.time() + mcp_auth.ACCESS_TTL - 60
    rt = run(pv2.load_refresh_token(client, tok.refresh_token))
    tok2 = run(pv2.exchange_refresh_token(client, rt, rt.scopes))
    assert run(pv2.load_refresh_token(client, tok.refresh_token)) is None
    assert run(pv2.load_access_token(tok2.access_token)) is not None
    run(pv2.revoke_token(at))
    # each JsonStore holds its own dict (one process per container), so the revocation is checked through a fresh store
    assert run(mcp_auth.DictAuthProvider(mcp_auth.JsonStore(path)).load_access_token(tok.access_token)) is None


# ─── services/mcp app build (corpus of this checkout, no download, no vision preload) ───

def test_mcp_space_app(tmp_path, monkeypatch):
    monkeypatch.setenv("MCP_ROOT", str(ROOT))
    monkeypatch.setenv("MCP_PRELOAD_VISION", "0")
    monkeypatch.setenv("AUTH_STORE_PATH", str(tmp_path / "auth.json"))
    monkeypatch.setenv("MCP_TOKEN", "static-hex")
    monkeypatch.setenv("MCP_USER", "u")
    monkeypatch.setenv("MCP_PASSWORD", "p")
    monkeypatch.delenv("DATA_REPO", raising=False)
    monkeypatch.delenv("MCP_PUBLIC_URL", raising=False)
    monkeypatch.setenv("VISION_ENCODER", "http")   # app.py setdefaults this; set it here so monkeypatch restores it
    mod = _load("_services_mcp", ROOT / "services" / "mcp" / "app.py")
    assert mod.public_url() == "http://localhost:8000"      # default without a tunnel hostname
    monkeypatch.setenv("MCP_PUBLIC_URL", "https://mcp.example.com/")
    assert mod.public_url() == "https://mcp.example.com"
    mod.fetch_data()  # no DATA_REPO: no-op
    with TestClient(mod.build_app()) as c:
        assert c.get("/health").json()["status"] == "ok"
        meta = c.get("/.well-known/oauth-authorization-server").json()
        assert meta["issuer"].rstrip("/") == "https://mcp.example.com" and meta["authorization_endpoint"].startswith("https://mcp.example.com/")
        assert "form" in c.get("/login?req=abc").text
        assert c.post("/login", data={"req": "abc", "username": "u", "password": "p"}).status_code == 401  # no pending req
        init = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
                "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}}}
        hdr = {"Accept": "application/json, text/event-stream"}
        assert c.post("/mcp", json=init, headers=hdr).status_code == 401
        r = c.post("/mcp", json=init, headers={**hdr, "Authorization": "Bearer static-hex"})
        assert r.status_code == 200 and r.json()["result"]["serverInfo"]["name"] == "pharma-corpus"
        r = c.post("/mcp", json={"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                                 "params": {"name": "calculate", "arguments": {"expression": "2*21"}}},
                   headers={**hdr, "Authorization": "Bearer static-hex"})
        assert r.status_code == 200 and r.json()["result"]["content"][0]["text"].startswith("42")
