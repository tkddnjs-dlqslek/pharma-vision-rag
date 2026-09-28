"""Bearer-token wrapper of the Modal MCP deployment, driven as a plain ASGI app: no network, no Modal, no corpus."""
from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("_modal_mcp", ROOT / "serverless" / "modal_mcp.py")
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod := mod)  # type: ignore[union-attr]


def _call(app, headers: dict[str, str] | None, scope_type: str = "http") -> tuple[int | None, bool]:
    sent = []

    async def inner(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    async def send(msg):
        sent.append(msg)

    wrapped = mod.require_bearer(inner, app)
    scope = {"type": scope_type, "headers": [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]}
    asyncio.run(wrapped(scope, None, send))
    status = next((m["status"] for m in sent if m["type"] == "http.response.start"), None)
    return status, any(m.get("body") == b"ok" for m in sent)


def test_accepts_matching_token():
    assert _call("s3cret", {"Authorization": "Bearer s3cret"}) == (200, True)


def test_rejects_missing_wrong_or_malformed():
    for h in (None, {"Authorization": "Bearer nope"}, {"Authorization": "s3cret"}, {"Authorization": "bearer s3cret"},
              {"X-Api-Key": "s3cret"}):
        status, passed = _call("s3cret", h)
        assert (status, passed) == (401, False), h


def test_empty_server_token_denies_everything():
    assert _call("", {"Authorization": "Bearer "}) == (401, False)


def test_non_http_scopes_pass_through():
    # lifespan events carry no headers; the wrapper must not block them
    assert _call("s3cret", None, scope_type="lifespan") == (200, True)
