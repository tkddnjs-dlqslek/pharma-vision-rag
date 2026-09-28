"""OAuth provider of the Modal MCP deployment over a plain dict: no network, no Modal, no corpus.

Covers the token verifier (static token, issued token, expired token, bad token), the credential check and the
register -> authorize -> login -> code -> token -> refresh chain.
"""
from __future__ import annotations

import asyncio
import importlib.util
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from mcp.server.auth.provider import AuthorizationParams
from mcp.shared.auth import OAuthClientInformationFull

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("_modal_mcp", ROOT / "serverless" / "modal_mcp.py")
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)  # type: ignore[union-attr]

run = asyncio.run
CLIENT = OAuthClientInformationFull(client_id="c1", client_secret="s", redirect_uris=["https://claude.ai/cb"],
                                    token_endpoint_auth_method="client_secret_post")


def provider(store=None):
    return mod.DictAuthProvider(store if store is not None else {}, static_token="static-hex", username="u", password="p")


def test_static_token_and_bad_token():
    pv = provider()
    assert run(pv.load_access_token("static-hex")).client_id == "static-token"
    assert run(pv.load_access_token("static-hey")) is None
    assert run(pv.load_access_token("")) is None
    assert run(mod.DictAuthProvider({}).load_access_token("")) is None  # no static token configured -> nothing matches


def test_credentials():
    pv = provider()
    assert pv.check_credentials("u", "p")
    assert not pv.check_credentials("u", "P") and not pv.check_credentials("U", "p") and not pv.check_credentials("", "")
    assert not mod.DictAuthProvider({}).check_credentials("", "")  # unset credentials never match


def _authorize(pv, store):
    run(pv.register_client(CLIENT))
    assert run(pv.get_client("c1")).client_secret == "s" and run(pv.get_client("nope")) is None
    params = AuthorizationParams(state="st", scopes=None, code_challenge="chal", redirect_uri="https://claude.ai/cb",
                                 redirect_uri_provided_explicitly=True, resource=None)
    login_url = run(pv.authorize(CLIENT, params))
    return parse_qs(urlparse(login_url).query)["req"][0]


def test_full_code_flow_and_refresh():
    store = {}
    pv = provider(store)
    req = _authorize(pv, store)
    assert pv.login(req, "u", "wrong") is None                 # bad password keeps the pending request
    redirect = pv.login(req, "u", "p")
    q = parse_qs(urlparse(redirect).query)
    assert redirect.startswith("https://claude.ai/cb?") and q["state"] == ["st"]
    code = q["code"][0]
    assert pv.login(req, "u", "p") is None                     # pending request is single use
    ac = run(pv.load_authorization_code(CLIENT, code))
    assert ac.code_challenge == "chal" and ac.subject == "u"
    other = CLIENT.model_copy(update={"client_id": "c2"})
    assert run(pv.load_authorization_code(other, code)) is None  # code bound to its client
    tok = run(pv.exchange_authorization_code(CLIENT, ac))
    assert run(pv.load_authorization_code(CLIENT, code)) is None  # code is single use
    at = run(pv.load_access_token(tok.access_token))
    assert at.client_id == "c1" and at.subject == "u" and at.expires_at > time.time() + mod.ACCESS_TTL - 60
    rt = run(pv.load_refresh_token(CLIENT, tok.refresh_token))
    tok2 = run(pv.exchange_refresh_token(CLIENT, rt, rt.scopes))
    assert run(pv.load_refresh_token(CLIENT, tok.refresh_token)) is None  # rotated
    assert run(pv.load_access_token(tok2.access_token)) is not None
    run(pv.revoke_token(at))
    assert run(pv.load_access_token(tok.access_token)) is None


def test_expired_records_are_dropped():
    store = {}
    pv = provider(store)
    req = _authorize(pv, store)
    ac = run(pv.load_authorization_code(CLIENT, parse_qs(urlparse(pv.login(req, "u", "p")).query)["code"][0]))
    tok = run(pv.exchange_authorization_code(CLIENT, ac))
    store[f"access:{tok.access_token}"]["expires_at"] = time.time() - 1
    assert run(pv.load_access_token(tok.access_token)) is None
    assert f"access:{tok.access_token}" not in store
    store[f"pending:x"] = {"client_id": "c1", "params": {}, "expires_at": time.time() - 1}
    assert pv.login("x", "u", "p") is None
