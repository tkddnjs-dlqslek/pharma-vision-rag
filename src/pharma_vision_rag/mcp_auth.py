"""OAuth 2.1 authorization server for the remote MCP deployments, shared by serverless/modal_mcp.py (state in a
modal.Dict) and services/mcp (state in JsonStore, a JSON file).

DictAuthProvider implements the mcp SDK's OAuthAuthorizationServerProvider over any dict-like store; attach_oauth
turns the stdio-built MCPServer into an OAuth resource + authorization server and adds the /login form. A static
bearer token (MCP_TOKEN) is accepted next to the issued tokens, for Claude Code.
"""
from __future__ import annotations

import hmac
import json
import os
import secrets
import threading
import time
from collections.abc import MutableMapping
from html import escape
from pathlib import Path

PUBLIC_URL = os.environ.get("MCP_PUBLIC_URL", "")   # the deployment's public origin, e.g. https://mcp.example.com
CODE_TTL, ACCESS_TTL, REFRESH_TTL = 300, 30 * 86400, 90 * 86400


class JsonStore(MutableMapping):
    """dict persisted to a JSON file after every write (services/mcp: no Modal Dict there).

    ponytail: the whole file is rewritten under a lock on each write; it holds a handful of tokens. An atomic
    os.replace keeps a crash from leaving a half-written file. Losing the file (container recreated without
    the volume) just means one more login.
    """

    def __init__(self, path: str | Path):
        self.path, self._lock = Path(path), threading.Lock()
        try:
            self._d = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self._d = {}

    def _flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self._d), encoding="utf-8")
        os.replace(tmp, self.path)

    def __getitem__(self, key):
        return self._d[key]

    def __setitem__(self, key, value) -> None:
        with self._lock:
            self._d[key] = value
            self._flush()

    def __delitem__(self, key) -> None:
        with self._lock:
            del self._d[key]
            self._flush()

    def __iter__(self):
        return iter(list(self._d))

    def __len__(self) -> int:
        return len(self._d)


def _eq(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode(), b.encode())


class DictAuthProvider:
    """OAuthAuthorizationServerProvider over a dict-like store (modal.Dict or JsonStore in production, {} in tests).

    Keys: client:<id>, pending:<req> (authorize params waiting for the login form), code:<code>, access:<tok>,
    refresh:<tok>. Every record carries expires_at and is dropped when read after it.
    ponytail: store calls are synchronous inside async handlers (a few ms each). Modal Dict entries also expire after
    7 days without any read or write, so an unused login is simply repeated.
    """

    def __init__(self, store, static_token: str = "", username: str = "", password: str = "", public_url: str = PUBLIC_URL):
        self.store, self.static_token, self.username, self.password = store, static_token, username, password
        self.public_url = public_url

    # store helpers
    def _get(self, key: str):
        rec = self.store.get(key)
        if rec is not None and rec.get("expires_at") is not None and rec["expires_at"] < time.time():
            self._pop(key)
            return None
        return rec

    def _pop(self, key: str):
        try:
            return self.store.pop(key)
        except KeyError:
            return None

    def check_credentials(self, username: str, password: str) -> bool:
        return bool(self.username and self.password) and _eq(username, self.username) and _eq(password, self.password)

    # OAuthAuthorizationServerProvider
    async def get_client(self, client_id: str):
        from mcp.shared.auth import OAuthClientInformationFull
        rec = self.store.get(f"client:{client_id}")
        return OAuthClientInformationFull.model_validate(rec) if rec else None

    async def register_client(self, client_info) -> None:
        self.store[f"client:{client_info.client_id}"] = client_info.model_dump(mode="json")

    async def authorize(self, client, params) -> str:
        """Park the request and send the browser to the login form; login() finishes the redirect."""
        req = secrets.token_urlsafe(24)
        self.store[f"pending:{req}"] = {"client_id": client.client_id, "params": params.model_dump(mode="json"),
                                        "expires_at": time.time() + CODE_TTL}
        return f"{self.public_url}/login?req={req}"

    def login(self, req: str, username: str, password: str) -> str | None:
        """Credentials + a live pending request -> redirect URL carrying the authorization code, else None."""
        if not self.check_credentials(username, password):
            return None
        pending = self._pop(f"pending:{req}") if req else None
        if not pending or pending["expires_at"] < time.time():
            return None
        from mcp.server.auth.provider import construct_redirect_uri
        p, code = pending["params"], secrets.token_urlsafe(32)
        self.store[f"code:{code}"] = {"code": code, "client_id": pending["client_id"], "scopes": p.get("scopes") or [],
                                      "code_challenge": p["code_challenge"], "redirect_uri": p["redirect_uri"],
                                      "redirect_uri_provided_explicitly": p["redirect_uri_provided_explicitly"],
                                      "resource": p.get("resource"), "subject": username,
                                      "expires_at": time.time() + CODE_TTL}
        return construct_redirect_uri(p["redirect_uri"], code=code, state=p.get("state"))

    async def load_authorization_code(self, client, authorization_code: str):
        from mcp.server.auth.provider import AuthorizationCode
        rec = self._get(f"code:{authorization_code}")
        return AuthorizationCode.model_validate(rec) if rec and rec["client_id"] == client.client_id else None

    def _issue(self, client_id: str, scopes: list[str], resource, subject):
        from mcp.shared.auth import OAuthToken
        access, refresh, now = secrets.token_urlsafe(32), secrets.token_urlsafe(32), int(time.time())
        self.store[f"access:{access}"] = {"token": access, "client_id": client_id, "scopes": scopes, "resource": resource,
                                          "subject": subject, "expires_at": now + ACCESS_TTL}
        self.store[f"refresh:{refresh}"] = {"token": refresh, "client_id": client_id, "scopes": scopes, "resource": resource,
                                            "subject": subject, "expires_at": now + REFRESH_TTL}
        return OAuthToken(access_token=access, expires_in=ACCESS_TTL, refresh_token=refresh,
                          scope=" ".join(scopes) or None)

    async def exchange_authorization_code(self, client, authorization_code):
        self._pop(f"code:{authorization_code.code}")  # single use
        return self._issue(client.client_id, authorization_code.scopes, authorization_code.resource, authorization_code.subject)

    async def load_refresh_token(self, client, refresh_token: str):
        from mcp.server.auth.provider import RefreshToken
        rec = self._get(f"refresh:{refresh_token}")
        return RefreshToken.model_validate(rec) if rec and rec["client_id"] == client.client_id else None

    async def exchange_refresh_token(self, client, refresh_token, scopes: list[str]):
        self._pop(f"refresh:{refresh_token.token}")  # rotate
        return self._issue(client.client_id, scopes, refresh_token.resource, refresh_token.subject)

    async def load_access_token(self, token: str):
        """The token verifier: the static MCP_TOKEN or an issued, unexpired access token."""
        from mcp.server.auth.provider import AccessToken
        if self.static_token and _eq(token, self.static_token):
            return AccessToken(token=token, client_id="static-token", scopes=[])
        rec = self._get(f"access:{token}")
        return AccessToken.model_validate(rec) if rec else None

    async def revoke_token(self, token) -> None:
        self._pop(f"access:{token.token}")
        self._pop(f"refresh:{token.token}")

    async def exchange_identity_assertion(self, client, params):  # SEP-990, not offered
        from mcp.server.auth.provider import TokenError
        raise TokenError("unsupported_grant_type")


LOGIN_FORM = """<!doctype html><html><head><meta charset="utf-8"><title>pharma-corpus MCP login</title>
<style>body{{font-family:system-ui;max-width:22rem;margin:4rem auto}}input{{width:100%;margin:.3rem 0 .8rem;padding:.5rem}}
button{{padding:.5rem 1rem}}.err{{color:#b00}}</style></head><body><h2>pharma-corpus MCP</h2>
<p>Sign in to connect this corpus to Claude.</p>{error}<form method="post">
<input type="hidden" name="req" value="{req}"><label>Username<input name="username" autocomplete="username" required></label>
<label>Password<input name="password" type="password" autocomplete="current-password" required></label>
<button type="submit">Sign in</button></form></body></html>"""


def attach_oauth(mcp, provider: DictAuthProvider) -> None:
    """Turn the stdio-built MCPServer into an OAuth resource + authorization server and add the /login form."""
    public_url = provider.public_url
    from mcp.server.auth.provider import ProviderTokenVerifier
    from mcp.server.auth.settings import AuthSettings, ClientRegistrationOptions, RevocationOptions
    from starlette.responses import HTMLResponse, RedirectResponse

    mcp.settings.auth = AuthSettings(issuer_url=public_url, resource_server_url=f"{public_url}/mcp",
                                     validate_token_resource=False,  # our own store issues the tokens
                                     client_registration_options=ClientRegistrationOptions(enabled=True),
                                     revocation_options=RevocationOptions(enabled=True))
    # MCPServer takes these only in its constructor, which mcp_server.py already ran; same two attributes it sets.
    mcp._auth_server_provider = provider
    mcp._token_verifier = ProviderTokenVerifier(provider)

    @mcp.custom_route("/login", methods=["GET", "POST"])
    async def login(request):
        if request.method == "GET":
            return HTMLResponse(LOGIN_FORM.format(req=escape(request.query_params.get("req", "")), error=""))
        form = await request.form()
        target = provider.login(str(form.get("req", "")), str(form.get("username", "")), str(form.get("password", "")))
        if target is None:
            return HTMLResponse(LOGIN_FORM.format(req=escape(str(form.get("req", ""))),
                                                  error='<p class="err">Wrong username or password, or the request expired.</p>'),
                                status_code=401)
        return RedirectResponse(target, status_code=302)
