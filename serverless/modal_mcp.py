"""Modal (CPU, scale to zero, no card) hosting of the MCP server over HTTPS, for claude.ai web, Claude Desktop
connectors and Claude Code.

Same server object as src/pharma_vision_rag/mcp_server.py (stdio), served with the streamable-HTTP transport. Two ways in:
  * OAuth 2.1 (what claude.ai / Claude Desktop custom connectors need): this file is also the authorization server.
    /.well-known/oauth-authorization-server, /register (dynamic client registration), /authorize -> /login (one shared
    username + password from the Modal secret), PKCE code -> /token (access 30 d, refresh 90 d). State lives in the
    Modal Dict ``pharma-mcp-auth`` so it survives restarts and is shared by all containers.
  * Static bearer ``MCP_TOKEN`` (Claude Code: ``claude mcp add --transport http ... --header "Authorization: Bearer ..."``).
The corpus lives in the Modal Volume ``pharma-corpus-data`` at the repo-relative paths; the image symlinks /app/data and
/app/eval to it, so mcp_server.py and modes/agentic.py find everything at ROOT=/app unchanged. search_pages calls the
deployed ``pharma-vision-encoder`` app and search_text the ``pharma-text-models`` app + Qdrant Cloud from inside Modal.

    modal volume create pharma-corpus-data && modal volume put pharma-corpus-data data/pdf/corpus /data/pdf/corpus ...
    modal secret create pharma-mcp-token --from-dotenv <file with MCP_TOKEN, MCP_USER, MCP_PASSWORD, QDRANT_CLOUD_URL, QDRANT_CLOUD_API_KEY>
    modal deploy serverless/modal_mcp.py          # prints the https://...modal.run URL; the endpoint is <url>/mcp
    python serverless/modal_mcp.py                # smoke test with the static token (MCP_URL, MCP_TOKEN from .env)
"""
from __future__ import annotations

import hmac
import importlib.util
import os
import secrets
import sys
import time
from html import escape
from pathlib import Path

import modal

APP_NAME = "pharma-mcp"
ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = "/app"            # mcp_server.py computes ROOT = parents[1] of its own file -> /app
VOLUME_MOUNT = "/vol"
PUBLIC_URL = os.environ.get("MCP_PUBLIC_URL", "https://tkddnjs-dlqslek--pharma-mcp-web.modal.run")  # = modal deploy output
CODE_TTL, ACCESS_TTL, REFRESH_TTL = 300, 30 * 86400, 90 * 86400

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("mcp>=2.2,<3", "numpy<3", "pypdfium2>=4.30", "Pillow>=10.4", "python-dotenv>=1.0", "qdrant-client>=1.12,<2")
    .run_commands(f"mkdir -p {REMOTE_ROOT} && ln -s {VOLUME_MOUNT}/data {REMOTE_ROOT}/data && ln -s {VOLUME_MOUNT}/eval {REMOTE_ROOT}/eval")
    # No ~/.modal.toml inside the container: force the Modal encoder instead of encoder_from_env's auto-detection.
    .env({"VISION_ENCODER": "modal", "PYTHONUNBUFFERED": "1"})
    .add_local_dir(ROOT / "src", remote_path=f"{REMOTE_ROOT}/src", ignore=["**/__pycache__", "**/*.pyc"])
)
corpus = modal.Volume.from_name("pharma-corpus-data", create_if_missing=True)
auth_store = modal.Dict.from_name("pharma-mcp-auth", create_if_missing=True)
app = modal.App(APP_NAME)


def _eq(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode(), b.encode())


class DictAuthProvider:
    """OAuthAuthorizationServerProvider over a dict-like store (modal.Dict in production, {} in tests).

    Keys: client:<id>, pending:<req> (authorize params waiting for the login form), code:<code>, access:<tok>,
    refresh:<tok>. Every record carries expires_at and is dropped when read after it.
    ponytail: store calls are synchronous inside async handlers (a few ms each on modal.Dict). Modal Dict entries also
    expire after 7 days without any read or write, so an unused login is simply repeated.
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


@app.function(image=image, volumes={VOLUME_MOUNT: corpus}, secrets=[modal.Secret.from_name("pharma-mcp-token")],
              cpu=2, memory=6144, scaledown_window=300, min_containers=0, timeout=600)
@modal.concurrent(max_inputs=16)   # one container, one resident 2.4 GB index, many MCP sessions
@modal.asgi_app()
def web():
    spec = importlib.util.spec_from_file_location("_mcp_server", f"{REMOTE_ROOT}/src/pharma_vision_rag/mcp_server.py")
    srv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(srv)  # type: ignore[union-attr]
    print(f"search_text dense path: {'on' if srv._dense is not None else 'off (BM25)'}", file=sys.stderr)
    srv._mode._tool_search_text("warm up", None)          # BM25 index (fallback path)
    try:                                                   # vision index resident now, not on the first search_pages
        from pharma_vision_rag.retriever.vision_local import LocalVisionIndex
        srv._vision = LocalVisionIndex(srv.VISION_INDEX_DIR, encoder=srv.encoder_from_env())
    except Exception as e:  # noqa: BLE001 - the tool reports the problem itself on first use
        print(f"vision index not preloaded: {e}", file=sys.stderr)
    attach_oauth(srv.mcp, DictAuthProvider(auth_store, os.environ.get("MCP_TOKEN", ""),
                                           os.environ.get("MCP_USER", ""), os.environ.get("MCP_PASSWORD", "")))
    # host="0.0.0.0" leaves the SDK's localhost-only Host check off (the Host here is *.modal.run); bearer auth on /mcp
    # comes from the SDK's RequireAuthMiddleware. Stateless JSON responses: no session affinity across containers.
    return srv.mcp.streamable_http_app(host="0.0.0.0", stateless_http=True, json_response=True)


def smoke(url: str, token: str) -> None:
    """Initialize, list tools, call the four tools with timings (run from the dev box)."""
    import asyncio

    from mcp import ClientSession
    from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client

    async def run():
        async with create_mcp_http_client(headers={"Authorization": f"Bearer {token}"}) as http:
            async with streamable_http_client(url, http_client=http) as (read, write, *_):
                async with ClientSession(read, write) as s:
                    t = time.time(); await s.initialize(); print(f"initialize {time.time() - t:.1f}s")
                    t = time.time(); tools = await s.list_tools(); print(f"list_tools {time.time() - t:.1f}s: {[x.name for x in tools.tools]}")
                    for name, args in [("search_text", {"query": "Dupixent sales Q2 2025"}),
                                       ("open_page", {"document_id": "sanofi_2025Q2_deck.pdf", "page": 5}),
                                       ("search_pages", {"query": "Beyfortus quarterly sales chart", "k": 3})]:
                        t = time.time(); r = await s.call_tool(name, args)
                        kinds = [c.type for c in r.content]
                        head = next((c.text[:160] for c in r.content if c.type == "text"), "")
                        print(f"{name} {time.time() - t:.1f}s error={r.is_error} {kinds} {head!r}")
    asyncio.run(run())


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    smoke(os.environ["MCP_URL"], os.environ["MCP_TOKEN"])
