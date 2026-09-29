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

import importlib.util
import os
import sys
import time
from pathlib import Path

import modal

APP_NAME = "pharma-mcp"
ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = "/app"            # mcp_server.py computes ROOT = parents[1] of its own file -> /app
VOLUME_MOUNT = "/vol"
PUBLIC_URL = os.environ.get("MCP_PUBLIC_URL", "https://tkddnjs-dlqslek--pharma-mcp-web.modal.run")  # = modal deploy output

# The provider lives in src/ (shared with services/mcp): on the dev box that is ROOT/src, inside the container /app/src.
sys.path[:0] = [str(ROOT / "src"), f"{REMOTE_ROOT}/src"]
from pharma_vision_rag.mcp_auth import ACCESS_TTL, CODE_TTL, REFRESH_TTL, DictAuthProvider, attach_oauth  # noqa: E402,F401

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
                                           os.environ.get("MCP_USER", ""), os.environ.get("MCP_PASSWORD", ""),
                                           public_url=PUBLIC_URL))
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
