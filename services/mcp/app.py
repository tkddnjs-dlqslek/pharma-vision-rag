"""The pharma corpus MCP server in a Docker container (any host), for claude.ai web, Claude Desktop and Claude Code.

Same MCPServer as src/pharma_vision_rag/mcp_server.py (stdio) served with the streamable-HTTP transport, the same
OAuth 2.1 login as serverless/modal_mcp.py (pharma_vision_rag.mcp_auth), with the token store in a JSON file instead
of a Modal Dict. Static bearer MCP_TOKEN still works (Claude Code).

Startup: the corpus files the server reads (data/pdf/corpus, eval/corpus.json, data/embeddings/v2/text/text_chunks.jsonl,
data/embeddings/vision_index_pooled) are downloaded from the HF dataset repo DATA_REPO (scripts/30_upload_corpus_data.py
fills it) into MCP_DATA_DIR; the Dockerfile symlinks ROOT/data and ROOT/eval there, so mcp_server.py (ROOT = the parent
of src/) finds them unchanged. search_pages and the dense search_text call the services/encoders container
(ENCODER_URL, ENCODER_TOKEN); the text vectors are in Qdrant Cloud (QDRANT_CLOUD_URL, QDRANT_CLOUD_API_KEY).

Env: MCP_TOKEN, MCP_USER, MCP_PASSWORD, MCP_PUBLIC_URL (public https origin, for OAuth redirects and metadata),
ENCODER_URL, ENCODER_TOKEN, QDRANT_CLOUD_URL, QDRANT_CLOUD_API_KEY, DATA_REPO, HF_TOKEN (private dataset).
Optional: AUTH_STORE_PATH (default /data/auth.json), MCP_DATA_DIR (default ROOT), PORT (default 8000),
MCP_PRELOAD_VISION=0 and MCP_ROOT=<checkout that already holds the data> (tests).
"""
from __future__ import annotations

import importlib.util
import os
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("MCP_ROOT") or Path(__file__).resolve().parent)
DATA_DIR = Path(os.environ.get("MCP_DATA_DIR") or ROOT)
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("VISION_ENCODER", "http")   # no RunPod keys or Modal token here: encoder_from_env must not guess
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")


def public_url() -> str:
    return (os.environ.get("MCP_PUBLIC_URL") or f"http://localhost:{os.environ.get('PORT', '8000')}").rstrip("/")


def fetch_data() -> None:
    """snapshot_download(DATA_REPO) into DATA_DIR; files already present and unchanged are skipped."""
    repo = os.environ.get("DATA_REPO", "").strip()
    if not repo:
        print("DATA_REPO not set: serving whatever is already under", DATA_DIR, file=sys.stderr)
        return
    from huggingface_hub import snapshot_download
    t = time.time()
    snapshot_download(repo_id=repo, repo_type="dataset", local_dir=str(DATA_DIR), token=os.environ.get("HF_TOKEN") or None)
    print(f"corpus data from {repo} ready in {time.time() - t:.0f}s", file=sys.stderr)


def build_app():
    from pharma_vision_rag.mcp_auth import DictAuthProvider, JsonStore, attach_oauth
    spec = importlib.util.spec_from_file_location("_mcp_server", ROOT / "src" / "pharma_vision_rag" / "mcp_server.py")
    srv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(srv)  # type: ignore[union-attr]
    print(f"search_text dense path: {'on' if srv._dense is not None else 'off (BM25)'}", file=sys.stderr)
    srv._mode._tool_search_text("warm up", None)          # BM25 index (fallback path)
    if os.environ.get("MCP_PRELOAD_VISION", "1") == "1":   # 2.4 GB index resident now, not on the first search_pages
        try:
            from pharma_vision_rag.retriever.vision_local import LocalVisionIndex
            srv._vision = LocalVisionIndex(srv.VISION_INDEX_DIR, encoder=srv.encoder_from_env())
        except Exception as e:  # noqa: BLE001 - the tool reports the problem itself on first use
            print(f"vision index not preloaded: {e}", file=sys.stderr)
    store = JsonStore(os.environ.get("AUTH_STORE_PATH", "/data/auth.json"))
    attach_oauth(srv.mcp, DictAuthProvider(store, os.environ.get("MCP_TOKEN", ""), os.environ.get("MCP_USER", ""),
                                           os.environ.get("MCP_PASSWORD", ""), public_url=public_url()))

    @srv.mcp.custom_route("/health", methods=["GET"])
    async def health(request):
        from starlette.responses import JSONResponse
        return JSONResponse({"status": "ok", "dense_text": srv._dense is not None, "vision_index": srv._vision is not None,
                             "oauth_clients": sum(k.startswith("client:") for k in store)})

    # host="0.0.0.0" leaves the SDK's localhost-only Host check off (the Host here is the tunnel or LAN name); bearer
    # auth on /mcp comes from the SDK's RequireAuthMiddleware. Stateless JSON responses: nothing to lose on a restart.
    return srv.mcp.streamable_http_app(host="0.0.0.0", stateless_http=True, json_response=True)


if __name__ == "__main__":
    import uvicorn
    fetch_data()
    uvicorn.run(build_app(), host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
