"""MCP server over the 27-document pharma corpus, so Claude Desktop / Claude Code can ask new questions.

Same tool bodies as the agentic benchmark (modes/agentic.py, via scripts/19's by-path load), served
over stdio. One long-lived process keeps the BM25 index and corpus warm, unlike the per-call CLI.

    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe src/pharma_vision_rag/mcp_server.py

Run it as a file, not with ``-m pharma_vision_rag.mcp_server``: the package path imports
modes/__init__ -> torch (76 s). stdout is the MCP channel, so nothing here may print.
"""
from __future__ import annotations

import base64
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
# Running as a file puts src/pharma_vision_rag on sys.path, where eval/, utils/ ... would shadow
# top-level names. Swap it for src/.
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != HERE]
sys.path.insert(0, str(ROOT / "src"))

from mcp.server.mcpserver import Image, MCPServer  # noqa: E402  (mcp 2.x; FastMCP was renamed MCPServer)
from mcp.server.mcpserver.exceptions import ToolError  # noqa: E402

VISION_INDEX_DIR = ROOT / "data" / "embeddings" / "vision_index"
MAX_IMAGE_BYTES = 1_000_000  # larger PNGs are re-encoded as JPEG

CORPUS_NOTE = (
    "The corpus is 27 English PDFs (1,709 pages): Sanofi quarterly results press releases and slide decks "
    "for 2024 and 2025 plus the Form 20-F for FY2024 and FY2025, and Q1-Q3 2025 results decks from Novartis, "
    "Roche and AstraZeneca. Questions may be in Korean, but the documents are English: always search in English."
)

INSTRUCTIONS = f"""{CORPUS_NOTE}
Workflow: list_documents to pick document ids for the named company and period, search_text (and search_pages
for chart-heavy slides) to find candidate pages, then open_page to read and verify every figure before answering.
Use calculate for any change, ratio or percentage. Quote figures with unit and period, cite document id and page
for each, and answer in the user's language. If the tools do not show the answer, say so instead of guessing."""


def _load_agentic():
    """Load modes/agentic.py by path, as scripts/19_agentic_tools.py does (package import pulls torch)."""
    spec = importlib.util.spec_from_file_location("_agentic", HERE / "modes" / "agentic.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


# client is never touched by the tool bodies; a placeholder avoids needing an API key.
_mode = _load_agentic().AgenticMode(client=object())
_vision = None  # lazy LocalVisionIndex

mcp = MCPServer("pharma-corpus", instructions=INSTRUCTIONS)


def _content(out: dict) -> str:
    if out["is_error"]:
        raise ToolError(out["content"])
    return out["content"]


@mcp.tool()
def list_documents() -> str:
    """List all 27 documents: id, company, reporting period, kind (pr = press release, deck = results slides,
    20F = annual report) and page count. Call first to pick the document ids for the company and period asked about."""
    return _content(_mode._tool_list_documents())


@mcp.tool(description=f"""Lexical (BM25) search over the parsed text and table chunks of the corpus. {CORPUS_NOTE}
Use English keywords (product names, metric names such as "net sales", "business EPS", periods such as "Q2 2025").
Returns up to k passages (400 chars each) with document id, page and score. Pass document_ids from list_documents to
restrict the search. Text extraction misses numbers drawn inside charts, so open_page the hit to verify figures.""")
def search_text(query: str, document_ids: list[str] | None = None, k: int = 5) -> str:
    return _content(_mode._tool_search_text(query, document_ids or None, max(1, min(int(k), 20))))


@mcp.tool(description=f"""Visual page search: ranks whole page images against the query with a vision embedding
model, which finds charts and slides that text search misses. {CORPUS_NOTE} Returns up to k (document id, page,
score) hits; open_page them to read the content. If vision search is not installed, use search_text instead.""")
def search_pages(query: str, document_ids: list[str] | None = None, k: int = 5) -> str:
    global _vision
    if _vision is None:
        try:
            from pharma_vision_rag.retriever.vision_local import LocalVisionIndex
        except ImportError:
            raise ToolError("Vision page search is not installed on this server (retriever/vision_local.py "
                            "missing). Use search_text instead.") from None
        if not LocalVisionIndex.available(VISION_INDEX_DIR):
            raise ToolError(f"Vision page search is not installed (no index at {VISION_INDEX_DIR}). "
                            "Use search_text instead.")
        _vision = LocalVisionIndex(VISION_INDEX_DIR)
    hits = _vision.search(query, k=max(1, min(int(k), 20)), document_ids=document_ids or None)
    return json.dumps([{"source": d, "page": p, "score": round(float(s), 3)} for d, p, s in hits], ensure_ascii=False)


@mcp.tool()
def open_page(document_id: str, page: int) -> list:
    """Render one page (1-based) of a document as an image so you can read it directly, including chart
    labels and table cells. Always open the page before quoting a number from it, and cite document_id and page."""
    content = _content(_mode._tool_open_page(document_id, page))
    img, header = content[0]["source"], content[1]["text"]
    doc = _mode._load_corpus()[document_id]
    data, fmt = base64.b64decode(img["data"]), "png"
    if len(data) > MAX_IMAGE_BYTES:  # photo-heavy slides (some Novartis pages ~1.7 MB as PNG)
        import io

        from PIL import Image as PILImage
        buf = io.BytesIO()
        PILImage.open(io.BytesIO(data)).save(buf, format="JPEG", quality=85)
        data, fmt = buf.getvalue(), "jpeg"
    return [f"{header} ({doc['label']}, {doc['pages']} pages)", Image(data=data, format=fmt)]


@mcp.tool()
def calculate(expression: str) -> str:
    """Evaluate arithmetic (numbers, + - * / % ** and parentheses only), e.g. "(3832-3303)/3303*100".
    Use it for every increase, decrease, ratio or percentage change instead of computing in your head."""
    return _content(_mode._tool_calculate(expression))


def main() -> None:
    _mode._tool_search_text("warm up", None)  # build/load the BM25 index before the first real call
    mcp.run("stdio")


if __name__ == "__main__":
    main()
