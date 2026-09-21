"""CLI wrapper around AgenticMode's tools, so the E4 agent can be a subagent instead of an API loop.

modes/agentic.py defines the six tools for the API version (EXPERIMENT_PLAN 3.3). We have no API
key, so the agent is a Claude Code subagent that calls these same tools as shell commands. The tool
bodies are reused from that file (loaded by path, per its own import warning) so both versions
behave identically: same BM25 pool, same page-bounds checks, same "caption index not built" error.

    python scripts/19_agentic_tools.py list_documents
    python scripts/19_agentic_tools.py search_text "Dupixent Q2 2025 sales" [--documents a.pdf b.pdf]
    python scripts/19_agentic_tools.py search_pages "net debt bridge"
    python scripts/19_agentic_tools.py open_page sanofi_2025Q2_deck.pdf 28
    python scripts/19_agentic_tools.py calculate "(3832-3303)/3303*100"

open_page differs from the API tool in transport only: it writes a PNG and prints the path for the
agent to Read, instead of returning a base64 image block.

The BM25 index is rebuilt per invocation (~2s for 15k chunks); with at most 10 tool calls per
question that is cheaper than keeping a server alive.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

IMG_DIR = ROOT / "data" / "images" / "agentic"
TARGET_WIDTH = 1400  # same as the generation tasks, so chart labels stay legible


def _load_agentic():
    """Load modes/agentic.py by path: importing it as a package member pulls torch (76s)."""
    spec = importlib.util.spec_from_file_location(
        "_agentic", ROOT / "src" / "pharma_vision_rag" / "modes" / "agentic.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="tool", required=True)
    sub.add_parser("list_documents")
    for name in ("search_text", "search_pages"):
        p = sub.add_parser(name)
        p.add_argument("query")
        p.add_argument("--documents", nargs="*", default=None)
    p = sub.add_parser("open_page")
    p.add_argument("document_id")
    p.add_argument("page", type=int)
    p = sub.add_parser("calculate")
    p.add_argument("expression")
    a = ap.parse_args()

    agentic = _load_agentic()
    # client is never touched by the tool bodies; pass a placeholder so no API key is needed.
    mode = agentic.AgenticMode(client=object())

    if a.tool == "open_page":
        doc = mode._load_corpus().get(a.document_id)
        if doc is None:
            sys.exit(f"unknown document_id {a.document_id!r}. Run list_documents first.")
        if not (1 <= a.page <= doc["pages"]):
            sys.exit(f"{a.document_id} has {doc['pages']} pages; page {a.page} is out of range.")
        IMG_DIR.mkdir(parents=True, exist_ok=True)
        dst = IMG_DIR / f"{a.document_id[:-4]}_p{a.page}.png"
        if not dst.exists():
            import pypdfium2 as pdfium
            pg = pdfium.PdfDocument(str(ROOT / "data" / "pdf" / "corpus" / a.document_id))[a.page - 1]
            pg.render(scale=TARGET_WIDTH / pg.get_width()).to_pil().convert("RGB").save(dst)
        print(dst)
        return

    if a.tool == "list_documents":
        out = mode._tool_list_documents()
    elif a.tool == "search_text":
        out = mode._tool_search_text(a.query, a.documents)
    elif a.tool == "search_pages":
        out = mode._tool_search_pages(a.query, a.documents)
    else:
        out = mode._tool_calculate(a.expression)

    content = out["content"]
    if isinstance(content, str) and content.startswith(("[", "{")):
        content = json.dumps(json.loads(content), ensure_ascii=False, indent=1)
    print(content)
    if out["is_error"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
