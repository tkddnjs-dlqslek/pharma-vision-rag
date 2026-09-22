"""Calls each MCP tool function directly: no transport, no network, no Claude.

Run: PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m pytest tests/test_mcp_server.py -q
Needs the local corpus PDFs and text_chunks.jsonl (same as the open_page tests in test_eval_generation.py).
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("_test_mcp_server", ROOT / "src" / "pharma_vision_rag" / "mcp_server.py")
srv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(srv)  # type: ignore[union-attr]
ToolError = srv.ToolError


def test_list_documents():
    docs = json.loads(srv.list_documents())
    assert len(docs) == 27 and {"id", "company", "period", "pages"} <= docs[0].keys()


def test_search_text_filter_and_k():
    hits = json.loads(srv.search_text("Dupixent sales", document_ids=["sanofi_2025Q2_pr.pdf"], k=3))
    assert 1 <= len(hits) <= 3 and all(h["source"] == "sanofi_2025Q2_pr.pdf" for h in hits)


def test_open_page_returns_header_and_image():
    header, image = srv.open_page("sanofi_2025Q1_pr.pdf", 1)
    assert header.startswith("sanofi_2025Q1_pr.pdf p.1")
    assert image.data[:4] == b"\x89PNG" and len(image.data) < srv.MAX_IMAGE_BYTES


def test_open_page_errors_are_tool_errors():
    with pytest.raises(ToolError):
        srv.open_page("does_not_exist.pdf", 1)
    with pytest.raises(ToolError):
        srv.open_page("sanofi_2025Q1_pr.pdf", 9999)


def test_calculate():
    assert float(srv.calculate("(3832-3303)/3303*100")) == pytest.approx(16.0157, abs=1e-3)
    with pytest.raises(ToolError):
        srv.calculate("__import__('os')")


def test_search_pages_answers_or_says_not_installed():
    try:
        hits = json.loads(srv.search_pages("Dupixent sales chart", k=2))
    except ToolError as e:
        assert "search_text" in str(e)
    else:
        assert len(hits) <= 2
