"""Raw Docling blocks -> retrieval chunks. Pure functions, no heavy imports.

A *block* is one Docling text item or one table (markdown), as extracted:
    {"source", "page", "block_type": "text"|"table", "block_index": int, "text"}
A *chunk* adds "context" (a one-line prefix used only for embedding) and may be a piece of a block
("block_index": "12.1").

Why each step exists (2026-09-19 baseline: R@5 0.40, appendix tables almost never retrieved):
  - footers dropped, short text merged forward: 36% of chunks were <40 chars and crowded the candidate pool
  - table pieces repeat the header rows: 213 of 376 table pieces had lost their column names
  - context prefix (document label, page, page title): Q1/Q2/Q3 releases are structural twins, and a bare
    table row does not say which quarter or which appendix it belongs to
"""
from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Any

MAX_CHUNK_CHARS = 1500
MIN_TEXT_LEN = 10      # drop fragments shorter than this even after merging
SHORT_TEXT_LEN = 40    # text blocks shorter than this are merged into the next block on the page

# Stable UUID namespace so re-indexing the same (source, block) upserts cleanly.
_NS = uuid.UUID("0f4cf7cb-9e3e-4cfa-a5d1-d9b64a4f2fe1")


def chunk_id(chunk: dict[str, Any]) -> str:
    # page is part of the id: block_index restarts in every Docling page window
    return str(uuid.uuid5(_NS, f"{chunk['source']}:{chunk['block_type']}:{chunk['page']}:{chunk['block_index']}"))


def _load_labels() -> dict[str, str]:
    manifest = Path(__file__).resolve().parents[3] / "eval" / "corpus.json"
    if not manifest.exists():
        return {}
    return {d["id"]: d["label"] for d in json.loads(manifest.read_text(encoding="utf-8"))}


DOC_LABELS = _load_labels()  # document id -> "Sanofi Q2 2025 results press release" (eval/corpus.json)
_FOOTER = re.compile(r"^(SANOFI\s+(PRESS RELEASE|FORM 20-F)|\d+\s+Investor Relations|Investor Relations$|PART I+$|ITEM \d)", re.I)
_TABLE_RULE = re.compile(r"^\|?\s*:?-{2,}")


def split_lines(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Split at line boundaries so markdown table rows stay intact; hard-cut only if one line is too long."""
    if len(text) <= max_chars:
        return [text]
    pieces, buf = [], ""
    for line in text.splitlines(keepends=True):
        while len(line) > max_chars:
            pieces.append((buf + line[:max_chars]).strip())
            buf, line = "", line[max_chars:]
        if len(buf) + len(line) > max_chars and buf:
            pieces.append(buf.strip())
            buf = ""
        buf += line
    if buf.strip():
        pieces.append(buf.strip())
    return pieces


def compact_table(md: str) -> str:
    """Docling pads markdown cells to a fixed width; strip the padding (a 9k-char table drops to ~4k)."""
    out = []
    for line in md.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        out.append("| " + " | ".join("---" if set(c) <= set("-:") and c else c for c in cells) + " |")
    return "\n".join(out)


def split_table(md: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Like split_lines, but every continuation piece starts with the table's header + rule rows."""
    md = compact_table(md)
    lines = md.splitlines()
    if len(md) <= max_chars or len(lines) < 3 or not _TABLE_RULE.match(lines[1]):
        return split_lines(md, max_chars)
    header = "\n".join(lines[:2])
    body = split_lines("\n".join(lines[2:]), max(max_chars - len(header) - 1, 200))
    return [f"{header}\n{piece}" for piece in body]


def build_chunks(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    blocks = [b for b in blocks if not (b["block_type"] == "text" and _FOOTER.match(b["text"].strip()))]
    # block_index restarts in every Docling page window, so page must come before it in the order
    blocks.sort(key=lambda b: (b["source"], b["page"] or 0, b["block_type"], b["block_index"]))

    titles: dict[tuple[str, Any], str] = {}
    for b in blocks:  # page title = first heading-sized text block on the page
        key = (b["source"], b["page"])
        if b["block_type"] == "text" and key not in titles and 15 <= len(b["text"]) <= 150:
            titles[key] = b["text"].replace("\n", " ")

    chunks: list[dict[str, Any]] = []

    def emit(b: dict[str, Any], text: str) -> None:
        label = DOC_LABELS.get(b["source"], b["source"])
        context = " | ".join(x for x in (label, f"page {b['page']}", titles.get((b["source"], b["page"]), "")) if x)
        pieces = split_table(text) if b["block_type"] == "table" else split_lines(text)
        for k, piece in enumerate(pieces):
            chunks.append({"text": piece, "context": context, "page": b["page"], "block_type": b["block_type"],
                           "block_index": b["block_index"] if k == 0 else f"{b['block_index']}.{k}",
                           "source": b["source"]})

    carry: list[str] = []
    carry_key: tuple[str, Any] | None = None
    carry_block: dict[str, Any] | None = None

    def flush_carry() -> None:
        nonlocal carry, carry_block
        if carry_block is not None and len(" ".join(carry)) >= MIN_TEXT_LEN:
            emit(carry_block, "\n".join(carry))
        carry, carry_block = [], None

    for b in blocks:
        text = b["text"].strip()
        if b["block_type"] == "table":
            if len(text) >= MIN_TEXT_LEN:
                emit(b, text)
            continue
        key = (b["source"], b["page"])
        if key != carry_key:
            flush_carry()
            carry_key = key
        if len(text) < SHORT_TEXT_LEN:
            carry.append(text)
            carry_block = carry_block or b
            continue
        emit(b, "\n".join(carry + [text]))
        carry, carry_block = [], None
    flush_carry()
    return chunks


def embed_text(chunk: dict[str, Any]) -> str:
    return f"{chunk['context']}\n{chunk['text']}" if chunk.get("context") else chunk["text"]


def _self_check() -> None:
    table = "| Product | Total | US |\n|---|---|---|\n" + "\n".join(f"| Drug{i} | {i} | {i * 2} |" for i in range(200))
    pieces = split_table(table)
    assert len(pieces) > 1 and all(p.startswith("| Product | Total | US |\n| --- |") for p in pieces)
    assert all(len(p) <= MAX_CHUNK_CHARS for p in pieces)
    assert compact_table("| a    |   b |\n|------|-----|\n| 1,0  |  x  |") == "| a | b |\n| --- | --- |\n| 1,0 | x |"
    assert sum(p.count("| Drug") for p in pieces) == 200, "rows lost"
    blocks = [
        {"source": "sanofi_2025Q2_pr.pdf", "page": 13, "block_type": "text", "block_index": 1, "text": "Appendix 1: Q2 2025 net sales by medicine"},
        {"source": "sanofi_2025Q2_pr.pdf", "page": 13, "block_type": "text", "block_index": 2, "text": "Dupixent"},
        {"source": "sanofi_2025Q2_pr.pdf", "page": 13, "block_type": "text", "block_index": 3, "text": "Dupixent sales were EUR 3,832 million in the second quarter."},
        {"source": "sanofi_2025Q2_pr.pdf", "page": 13, "block_type": "text", "block_index": 4, "text": "SANOFI PRESS RELEASE Q2 2025 13"},
        {"source": "sanofi_2025Q2_pr.pdf", "page": 13, "block_type": "table", "block_index": 0, "text": table},
        {"source": "sanofi_2025Q2_pr.pdf", "page": 14, "block_type": "text", "block_index": 5, "text": "End."},
    ]
    out = build_chunks(blocks)
    texts = [c["text"] for c in out if c["block_type"] == "text"]
    assert not any("PRESS RELEASE" in t for t in texts), "footer kept"
    assert any(t.startswith("Dupixent\nDupixent sales") for t in texts), "short heading not merged forward"
    assert all(c["context"].startswith(DOC_LABELS.get("sanofi_2025Q2_pr.pdf", "sanofi_2025Q2_pr.pdf"))
               and "page 13" in c["context"] for c in out if c["page"] == 13)
    assert "Appendix 1" in out[0]["context"]
    assert not any(c["text"] == "End." for c in out), "sub-minimum fragment kept"
    assert len({(c["block_type"], c["block_index"]) for c in out}) == len(out), "duplicate ids"
    print("chunking self-check ok:", len(out), "chunks")


if __name__ == "__main__":
    _self_check()
