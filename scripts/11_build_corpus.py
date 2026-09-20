"""Build the indexed corpus (Phase 2 v2, 2026-09-20): 27 full documents, ~1,700 pages.

Copies the raw downloads into data/pdf/corpus/ under uniform names and writes eval/corpus.json
(the single source of truth for document ids, labels and page counts; committed).

v1 indexed 172 pages (three press releases + a 50-page 20-F extract + three 15-page deck extracts). That fits
in a long context window, so RAG was hard to justify. v2 uses the full documents, the prior year and three
peers: near-duplicate pages across quarters, years and companies are what makes retrieval hard in practice.

`--migrate-v1` rewrites eval/questions.jsonl and eval/page_inventory.csv from v1 names/extract page numbers
to v2 (idempotent: rows already in v2 form are left alone).

Usage:
    PYTHONIOENCODING=utf-8 python scripts/11_build_corpus.py [--migrate-v1]
"""
from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

import pypdfium2 as pdfium

ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = ROOT / "data" / "pdf"
RAW = PDF_DIR / "raw"
OUT = PDF_DIR / "corpus"
MANIFEST = ROOT / "eval" / "corpus.json"

KIND = {"pr": "results press release", "deck": "results presentation slides", "20F": "Form 20-F annual report"}


def _doc(doc_id: str, raw: Path, company: str, period: str, kind: str) -> dict:
    return {"id": doc_id, "raw": raw, "company": company, "period": period, "kind": kind,
            "label": f"{company} {period} {KIND[kind]}"}


DOCS: list[dict] = [
    *[_doc(f"sanofi_2025Q{q}_pr.pdf", PDF_DIR / f"Q{q}.pdf", "Sanofi", f"Q{q} 2025", "pr") for q in (1, 2, 3)],
    *[_doc(f"sanofi_2025Q{q}_deck.pdf", RAW / f"Q{q}_deck_full.pdf", "Sanofi", f"Q{q} 2025", "deck") for q in (1, 2, 3)],
    _doc("sanofi_2025Q4_pr.pdf", RAW / "sanofi_2025Q4_pr.pdf", "Sanofi", "Q4 and full-year 2025", "pr"),
    _doc("sanofi_2025Q4_deck.pdf", RAW / "sanofi_2025Q4_deck.pdf", "Sanofi", "Q4 and full-year 2025", "deck"),
    _doc("sanofi_20F_FY2025.pdf", PDF_DIR / "Form 20-F 2025 (Oct 2025).pdf", "Sanofi", "FY2025", "20F"),
    *[_doc(f"sanofi_2024Q{q}_{k}.pdf", RAW / f"sanofi_2024Q{q}_{k}.pdf", "Sanofi",
           f"Q{q} 2024" if q < 4 else "Q4 and full-year 2024", k) for q in (1, 2, 3, 4) for k in ("pr", "deck")],
    _doc("sanofi_20F_FY2024.pdf", RAW / "sanofi_20F_FY2024.pdf", "Sanofi", "FY2024", "20F"),
    *[_doc(f"{c.lower()}_2025Q{q}_deck.pdf", RAW / f"{c.lower()}_2025Q{q}_deck.pdf", c, f"Q{q} 2025", "deck")
      for c in ("Novartis", "Roche", "AstraZeneca") for q in (1, 2, 3)],
]

# v1 extract page -> raw page (from the retired scripts/11_build_extracts.py)
_V1_20F = [25, 26, 27, 28, 30, 31, 32, 33, 34, 39, 40, 44, 45, 46, 47, 48, 49, 51, 52, 55, 64, 65, 68, 70, 71, 77,
           78, 79, 80, 81, 82, 85, 86, 87, 92, 117, 133, 145, 175, 196, 197, 198, 202, 231, 235, 254, 256, 288, 290, 292]
_V1_DECK = {1: [5, 6, 7, 9, 12, 14, 19, 20, 23, 24, 25, 26, 27, 28, 30],
            2: [5, 6, 7, 8, 9, 10, 12, 14, 17, 21, 24, 25, 26, 27, 28],
            3: [5, 6, 7, 8, 9, 10, 12, 13, 17, 18, 24, 25, 26, 27, 28]}


def v1_to_v2(source: str, page: int) -> tuple[str, int]:
    if source == "20F_extract.pdf":
        return "sanofi_20F_FY2025.pdf", _V1_20F[page - 1]
    for q in (1, 2, 3):
        if source == f"Q{q}.pdf":
            return f"sanofi_2025Q{q}_pr.pdf", page
        if source == f"Q{q}_deck.pdf":
            return f"sanofi_2025Q{q}_deck.pdf", _V1_DECK[q][page - 1]
    return source, page  # already v2


def build() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for d in DOCS:
        if not d["raw"].exists():
            raise SystemExit(f"missing raw PDF: {d['raw']}")
        dst = OUT / d["id"]
        if not dst.exists() or dst.stat().st_size != d["raw"].stat().st_size:
            shutil.copyfile(d["raw"], dst)
        pages = len(pdfium.PdfDocument(str(dst)))
        manifest.append({k: d[k] for k in ("id", "company", "period", "kind", "label")} | {"pages": pages})
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(manifest)} documents, {sum(m['pages'] for m in manifest)} pages -> {OUT}\nmanifest -> {MANIFEST}")


def migrate_v1() -> None:
    qpath = ROOT / "eval" / "questions.jsonl"
    qs = [json.loads(l) for l in open(qpath, encoding="utf-8") if l.strip()]
    conv = lambda g: dict(zip(("source", "page"), v1_to_v2(g["source"], g["page"])))  # noqa: E731
    for q in qs:
        q["gold_pages"] = [conv(g) for g in q["gold_pages"]]
        if q.get("gold_groups"):
            q["gold_groups"] = [[conv(g) for g in grp] for grp in q["gold_groups"]]
    open(qpath, "w", encoding="utf-8").writelines(json.dumps(q, ensure_ascii=False) + "\n" for q in qs)

    ipath = ROOT / "eval" / "page_inventory.csv"
    rows = list(csv.DictReader(open(ipath, encoding="utf-8")))
    for r in rows:
        r["source"], page = v1_to_v2(r["source"], int(r["page"]))
        r["page"] = page
    with open(ipath, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["source", "page", "block_type", "note"])
        w.writeheader()
        w.writerows(rows)
    print(f"migrated {len(qs)} questions and {len(rows)} inventory rows to v2 ids/pages")


def _self_check() -> None:
    assert v1_to_v2("Q2.pdf", 13) == ("sanofi_2025Q2_pr.pdf", 13)
    assert v1_to_v2("20F_extract.pdf", 28) == ("sanofi_20F_FY2025.pdf", 79)
    assert v1_to_v2("Q1_deck.pdf", 4) == ("sanofi_2025Q1_deck.pdf", 9)
    assert v1_to_v2("sanofi_2025Q1_deck.pdf", 9) == ("sanofi_2025Q1_deck.pdf", 9), "migration must be idempotent"
    assert len({d["id"] for d in DOCS}) == len(DOCS) == 27


if __name__ == "__main__":
    _self_check()
    build()
    if "--migrate-v1" in sys.argv:
        migrate_v1()
