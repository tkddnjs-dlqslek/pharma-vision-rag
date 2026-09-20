"""Validate eval/questions.jsonl against the schema, type distribution and page inventory.

Checks (fail):
  - 30 questions, unique ids, required fields, type in A/B/C/D, block_type in chart/table/text
  - distribution: A10 / B10 / C5 / D5, explicit 25 / ambiguous 5
  - gold_pages lists every acceptable page; type D also has gold_groups (one group of alternates per hop)
  - every gold page exists in eval/corpus.json (document id + page range)
  - type A gold pages include at least one page tagged chart in the inventory
  - every answer_key string is found in the extracted text of at least one gold page,
    unless needs_review or visual_only is true (chart labels rasterized / values read from bar height)
  - explicit B/C questions: no non-gold page may contain every answer_key (incomplete gold deflates recall)
Report (warn only):
  - answer leakage: type A answer_keys that also appear on non-gold pages of any press release
    (text-only path could answer from the press release instead of the chart)

Usage:
    PYTHONIOENCODING=utf-8 python scripts/12_validate_questions.py
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import pypdfium2 as pdfium

ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = ROOT / "data" / "pdf" / "corpus"
CORPUS = json.loads((ROOT / "eval" / "corpus.json").read_text(encoding="utf-8"))
PAGES = {d["id"]: d["pages"] for d in CORPUS}
PRESS_RELEASES = [d["id"] for d in CORPUS if d["kind"] == "pr"]
QUESTIONS = ROOT / "eval" / "questions.jsonl"
INVENTORY = ROOT / "eval" / "page_inventory.csv"

REQUIRED = {"id", "type", "block_type", "q_ko", "q_en", "answer", "answer_keys",
            "gold_pages", "period_spec", "needs_review", "visual_only", "notes"}
EXPECTED_TYPES = {"A": 10, "B": 10, "C": 5, "D": 5}
EXPECTED_PERIOD = {"explicit": 25, "ambiguous": 5}


def load_inventory() -> dict[tuple[str, int], str]:
    with open(INVENTORY, encoding="utf-8") as f:
        return {(r["source"], int(r["page"])): r["block_type"] for r in csv.DictReader(f)}


_text_cache: dict[str, list[str]] = {}


def page_text(source: str, page: int) -> str:
    if source not in _text_cache:
        doc = pdfium.PdfDocument(str(PDF_DIR / source))
        _text_cache[source] = [doc[i].get_textpage().get_text_range() or "" for i in range(len(doc))]
    return _text_cache[source][page - 1]


def norm(s: str) -> str:
    return " ".join(s.split())


def main() -> int:
    inv = load_inventory()
    qs = [json.loads(l) for l in open(QUESTIONS, encoding="utf-8") if l.strip()]
    errors: list[str] = []
    warns: list[str] = []

    if len(qs) != 30:
        errors.append(f"expected 30 questions, got {len(qs)}")
    ids = [q["id"] for q in qs]
    if len(set(ids)) != len(ids):
        errors.append("duplicate ids")

    for q in qs:
        qid = q.get("id", "?")
        missing = REQUIRED - set(q)
        if missing:
            errors.append(f"{qid}: missing fields {sorted(missing)}")
            continue
        if q["type"] not in EXPECTED_TYPES:
            errors.append(f"{qid}: bad type {q['type']}")
        if q["block_type"] not in {"chart", "table", "text"}:
            errors.append(f"{qid}: bad block_type {q['block_type']}")
        if q["period_spec"] not in EXPECTED_PERIOD:
            errors.append(f"{qid}: bad period_spec")
        if not q["gold_pages"]:
            errors.append(f"{qid}: no gold_pages")
        for gp in q["gold_pages"]:
            if not 1 <= gp["page"] <= PAGES.get(gp["source"], 0):
                errors.append(f"{qid}: gold page {gp['source']} p{gp['page']} not in eval/corpus.json")
        if q.get("gold_groups"):  # multi-hop: groups must partition gold_pages exactly
            flat = [(g["source"], g["page"]) for grp in q["gold_groups"] for g in grp]
            if sorted(flat) != sorted((g["source"], g["page"]) for g in q["gold_pages"]):
                errors.append(f"{qid}: gold_groups and gold_pages list different pages")
        elif q["type"] == "D":
            errors.append(f"{qid}: multi-hop question needs gold_groups (one group per hop)")
        if q["type"] == "A":
            if not any(inv.get((gp["source"], gp["page"])) == "chart" for gp in q["gold_pages"]):
                errors.append(f"{qid}: type A but no gold page tagged chart")
        # answer keys must be on a gold page (text layer) unless visual-only
        if not (q["needs_review"] or q["visual_only"]):
            for key in q["answer_keys"]:
                found = any(norm(key) in norm(page_text(gp["source"], gp["page"])) for gp in q["gold_pages"])
                if not found:
                    errors.append(f"{qid}: answer_key '{key}' not found in text of any gold page")
        # gold completeness: any non-gold page whose text holds ALL answer keys is a missing gold page
        if q["type"] in "BC" and q["period_spec"] == "explicit" and q["answer_keys"] and not q["visual_only"]:
            gold_set = {(gp["source"], gp["page"]) for gp in q["gold_pages"]}
            for (src, pg) in ((d, n) for d, total in PAGES.items() for n in range(1, total + 1)):
                if (src, pg) not in gold_set and all(norm(k) in norm(page_text(src, pg)) for k in q["answer_keys"]):
                    errors.append(f"{qid}: {src} p{pg} contains every answer_key but is not in gold_pages")
        # leakage report for chart questions
        if q["type"] == "A" and q["period_spec"] == "explicit":
            gold = {(gp["source"], gp["page"]) for gp in q["gold_pages"]}
            for key in q["answer_keys"]:
                if len(key.replace(",", "").replace(".", "").replace("-", "")) < 3:
                    continue  # too short to be meaningful
                for src in PRESS_RELEASES:
                    hits = [p for p in range(1, PAGES[src] + 1)
                            if (src, p) not in gold and norm(key) in norm(page_text(src, p))]
                    if hits:
                        warns.append(f"{qid}: key '{key}' also on {src} pages {hits} (leakage risk)")

    tc = Counter(q["type"] for q in qs)
    if dict(tc) != EXPECTED_TYPES:
        errors.append(f"type distribution {dict(tc)} != {EXPECTED_TYPES}")
    pc = Counter(q["period_spec"] for q in qs)
    if dict(pc) != EXPECTED_PERIOD:
        errors.append(f"period distribution {dict(pc)} != {EXPECTED_PERIOD}")
    nr = [q["id"] for q in qs if q["needs_review"]]

    print(f"questions: {len(qs)}  types: {dict(tc)}  period: {dict(pc)}  needs_review: {nr}")
    for w in warns:
        print("WARN", w)
    for e in errors:
        print("FAIL", e)
    print("OK" if not errors else f"{len(errors)} error(s)")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
