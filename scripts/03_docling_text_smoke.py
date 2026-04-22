"""Phase 1-B1 smoke test — index Q1.pdf into Qdrant and search.

Expected:
    - Docling extracts ~300+ text + ~15-20 table chunks from Q1.pdf
    - All chunks embed with BGE-M3 (1024-d, cosine)
    - Upsert into Qdrant collection ``pharma_text`` on port 6335
    - Query "2025 Q1 Dupixent sales" → top hits include pages with €3.5B / €3,480m
    - Query "Business EPS" → top hits mention €1.79 or 17%

Usage:
    python scripts/03_docling_text_smoke.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env")

from pharma_vision_rag.retriever import DoclingTextRetriever  # noqa: E402

PDF_PATH = ROOT / "data" / "pdf" / "Q1.pdf"


def main() -> None:
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    retriever = DoclingTextRetriever(
        qdrant_url=os.environ.get("QDRANT_URL", "http://localhost:6335"),
        qdrant_api_key=os.environ.get("QDRANT_API_KEY") or None,
    )

    # ─── Index ────────────────────────────────────────────────────
    print(f"\n=== Indexing {PDF_PATH.name} ===")
    stats = retriever.index(PDF_PATH, source="Q1.pdf")
    print(f"chunks: {stats['chunks']}  ({stats['by_type']})")
    print(f"collection: {stats['collection']}")

    # ─── Searches ─────────────────────────────────────────────────
    queries = [
        ("What was Dupixent Q1 2025 sales?",      ["3.5", "3,480", "3.48", "dupixent"]),
        ("Sanofi Q1 2025 business EPS growth",    ["1.79", "17.0", "eps"]),
        ("2025 Q1 Beyfortus vaccines revenue",    ["beyfortus", "vaccines"]),
    ]
    successes = 0
    for query, keywords in queries:
        print(f"\n=== Q: {query} ===")
        hits = retriever.search(query, k=3)
        for rank, h in enumerate(hits, start=1):
            snippet = h["text"][:200].replace("\n", " / ")
            print(f"  #{rank} page={h['page']}  type={h['block_type']}  score={h['score']:.3f}")
            print(f"      {snippet}")
        # Accept if any top-3 hit contains any expected keyword
        combined = " ".join(h["text"].lower() for h in hits)
        hit_kw = any(kw.lower() in combined for kw in keywords)
        status = "OK  " if hit_kw else "FAIL"
        print(f"  {status} keyword present in top-3: {keywords}")
        successes += int(hit_kw)

    print(f"\n=== Acceptance: {successes}/{len(queries)} queries passed ===")


if __name__ == "__main__":
    main()
