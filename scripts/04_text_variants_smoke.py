"""Phase 1-B2 smoke test — compare baseline / QT / HyDE text retrievers.

Focus case: "business EPS growth" — the known baseline failure from
scripts/03_docling_text_smoke.py. HyDE should recover it by embedding
a report-style hypothetical passage that lands near "Business EPS was €1.79,
up 17.0%".

Other cases:
- KR -> EN query via QT (cross-lingual sanity).
- Dupixent named-entity query (all 3 variants should succeed).

Usage:
    python scripts/04_text_variants_smoke.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env")

from pharma_vision_rag.retriever import (  # noqa: E402
    DoclingTextRetriever,
    TextBaselineRetriever,
    TextHyDERetriever,
    TextQTRetriever,
)

EPS_KEYWORDS = {"1.79", "17.0", "15.7", "eps"}
DUPIXENT_KEYWORDS = {"3.5", "3,480", "dupixent", "20.3"}


def show_hits(label: str, hits, max_chars: int = 150) -> None:
    print(f"  [{label}]")
    for rank, h in enumerate(hits, start=1):
        snippet = h["text"][:max_chars].replace("\n", " / ")
        print(f"    #{rank} p{h['page']} {h['block_type']:5s} {h['score']:.3f}  {snippet}")


def keyword_hit(hits, keywords) -> bool:
    return any(kw.lower() in " ".join(h["text"].lower() for h in hits) for kw in keywords)


def main() -> None:
    base = DoclingTextRetriever(
        qdrant_url=os.environ.get("QDRANT_URL", "http://localhost:6335"),
        qdrant_api_key=os.environ.get("QDRANT_API_KEY") or None,
    )
    variants = {
        "baseline": TextBaselineRetriever(base),
        "qt":       TextQTRetriever(base),
        "hyde":     TextHyDERetriever(base),
    }

    cases = [
        {
            "name": "EPS growth (abstract, baseline failure case)",
            "query": "Sanofi Q1 2025 business EPS growth",
            "keywords": EPS_KEYWORDS,
        },
        {
            "name": "Dupixent Q1 sales (named entity, baseline OK)",
            "query": "What was Dupixent Q1 2025 sales?",
            "keywords": DUPIXENT_KEYWORDS,
        },
        {
            "name": "Korean cross-lingual: Dupixent",
            "query": "2025년 1분기 Dupixent 매출 증가율은?",
            "keywords": DUPIXENT_KEYWORDS,
        },
    ]

    summary = []
    for case in cases:
        print(f"\n=== {case['name']} ===")
        print(f"Q: {case['query']}")
        row = {"case": case["name"]}
        for vname, variant in variants.items():
            hits = variant.search(case["query"], k=3)
            rewritten = hits[0]["rewritten_query"] if hits else "(no hits)"
            ok = keyword_hit(hits, case["keywords"])
            print(f"\n  --- {vname} ---")
            if rewritten != case["query"]:
                print(f"  rewritten: {rewritten[:200]}")
            show_hits(f"top-3, kw hit = {'OK' if ok else 'FAIL'}", hits)
            row[vname] = ok
        summary.append(row)

    print("\n" + "=" * 70)
    print(f"{'Case':55s} {'baseline':>10s} {'qt':>6s} {'hyde':>6s}")
    print("-" * 70)
    for row in summary:
        mark = lambda b: " OK " if b else "FAIL"
        print(f"{row['case'][:55]:55s} {mark(row['baseline']):>10s} {mark(row['qt']):>6s} {mark(row['hyde']):>6s}")
    print("=" * 70)

    hyde_recovers_eps = summary[0]["hyde"] and not summary[0]["baseline"]
    print(f"\nHyDE recovers the baseline EPS failure? {'YES — the promised ablation story works' if hyde_recovers_eps else 'no (re-tune HYDE_SYSTEM or pick a different failure case)'}")


if __name__ == "__main__":
    main()
