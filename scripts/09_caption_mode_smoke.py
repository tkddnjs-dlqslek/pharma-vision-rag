"""Phase 1 smoke test — caption mode end-to-end.

Pre-flight:
    1. scripts/08_index_captions.py was run for at least Q1.pdf.

Usage:
    PYTHONIOENCODING=utf-8 python scripts/09_caption_mode_smoke.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env")

from pharma_vision_rag.modes import CaptionMode  # noqa: E402
from pharma_vision_rag.retriever import CaptionRetriever  # noqa: E402


def main() -> None:
    retriever = CaptionRetriever(
        qdrant_url=os.environ.get("QDRANT_URL", "http://localhost:6335"),
        qdrant_api_key=os.environ.get("QDRANT_API_KEY") or None,
    )
    mode = CaptionMode(retriever=retriever)
    print(f"Mode: {mode.name}\nGenerator: {mode.generator.model}\n")

    queries = [
        ("EN named entity",  "What was Dupixent Q1 2025 sales?",          ["3.5", "3,480", "20.3"]),
        ("EN abstract",      "Sanofi Q1 2025 business EPS growth",        ["1.79", "17", "eps"]),
        ("KR cross-lingual", "2025년 1분기 Beyfortus 백신 매출은?",         ["beyfortus", "vaccine", "284"]),
    ]

    rows = []
    for label, q, keywords in queries:
        print(f"\n=== {label} ===")
        print(f"Q: {q}")
        result = mode.answer(q, k=3)
        print(f"A: {result['answer']}")
        print(f"  hit pages: {[(h.get('source'), h.get('page')) for h in result['hits']]}")
        print(f"  scores:    {[round(h.get('score', 0), 3) for h in result['hits']]}")
        print(f"  usage:     {result['usage']}")
        ans_lc = result["answer"].lower()
        ok = any(kw.lower() in ans_lc for kw in keywords)
        rows.append((label, ok))
        print(f"  keyword hit: {'OK' if ok else 'FAIL'}  ({keywords})")

    print("\n" + "=" * 70)
    for label, ok in rows:
        print(f"  {label[:40]:40s}  {'OK' if ok else 'FAIL'}")
    print("=" * 70)

    try:
        from langfuse import get_client
        get_client().flush()
        print("\nLangfuse traces flushed.")
    except Exception:
        pass


if __name__ == "__main__":
    main()
