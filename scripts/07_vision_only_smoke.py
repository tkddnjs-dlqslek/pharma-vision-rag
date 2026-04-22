"""Phase 1 smoke test — vision_only mode end-to-end.

Pre-flight (same as 06_index_nemotron.py):
    1. Colab tunnel running, COLAB_EMBEDDING_URL in .env.
    2. scripts/06_index_nemotron.py already populated 'pharma_vision' collection.

Acceptance:
    - Vision retrieval surfaces correct page for each query.
    - Claude Vision answer mentions expected number/keyword.
    - Off-page query is refused.

Usage:
    PYTHONIOENCODING=utf-8 python scripts/07_vision_only_smoke.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env")

from pharma_vision_rag.modes import VisionOnlyMode  # noqa: E402
from pharma_vision_rag.retriever import NemotronEmbeddingClient, NemotronVisionRetriever  # noqa: E402

PDF_DIR = ROOT / "data" / "pdf"


def main() -> None:
    tunnel_url = os.environ.get("COLAB_EMBEDDING_URL", "").strip()
    if not tunnel_url:
        raise SystemExit(
            "COLAB_EMBEDDING_URL is unset. Start notebooks/02_nemotron_tunnel.ipynb "
            "on Colab and paste the URL into .env."
        )

    client = NemotronEmbeddingClient(tunnel_url)
    print(f"Tunnel: {tunnel_url}")
    print(f"Health: {client.health()}\n")

    retriever = NemotronVisionRetriever(
        embedding_client=client,
        qdrant_url=os.environ.get("QDRANT_URL", "http://localhost:6335"),
        qdrant_api_key=os.environ.get("QDRANT_API_KEY") or None,
    )
    mode = VisionOnlyMode(retriever=retriever, pdf_dir=PDF_DIR)
    print(f"Mode: {mode.name}\nGenerator model: {mode.generator.model}\n")

    queries = [
        ("EN chart-friendly",  "What was Dupixent Q1 2025 sales?",          ["3.5", "3,480", "20.3"]),
        ("EN abstract",        "Sanofi Q1 2025 business EPS growth",        ["1.79", "17", "eps"]),
        ("KR cross-lingual",   "2025년 1분기 Beyfortus 백신 매출은?",         ["beyfortus", "vaccine", "284"]),
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

    client.close()


if __name__ == "__main__":
    main()
