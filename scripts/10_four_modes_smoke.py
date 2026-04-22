"""Phase 1 final smoke — all 4 modes × 3 queries = 12 Langfuse traces.

Pre-flight:
    1. Local Qdrant up (docker compose up -d qdrant).
    2. text retriever indexed: scripts/03_docling_text_smoke.py already ran.
    3. captions indexed: scripts/08_index_captions.py already ran.
    4. vision retriever indexed: scripts/06_index_nemotron.py already ran (needs Colab tunnel).
    5. .env has COLAB_EMBEDDING_URL set (still pointing at the live tunnel).

Acceptance:
    - All 12 (mode, query) combinations produce an answer.
    - Each mode's keyword-hit rate is recorded.
    - Hybrid graph (LangGraph) + Hybrid mode (plain) both run.
    - 12+ traces visible in Langfuse cloud dashboard.

Usage:
    PYTHONIOENCODING=utf-8 python scripts/10_four_modes_smoke.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env")

from pharma_vision_rag.modes import CaptionMode, HybridMode, TextOnlyMode, VisionOnlyMode  # noqa: E402
from pharma_vision_rag.retriever import (  # noqa: E402
    CaptionRetriever,
    DoclingTextRetriever,
    NemotronEmbeddingClient,
    NemotronVisionRetriever,
    TextBaselineRetriever,
)
from pharma_vision_rag.router import HybridGraph  # noqa: E402

PDF_DIR = ROOT / "data" / "pdf"

QUERIES = [
    ("EN named entity (table)", "What was Dupixent Q1 2025 sales?",          ["3.5", "3,480", "20.3"]),
    ("EN abstract (prose)",     "Sanofi Q1 2025 business EPS growth",        ["1.79", "17", "eps"]),
    ("KR cross-lingual",        "2025년 1분기 Beyfortus 백신 매출은?",         ["beyfortus", "vaccine", "284"]),
]


def _hit_kw(answer: str, keywords) -> bool:
    a = answer.lower()
    return any(kw.lower() in a for kw in keywords)


def main() -> None:
    qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6335")
    qdrant_api_key = os.environ.get("QDRANT_API_KEY") or None
    tunnel_url = os.environ.get("COLAB_EMBEDDING_URL", "").strip()
    if not tunnel_url:
        raise SystemExit(
            "COLAB_EMBEDDING_URL is unset. Start notebooks/02_nemotron_tunnel.ipynb "
            "and paste the URL into .env."
        )

    # ─── Wire shared resources ────────────────────────────────────
    docling = DoclingTextRetriever(qdrant_url=qdrant_url, qdrant_api_key=qdrant_api_key)
    text_baseline = TextBaselineRetriever(docling)

    caption_retriever = CaptionRetriever(qdrant_url=qdrant_url, qdrant_api_key=qdrant_api_key)

    nemotron_client = NemotronEmbeddingClient(tunnel_url)
    print(f"Nemotron tunnel health: {nemotron_client.health()}\n")
    nemotron = NemotronVisionRetriever(
        embedding_client=nemotron_client,
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
    )

    # ─── Instantiate all 4 modes (+ HybridGraph) ──────────────────
    modes = {
        "text_only":   TextOnlyMode(retriever=text_baseline),
        "vision_only": VisionOnlyMode(retriever=nemotron, pdf_dir=PDF_DIR),
        "caption":     CaptionMode(retriever=caption_retriever),
        "hybrid":      HybridMode(text_retriever=text_baseline, vision_retriever=nemotron, pdf_dir=PDF_DIR),
        "hybrid_lg":   HybridGraph(text_retriever=text_baseline, vision_retriever=nemotron, pdf_dir=PDF_DIR),
    }

    # ─── Run 5 modes × 3 queries ──────────────────────────────────
    results: dict[str, dict[str, bool]] = {m: {} for m in modes}
    for mode_name, mode in modes.items():
        print(f"\n{'=' * 70}\nMode: {mode_name}\n{'=' * 70}")
        for label, q, kws in QUERIES:
            print(f"\n  [{label}] {q}")
            try:
                r = mode.answer(q, k=3)
                ok = _hit_kw(r["answer"], kws)
                results[mode_name][label] = ok
                preview = r["answer"][:160].replace("\n", " ")
                print(f"    A: {preview}")
                print(f"    keyword hit: {'OK' if ok else 'FAIL'}  usage={r.get('usage', {})}")
            except Exception as e:  # noqa: BLE001
                results[mode_name][label] = False
                print(f"    ERROR: {type(e).__name__}: {e}")

    # ─── Summary table ────────────────────────────────────────────
    print("\n" + "=" * 90)
    print(f"{'Query':40s}  " + "  ".join(f"{m:>12s}" for m in modes))
    print("-" * 90)
    for label, _, _ in QUERIES:
        cells = "  ".join(f"{('OK' if results[m].get(label) else 'FAIL'):>12s}" for m in modes)
        print(f"{label[:40]:40s}  {cells}")
    print("=" * 90)

    try:
        from langfuse import get_client
        get_client().flush()
        print("\nLangfuse traces flushed → check https://us.cloud.langfuse.com")
    except Exception:
        pass

    nemotron_client.close()


if __name__ == "__main__":
    main()
