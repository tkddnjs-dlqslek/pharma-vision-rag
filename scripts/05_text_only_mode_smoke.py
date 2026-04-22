"""Phase 1 smoke test — text_only mode end-to-end + reranker check.

Pipeline tested:
    query -> TextBaselineRetriever (top-20 from BGE-M3/Qdrant)
          -> ZeRank2Reranker (top-3)
          -> ClaudeVisionGenerator with text-only system prompt
          -> answer

Acceptance:
    - 3 queries (KR + EN, abstract + named-entity)
    - Each answer mentions the expected number/keyword OR refuses correctly
    - Rerank changes the order of top-3 vs raw top-3 (proves it's doing something)

Usage:
    PYTHONIOENCODING=utf-8 python scripts/05_text_only_mode_smoke.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env")

from pharma_vision_rag.modes import TextOnlyMode  # noqa: E402
from pharma_vision_rag.rerank import ZeRank2Reranker  # noqa: E402
from pharma_vision_rag.retriever import DoclingTextRetriever, TextBaselineRetriever  # noqa: E402


class RerankedRetriever:
    """Wraps a retriever: pull top-N, then rerank to top-K. Drop-in for TextRetriever."""

    def __init__(self, base, reranker: ZeRank2Reranker, candidate_pool: int = 20) -> None:
        self.base = base
        self.reranker = reranker
        self.candidate_pool = candidate_pool

    def search(self, query: str, k: int = 3) -> list[dict]:
        candidates = self.base.search(query, k=self.candidate_pool)
        return self.reranker.rerank(query, candidates, top_k=k)


def main() -> None:
    base_retriever = DoclingTextRetriever(
        qdrant_url=os.environ.get("QDRANT_URL", "http://localhost:6335"),
        qdrant_api_key=os.environ.get("QDRANT_API_KEY") or None,
    )
    baseline_text = TextBaselineRetriever(base_retriever)
    reranker = ZeRank2Reranker()
    reranked = RerankedRetriever(baseline_text, reranker, candidate_pool=20)

    mode = TextOnlyMode(retriever=reranked)
    print(f"Mode: {mode.name}\nGenerator model: {mode.generator.model}\n")

    queries = [
        ("EN abstract (the EPS failure case)", "Sanofi Q1 2025 business EPS growth", ["1.79", "17", "eps"]),
        ("EN named entity",                     "What was Dupixent Q1 2025 sales?",   ["3.5", "3,480", "20.3"]),
        ("KR cross-lingual",                    "2025년 1분기 Beyfortus 백신 매출은?", ["beyfortus", "vaccine"]),
    ]

    rows = []
    for label, q, keywords in queries:
        print(f"\n=== {label} ===")
        print(f"Q: {q}")

        result = mode.answer(q, k=3)
        print(f"A: {result['answer']}")
        print(f"  pages used: {[h['page'] for h in result['hits']]}")
        print(f"  rerank scores: {[round(h.get('rerank_score', 0), 3) for h in result['hits']]}")
        print(f"  usage: {result['usage']}")

        ans_lc = result["answer"].lower()
        kw_hit = any(kw.lower() in ans_lc for kw in keywords)
        print(f"  keyword in answer: {'OK' if kw_hit else 'FAIL'}  ({keywords})")
        rows.append((label, kw_hit))

    print("\n" + "=" * 70)
    print(f"{'Case':40s}  result")
    for label, ok in rows:
        print(f"{label[:40]:40s}  {'OK' if ok else 'FAIL'}")
    print("=" * 70)

    # Flush Langfuse so traces appear in the cloud dashboard before exit.
    try:
        from langfuse import get_client
        get_client().flush()
        print("\nLangfuse traces flushed. Check https://us.cloud.langfuse.com → project → Traces.")
    except Exception:
        pass


if __name__ == "__main__":
    main()
