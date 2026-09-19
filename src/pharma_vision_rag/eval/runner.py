"""Retrieval benchmark runner (EXPERIMENT_PLAN §3 E1/E2, retrieval half; deterministic -> 1 run).

Runs every question in eval/questions.jsonl in KR and EN through each available retrieval variant,
scores page-level Recall@1/3/5 and NDCG@5 against gold_pages, writes one row per (variant, question, lang).

Variants (skipped with a message when their index/inputs are missing):
    text           BGE-M3 dense over Docling chunks
    text_rerank    text top-30 chunks -> bge-reranker-v2-m3
    vision         Nemotron multi-vector; query embeddings read offline from data/embeddings/*/queries
    caption        BGE-M3 over Haiku page captions
    hybrid         RRF(text, vision) with the keyword router weights from modes/hybrid.py
QT / HyDE (Haiku query rewriting) and the generation + judge stage need ANTHROPIC_API_KEY; added with --generate later.

Usage:
    PYTHONIOENCODING=utf-8 python -m pharma_vision_rag.eval.runner --mode all
    PYTHONIOENCODING=utf-8 python -m pharma_vision_rag.eval.runner --mode text,text_rerank
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
from dotenv import load_dotenv

from pharma_vision_rag.eval.metrics import first_gold_rank, gold_groups, ndcg_at_k, ranked_pages, recall_at_k

ROOT = Path(__file__).resolve().parents[3]
load_dotenv(ROOT / ".env")
QUESTIONS = ROOT / "eval" / "questions.jsonl"
RESULTS = ROOT / "eval" / "results"
EMB_DIR = ROOT / "data" / "embeddings"
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6335")
CHUNK_POOL = 30   # chunks fetched before collapsing to pages / reranking
PAGE_POOL = 8     # pages per retriever fed to RRF (matches HybridMode)

Search = Callable[[str, str], list[dict[str, Any]]]  # (question_id_lang, query_text) -> hits


class OfflineQueryClient:
    """Stands in for NemotronEmbeddingClient: query embeddings precomputed by scripts/embed_pages_gpu.py."""

    def __init__(self, root: Path) -> None:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        self._by_text = {q["text"]: root / "queries" / f"{q['id']}.npy" for q in manifest["queries"]}

    def embed_query(self, query: str) -> np.ndarray:
        return np.load(self._by_text[query]).astype(np.float32)


def _collection_count(name: str) -> int:
    from qdrant_client import QdrantClient
    c = QdrantClient(url=QDRANT_URL, check_compatibility=False)
    return c.count(name, exact=True).count if c.collection_exists(name) else 0


def build_variants(wanted: list[str]) -> dict[str, Search]:
    variants: dict[str, Search] = {}
    need_text = {"text", "text_rerank", "hybrid"} & set(wanted)
    need_vision = {"vision", "hybrid"} & set(wanted)
    text = vision = None

    if need_text:
        if _collection_count("pharma_text"):
            from pharma_vision_rag.retriever.docling_text import DoclingTextRetriever
            text = DoclingTextRetriever(qdrant_url=QDRANT_URL)
        else:
            print("SKIP text variants: pharma_text is empty (run scripts/14_index_text_all.py)")
    if need_vision:
        emb_roots = sorted(p.parent for p in EMB_DIR.glob("*/manifest.json"))
        if emb_roots and _collection_count("pharma_vision"):
            from pharma_vision_rag.retriever.nemotron import NemotronVisionRetriever
            vision = NemotronVisionRetriever(OfflineQueryClient(emb_roots[-1]), qdrant_url=QDRANT_URL)
        else:
            print("SKIP vision variants: no data/embeddings/*/manifest.json or pharma_vision empty (run scripts/13)")

    if text and "text" in wanted:
        variants["text"] = lambda q: text.search(q, k=CHUNK_POOL)
    if text and "text_rerank" in wanted:
        from pharma_vision_rag.rerank.zerank2 import ZeRank2Reranker
        rr = ZeRank2Reranker()
        variants["text_rerank"] = lambda q: rr.rerank(q, text.search(q, k=CHUNK_POOL), top_k=CHUNK_POOL)
    if vision and "vision" in wanted:
        variants["vision"] = lambda q: vision.search(q, k=PAGE_POOL)
    if "caption" in wanted:
        if _collection_count("pharma_caption"):
            from pharma_vision_rag.retriever.caption import CaptionRetriever
            cap = CaptionRetriever(qdrant_url=QDRANT_URL)
            variants["caption"] = lambda q: cap.search(q, k=PAGE_POOL)
        else:
            print("SKIP caption: pharma_caption is empty (run scripts/08_index_captions.py, needs ANTHROPIC_API_KEY)")
    if text and vision and "hybrid" in wanted:
        from pharma_vision_rag.modes.hybrid import route, rrf_merge

        def hybrid(q: str) -> list[dict[str, Any]]:
            w = route(q)
            t_pages = [{"source": s, "page": p} for s, p in ranked_pages(text.search(q, k=CHUNK_POOL))[:PAGE_POOL]]
            return rrf_merge(t_pages, vision.search(q, k=PAGE_POOL), w["w_text"], w["w_vision"])
        variants["hybrid"] = hybrid
    return variants


def run(variants: dict[str, Search]) -> list[dict[str, Any]]:
    qs = [json.loads(l) for l in QUESTIONS.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = []
    for name, search in variants.items():
        for q in qs:
            gold = gold_groups(q)
            for lang in ("ko", "en"):
                pages = ranked_pages(search(q[f"q_{lang}"]))
                rows.append({
                    "variant": name, "id": q["id"], "type": q["type"], "lang": lang, "period_spec": q["period_spec"],
                    "r@1": round(recall_at_k(pages, gold, 1), 4), "r@3": round(recall_at_k(pages, gold, 3), 4),
                    "r@5": round(recall_at_k(pages, gold, 5), 4), "ndcg@5": round(ndcg_at_k(pages, gold, 5), 4),
                    "first_gold_rank": first_gold_rank(pages, gold) or "",
                    "top5": " ".join(f"{s}:{p}" for s, p in pages[:5]),
                })
        print(f"{name}: {len(qs) * 2} queries done", flush=True)
    return rows


def summarize(rows: list[dict[str, Any]]) -> None:
    def table(title: str, key: Callable[[dict], str]) -> None:
        groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for r in rows:
            groups[(r["variant"], key(r))].append(r)
        print(f"\n{title}\n{'variant':<13}{'group':<8}{'n':>4}{'R@1':>7}{'R@3':>7}{'R@5':>7}{'NDCG@5':>8}")
        for (v, g), rs in sorted(groups.items()):
            m = {k: sum(float(r[k]) for r in rs) / len(rs) for k in ("r@1", "r@3", "r@5", "ndcg@5")}
            print(f"{v:<13}{g:<8}{len(rs):>4}{m['r@1']:>7.2f}{m['r@3']:>7.2f}{m['r@5']:>7.2f}{m['ndcg@5']:>8.2f}")
    table("overall", lambda r: "all")
    table("by language", lambda r: r["lang"])
    table("by question type", lambda r: r["type"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="all", help="all or comma list: text,text_rerank,vision,caption,hybrid")
    a = ap.parse_args()
    wanted = ["text", "text_rerank", "vision", "caption", "hybrid"] if a.mode == "all" else a.mode.split(",")
    rows = run(build_variants(wanted))
    if not rows:
        raise SystemExit("no variant available")
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "retrieval.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summarize(rows)
    print(f"\n{len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
