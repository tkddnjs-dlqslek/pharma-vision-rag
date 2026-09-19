"""Page-level retrieval metrics against gold_pages (EXPERIMENT_PLAN §6).

A "page" is a (source, page) tuple. Retrievers return chunks or pages; ``ranked_pages``
collapses them to a de-duplicated page ranking first (best-ranked chunk wins).

Scoring rules:
  - explicit questions: recall@k = fraction of gold pages in the top k (multi-hop D has several).
  - ambiguous questions: any one gold page counts as a full hit (recall 1.0, ideal DCG of one page).
"""
from __future__ import annotations

import math
from typing import Any

Page = tuple[str, int]


def ranked_pages(hits: list[dict[str, Any]]) -> list[Page]:
    seen: dict[Page, None] = {}
    for h in hits:
        if h.get("source") is None or h.get("page") is None:
            continue
        seen.setdefault((h["source"], int(h["page"])), None)
    return list(seen)


def recall_at_k(ranked: list[Page], gold: set[Page], k: int, ambiguous: bool = False) -> float:
    found = len(gold & set(ranked[:k]))
    if ambiguous:
        return 1.0 if found else 0.0
    return found / len(gold)


def ndcg_at_k(ranked: list[Page], gold: set[Page], k: int, ambiguous: bool = False) -> float:
    gains = [1.0 if p in gold else 0.0 for p in ranked[:k]]
    if ambiguous:  # only the first gold page earns gain
        first = next((i for i, g in enumerate(gains) if g), None)
        gains = [1.0 if i == first else 0.0 for i in range(len(gains))]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    n_ideal = 1 if ambiguous else min(len(gold), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(n_ideal))
    return dcg / idcg if idcg else 0.0


def first_gold_rank(ranked: list[Page], gold: set[Page]) -> int | None:
    return next((i for i, p in enumerate(ranked, start=1) if p in gold), None)


def _self_check() -> None:
    g = {("a.pdf", 1), ("a.pdf", 2)}
    r = [("x.pdf", 9), ("a.pdf", 2), ("a.pdf", 1)]
    assert ranked_pages([{"source": "a.pdf", "page": 1}, {"source": "a.pdf", "page": 1}, {"source": None, "page": 1}]) == [("a.pdf", 1)]
    assert recall_at_k(r, g, 1) == 0.0 and recall_at_k(r, g, 2) == 0.5 and recall_at_k(r, g, 3) == 1.0
    assert recall_at_k(r, g, 2, ambiguous=True) == 1.0
    assert ndcg_at_k([("a.pdf", 1), ("a.pdf", 2)], g, 5) == 1.0
    assert abs(ndcg_at_k(r, g, 5, ambiguous=True) - 1 / math.log2(3)) < 1e-9
    assert ndcg_at_k([("x.pdf", 9)], g, 5) == 0.0
    assert first_gold_rank(r, g) == 2 and first_gold_rank([("x.pdf", 9)], g) is None
    print("metrics self-check ok")


if __name__ == "__main__":
    _self_check()
