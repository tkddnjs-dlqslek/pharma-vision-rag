"""Page-level retrieval metrics against gold page *groups* (EXPERIMENT_PLAN §6).

A "page" is a (source, page) tuple. Retrievers return chunks or pages; ``ranked_pages`` collapses them
to a de-duplicated page ranking first (best-ranked chunk wins).

Gold is a list of groups. Every page in a group is an equally valid place to find that piece of the
answer (the same figure printed in an appendix table, the narrative and a slide), so one hit satisfies
the group. Single-hop and ambiguous-period questions have one group; multi-hop (D) has one group per hop.
Counting alternates as separate targets would punish a retriever for finding the answer only once.

    recall@k = satisfied groups / groups
    ndcg@k   = binary gain for the first page that satisfies each group, ideal = min(groups, k) hits up front
"""
from __future__ import annotations

import math
from typing import Any

Page = tuple[str, int]
Groups = list[set[Page]]


def ranked_pages(hits: list[dict[str, Any]]) -> list[Page]:
    seen: dict[Page, None] = {}
    for h in hits:
        if h.get("source") is None or h.get("page") is None:
            continue
        seen.setdefault((h["source"], int(h["page"])), None)
    return list(seen)


def gold_groups(question: dict[str, Any]) -> Groups:
    raw = question.get("gold_groups") or [question["gold_pages"]]
    return [{(g["source"], int(g["page"])) for g in group} for group in raw]


def _gains(ranked: list[Page], groups: Groups, k: int) -> list[float]:
    open_groups = list(groups)
    gains = []
    for page in ranked[:k]:
        hit = next((g for g in open_groups if page in g), None)
        if hit is not None:
            open_groups.remove(hit)
        gains.append(1.0 if hit is not None else 0.0)
    return gains


def recall_at_k(ranked: list[Page], groups: Groups, k: int) -> float:
    return sum(_gains(ranked, groups, k)) / len(groups)


def ndcg_at_k(ranked: list[Page], groups: Groups, k: int) -> float:
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(_gains(ranked, groups, k)))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(groups), k)))
    return dcg / idcg


def first_gold_rank(ranked: list[Page], groups: Groups) -> int | None:
    allowed = set().union(*groups)
    return next((i for i, p in enumerate(ranked, start=1) if p in allowed), None)


def _self_check() -> None:
    a1, a2, b1, x = ("a.pdf", 1), ("a.pdf", 2), ("b.pdf", 1), ("x.pdf", 9)
    assert ranked_pages([{"source": "a.pdf", "page": 1}, {"source": "a.pdf", "page": 1}, {"source": None, "page": 1}]) == [a1]
    one = [{a1, a2}]                      # two alternates for a single answer
    assert recall_at_k([x, a2, a1], one, 1) == 0.0 and recall_at_k([x, a2, a1], one, 2) == 1.0
    assert ndcg_at_k([a2, a1], one, 5) == 1.0, "second alternate must not earn extra gain or lower the ideal"
    assert abs(ndcg_at_k([x, a1], one, 5) - 1 / math.log2(3)) < 1e-9
    hops = [{a1, a2}, {b1}]               # multi-hop: both groups needed
    assert recall_at_k([a1, a2, x], hops, 3) == 0.5 and recall_at_k([a1, b1], hops, 2) == 1.0
    assert ndcg_at_k([a1, b1], hops, 5) == 1.0 and ndcg_at_k([x], hops, 5) == 0.0
    assert first_gold_rank([x, b1], hops) == 2 and first_gold_rank([x], hops) is None
    assert gold_groups({"gold_pages": [{"source": "a.pdf", "page": 1}]}) == [{a1}]
    assert gold_groups({"gold_pages": [], "gold_groups": [[{"source": "b.pdf", "page": "1"}]]}) == [{b1}]
    print("metrics self-check ok")


if __name__ == "__main__":
    _self_check()
