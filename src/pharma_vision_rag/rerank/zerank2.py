"""ZeRank2-style cross-encoder reranker.

The PRD calls this "ZeRank2" as a label; the underlying model is
``BAAI/bge-reranker-v2-m3`` — a multilingual cross-encoder (568M params,
MIT-licensed) that consistently scores in the top tier on multilingual
rerank benchmarks. Multilingual support is essential for our KR/EN
cross-lingual question set.

Cross-encoder pattern: takes (query, document) pairs and returns a
relevance score in a unified range that is comparable across all candidates,
regardless of which retriever surfaced them. This is what enables fair fusion
of BGE-M3 (text) and Nemotron (vision) candidate lists in Hybrid mode.
"""
from __future__ import annotations

import logging
from typing import Any

from sentence_transformers import CrossEncoder

log = logging.getLogger(__name__)

DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"


class ZeRank2Reranker:
    """Cross-encoder reranker over candidate dicts."""

    def __init__(self, model_id: str = DEFAULT_MODEL, max_length: int = 512) -> None:
        log.info("Loading reranker %s", model_id)
        self.model = CrossEncoder(model_id, max_length=max_length)
        self.model_id = model_id

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Score every (query, candidate.text) pair and return them re-sorted.

        ``top_k=None`` returns all candidates (re-sorted). ``top_k=N`` returns
        the N highest-scoring. Each returned candidate gets a ``rerank_score``
        field added in-place.
        """
        if not candidates:
            return []

        pairs = [[query, c.get("text", "")] for c in candidates]
        scores = self.model.predict(pairs, show_progress_bar=False)

        for c, s in zip(candidates, scores, strict=True):
            c["rerank_score"] = float(s)

        ranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
        return ranked if top_k is None else ranked[:top_k]
