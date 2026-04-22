"""Baseline text retriever — raw query straight to BGE-M3 + Qdrant.

Thin wrapper over DoclingTextRetriever, exists so all 3 variants
(baseline / QT / HyDE) share the same surface:

    variant.search(query, k) -> list[hit]

Each hit is annotated with ``variant`` and ``rewritten_query`` so downstream
analysis can distinguish results.
"""
from __future__ import annotations

from typing import Any

from pharma_vision_rag.retriever.docling_text import DoclingTextRetriever


class TextBaselineRetriever:
    """No query transformation — pass the query through as-is."""

    variant_name = "baseline"

    def __init__(self, base: DoclingTextRetriever) -> None:
        self.base = base

    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        hits = self.base.search(query, k=k)
        for h in hits:
            h["variant"] = self.variant_name
            h["original_query"] = query
            h["rewritten_query"] = query
        return hits
