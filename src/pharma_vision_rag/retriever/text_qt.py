"""Query Translation (QT) retriever.

Pipeline:
    Korean query -> Claude Haiku translation -> English query -> BGE-M3 search.

Why: even though BGE-M3 is multilingual, translating to the document language
(English, for Sanofi's filings) typically improves dense retrieval precision
on numeric and named-entity phrases.
"""
from __future__ import annotations

from typing import Any

from pharma_vision_rag.generator import ClaudeTextGenerator
from pharma_vision_rag.retriever.docling_text import DoclingTextRetriever

QT_SYSTEM = """You translate Korean financial questions about Sanofi into English.

Rules:
- Preserve all named entities verbatim (Dupixent, Beyfortus, Sanofi, ALTUVIIIO, etc.).
- Preserve period labels verbatim (Q1 2025, FY2025, H1 2025).
- Preserve numeric units (€, %, billion, million).
- Translate only the natural-language portions.
- Output ONLY the English translation. No quotation marks, no preamble."""


class TextQTRetriever:
    """Translate Korean queries to English, then search."""

    variant_name = "qt"

    def __init__(self, base: DoclingTextRetriever, translator: ClaudeTextGenerator | None = None) -> None:
        self.base = base
        self.translator = translator or ClaudeTextGenerator()

    def translate(self, query: str) -> str:
        return self.translator.generate(system=QT_SYSTEM, user=query, max_tokens=128)["answer"]

    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        translated = self.translate(query)
        hits = self.base.search(translated, k=k)
        for h in hits:
            h["variant"] = self.variant_name
            h["original_query"] = query
            h["rewritten_query"] = translated
        return hits
