"""HyDE (Hypothetical Document Embeddings) retriever.

Pipeline:
    query -> Claude Haiku generates a 1-2 sentence fake passage in
    Sanofi-report style -> embed that passage with BGE-M3 -> Qdrant search.

Why: the query lives in "question space"; documents live in "declarative
report space". Embedding a hypothetical report-style answer lands closer to
real passages (especially ones with concrete €/% numbers), which is exactly
the failure mode for the baseline on abstract queries like "EPS growth".
The fake numbers are fine — we only need *document-style embeddings*, not
factual correctness; retrieval is doc-to-doc similarity.
"""
from __future__ import annotations

from typing import Any

from pharma_vision_rag.generator import ClaudeTextGenerator
from pharma_vision_rag.retriever.docling_text import DoclingTextRetriever

HYDE_SYSTEM = """You draft short passages that *look like they came from a Sanofi financial report*, to be used for document-similarity search.

Given a question, write 1-2 sentences in the declarative style of Sanofi's Q1 2025 press release. Use:
- concrete currency units (€X billion, €Y million)
- percent changes (+Z% at CER, +Z% reported)
- period labels (Q1 2025, FY2025, H1 2025)
- product names as appropriate (Dupixent, Beyfortus, ALTUVIIIO)

Invent plausible-looking numbers if needed — factual accuracy does NOT matter, only the textual shape. Output ONLY the passage. No preamble, no quotation marks."""


class TextHyDERetriever:
    """Generate hypothetical answer passage, then search with its embedding."""

    variant_name = "hyde"

    def __init__(self, base: DoclingTextRetriever, generator: ClaudeTextGenerator | None = None) -> None:
        self.base = base
        self.generator = generator or ClaudeTextGenerator()

    def draft_passage(self, query: str) -> str:
        return self.generator.generate(system=HYDE_SYSTEM, user=query, max_tokens=160)["answer"]

    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        passage = self.draft_passage(query)
        hits = self.base.search(passage, k=k)
        for h in hits:
            h["variant"] = self.variant_name
            h["original_query"] = query
            h["rewritten_query"] = passage
        return hits
