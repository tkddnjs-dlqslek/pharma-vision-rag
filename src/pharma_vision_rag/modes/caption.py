"""Caption mode — Haiku captions retrieved by BGE-M3, Sonnet answers from captions.

Same generation surface as text_only (text-only context to Sonnet) but the
retrieved "passages" are synthetic captions, not Docling-extracted text/tables.
This isolates the "VLM caption" baseline from structured PDF parsing.

If the caption was thorough, this mode should answer most questions cheaply.
Where the caption omitted a number that the chart had, we lose information —
that gap vs vision_only mode is the Phase 3 ablation point.
"""
from __future__ import annotations

from typing import Any

from pharma_vision_rag.generator import ClaudeVisionGenerator
from pharma_vision_rag.modes.text_only import format_context
from pharma_vision_rag.retriever.caption import CaptionRetriever

try:
    from langfuse import get_client, observe  # type: ignore
except ImportError:
    def observe(*args, **kwargs):  # type: ignore
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        def deco(fn):
            return fn
        return deco

    def get_client():  # type: ignore
        return None


CAPTION_MODE_SYSTEM_PROMPT = """You are a pharma analyst answering questions about Sanofi's public disclosures.

You receive captions of pages from Sanofi's reports. Each caption was machine-generated to summarize that page's contents (numbers, products, periods, table rows). The captions are your only source.

Rules:
- Answer ONLY from the captions. If the answer is not in them, reply "정보를 찾을 수 없습니다." (Korean) or "Not found in the provided captions." (English).
- Quote numbers verbatim with currency/units (€3.5 billion, 20.3%) and period labels (FY2025, Q1 2025).
- Note: captions may have missed details visible only in charts. If a caption mentions a topic but lacks the specific number, say so explicitly rather than guessing.
- Keep the answer to 2-3 sentences. No preamble."""


class CaptionMode:
    """Caption retrieval -> Sonnet answers from caption text."""

    name = "caption"

    def __init__(
        self,
        retriever: CaptionRetriever,
        generator: ClaudeVisionGenerator | None = None,
    ) -> None:
        self.retriever = retriever
        self.generator = generator or ClaudeVisionGenerator(system_prompt=CAPTION_MODE_SYSTEM_PROMPT)

    @observe(name="mode.caption.answer")
    def answer(self, query: str, k: int = 5) -> dict[str, Any]:
        hits = self.retriever.search(query, k=k)
        context = format_context(hits)
        user_msg = f"Question: {query}\n\nCaptions:\n{context}"
        result = self.generator.generate(query=user_msg, images=None)

        client = get_client()
        if client is not None:
            try:
                client.update_current_trace(
                    name=f"mode.{self.name}",
                    input={"query": query, "k": k},
                    output={"answer": result["answer"]},
                    metadata={
                        "mode": self.name,
                        "model": self.generator.model,
                        "stop_reason": result["stop_reason"],
                        "usage": result["usage"],
                        "n_hits": len(hits),
                        "hit_pages": [(h.get("source"), h.get("page")) for h in hits],
                    },
                )
            except Exception:  # noqa: BLE001
                pass

        return {
            "mode": self.name,
            "query": query,
            "answer": result["answer"],
            "hits": hits,
            "usage": result["usage"],
            "stop_reason": result["stop_reason"],
        }
