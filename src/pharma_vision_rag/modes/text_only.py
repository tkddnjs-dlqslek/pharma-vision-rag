"""Text-only mode — retriever returns text/table chunks, Sonnet answers from text.

This is the cheapest of the 4 modes. No image tokens. Strong on prose questions
and structured tables (Docling exports tables to markdown), weak on chart-only
information that BGE-M3 / Sonnet can't read from text alone.

Composes any ``TextRetriever`` (baseline / QT / HyDE) with ``ClaudeVisionGenerator``
in text-only mode (no images sent).
"""
from __future__ import annotations

from typing import Any, Protocol

from pharma_vision_rag.generator import ClaudeVisionGenerator

# Optional Langfuse integration. If env vars LANGFUSE_PUBLIC_KEY/SECRET_KEY/BASE_URL
# are missing, get_client() returns a no-op stub and @observe just calls the function.
try:
    from langfuse import get_client, observe  # type: ignore
except ImportError:
    def observe(*args, **kwargs):  # type: ignore
        # Bare decorator: @observe
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        # Parametrized: @observe(name=...)
        def deco(fn):
            return fn
        return deco

    def get_client():  # type: ignore
        return None

TEXT_ONLY_SYSTEM_PROMPT = """You are a pharma analyst answering questions about Sanofi's public disclosures (Form 20-F, quarterly press releases, clinical trial summaries).

You will receive a question and a set of retrieved text passages. Each passage is labeled with its source page and block type (text or table).

Rules:
- Answer ONLY from the provided passages. If the answer is not in them, reply "정보를 찾을 수 없습니다." (Korean question) or "Not found in the provided passages." (English question).
- Quote numbers verbatim with currency/units (€3.5 billion, 20.3%) and period labels (FY2025, Q1 2025, H1 2025).
- Prefer the more specific passage when multiple cover the same metric at different granularity.
- Keep the answer to 2-3 sentences. No preamble ("Based on the provided...", "According to...")."""


class TextRetriever(Protocol):
    """Duck-typed: anything with ``search(query, k) -> list[dict]`` qualifies."""

    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]: ...


def format_context(hits: list[dict[str, Any]]) -> str:
    """Render retrieved chunks as a labeled passage list for the LLM."""
    parts = []
    for i, h in enumerate(hits, start=1):
        page = h.get("page", "?")
        block_type = h.get("block_type", "?")
        source = h.get("source", "?")
        parts.append(f"[Passage {i} — {source} p{page} {block_type}]\n{h['text']}")
    return "\n\n".join(parts)


class TextOnlyMode:
    """Retriever -> formatted text context -> Claude Sonnet -> answer."""

    name = "text_only"

    def __init__(
        self,
        retriever: TextRetriever,
        generator: ClaudeVisionGenerator | None = None,
    ) -> None:
        self.retriever = retriever
        self.generator = generator or ClaudeVisionGenerator(system_prompt=TEXT_ONLY_SYSTEM_PROMPT)

    @observe(name="mode.text_only.answer")
    def answer(self, query: str, k: int = 5) -> dict[str, Any]:
        hits = self.retriever.search(query, k=k)
        context = format_context(hits)
        user_msg = f"Question: {query}\n\nPassages:\n{context}"
        result = self.generator.generate(query=user_msg, images=None)

        # Annotate the Langfuse trace with structured metadata so the dashboard
        # can group/filter by mode, model, retriever variant, etc.
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
                        "retriever_variant": getattr(self.retriever, "variant_name", "unknown"),
                        "stop_reason": result["stop_reason"],
                        "usage": result["usage"],
                        "n_hits": len(hits),
                        "hit_pages": [h.get("page") for h in hits],
                    },
                )
            except Exception:  # noqa: BLE001
                # Never let observability break inference.
                pass

        return {
            "mode": self.name,
            "query": query,
            "answer": result["answer"],
            "hits": hits,
            "usage": result["usage"],
            "stop_reason": result["stop_reason"],
        }
