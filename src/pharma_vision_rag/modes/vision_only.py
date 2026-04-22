"""Vision-only mode — Nemotron retrieves pages, Claude Vision answers from images.

Pipeline:
    query
      -> NemotronVisionRetriever.search(query, k)  [MaxSim over Qdrant multi-vector]
      -> render top-k pages back to PIL images (from local PDF files)
      -> ClaudeVisionGenerator with images
      -> answer

Strong on chart/table-only information. Weaker on prose questions where the
text retriever's structured chunks would be more efficient.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from pharma_vision_rag.generator import ClaudeVisionGenerator
from pharma_vision_rag.retriever.nemotron import NemotronVisionRetriever
from pharma_vision_rag.utils.pdf import render_page

# Optional Langfuse integration (silent fallback if not installed).
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


VISION_ONLY_SYSTEM_PROMPT = """You are a pharma analyst answering questions about Sanofi's public disclosures (Form 20-F, quarterly press releases, clinical trial summaries).

You will receive a question and several full-page images from those documents (charts, tables, narrative prose).

Rules:
- Answer ONLY from the provided pages. If the answer is not visible on any page, reply "정보를 찾을 수 없습니다." (Korean) or "Not found in the provided pages." (English).
- Quote numbers verbatim with currency/units (€3.5 billion, 20.3%) and period labels (FY2025, Q1 2025).
- Prefer the more specific page when multiple show the same metric at different granularity.
- Keep the answer to 2-3 sentences. No preamble."""


class VisionOnlyMode:
    """Nemotron retrieval -> Claude Vision generation."""

    name = "vision_only"

    def __init__(
        self,
        retriever: NemotronVisionRetriever,
        pdf_dir: str | Path,
        generator: ClaudeVisionGenerator | None = None,
        render_scale: float = 1.5,
    ) -> None:
        self.retriever = retriever
        self.pdf_dir = Path(pdf_dir)
        self.generator = generator or ClaudeVisionGenerator(system_prompt=VISION_ONLY_SYSTEM_PROMPT)
        self.render_scale = render_scale

    def _hits_to_images(self, hits: list[dict[str, Any]]):
        """Render each hit's page back to a PIL image using local PDF files."""
        images = []
        for h in hits:
            pdf_path = self.pdf_dir / h["source"]
            images.append(render_page(pdf_path, page_number=h["page"], scale=self.render_scale))
        return images

    @observe(name="mode.vision_only.answer")
    def answer(self, query: str, k: int = 3) -> dict[str, Any]:
        hits = self.retriever.search(query, k=k)
        images = self._hits_to_images(hits)
        result = self.generator.generate(query=query, images=images)

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
                        "hit_scores": [round(h.get("score", 0), 3) for h in hits],
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
