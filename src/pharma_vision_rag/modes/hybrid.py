"""Hybrid mode — fuse text + vision retrievers, answer with full page images.

Pipeline:
    query
      -> route(query): heuristic weights {w_text, w_vision} based on intent
      -> parallel: text_retriever.search(top_n_text) + vision_retriever.search(top_n_vision)
      -> RRF (reciprocal rank fusion) merge into a single ranked page list
      -> render top-k unique pages back to images
      -> ClaudeVisionGenerator with images
      -> answer

Why fuse: text retrieval is precise on prose / structured tables; vision is
strong on charts and visually-encoded numbers. RRF doesn't require the two
retrievers' scores to be on the same scale (which they aren't — BGE-M3 cosine
vs Nemotron MaxSim).
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Protocol

from pharma_vision_rag.generator import ClaudeVisionGenerator
from pharma_vision_rag.retriever.nemotron import NemotronVisionRetriever
from pharma_vision_rag.utils.pdf import render_page

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


HYBRID_SYSTEM_PROMPT = """You are a pharma analyst answering questions about Sanofi's public disclosures.

You receive a question and a set of full-page images surfaced by a hybrid retriever (text + vision). Pages may include any mix of charts, tables, and prose.

Rules:
- Answer ONLY from the provided pages. If the answer is not visible, reply "정보를 찾을 수 없습니다." (Korean) or "Not found in the provided pages." (English).
- Quote numbers verbatim with currency/units (€3.5 billion, 20.3%) and period labels (FY2025, Q1 2025).
- Prefer the more specific page when multiple show the same metric at different granularity.
- Keep the answer to 2-3 sentences. No preamble."""


# ─── Router ─────────────────────────────────────────────────────────────


# Keywords that hint a question is dominantly about charts / tables / numbers.
# Used to bias the candidate pool size, not to short-circuit either retriever.
CHART_KEYWORDS = (
    "chart", "graph", "trend", "growth", "rate", "yoy", "qoq", "%",
    "차트", "그래프", "추이", "성장률", "증가율", "변화율",
)
TABLE_KEYWORDS = (
    "table", "breakdown", "by region", "by indication", "by segment",
    "표", "지역별", "부문별",
)
PROSE_KEYWORDS = (
    "explain", "describe", "mechanism", "indication", "definition", "summary",
    "설명", "정의", "요약", "메커니즘",
)


def route(query: str) -> dict[str, Any]:
    """Heuristic router: returns {w_text, w_vision, rationale}."""
    q = query.lower()
    chart = any(k in q for k in CHART_KEYWORDS)
    table = any(k in q for k in TABLE_KEYWORDS)
    prose = any(k in q for k in PROSE_KEYWORDS)

    if chart and not prose:
        return {"w_text": 0.3, "w_vision": 0.7, "rationale": "chart-leaning"}
    if table and not prose:
        return {"w_text": 0.5, "w_vision": 0.5, "rationale": "table — both useful"}
    if prose and not (chart or table):
        return {"w_text": 0.7, "w_vision": 0.3, "rationale": "prose-leaning"}
    return {"w_text": 0.5, "w_vision": 0.5, "rationale": "balanced (default)"}


# ─── Reciprocal Rank Fusion ────────────────────────────────────────────


def rrf_merge(
    text_hits: list[dict[str, Any]],
    vision_hits: list[dict[str, Any]],
    w_text: float = 0.5,
    w_vision: float = 0.5,
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    """Combine two ranked lists by reciprocal rank fusion (RRF).

    A page that appears in both lists is rewarded twice. Ties broken by the
    higher fused score. Each result carries ``fused_score`` and ``sources``
    (which retrievers contributed)."""
    scores: dict[tuple[Any, Any], float] = defaultdict(float)
    sources: dict[tuple[Any, Any], list[str]] = defaultdict(list)
    payloads: dict[tuple[Any, Any], dict[str, Any]] = {}

    for rank, h in enumerate(text_hits, start=1):
        key = (h.get("source"), h.get("page"))
        if key == (None, None):
            continue
        scores[key] += w_text / (rrf_k + rank)
        sources[key].append("text")
        payloads.setdefault(key, dict(h))

    for rank, h in enumerate(vision_hits, start=1):
        key = (h.get("source"), h.get("page"))
        if key == (None, None):
            continue
        scores[key] += w_vision / (rrf_k + rank)
        sources[key].append("vision")
        if key not in payloads:
            payloads[key] = dict(h)

    fused = []
    for key in sorted(scores, key=lambda k: scores[k], reverse=True):
        item = payloads[key]
        item["fused_score"] = scores[key]
        item["sources"] = sources[key]
        fused.append(item)
    return fused


# ─── Mode ──────────────────────────────────────────────────────────────


class TextRetrieverProto(Protocol):
    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]: ...


class HybridMode:
    """Text retrieval + vision retrieval -> RRF -> page images -> Claude Vision."""

    name = "hybrid"

    def __init__(
        self,
        text_retriever: TextRetrieverProto,
        vision_retriever: NemotronVisionRetriever,
        pdf_dir: str | Path,
        generator: ClaudeVisionGenerator | None = None,
        n_per_retriever: int = 8,
        render_scale: float = 1.5,
    ) -> None:
        self.text_retriever = text_retriever
        self.vision_retriever = vision_retriever
        self.pdf_dir = Path(pdf_dir)
        self.generator = generator or ClaudeVisionGenerator(system_prompt=HYBRID_SYSTEM_PROMPT)
        self.n_per_retriever = n_per_retriever
        self.render_scale = render_scale

    def _hits_to_images(self, fused: list[dict[str, Any]]):
        images = []
        for h in fused:
            pdf_path = self.pdf_dir / h["source"]
            images.append(render_page(pdf_path, page_number=h["page"], scale=self.render_scale))
        return images

    @observe(name="mode.hybrid.answer")
    def answer(self, query: str, k: int = 3) -> dict[str, Any]:
        routing = route(query)

        text_hits = self.text_retriever.search(query, k=self.n_per_retriever)
        vision_hits = self.vision_retriever.search(query, k=self.n_per_retriever)

        fused = rrf_merge(
            text_hits=text_hits,
            vision_hits=vision_hits,
            w_text=routing["w_text"],
            w_vision=routing["w_vision"],
        )[:k]

        images = self._hits_to_images(fused)
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
                        "routing": routing,
                        "n_text_candidates": len(text_hits),
                        "n_vision_candidates": len(vision_hits),
                        "fused_pages": [(h.get("source"), h.get("page"), h.get("sources"), round(h.get("fused_score", 0), 4)) for h in fused],
                    },
                )
            except Exception:  # noqa: BLE001
                pass

        return {
            "mode": self.name,
            "query": query,
            "answer": result["answer"],
            "hits": fused,
            "routing": routing,
            "text_hits": text_hits,
            "vision_hits": vision_hits,
            "usage": result["usage"],
            "stop_reason": result["stop_reason"],
        }
