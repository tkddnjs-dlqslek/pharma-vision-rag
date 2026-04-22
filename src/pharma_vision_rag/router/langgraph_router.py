"""HybridGraph — LangGraph implementation of the hybrid pipeline.

Graph (parallel fan-out + join):

                 ┌─► retrieve_text ──┐
    route ──────►│                    ├──► merge ──► render ──► answer
                 └─► retrieve_vision ─┘

Each node sets a slice of the shared state. Parallel fan-out is handled by
``add_edge`` from a single source to multiple targets and by LangGraph's
default reducer (last-write-wins per field; we use disjoint fields so it's safe).

Functionally identical to ``modes.hybrid.HybridMode``, but the explicit nodes
show up in Langfuse as nested spans, which makes the agentic flow legible in
the dashboard and gives a portfolio talking point.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from pharma_vision_rag.generator import ClaudeVisionGenerator
from pharma_vision_rag.modes.hybrid import HYBRID_SYSTEM_PROMPT, route, rrf_merge
from pharma_vision_rag.retriever.nemotron import NemotronVisionRetriever
from pharma_vision_rag.utils.pdf import render_page

try:
    from langfuse import get_client  # type: ignore
except ImportError:
    def get_client():  # type: ignore
        return None


class HybridState(TypedDict, total=False):
    query: str
    k: int
    n_per_retriever: int
    routing: dict[str, Any]
    text_hits: list[dict[str, Any]]
    vision_hits: list[dict[str, Any]]
    fused: list[dict[str, Any]]
    images: list[Any]
    answer: str
    usage: dict[str, int]
    stop_reason: str


class HybridGraph:
    """LangGraph wrapper around the hybrid pipeline."""

    name = "hybrid_graph"

    def __init__(
        self,
        text_retriever,
        vision_retriever: NemotronVisionRetriever,
        pdf_dir: str | Path,
        generator: ClaudeVisionGenerator | None = None,
    ) -> None:
        self.text_retriever = text_retriever
        self.vision_retriever = vision_retriever
        self.pdf_dir = Path(pdf_dir)
        self.generator = generator or ClaudeVisionGenerator(system_prompt=HYBRID_SYSTEM_PROMPT)
        self.graph = self._build()

    def _build(self):
        g = StateGraph(HybridState)
        g.add_node("route", self._route)
        g.add_node("retrieve_text", self._retrieve_text)
        g.add_node("retrieve_vision", self._retrieve_vision)
        g.add_node("merge", self._merge)
        g.add_node("render", self._render)
        g.add_node("answer", self._answer)

        g.add_edge(START, "route")
        g.add_edge("route", "retrieve_text")
        g.add_edge("route", "retrieve_vision")
        g.add_edge("retrieve_text", "merge")
        g.add_edge("retrieve_vision", "merge")
        g.add_edge("merge", "render")
        g.add_edge("render", "answer")
        g.add_edge("answer", END)
        return g.compile()

    # ─── Node implementations ─────────────────────────────────────────

    def _route(self, state: HybridState) -> dict[str, Any]:
        return {"routing": route(state["query"])}

    def _retrieve_text(self, state: HybridState) -> dict[str, Any]:
        return {"text_hits": self.text_retriever.search(state["query"], k=state.get("n_per_retriever", 8))}

    def _retrieve_vision(self, state: HybridState) -> dict[str, Any]:
        return {"vision_hits": self.vision_retriever.search(state["query"], k=state.get("n_per_retriever", 8))}

    def _merge(self, state: HybridState) -> dict[str, Any]:
        routing = state["routing"]
        fused = rrf_merge(
            text_hits=state.get("text_hits", []),
            vision_hits=state.get("vision_hits", []),
            w_text=routing["w_text"],
            w_vision=routing["w_vision"],
        )[: state.get("k", 3)]
        return {"fused": fused}

    def _render(self, state: HybridState) -> dict[str, Any]:
        images = []
        for h in state["fused"]:
            pdf_path = self.pdf_dir / h["source"]
            images.append(render_page(pdf_path, page_number=h["page"], scale=1.5))
        return {"images": images}

    def _answer(self, state: HybridState) -> dict[str, Any]:
        result = self.generator.generate(query=state["query"], images=state["images"])
        return {
            "answer": result["answer"],
            "usage": result["usage"],
            "stop_reason": result["stop_reason"],
        }

    # ─── Public API (mirrors HybridMode.answer) ───────────────────────

    def answer(self, query: str, k: int = 3, n_per_retriever: int = 8) -> dict[str, Any]:
        final_state: HybridState = self.graph.invoke({  # type: ignore
            "query": query,
            "k": k,
            "n_per_retriever": n_per_retriever,
        })

        client = get_client()
        if client is not None:
            try:
                client.update_current_trace(
                    name=f"mode.{self.name}",
                    input={"query": query, "k": k},
                    output={"answer": final_state["answer"]},
                    metadata={
                        "mode": self.name,
                        "model": self.generator.model,
                        "stop_reason": final_state.get("stop_reason"),
                        "usage": final_state.get("usage"),
                        "routing": final_state.get("routing"),
                        "fused_pages": [(h.get("source"), h.get("page"), h.get("sources")) for h in final_state.get("fused", [])],
                    },
                )
            except Exception:  # noqa: BLE001
                pass

        return {
            "mode": self.name,
            "query": query,
            "answer": final_state["answer"],
            "hits": final_state.get("fused", []),
            "routing": final_state.get("routing"),
            "text_hits": final_state.get("text_hits", []),
            "vision_hits": final_state.get("vision_hits", []),
            "usage": final_state.get("usage", {}),
            "stop_reason": final_state.get("stop_reason"),
        }
