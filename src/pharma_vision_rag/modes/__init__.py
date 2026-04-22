"""End-to-end RAG modes — wire a retriever to a generator and answer queries.

Four modes per the PRD: text_only, vision_only, caption, hybrid.
All share the surface ``mode.answer(query, k) -> dict``.
"""
from pharma_vision_rag.modes.caption import CaptionMode
from pharma_vision_rag.modes.hybrid import HybridMode, route, rrf_merge
from pharma_vision_rag.modes.text_only import TextOnlyMode
from pharma_vision_rag.modes.vision_only import VisionOnlyMode

__all__ = ["CaptionMode", "HybridMode", "TextOnlyMode", "VisionOnlyMode", "route", "rrf_merge"]
