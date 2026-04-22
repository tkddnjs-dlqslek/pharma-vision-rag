"""End-to-end RAG modes — wire a retriever to a generator and answer queries.

Four modes per the PRD: text_only, vision_only, caption, hybrid.
All share the surface ``mode.answer(query, k) -> dict``.
"""
from pharma_vision_rag.modes.text_only import TextOnlyMode

__all__ = ["TextOnlyMode"]
