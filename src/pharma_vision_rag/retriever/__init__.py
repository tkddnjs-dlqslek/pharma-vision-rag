"""Retrievers — turn queries into ranked candidate chunks / pages.

Text-path variants (baseline / QT / HyDE) all wrap ``DoclingTextRetriever``
and share the surface ``search(query, k) -> list[hit]``.
"""
from pharma_vision_rag.retriever.caption import CaptionIndexer, CaptionRetriever
from pharma_vision_rag.retriever.docling_text import DoclingTextRetriever
from pharma_vision_rag.retriever.nemotron import NemotronEmbeddingClient, NemotronVisionRetriever
from pharma_vision_rag.retriever.text_baseline import TextBaselineRetriever
from pharma_vision_rag.retriever.text_hyde import TextHyDERetriever
from pharma_vision_rag.retriever.text_qt import TextQTRetriever

__all__ = [
    "CaptionIndexer",
    "CaptionRetriever",
    "DoclingTextRetriever",
    "NemotronEmbeddingClient",
    "NemotronVisionRetriever",
    "TextBaselineRetriever",
    "TextHyDERetriever",
    "TextQTRetriever",
]
