"""Retrievers — turn queries into ranked candidate chunks / pages.

Text-path variants (baseline / QT / HyDE) all wrap ``DoclingTextRetriever``
and share the surface ``search(query, k) -> list[hit]``.

Names are imported lazily (PEP 562): the text retrievers pull in docling, torch and sentence-transformers,
which light submodules (``bm25``, ``vision_local``) must not pay for.
"""
from importlib import import_module

_WHERE = {
    "CaptionIndexer": "caption",
    "CaptionRetriever": "caption",
    "DoclingTextRetriever": "docling_text",
    "NemotronEmbeddingClient": "nemotron",
    "NemotronVisionRetriever": "nemotron",
    "TextBaselineRetriever": "text_baseline",
    "TextHyDERetriever": "text_hyde",
    "TextQTRetriever": "text_qt",
    "LocalVisionIndex": "vision_local",
}

__all__ = list(_WHERE)


def __getattr__(name: str):
    if name not in _WHERE:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(f"{__name__}.{_WHERE[name]}"), name)
