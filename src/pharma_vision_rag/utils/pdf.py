"""PDF rendering helpers using pypdfium2 (Windows-friendly, no Poppler)."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pypdfium2 as pdfium
from PIL import Image


def page_count(pdf_path: str | Path) -> int:
    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        return len(pdf)
    finally:
        pdf.close()


def render_page(pdf_path: str | Path, page_number: int, scale: float = 1.5) -> Image.Image:
    """Render one page (1-indexed) to a PIL Image.

    ``scale=1.5`` ≈ 150 DPI (1.0 is 72 DPI). Claude Vision does well at 1.5 without
    inflating token counts unnecessarily.
    """
    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        return pdf[page_number - 1].render(scale=scale).to_pil()
    finally:
        pdf.close()


def iter_pages(pdf_path: str | Path, scale: float = 1.5) -> Iterator[tuple[int, Image.Image]]:
    """Yield ``(page_number_1indexed, PIL.Image)`` pairs for every page."""
    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        for i, page in enumerate(pdf, start=1):
            yield i, page.render(scale=scale).to_pil()
    finally:
        pdf.close()
