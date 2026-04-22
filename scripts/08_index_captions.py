"""One-time indexing — Haiku captions every page, BGE-M3 embeds, Qdrant stores.

Cost estimate: Haiku $1/$5 per 1M tokens. Per page ~ image-input ~1500 + caption-output ~250 = ~$0.0028.
    Q1.pdf (23 pages)            ~ $0.07
    All 4 PDFs (~110 pages)      ~ $0.31

Usage:
    PYTHONIOENCODING=utf-8 python scripts/08_index_captions.py [pdf_name ...]
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env")

from pharma_vision_rag.retriever import CaptionIndexer  # noqa: E402

PDF_DIR = ROOT / "data" / "pdf"
DEFAULT_PDFS = ["Q1.pdf"]


def main(argv: list[str]) -> None:
    pdfs = argv[1:] if len(argv) > 1 else DEFAULT_PDFS

    indexer = CaptionIndexer(
        qdrant_url=os.environ.get("QDRANT_URL", "http://localhost:6335"),
        qdrant_api_key=os.environ.get("QDRANT_API_KEY") or None,
    )

    grand_in = grand_out = 0
    for name in pdfs:
        path = PDF_DIR / name
        if not path.exists():
            print(f"  SKIP missing: {path}")
            continue
        print(f"\n=== Captioning {name} ===")
        stats = indexer.index_pdf(path, source=name)
        print(
            f"  pages: {stats['captioned']}  in={stats['input_tokens']} out={stats['output_tokens']}"
            f"  collection={stats['collection']}"
        )
        grand_in += stats["input_tokens"]
        grand_out += stats["output_tokens"]

    cost = grand_in * 1.0 / 1_000_000 + grand_out * 5.0 / 1_000_000
    print(f"\nTotal Haiku tokens: in={grand_in} out={grand_out}  ~= ${cost:.4f}")


if __name__ == "__main__":
    main(sys.argv)
