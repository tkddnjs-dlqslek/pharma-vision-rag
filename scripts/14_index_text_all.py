"""Index the full 7-file corpus into the ``pharma_text`` collection (Docling + BGE-M3).

Phase 1 hit ``std::bad_alloc`` inside Docling on dense pages (Q1 p16/17/21) and the first Phase 2 run
was OOM-killed converting a whole file (16 GB RAM, ~3 GB free). Strategy: convert in fixed page windows
(WINDOW pages at a time, bounded memory); a window that raises is retried one page at a time; pages that
still fail are reported as known gaps in the text index.

Local CPU only, no API. BGE-M3 (~2.3 GB) downloads on first run.

Usage:
    PYTHONIOENCODING=utf-8 python scripts/14_index_text_all.py            # all 7 files
    PYTHONIOENCODING=utf-8 python scripts/14_index_text_all.py Q2.pdf     # subset
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env")

from pharma_vision_rag.retriever import DoclingTextRetriever  # noqa: E402
from pharma_vision_rag.utils.pdf import page_count  # noqa: E402

PDF_DIR = ROOT / "data" / "pdf"
CORPUS = ["Q1.pdf", "Q2.pdf", "Q3.pdf", "20F_extract.pdf", "Q1_deck.pdf", "Q2_deck.pdf", "Q3_deck.pdf"]
WINDOW = 5  # pages per Docling conversion; ponytail: fixed, raise on a 32 GB box


def index_file(retriever: DoclingTextRetriever, name: str) -> dict:
    path = PDF_DIR / name
    n_pages = page_count(path)
    t0 = time.time()
    chunks, by_type, failed = 0, {"text": 0, "table": 0}, []

    def run(rng: tuple[int, int]) -> None:
        nonlocal chunks
        s = retriever.index(path, source=name, page_range=rng)
        chunks += s["chunks"]
        for k in by_type:
            by_type[k] += s.get("by_type", {}).get(k, 0)

    for start in range(1, n_pages + 1, WINDOW):
        rng = (start, min(start + WINDOW - 1, n_pages))
        try:
            run(rng)
        except Exception as e:  # noqa: BLE001 — Docling raises RuntimeError/MemoryError on bad_alloc
            print(f"  pages {rng} failed ({type(e).__name__}: {str(e)[:60]}); retrying one page at a time")
            for p in range(rng[0], rng[1] + 1):
                try:
                    run((p, p))
                except Exception as e2:  # noqa: BLE001
                    failed.append(p)
                    print(f"    p{p} failed: {type(e2).__name__}: {str(e2)[:60]}")
        print(f"  pages {rng[0]}-{rng[1]} done  ({time.time() - t0:.0f}s, {chunks} chunks so far)", flush=True)

    return {"source": name, "chunks": chunks, "by_type": by_type, "failed_pages": failed,
            "pages": n_pages, "seconds": round(time.time() - t0)}


def main(argv: list[str]) -> None:
    names = argv[1:] or CORPUS
    retriever = DoclingTextRetriever(
        qdrant_url=os.environ.get("QDRANT_URL", "http://localhost:6335"),
        qdrant_api_key=os.environ.get("QDRANT_API_KEY") or None,
    )
    results = []
    for name in names:
        print(f"\n=== {name} ===")
        stats = index_file(retriever, name)
        print(f"  {stats['chunks']} chunks {stats['by_type']}  "
              f"pages {stats['pages']}  failed {stats['failed_pages']}  {stats['seconds']}s")
        results.append(stats)

    total = retriever.client.count(retriever.collection, exact=True).count
    print(f"\ncollection {retriever.collection}: {total} points")
    gaps = {r["source"]: r["failed_pages"] for r in results if r["failed_pages"]}
    print("failed pages:", gaps or "none")


if __name__ == "__main__":
    main(sys.argv)
