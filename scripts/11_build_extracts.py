"""Build the indexed corpus extracts from raw PDFs (Phase 2, EXPERIMENT_PLAN §5 step 1).

Raw -> extract:
    data/pdf/Form 20-F 2025 (Oct 2025).pdf  (300p) -> data/pdf/20F_extract.pdf (50p)
    data/pdf/raw/Q{n}_deck_full.pdf          (41-43p) -> data/pdf/Q{n}_deck.pdf  (15p each)

Page selections are fixed constants below; rationale lives in docs/CORPUS.md.
gold_pages in eval/questions.jsonl use the *extract* page numbers (1-based).
Re-running is idempotent (overwrites).

Usage:
    PYTHONIOENCODING=utf-8 python scripts/11_build_extracts.py
"""
from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader, PdfWriter

ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = ROOT / "data" / "pdf"

# (raw path, extract path, 1-based raw page list)
EXTRACTS: list[tuple[Path, Path, list[int]]] = [
    (
        PDF_DIR / "Form 20-F 2025 (Oct 2025).pdf",
        PDF_DIR / "20F_extract.pdf",
        # text 14 / mixed 8 / table 28 = 50
        [25, 26, 27, 28, 30, 31, 32, 33, 34, 39, 40, 44, 45, 46, 47, 48, 49, 51, 52, 55,
         64, 65, 68, 70, 71, 77, 78, 79, 80, 81, 82, 85, 86, 87, 92, 117, 133, 145, 175,
         196, 197, 198, 202, 231, 235, 254, 256, 288, 290, 292],
    ),
    (
        PDF_DIR / "raw" / "Q1_deck_full.pdf",
        PDF_DIR / "Q1_deck.pdf",
        [5, 6, 7, 9, 12, 14, 19, 20, 23, 24, 25, 26, 27, 28, 30],
    ),
    (
        PDF_DIR / "raw" / "Q2_deck_full.pdf",
        PDF_DIR / "Q2_deck.pdf",
        [5, 6, 7, 8, 9, 10, 12, 14, 17, 21, 24, 25, 26, 27, 28],
    ),
    (
        PDF_DIR / "raw" / "Q3_deck_full.pdf",
        PDF_DIR / "Q3_deck.pdf",
        [5, 6, 7, 8, 9, 10, 12, 13, 17, 18, 24, 25, 26, 27, 28],
    ),
]


def build(raw: Path, out: Path, pages: list[int]) -> None:
    assert len(pages) == len(set(pages)) and pages == sorted(pages), f"bad page list for {out.name}"
    reader = PdfReader(str(raw))
    assert max(pages) <= len(reader.pages), f"{raw.name}: page {max(pages)} > {len(reader.pages)}"
    writer = PdfWriter()
    for p in pages:
        writer.add_page(reader.pages[p - 1])
    with open(out, "wb") as f:
        writer.write(f)
    print(f"{out.name}: {len(pages)}p  <- {raw.name} ({len(reader.pages)}p)")
    print("  extract->raw:", ", ".join(f"{i}->{p}" for i, p in enumerate(pages, 1)))


def main() -> None:
    for raw, out, pages in EXTRACTS:
        if not raw.exists():
            raise SystemExit(f"missing raw PDF: {raw}")
        build(raw, out, pages)


if __name__ == "__main__":
    main()
