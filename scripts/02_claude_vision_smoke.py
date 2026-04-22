"""Phase 1 smoke test — ClaudeVisionGenerator against Q1.pdf page 1.

Expected:
    - Page 1 contains "Dupixent sales were €3.5 billion, up 20.3%"
    - Query "2025 Q1 Dupixent 매출은?" should yield an answer mentioning €3.5 billion
    - Second query re-using the same page should show cache_read_input_tokens > 0

Usage:
    python scripts/02_claude_vision_smoke.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pypdfium2 as pdfium
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

load_dotenv(ROOT / ".env")

from pharma_vision_rag.generator import ClaudeVisionGenerator  # noqa: E402

PDF_PATH = ROOT / "data" / "pdf" / "Q1.pdf"


def render_pdf_page(pdf_path: Path, page_number: int, scale: float = 1.5):
    """Render one PDF page (1-indexed) to a PIL Image using pypdfium2.

    pypdfium2 is Windows-friendly — no Poppler required.
    """
    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        page = pdf[page_number - 1]
        return page.render(scale=scale).to_pil()
    finally:
        pdf.close()


def main() -> None:
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    # Mimic real Phase 3 retrieval: top-3 pages per query.
    # Sonnet 4.6 min cache prefix = 2048 tokens; 3 images easily exceed this.
    pages = [render_pdf_page(PDF_PATH, page_number=n) for n in (1, 2, 3)]
    for i, p in enumerate(pages, start=1):
        print(f"Rendered page {i}: {p.size} (w, h)")
    print()

    generator = ClaudeVisionGenerator()
    print(f"Model: {generator.model}\n")

    # ─── Query 1 (KR, first query, cache write expected) ─────────
    q1 = "2025 Q1 Dupixent 매출은 얼마인가요?"
    print(f"Q1 (KR): {q1}")
    r1 = generator.generate(query=q1, images=pages)
    print(f"A: {r1['answer']}")
    print(f"  usage: {r1['usage']}")
    print(f"  stop_reason: {r1['stop_reason']}\n")

    # ─── Query 2 (EN, same pages, cache read expected) ───────────
    q2 = "What was Sanofi's Q1 2025 business EPS and how much did it grow?"
    print(f"Q2 (EN): {q2}")
    r2 = generator.generate(query=q2, images=pages)
    print(f"A: {r2['answer']}")
    print(f"  usage: {r2['usage']}")
    print(f"  stop_reason: {r2['stop_reason']}\n")

    # ─── Query 3 (answer not on pages — should refuse) ───────────
    q3 = "What is the projected 2030 Dupixent revenue guidance?"
    print(f"Q3 (EN, off-page): {q3}")
    r3 = generator.generate(query=q3, images=pages)
    print(f"A: {r3['answer']}")
    print(f"  usage: {r3['usage']}")
    print(f"  stop_reason: {r3['stop_reason']}\n")

    # ─── Acceptance ───────────────────────────────────────────────
    print("=== Acceptance ===")
    a1 = r1["answer"]
    checks = {
        "Q1 mentions 3.5 (Dupixent Q1 sales)": "3.5" in a1,
        "Q2 mentions 1.79 (business EPS)": "1.79" in r2["answer"],
        "Q3 refuses (not found / 찾을 수 없)": any(
            kw in r3["answer"].lower() for kw in ["not found", "찾을 수 없", "no information", "not on"]
        ),
        "Q2 has cache reads > 0 (system + image cached from Q1)": r2["usage"]["cache_read_input_tokens"] > 0,
    }
    for name, ok in checks.items():
        print(f"  {'OK  ' if ok else 'FAIL'} {name}")

    # Cost snapshot
    total_in = sum(r["usage"]["input_tokens"] for r in (r1, r2, r3))
    total_out = sum(r["usage"]["output_tokens"] for r in (r1, r2, r3))
    total_cw = sum(r["usage"]["cache_creation_input_tokens"] for r in (r1, r2, r3))
    total_cr = sum(r["usage"]["cache_read_input_tokens"] for r in (r1, r2, r3))
    print(f"\nTotal across 3 queries: input={total_in}, output={total_out}, cache_write={total_cw}, cache_read={total_cr}")


if __name__ == "__main__":
    main()
