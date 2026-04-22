"""Phase 0 smoke test — Docling layout extraction on Q1.pdf.

Usage:
    python scripts/00_docling_smoke.py

Expected outcome:
    - Docling parses data/pdf/Q1.pdf without error
    - Reports text block count, table count, figure count
    - Dumps first page markdown to stdout (first 500 chars)
    - Confirms tables are extracted as markdown (not OCR noise)

Success = Docling handles Sanofi press releases cleanly. If this passes,
the Phase 1 text retriever (Docling + BGE-M3) will work.
"""
from pathlib import Path
from docling.document_converter import DocumentConverter

PDF_PATH = Path(__file__).parent.parent / "data" / "pdf" / "Q1.pdf"


def main() -> None:
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    print(f"Parsing: {PDF_PATH.name}")
    converter = DocumentConverter()
    result = converter.convert(str(PDF_PATH))
    doc = result.document

    # Structure inventory
    texts = getattr(doc, "texts", [])
    tables = getattr(doc, "tables", [])
    pictures = getattr(doc, "pictures", [])
    pages = getattr(doc, "pages", {})

    print(f"\n=== Structure ===")
    print(f"  pages:    {len(pages) if hasattr(pages, '__len__') else '?'}")
    print(f"  texts:    {len(texts)}")
    print(f"  tables:   {len(tables)}")
    print(f"  pictures: {len(pictures)}")

    # First text block
    if texts:
        first_text = texts[0]
        content = getattr(first_text, "text", str(first_text))[:300]
        print(f"\n=== First text block ===\n{content}")

    # First table as markdown
    if tables:
        first_table = tables[0]
        try:
            md = first_table.export_to_markdown(doc=doc)
        except TypeError:
            md = first_table.export_to_markdown()
        except Exception as e:  # noqa: BLE001
            md = f"(export failed: {e})"
        print(f"\n=== First table (markdown) ===\n{md[:800]}")

    # Full document markdown preview
    full_md = doc.export_to_markdown()
    print(f"\n=== Page markdown (first 500 chars) ===\n{full_md[:500]}")

    # Acceptance checks
    print("\n=== Acceptance ===")
    checks = {
        "texts > 0": len(texts) > 0,
        "tables >= 1 (press release usually has at least one)": len(tables) >= 1,
        "markdown length > 500": len(full_md) > 500,
    }
    for name, ok in checks.items():
        print(f"  {'OK ' if ok else 'FAIL '} {name}")


if __name__ == "__main__":
    main()
