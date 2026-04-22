"""One-time indexing — push every page of given PDF(s) through the Colab
Nemotron tunnel and upsert their multi-vector embeddings into Qdrant.

Usage:
    PYTHONIOENCODING=utf-8 python scripts/06_index_nemotron.py [pdf_name ...]

If no PDF names are passed, defaults to Q1.pdf only (smallest, cheapest).

Pre-flight:
    1. notebooks/02_nemotron_tunnel.ipynb is running on Colab.
    2. .env has COLAB_EMBEDDING_URL set to the printed ngrok URL.
    3. local Qdrant is up at QDRANT_URL (docker compose up -d qdrant).

Cost estimate (free Colab T4, ngrok free):
    ~5 sec per page upload + Nemotron forward pass + 5 MB transfer.
    Q1.pdf (23 pages) ~= 2 minutes.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env")

from pharma_vision_rag.retriever import NemotronEmbeddingClient, NemotronVisionRetriever  # noqa: E402

PDF_DIR = ROOT / "data" / "pdf"
DEFAULT_PDFS = ["Q1.pdf"]


def main(argv: list[str]) -> None:
    tunnel_url = os.environ.get("COLAB_EMBEDDING_URL", "").strip()
    if not tunnel_url:
        raise SystemExit(
            "COLAB_EMBEDDING_URL is unset. Start notebooks/02_nemotron_tunnel.ipynb "
            "on Colab, copy the printed URL into .env, then re-run."
        )

    pdfs = argv[1:] if len(argv) > 1 else DEFAULT_PDFS

    print(f"Tunnel: {tunnel_url}")
    client = NemotronEmbeddingClient(tunnel_url)
    health = client.health()
    print(f"Health: {health}\n")

    retriever = NemotronVisionRetriever(
        embedding_client=client,
        qdrant_url=os.environ.get("QDRANT_URL", "http://localhost:6335"),
        qdrant_api_key=os.environ.get("QDRANT_API_KEY") or None,
    )

    for name in pdfs:
        path = PDF_DIR / name
        if not path.exists():
            print(f"  SKIP missing: {path}")
            continue
        print(f"\n=== Indexing {name} ===")
        stats = retriever.index_pdf(path, source=name)
        print(f"  pages: {stats['pages']}  patches: {stats['total_patches']}  collection: {stats['collection']}")

    client.close()
    print("\nDone.")


if __name__ == "__main__":
    main(sys.argv)
