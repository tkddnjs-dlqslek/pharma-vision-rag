"""Text retriever — Docling for PDF parsing, BGE-M3 for embeddings, Qdrant for search.

Responsibilities
----------------
1. Parse a pharma PDF with Docling → extract text blocks + tables (as markdown).
2. Embed each chunk with BGE-M3 (dense, 1024-d, cosine).
3. Upsert into Qdrant collection ``pharma_text`` with page/block metadata.
4. Serve ``search(query, k)`` returning ranked chunks.

This is the *baseline* text retriever. Query-transformation variants (QT, HyDE)
compose around this class by rewriting the query before calling ``search``.
"""
from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import AcceleratorOptions, PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)

DEFAULT_COLLECTION = "pharma_text"
DEFAULT_EMBED_MODEL = "BAAI/bge-m3"
EMBED_DIM = 1024
MIN_TEXT_LEN = 10  # skip fragments shorter than this

# Stable UUID namespace so re-indexing the same (source, block) upserts cleanly.
_NS = uuid.UUID("0f4cf7cb-9e3e-4cfa-a5d1-d9b64a4f2fe1")


def _chunk_id(source: str, block_type: str, page: int | None, block_index: int) -> str:
    # page is part of the id: per-page fallback conversions restart block_index at 0
    return str(uuid.uuid5(_NS, f"{source}:{block_type}:{page}:{block_index}"))


class DoclingTextRetriever:
    """Index PDFs via Docling + BGE-M3, search by dense cosine similarity."""

    def __init__(
        self,
        qdrant_url: str,
        qdrant_api_key: str | None = None,
        collection: str = DEFAULT_COLLECTION,
        embed_model: str = DEFAULT_EMBED_MODEL,
        grpc_port: int = 6336,
    ) -> None:
        self.collection = collection
        # Prefer gRPC for consistency with the multi-vector path and to avoid
        # REST workers getting stuck under load.
        from urllib.parse import urlparse
        host = urlparse(qdrant_url).hostname or "localhost"
        self.client = QdrantClient(
            host=host,
            grpc_port=grpc_port,
            prefer_grpc=True,
            api_key=qdrant_api_key or None,
            check_compatibility=False,
            timeout=60,
        )
        log.info("Loading embedding model %s (~2.3 GB on first run)", embed_model)
        self.embedder = SentenceTransformer(embed_model)
        # Corpus is born-digital: no OCR (RapidOCR models + page rasters were the main RAM cost,
        # and the 2026-09-16 run was OOM-killed on a 16 GB box). Table structure stays on.
        pipeline = PdfPipelineOptions(do_ocr=False, do_table_structure=True)
        # Docling defaults to 4 threads; the indexing box has 16 logical cores.
        pipeline.accelerator_options = AcceleratorOptions(num_threads=8, device="cpu")
        self.converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline)}
        )

    # ─── Index ────────────────────────────────────────────────────────────

    def ensure_collection(self) -> None:
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
            )
            log.info("Created collection %s (dim=%d)", self.collection, EMBED_DIM)

    def _extract_chunks(
        self, pdf_path: Path, source: str, page_range: tuple[int, int] | None = None
    ) -> list[dict[str, Any]]:
        """Parse one PDF (optionally a 1-based inclusive page range) and return chunk dicts."""
        log.info("Parsing %s with Docling (pages %s)", pdf_path.name, page_range or "all")
        kwargs = {"page_range": page_range} if page_range else {}
        doc = self.converter.convert(str(pdf_path), **kwargs).document

        chunks: list[dict[str, Any]] = []

        for idx, item in enumerate(getattr(doc, "texts", [])):
            text = (getattr(item, "text", "") or "").strip()
            if len(text) < MIN_TEXT_LEN:
                continue
            prov = getattr(item, "prov", None)
            page = prov[0].page_no if prov else None
            chunks.append({
                "text": text,
                "page": page,
                "block_type": "text",
                "block_index": idx,
                "source": source,
            })

        for idx, table in enumerate(getattr(doc, "tables", [])):
            try:
                md = table.export_to_markdown(doc=doc)
            except TypeError:
                md = table.export_to_markdown()
            except Exception as e:  # noqa: BLE001
                log.warning("Table %d export failed: %s", idx, e)
                continue
            md = (md or "").strip()
            if len(md) < MIN_TEXT_LEN:
                continue
            prov = getattr(table, "prov", None)
            page = prov[0].page_no if prov else None
            chunks.append({
                "text": md,
                "page": page,
                "block_type": "table",
                "block_index": idx,
                "source": source,
            })

        log.info("%s: extracted %d chunks (text + table)", pdf_path.name, len(chunks))
        return chunks

    def index(
        self,
        pdf_path: str | Path,
        source: str | None = None,
        batch_size: int = 32,
        page_range: tuple[int, int] | None = None,
    ) -> dict[str, Any]:
        """Parse ``pdf_path`` (or a page range), embed, upsert. Idempotent via stable UUIDs."""
        pdf_path = Path(pdf_path)
        source = source or pdf_path.name
        self.ensure_collection()

        chunks = self._extract_chunks(pdf_path, source=source, page_range=page_range)
        if not chunks:
            return {"source": source, "chunks": 0, "collection": self.collection}

        texts = [c["text"] for c in chunks]
        vectors = self.embedder.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        points = [
            PointStruct(
                id=_chunk_id(c["source"], c["block_type"], c["page"], c["block_index"]),
                vector=v.tolist(),
                payload=c,
            )
            for c, v in zip(chunks, vectors, strict=True)
        ]
        self.client.upsert(collection_name=self.collection, points=points)

        by_type = {"text": 0, "table": 0}
        for c in chunks:
            by_type[c["block_type"]] += 1
        return {
            "source": source,
            "chunks": len(chunks),
            "by_type": by_type,
            "collection": self.collection,
        }

    # ─── Search ───────────────────────────────────────────────────────────

    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """Return top-``k`` chunks ranked by cosine similarity to ``query``."""
        q_vec = self.embedder.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0].tolist()

        result = self.client.query_points(
            collection_name=self.collection,
            query=q_vec,
            limit=k,
            with_payload=True,
        )
        hits = []
        for point in result.points:
            payload = point.payload or {}
            hits.append({
                "score": point.score,
                "text": payload.get("text", ""),
                "page": payload.get("page"),
                "block_type": payload.get("block_type"),
                "source": payload.get("source"),
                "id": point.id,
            })
        return hits
