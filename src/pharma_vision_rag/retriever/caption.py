"""Caption-based retriever — Haiku writes per-page captions, BGE-M3 indexes them.

Indexing pipeline:
    page image -> Claude Haiku Vision generates a 3-6 sentence text description
                  (heavy on numbers, products, periods, table contents)
               -> BGE-M3 embeds the caption
               -> Qdrant collection ``pharma_caption`` (one point per page)

Search pipeline:
    query -> BGE-M3 embedding -> Qdrant cosine search -> top-k caption hits
    (each hit carries the source/page so the mode can fetch the caption text
     OR the original page image at answer time)

Why this exists: it isolates the "VLM caption" baseline from the dense text
retriever (Docling chunks). Lets us measure how much information is lost when
we replace structured PDF parsing with a free-form VLM description.
"""
from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

import anthropic
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from pharma_vision_rag.generator.claude_vision import _image_to_base64

log = logging.getLogger(__name__)

DEFAULT_COLLECTION = "pharma_caption"
DEFAULT_EMBED_MODEL = "BAAI/bge-m3"
EMBED_DIM = 1024
DEFAULT_CAPTION_MODEL = "claude-haiku-4-5"

CAPTION_SYSTEM_PROMPT = """You extract structured, retrieval-friendly captions from Sanofi financial-report pages.

Given a page image, output 3-6 sentences that capture:
- the page topic / header / section name
- every product name visible (Dupixent, Beyfortus, ALTUVIIIO, etc.)
- every number with its currency/unit and period (€3,480m Q1 2025, +20.3% at CER, etc.)
- table contents — list each row's key numbers
- chart annotations — axis labels, data points, growth rates

Be exhaustive on numbers; brief on visual styling. Output ONLY the caption — no preamble, no quotation marks."""

_NS = uuid.UUID("c1be0a3a-ec0a-44e8-9a45-7e6a5d33b08c")


def _caption_id(source: str, page: int) -> str:
    return str(uuid.uuid5(_NS, f"{source}:caption:page:{page}"))


class CaptionIndexer:
    """One-time: render each page, ask Haiku for a caption, embed + upsert."""

    def __init__(
        self,
        qdrant_url: str,
        qdrant_api_key: str | None = None,
        collection: str = DEFAULT_COLLECTION,
        embed_model: str = DEFAULT_EMBED_MODEL,
        anthropic_client: anthropic.Anthropic | None = None,
        caption_model: str = DEFAULT_CAPTION_MODEL,
    ) -> None:
        self.collection = collection
        self.qdrant = QdrantClient(url=qdrant_url, api_key=qdrant_api_key or None, check_compatibility=False)
        self.embedder = SentenceTransformer(embed_model)
        self.anthropic = anthropic_client or anthropic.Anthropic()
        self.caption_model = caption_model

    def ensure_collection(self) -> None:
        if not self.qdrant.collection_exists(self.collection):
            self.qdrant.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
            )
            log.info("Created caption collection %s", self.collection)

    def caption_image(self, image: Image.Image) -> tuple[str, dict[str, int]]:
        """Returns (caption_text, usage)."""
        media_type, data = _image_to_base64(image)
        response = self.anthropic.messages.create(
            model=self.caption_model,
            max_tokens=512,
            system=CAPTION_SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}},
                    {"type": "text", "text": "Caption this page."},
                ],
            }],
        )
        text = next((b.text for b in response.content if b.type == "text"), "").strip()
        usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }
        return text, usage

    def index_pdf(self, pdf_path: str | Path, source: str | None = None) -> dict[str, Any]:
        from pharma_vision_rag.utils.pdf import iter_pages

        pdf_path = Path(pdf_path)
        source = source or pdf_path.name
        self.ensure_collection()

        captions = []
        total_in = total_out = 0
        for page_num, img in iter_pages(pdf_path):
            caption, usage = self.caption_image(img)
            total_in += usage["input_tokens"]
            total_out += usage["output_tokens"]
            captions.append({"source": source, "page": page_num, "caption": caption})
            log.info("Captioned %s p%d (%d chars, in=%d out=%d)", source, page_num, len(caption), usage["input_tokens"], usage["output_tokens"])

        if not captions:
            return {"source": source, "captioned": 0}

        vectors = self.embedder.encode(
            [c["caption"] for c in captions],
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        points = [
            PointStruct(
                id=_caption_id(c["source"], c["page"]),
                vector=v.tolist(),
                payload={
                    "source": c["source"],
                    "page": c["page"],
                    "caption": c["caption"],
                    "text": c["caption"],  # alias so generic format_context works
                    "block_type": "caption",
                },
            )
            for c, v in zip(captions, vectors, strict=True)
        ]
        self.qdrant.upsert(collection_name=self.collection, points=points)

        return {
            "source": source,
            "captioned": len(captions),
            "input_tokens": total_in,
            "output_tokens": total_out,
            "collection": self.collection,
        }


class CaptionRetriever:
    """Search the caption collection by BGE-M3 dense cosine similarity."""

    variant_name = "caption"

    def __init__(
        self,
        qdrant_url: str,
        qdrant_api_key: str | None = None,
        collection: str = DEFAULT_COLLECTION,
        embed_model: str = DEFAULT_EMBED_MODEL,
    ) -> None:
        self.collection = collection
        self.qdrant = QdrantClient(url=qdrant_url, api_key=qdrant_api_key or None, check_compatibility=False)
        self.embedder = SentenceTransformer(embed_model)

    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        q_vec = self.embedder.encode([query], normalize_embeddings=True, show_progress_bar=False)[0].tolist()
        result = self.qdrant.query_points(
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
                "text": payload.get("text", payload.get("caption", "")),
                "page": payload.get("page"),
                "source": payload.get("source"),
                "block_type": payload.get("block_type", "caption"),
                "id": point.id,
                "variant": self.variant_name,
            })
        return hits
