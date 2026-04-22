# VARAG Architecture Review

**Reviewed**: 2026-04-22
**Source**: https://github.com/adithya-s-k/VARAG (shallow clone at `c:/Users/user/Desktop/VARAG-ref/`)
**Purpose**: Decide how much of VARAG to reuse in `pharma-vision-rag`.

---

## 1. Four RAG modes (`varag/rag/`)

| Class | File | Key methods | What it does |
|---|---|---|---|
| `SimpleRAG` | `_simpleRAG.py` | `index(path, chunking_strategy)`, `search(query, k)` | Text-only. SentenceTransformer for chunks + embeds. LanceDB. |
| `VisionRAG` | `_visionRAG.py` | `index(path)`, `search(query, k)` | PDF→image, Jina-CLIP single-vector. Page-level retrieval. |
| `ColpaliRAG` | `_colpaliRAG.py` | `index(path)`, `search(query, k, limit)` | ColPali token-level patch embeddings. Optional token pooling. |
| `HybridColpaliRAG` | `_hybridColpaliRAG.py` | `index(path)`, `search(query, k, use_image_search)` | Coarse filter by Jina-CLIP single-vector, rerank by ColPali late interaction. |

All use **LanceDB** (not Qdrant) with PyArrow schemas.

## 2. LLM / VLM abstraction

- `BaseLLM.query(query: str) -> str`
- `BaseVLM.__call__(image: Image, query: str) -> str`
- Providers: `OpenAILLM/VLM`, `LiteLLM/LiteLLMVLM` (LiteLLM supports Claude too)
- Swap by import in `__init__.py` or env check in `demo.py`

## 3. Hybrid-mode data flow

```
Indexing:
  PDF → pages (Poppler/PyMuPDF)
  each page → ColPali token-level patch embedding (flatten + shape stored)
  each page → Jina-CLIP 768-d single vector
  → LanceDB row with both vectors

Search:
  query → ColPali token embeddings (pooled)
  query → Jina-CLIP single vector
  if use_image_search: initial top-k by image_vector (fast)
  else: initial top-k by image_vector (still image_vector — both branches use image_vector)
  rerank top-k by ColPali MaxSim
  return images + metadata
```

**Key insight**: VARAG "hybrid" = ColPali + Jina-CLIP cheap prefilter, **all vision, no text retriever**. Our "hybrid" = BGE-M3 text + Nemotron vision. **Architecturally different** — we can't reuse their hybrid logic, only the pattern of `coarse filter → rerank`.

## 4. Reusable pieces

**From `varag/utils.py`**:
- `get_model_colpali()` — factory pattern worth copying for our Nemotron loader
- `ColPaliSimilarityMapper` — attention-map visualization (useful for Phase 3 failure analysis)
- PDF↔image helpers, base64 I/O

**From `demo.py`**:
- 4-tab Gradio layout
- Per-mode LanceDB tables
- Progress bars + metrics display

**From `varag/chunking/`**:
- `FixedTokenChunker` interface worth keeping (token-based default=1000)

**Prompt templates**: Not centralized. Each `search()` returns raw results; generation prompt is inline in `demo.py`. We rewrite for pharma domain anyway.

## 5. Incompatible / outdated pieces

- **OpenAI hardcoded** in `_simpleRAG.py:14` — need Claude swap
- **ColPali via `get_model_colpali()`** — need full replacement with Nemotron ColEmbed V2 Colab tunnel
- **LanceDB everywhere** — we're on Qdrant; schemas and queries differ
- **Chunk size 1000 tokens** — tune for pharma (IR slides have short structured blocks)
- **Base64 PNG in DB** — memory-heavy at pharma scan resolution; we store paths instead

## 6. Integration recommendation

**Do not fork as submodule.** Reuse ~15-20% by selective copy. Architecture diverges too much (Qdrant vs LanceDB, Claude vs OpenAI, BGE-M3 vs Jina-CLIP, Nemotron vs ColPali, true text+vision hybrid vs all-vision hybrid).

**Files to copy / port**:
- `varag/utils.py` → our `utils/pdf.py` — PDF→image, base64, path helpers
- `BaseLLM`/`BaseVLM` pattern → our `generator/base.py` (Claude-first)
- `FixedTokenChunker` logic → our `retriever/chunking.py`
- `demo.py` Gradio tab layout → skeleton for our 3-UI (demo / batch dashboard / page viewer)

**Files to build fresh**:
- All 4 mode classes (`modes/*.py`)
- All retrievers (`retriever/*.py`)
- Rerankers (new — VARAG doesn't have a dedicated reranker)
- LangGraph router (`router/` — VARAG has no router)

**Final take**: VARAG is inspiration, not scaffolding. The 4-mode layout and class shape transfer; the internals do not.
