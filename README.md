# pharma-vision-rag

> Pharmaceutical-domain multimodal hybrid RAG benchmark with 2026 SOTA stack.
> Cross-lingual (KR/EN) evaluation on Sanofi FY2025 corpus.

**Status**: 🚧 Phase 0 — environment setup. See [plan.md](plan.md) for roadmap.

---

## What this project does

Builds and benchmarks four RAG pipelines on pharma PDFs dense with charts, tables, and numeric data:

| Mode | Retrieval | Strength |
|---|---|---|
| **Text-only** | Docling → BGE-M3 | Prose questions, cheap |
| **Vision-only** | pdf2image → Nemotron ColEmbed V2 | Chart/table questions |
| **Caption** | Per-page VLM caption → BGE-M3 | All questions, expensive |
| **Hybrid** ★ | Text + Vision with LangGraph router + reranker | Best of both |

The main contribution is the **Hybrid mode** — a LangGraph routing graph that chooses text vs vision retrieval based on query intent, reranks combined candidates, and feeds top-3 to Claude Sonnet Vision for final answer.

## Why this matters

Traditional RAG (OCR → chunk → embed) fails on pharma PDFs because critical numbers (e.g., "Q3 Dupixent sales €3.5B") exist only in charts and tables that OCR destroys. Vision-only RAG helps but is weak on prose and costs 1000× more storage. This project quantifies the trade-off and shows when each mode wins.

## Tech stack (2026 SOTA)

- **Visual retriever**: Nemotron ColEmbed V2 (4B default, 3rd on ViDoRe V3) — swappable to 8B (#1) or 3B (T4-friendly)
- **Text retriever**: BGE-M3 (multilingual, ColBERT-style multi-vector)
- **Layout + OCR**: IBM Docling
- **Reranker**: ZeRank2 vs `llama-nemotron-rerank-vl-1b-v2` (A/B)
- **Generator**: Claude Sonnet 4.7 Vision (+ Qwen3-VL-8B comparator)
- **Vector DB**: Qdrant (multi-vector native)
- **Router**: LangGraph
- **Observability**: Langfuse
- **UI**: Gradio
- **Eval**: custom (ViDoRe V3 metrics reused)
- **Compute**: Google Colab Pro for Nemotron inference

## Evaluation corpus

Sanofi 2025 public disclosures (not committed to repo, see `data/pdf/`):
- Form 20-F 2025 (FY2025 annual, ~50 pages extracted)
- Q1/Q2/Q3 2025 Press Releases (23/26/28 pages)

30 Korean questions (with English parallels) across chart/table/text/multi-hop types, 25 with specific periods and 5 temporally ambiguous for disambiguation experiments.

## Quick start

> Not runnable yet. Phase 0 scaffolding in progress.

```bash
# Clone
git clone https://github.com/tkddnjs-dlqslek/pharma-vision-rag
cd pharma-vision-rag

# Copy env template and fill in keys
cp .env.example .env
# edit .env: ANTHROPIC_API_KEY, HF_TOKEN, LANGFUSE_*

# Start Qdrant
docker compose up -d qdrant

# Install deps
python -m venv venv && source venv/bin/activate  # on Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run Nemotron embedding on Colab (see notebooks/01_nemotron_smoke_test.ipynb)
# then dump embeddings into local Qdrant
```

## Repository layout

```
pharma-vision-rag/
├── src/pharma_vision_rag/
│   ├── retriever/      # BGE-M3 (baseline/QT/HyDE) + Nemotron wrappers
│   ├── rerank/         # ZeRank2, Nemotron rerank
│   ├── generator/      # Claude Vision wrapper
│   ├── modes/          # text_only, vision_only, caption, hybrid
│   ├── router/         # LangGraph routing graph
│   ├── eval/           # Recall/NDCG/Judge eval pipeline
│   └── utils/
├── notebooks/          # Colab notebooks for Nemotron inference
├── data/
│   ├── pdf/            # Source PDFs (gitignored)
│   └── samples/        # Public-license sample excerpts
├── scripts/            # CLI entry points
├── docs/               # Reports, blog drafts, design notes
├── docker-compose.yml
├── requirements.txt
├── plan.md             # Execution roadmap
└── pharma-vision-rag-PRD.md
```

## Roadmap

- **Phase 0** (in progress): environment + scaffolding
- **Phase 1**: swap VARAG internals with 2026 SOTA models
- **Phase 2**: build Korean question set + eval pipeline
- **Phase 3**: full benchmark + text-optimization ablation (baseline vs +QT vs +HyDE) + KR/EN breakdown
- **Phase 4**: README/blog/demo deploy

Details in [plan.md](plan.md).

## Planned deliverables

- Benchmark report: 4 modes × 30 questions × 2 languages, with text-retrieval ablation
- Gradio UIs: demo page, batch evaluation dashboard, page viewer
- Blog posts in Korean (LinkedIn) and English (Medium)
- HuggingFace Spaces demo
- Hypothesis (to be verified): *Vision retrieval closes the KR/EN language gap even after aggressive text-side optimization (QT + HyDE).*

## License

This project's own code (benchmark pipeline, question sets, evaluation scripts) is **MIT-licensed**.

**Third-party model licenses** (respect these when using/forking):
- **Nemotron ColEmbed V2** (`nvidia/nemotron-colembed-vl-4b-v2`): CC-BY-NC-4.0 — **non-commercial/research use only**. This benchmark project is non-commercial. Commercial users should switch to NVIDIA's commercial variants (e.g., `llama-3_2-nemoretriever-1b-vlm-embed-v1`).
- **BGE-M3**: MIT
- **Claude API**: Anthropic commercial terms
- **Docling**: MIT
- **Sanofi PDFs**: treated as user-provided inputs and not redistributed in this repo.

## Author

김상원 (Sangwon Kim) — Sanofi pharma intern, AI engineering portfolio project.
