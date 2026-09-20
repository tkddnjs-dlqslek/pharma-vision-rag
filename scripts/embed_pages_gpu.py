"""GPU-box job (RunPod): Nemotron ColEmbed page/query embeddings + Docling text blocks for the whole corpus.

Standalone: no repo imports. Input dir holds the corpus PDFs, corpus.json and questions.jsonl
(bundle them with scripts/16_make_runpod_bundle.py). Output dir:
    manifest.json                     model id, render scale, per-page patch counts, queries
    pages/<doc id>/<page>.npy         fp16 [N_patches, 3072]  (full resolution: the source of truth)
    queries/<id>_<ko|en>.npy          fp16 [N_tokens, 3072]
    text_blocks.jsonl                 raw Docling text/table blocks (same schema as retriever/docling_text.py)
    zips/<doc id>.zip, zips/meta.zip  one archive per document so a ~12 GB result downloads in resumable pieces

Locally: unzip everything into data/embeddings/v2/, copy text_blocks.jsonl to data/embeddings/, then
scripts/15_rebuild_text_index.py (text) and scripts/13_score_vision_exact.py (vision).

RunPod recipe (PyTorch template, 24 GB GPU, 40 GB volume):
    cd /workspace && mkdir -p input && unzip -o runpod_input.zip -d input
    pip install -q "transformers>=4.45,<5" accelerate einops sentencepiece pypdfium2 Pillow huggingface_hub docling
    python input/embed_pages_gpu.py --input input --out smoke --smoke     # ~3 min: 2 pages, queries, Docling on 2 pages
    python input/embed_pages_gpu.py --input input --out out --batch 4     # full run; resume-safe
    # no HF token needed: the model repo is public (gated=False, checked 2026-09-20; NVIDIA non-commercial license)

Steps can be run separately with --only embed|docling|zip.
"""
from __future__ import annotations

import argparse
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import pypdfium2 as pdfium

MODEL_ID = "nvidia/llama-nemotron-colembed-vl-3b-v2"  # ViDoRe V3 rank 6; 4b/8b variants need more VRAM


# ─── embeddings ───────────────────────────────────────────────────────────

def load_model():
    import torch
    from transformers import AutoModel
    assert torch.cuda.is_available(), "no CUDA device"
    print(f"device {torch.cuda.get_device_name(0)}")
    model = AutoModel.from_pretrained(MODEL_ID, device_map="cuda", trust_remote_code=True,
                                      torch_dtype=torch.bfloat16).eval()
    # internal ViT is fp32-pinned; cast float params/buffers only (see CLAUDE.md known issues)
    floats = (torch.float32, torch.float16, torch.bfloat16, torch.float64)
    for t in list(model.parameters()) + list(model.buffers()):
        if t.dtype in floats:
            t.data = t.data.to(torch.bfloat16)
    return model


def as_list(out):
    if isinstance(out, list):
        return out
    return [out[i] for i in range(out.shape[0])] if out.dim() == 3 else [out]


def to_fp16(t) -> np.ndarray:
    import torch
    return t.detach().to(torch.float16).cpu().numpy()


def embed_pages(model, inp: Path, out: Path, docs: list[str], scale: float, batch: int, max_pages: int | None):
    import torch
    pages, t0 = [], time.time()
    for doc in docs:
        out_dir = out / "pages" / doc
        out_dir.mkdir(parents=True, exist_ok=True)
        pdf = pdfium.PdfDocument(str(inp / doc))
        n = min(len(pdf), max_pages or len(pdf))
        todo = [p for p in range(1, n + 1) if not (out_dir / f"{p}.npy").exists()]
        for i in range(0, len(todo), batch):
            part = todo[i:i + batch]
            imgs = [pdf[p - 1].render(scale=scale).to_pil().convert("RGB") for p in part]
            with torch.no_grad():
                embs = as_list(model.forward_images(imgs, batch_size=len(imgs)))
            for p, e in zip(part, embs):
                np.save(out_dir / f"{p}.npy", to_fp16(e))
        for p in range(1, n + 1):
            arr = np.load(out_dir / f"{p}.npy", mmap_mode="r")
            pages.append({"source": doc, "page": p, "n_patches": int(arr.shape[0])})
        pdf.close()
        print(f"{doc}: {n} pages  (total {len(pages)}, {time.time() - t0:.0f}s)", flush=True)
    counts = [p["n_patches"] for p in pages]
    print(f"pages {len(counts)}  patches/page min {min(counts)} mean {sum(counts) / len(counts):.0f} max {max(counts)}"
          f"  -> {sum(counts) * 3072 * 2 / 1e9:.1f} GB fp16")
    return pages


def embed_queries(model, questions: Path, out: Path):
    import torch
    (out / "queries").mkdir(parents=True, exist_ok=True)
    qs = [json.loads(l) for l in questions.read_text(encoding="utf-8").splitlines() if l.strip()]
    items = [(f"{q['id']}_ko", q["q_ko"]) for q in qs] + [(f"{q['id']}_en", q["q_en"]) for q in qs]
    with torch.no_grad():
        embs = as_list(model.forward_queries([t for _, t in items], batch_size=8))
    meta = []
    for (name, text), e in zip(items, embs):
        arr = to_fp16(e)
        np.save(out / "queries" / f"{name}.npy", arr)
        meta.append({"id": name, "n_tokens": int(arr.shape[0]), "text": text})
    print(f"{len(items)} queries embedded")
    return meta


# ─── Docling text blocks (mirrors DoclingTextRetriever._raw_blocks; keep the two in sync) ───

def docling_blocks(inp: Path, out: Path, docs: list[str], max_pages: int | None) -> None:
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import AcceleratorOptions, PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
    opts = PdfPipelineOptions(do_ocr=False, do_table_structure=True)  # born-digital corpus
    opts.accelerator_options = AcceleratorOptions(num_threads=8, device="cuda")
    conv = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})

    path = out / "text_blocks.jsonl"
    done = {json.loads(l)["source"] for l in open(path, encoding="utf-8")} if path.exists() else set()
    for doc in docs:
        if doc in done:
            continue
        t0 = time.time()
        kwargs = {"page_range": (1, max_pages)} if max_pages else {}
        d = conv.convert(str(inp / doc), **kwargs).document
        blocks: list[dict] = []

        def add(item, kind: str, idx: int, text: str) -> None:
            prov = getattr(item, "prov", None)
            if text.strip():
                blocks.append({"source": doc, "page": prov[0].page_no if prov else None,
                               "block_type": kind, "block_index": idx, "text": text.strip()})

        for idx, item in enumerate(getattr(d, "texts", [])):
            add(item, "text", idx, getattr(item, "text", "") or "")
        for idx, table in enumerate(getattr(d, "tables", [])):
            try:
                md = table.export_to_markdown(doc=d)
            except TypeError:
                md = table.export_to_markdown()
            add(table, "table", idx, md or "")
        with open(path, "a", encoding="utf-8") as f:  # appended per document -> resume-safe
            f.writelines(json.dumps(b, ensure_ascii=False) + "\n" for b in blocks)
        print(f"{doc}: {len(blocks)} blocks  ({time.time() - t0:.0f}s)", flush=True)


# ─── packaging ────────────────────────────────────────────────────────────

def make_zips(out: Path, docs: list[str]) -> None:
    zdir = out / "zips"
    zdir.mkdir(exist_ok=True)
    for doc in docs:  # npy is incompressible: store, don't deflate
        with zipfile.ZipFile(zdir / f"{doc}.zip", "w", zipfile.ZIP_STORED) as z:
            for f in sorted((out / "pages" / doc).glob("*.npy"), key=lambda p: int(p.stem)):
                z.write(f, f"pages/{doc}/{f.name}")
    with zipfile.ZipFile(zdir / "meta.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for f in [out / "manifest.json", out / "text_blocks.jsonl", *sorted((out / "queries").glob("*.npy"))]:
            if f.exists():
                z.write(f, str(f.relative_to(out)).replace("\\", "/"))
    zips = list(zdir.glob("*.zip"))
    print(f"{len(zips)} zips in {zdir}  ({sum(f.stat().st_size for f in zips) / 1e9:.1f} GB) — download all of them")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("out"))
    ap.add_argument("--scale", type=float, default=1.5, help="pypdfium2 render scale (1.5 ~ 150 DPI)")
    ap.add_argument("--batch", type=int, default=2, help="images per forward_images call (2 on 16 GB, 4 on 24 GB+)")
    ap.add_argument("--only", choices=["embed", "docling", "zip"], help="run a single step")
    ap.add_argument("--smoke", action="store_true", help="first document, 2 pages: verify the whole stack in minutes")
    a = ap.parse_args()

    corpus = json.loads((a.input / "corpus.json").read_text(encoding="utf-8"))
    docs = [d["id"] for d in corpus]
    missing = [n for n in docs + ["questions.jsonl"] if not (a.input / n).exists()]
    if missing:
        raise SystemExit(f"missing in {a.input}: {missing}")
    max_pages = None
    if a.smoke:
        docs, max_pages = docs[:1], 2
    a.out.mkdir(parents=True, exist_ok=True)

    if a.only in (None, "embed"):
        import torch
        model = load_model()
        pages = embed_pages(model, a.input, a.out, docs, a.scale, a.batch, max_pages)
        queries = embed_queries(model, a.input / "questions.jsonl", a.out)
        (a.out / "manifest.json").write_text(json.dumps(
            {"model_id": MODEL_ID, "render_scale": a.scale, "dim": 3072, "pages": pages, "queries": queries},
            ensure_ascii=False, indent=1), encoding="utf-8")
        del model
        torch.cuda.empty_cache()  # leave the GPU to Docling
    if a.only in (None, "docling"):
        docling_blocks(a.input, a.out, docs, max_pages)
    if a.smoke:
        arr = np.load(a.out / "pages" / docs[0] / "1.npy")
        assert arr.ndim == 2 and arr.shape[1] == 3072 and arr.dtype == np.float16, f"bad page embedding {arr.shape} {arr.dtype}"
        n_blocks = sum(1 for _ in open(a.out / "text_blocks.jsonl", encoding="utf-8"))
        assert n_blocks > 0, "Docling produced no blocks"
        total_pages = sum(d["pages"] for d in corpus)
        print(f"SMOKE OK: page emb {arr.shape} {arr.dtype}, {n_blocks} Docling blocks. Full corpus at this patch "
              f"count = {arr.shape[0] * 3072 * 2 * total_pages / 1e9:.0f} GB fp16 — check the volume size before the full run")
        return
    if a.only in (None, "zip"):
        make_zips(a.out, docs)


if __name__ == "__main__":
    main()
