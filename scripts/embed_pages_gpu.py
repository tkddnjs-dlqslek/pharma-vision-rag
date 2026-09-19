"""Nemotron ColEmbed V2 batch page + query embedding. Runs on a GPU box (RunPod), not locally.

Standalone: no repo imports. Embeds every page of the 7-file corpus (render_scale 1.5) and the
60 benchmark queries, writes fp16 .npy files + manifest.json, zips to embeddings.zip.
Local side: scripts/13_load_nemotron_npz.py upserts the zip into Qdrant (pharma_vision).

RunPod recipe (PyTorch template, any 16 GB+ GPU; ~10 min on RTX 4090, ~15 min on T4-class):
    # on the pod
    pip install -q "transformers>=4.45,<5" accelerate einops sentencepiece pypdfium2 Pillow huggingface_hub
    export HF_TOKEN=hf_...                      # gated: accept the NVIDIA license on the model page first
    # flash-attn is optional (model card recommends it; Phase 1 ran without it on a T4)
    mkdir -p /workspace/input                   # then upload from local:
    #   runpodctl send data/pdf/*.pdf eval/questions.jsonl scripts/embed_pages_gpu.py   -> receive on pod into /workspace/input
    #   (or scp -P <port> ... root@<ip>:/workspace/input/)
    python /workspace/input/embed_pages_gpu.py --input /workspace/input --out /workspace/smoke --smoke   # 1 min check first
    python /workspace/input/embed_pages_gpu.py --input /workspace/input --out /workspace/embeddings
    # download /workspace/embeddings.zip (runpodctl send embeddings.zip on the pod, receive locally into data/embeddings/)

Resume-safe: existing .npy files are skipped, so a crashed run can be restarted.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pypdfium2 as pdfium
import torch
from transformers import AutoModel

MODEL_ID = "nvidia/llama-nemotron-colembed-vl-3b-v2"  # ViDoRe V3 rank 6; 4b/8b variants need more VRAM
CORPUS = ["Q1.pdf", "Q2.pdf", "Q3.pdf", "20F_extract.pdf", "Q1_deck.pdf", "Q2_deck.pdf", "Q3_deck.pdf"]


def load_model(device: str):
    if os.environ.get("HF_TOKEN"):
        from huggingface_hub import login
        login(token=os.environ["HF_TOKEN"])
    model = AutoModel.from_pretrained(
        MODEL_ID, device_map=device, trust_remote_code=True, torch_dtype=torch.bfloat16,
    ).eval()
    # internal ViT is fp32-pinned; cast float params/buffers only (see CLAUDE.md known issues)
    float_dtypes = (torch.float32, torch.float16, torch.bfloat16, torch.float64)
    for p in model.parameters():
        if p.dtype in float_dtypes:
            p.data = p.data.to(torch.bfloat16)
    for b in model.buffers():
        if b.dtype in float_dtypes:
            b.data = b.data.to(torch.bfloat16)
    return model


def as_list(out):
    """forward_images / forward_queries -> list of per-item tensors."""
    if isinstance(out, list):
        return out
    if hasattr(out, "dim") and out.dim() == 3:
        return [out[i] for i in range(out.shape[0])]
    return [out]


def to_fp16(t: torch.Tensor) -> np.ndarray:
    return t.detach().to(torch.float16).cpu().numpy()


def render_pages(pdf_path: Path, scale: float):
    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        for i in range(len(pdf)):
            yield i + 1, pdf[i].render(scale=scale).to_pil().convert("RGB")
    finally:
        pdf.close()


def embed_pages(model, inp: Path, out: Path, scale: float, batch: int, manifest: dict,
                sources: list[str] = CORPUS, max_pages: int | None = None) -> None:
    t0, done = time.time(), 0
    for source in sources:
        out_dir = out / "pages" / source
        out_dir.mkdir(parents=True, exist_ok=True)
        pending: list[tuple[int, object]] = []

        def flush() -> None:
            nonlocal done
            if not pending:
                return
            with torch.no_grad():
                embs = as_list(model.forward_images([im for _, im in pending], batch_size=len(pending)))
            for (page, _), emb in zip(pending, embs):
                arr = to_fp16(emb)
                np.save(out_dir / f"{page}.npy", arr)
                manifest["pages"].append({"source": source, "page": page, "n_patches": int(arr.shape[0])})
                done += 1
            pending.clear()

        for page, img in render_pages(inp / source, scale):
            if max_pages and page > max_pages:
                break
            f = out_dir / f"{page}.npy"
            if f.exists():
                arr = np.load(f, mmap_mode="r")
                manifest["pages"].append({"source": source, "page": page, "n_patches": int(arr.shape[0])})
                done += 1
                continue
            pending.append((page, img))
            if len(pending) >= batch:
                flush()
        flush()
        print(f"{source}: done  (total {done} pages, {time.time() - t0:.0f}s)", flush=True)
    n = [p["n_patches"] for p in manifest["pages"]]
    print(f"pages {len(n)}  patches/page min {min(n)} mean {sum(n) / len(n):.0f} max {max(n)}")


def embed_queries(model, questions: Path, out: Path, manifest: dict) -> None:
    (out / "queries").mkdir(parents=True, exist_ok=True)
    qs = [json.loads(l) for l in questions.read_text(encoding="utf-8").splitlines() if l.strip()]
    items = [(f"{q['id']}_ko", q["q_ko"]) for q in qs] + [(f"{q['id']}_en", q["q_en"]) for q in qs]
    with torch.no_grad():
        embs = as_list(model.forward_queries([t for _, t in items], batch_size=8))
    for (name, text), emb in zip(items, embs):
        arr = to_fp16(emb)
        np.save(out / "queries" / f"{name}.npy", arr)
        manifest["queries"].append({"id": name, "n_tokens": int(arr.shape[0]), "text": text})
    print(f"{len(items)} queries embedded")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True, help="dir with the 7 corpus PDFs and questions.jsonl")
    ap.add_argument("--out", type=Path, default=Path("embeddings"))
    ap.add_argument("--scale", type=float, default=1.5, help="pypdfium2 render scale (1.5 ~ 150 DPI)")
    ap.add_argument("--batch", type=int, default=2, help="images per forward_images call (2 on 16 GB, 4 on 24 GB+)")
    ap.add_argument("--no-zip", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="2 pages of Q1.pdf + all queries, no zip: verify the stack before the full run")
    a = ap.parse_args()

    missing = [n for n in CORPUS + ["questions.jsonl"] if not (a.input / n).exists()]
    if missing:
        raise SystemExit(f"missing in {a.input}: {missing}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {device}  {torch.cuda.get_device_name(0) if device == 'cuda' else ''}")
    model = load_model(device)

    manifest = {"model_id": MODEL_ID, "render_scale": a.scale, "dim": 3072, "pages": [], "queries": []}
    if a.smoke:
        embed_pages(model, a.input, a.out, a.scale, a.batch, manifest, sources=["Q1.pdf"], max_pages=2)
    else:
        embed_pages(model, a.input, a.out, a.scale, a.batch, manifest)
    embed_queries(model, a.input / "questions.jsonl", a.out, manifest)
    (a.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")

    if a.smoke:
        arr = np.load(a.out / "pages" / "Q1.pdf" / "1.npy")
        assert arr.ndim == 2 and arr.shape[1] == 3072 and arr.dtype == np.float16, f"unexpected page embedding {arr.shape} {arr.dtype}"
        print(f"SMOKE OK: page emb {arr.shape} {arr.dtype}, {len(manifest['queries'])} queries")
        return
    if not a.no_zip:
        zip_path = shutil.make_archive(str(a.out), "zip", a.out)
        print(f"{zip_path}  {Path(zip_path).stat().st_size / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
