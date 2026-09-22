"""Bundle everything the GPU box needs into one upload: data/embeddings/runpod_input.zip.

Contents (flat): every corpus PDF from data/pdf/corpus/, eval/corpus.json, eval/questions.jsonl,
scripts/embed_pages_gpu.py, and for the compact vision index scripts/24_vision_index_gpu.py with the two numpy-only
modules it imports (retriever/vision_local.py, eval/metrics.py). Re-run after changing any of them.

Usage:
    PYTHONIOENCODING=utf-8 python scripts/16_make_runpod_bundle.py
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "embeddings" / "runpod_input.zip"


def main() -> None:
    corpus = json.loads((ROOT / "eval" / "corpus.json").read_text(encoding="utf-8"))
    files = [ROOT / "data" / "pdf" / "corpus" / d["id"] for d in corpus]
    files += [ROOT / "eval" / "corpus.json", ROOT / "eval" / "questions.jsonl", ROOT / "scripts" / "embed_pages_gpu.py",
              ROOT / "scripts" / "24_vision_index_gpu.py", ROOT / "src" / "pharma_vision_rag" / "retriever" / "vision_local.py",
              ROOT / "src" / "pharma_vision_rag" / "eval" / "metrics.py"]
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        raise SystemExit(f"missing: {missing}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, f.name)
    print(f"{len(files)} files ({len(corpus)} PDFs, {sum(d['pages'] for d in corpus)} pages) -> {OUT}  "
          f"{OUT.stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
