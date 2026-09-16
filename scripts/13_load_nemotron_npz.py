"""Load Nemotron page embeddings produced by notebooks/03_nemotron_batch_index.ipynb into Qdrant.

Input: embeddings.zip (or an already-extracted directory) with
    manifest.json
    pages/<source>/<page>.npy      fp16 [N_patches, 3072]
    queries/<id>_<lang>.npy        fp16 [N_tokens, 3072]  (kept on disk for the eval runner; not upserted)

Point IDs reuse NemotronVisionRetriever._page_id(source, page) -> idempotent, re-runs overwrite.
Upserts go over gRPC one page per call (~14 MB fp32 each); REST would inflate to ~100 MB JSON.

Usage:
    PYTHONIOENCODING=utf-8 python scripts/13_load_nemotron_npz.py data/embeddings/embeddings.zip
    PYTHONIOENCODING=utf-8 python scripts/13_load_nemotron_npz.py data/embeddings/extracted_dir
    PYTHONIOENCODING=utf-8 python scripts/13_load_nemotron_npz.py <zip_or_dir> [collection_name]   # default pharma_vision
"""
from __future__ import annotations

import json
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from qdrant_client.models import PointStruct

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env")

from pharma_vision_rag.retriever.nemotron import DEFAULT_COLLECTION, EMBED_DIM, NemotronVisionRetriever, _page_id  # noqa: E402

EMB_DIR = ROOT / "data" / "embeddings"


def resolve_input(arg: str) -> Path:
    p = Path(arg)
    if p.is_dir():
        return p
    if p.suffix == ".zip":
        out = EMB_DIR / p.stem
        if not (out / "manifest.json").exists():
            out.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(p) as z:
                z.extractall(out)
            print(f"extracted {p.name} -> {out}")
        return out
    raise SystemExit(f"not a zip or directory: {arg}")


def main(argv: list[str]) -> None:
    if len(argv) < 2:
        raise SystemExit(__doc__)
    root = resolve_input(argv[1])
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    print(f"model {manifest['model_id']}  render_scale {manifest['render_scale']}  "
          f"pages {len(manifest['pages'])}  queries {len(manifest['queries'])}")

    retriever = NemotronVisionRetriever(
        embedding_client=None,  # offline: no tunnel needed for upsert
        qdrant_url=os.environ.get("QDRANT_URL", "http://localhost:6335"),
        qdrant_api_key=os.environ.get("QDRANT_API_KEY") or None,
        collection=argv[2] if len(argv) > 2 else DEFAULT_COLLECTION,
    )
    retriever.ensure_collection()

    n = 0
    for entry in manifest["pages"]:
        source, page = entry["source"], int(entry["page"])
        arr = np.load(root / "pages" / source / f"{page}.npy").astype(np.float32)
        assert arr.ndim == 2 and arr.shape[1] == EMBED_DIM, f"{source} p{page}: bad shape {arr.shape}"
        retriever.qdrant.upsert(
            collection_name=retriever.collection,
            points=[PointStruct(
                id=_page_id(source, page),
                vector=arr.tolist(),
                payload={"source": source, "page": page, "n_patches": int(arr.shape[0]),
                         "render_scale": manifest["render_scale"]},
            )],
        )
        n += 1
        if n % 10 == 0 or n == len(manifest["pages"]):
            print(f"  upserted {n}/{len(manifest['pages'])}  ({source} p{page}, {arr.shape[0]} patches)")

    count = retriever.qdrant.count(retriever.collection, exact=True).count
    print(f"collection {retriever.collection}: {count} points")
    print(f"query embeddings left at {root / 'queries'} for the eval runner")


if __name__ == "__main__":
    main(sys.argv)
