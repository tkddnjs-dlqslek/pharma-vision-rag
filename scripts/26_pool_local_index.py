"""Shrink a downloaded vision index locally: int8 (9.5 GB) -> pooled_int8 (2.4 GB), no GPU, bounded RAM.

Same pooling and quantization as scripts/24_vision_index_gpu.py, applied to an index that is already int8, so the
result carries one extra rounding step. Use it when the pod wrote the wrong variant and re-embedding is not worth it;
check the cost with scripts/25_check_vision_index.py afterwards (it prints R@5 against the reference rankings).

    PYTHONIOENCODING=utf-8 PYTHONPATH=src .venv/Scripts/python.exe scripts/26_pool_local_index.py \
        --src data/embeddings/vision_index --out data/embeddings/vision_index_pooled
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from pharma_vision_rag.retriever.vision_local import pool_adjacent, quantize_int8  # noqa: E402

PAGES_PER_CHUNK = 40  # ~72k rows: one float32 buffer of about 0.9 GB


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=ROOT / "data/embeddings/vision_index")
    ap.add_argument("--out", type=Path, default=ROOT / "data/embeddings/vision_index_pooled")
    ap.add_argument("--pool", type=int, default=4)
    a = ap.parse_args()

    meta = json.loads((a.src / "meta.json").read_text(encoding="utf-8"))
    if meta["dtype"] != "int8" or meta.get("pool_factor", 1) != 1:
        raise SystemExit(f"expected an unpooled int8 index, got {meta['dtype']} pool={meta.get('pool_factor')}")
    vec = np.load(a.src / "vectors.npy", mmap_mode="r")
    scales = np.load(a.src / "scales.npy")
    off = np.load(a.src / "offsets.npy")
    dim = int(meta["dim"])
    n_pages = len(off) - 1
    pooled_off = np.concatenate([[0], np.cumsum([-(-int(off[i + 1] - off[i]) // a.pool) for i in range(n_pages)])])
    a.out.mkdir(parents=True, exist_ok=True)

    out_vec = np.lib.format.open_memmap(a.out / "vectors.npy", mode="w+", dtype=np.int8,
                                        shape=(int(pooled_off[-1]), dim))
    out_sc = np.lib.format.open_memmap(a.out / "scales.npy", mode="w+", dtype=np.float32, shape=(int(pooled_off[-1]),))
    t0 = time.time()
    for a0 in range(0, n_pages, PAGES_PER_CHUNK):
        b0 = min(a0 + PAGES_PER_CHUNK, n_pages)
        lo, hi = int(off[a0]), int(off[b0])
        x = vec[lo:hi].astype(np.float32) * scales[lo:hi, None]          # dequantize
        pooled, _ = pool_adjacent(x, off[a0:b0 + 1] - lo, a.pool)
        plo, phi = int(pooled_off[a0]), int(pooled_off[b0])
        out_vec[plo:phi], out_sc[plo:phi] = quantize_int8(pooled)
        print(f"  pages {b0}/{n_pages}  ({time.time() - t0:.0f}s)", flush=True)
    out_vec.flush()
    out_sc.flush()

    np.save(a.out / "offsets.npy", pooled_off.astype(np.int64))
    (a.out / "pages.json").write_text((a.src / "pages.json").read_text(encoding="utf-8"), encoding="utf-8")
    meta = {**meta, "variant": "pooled_int8", "pool_factor": a.pool, "n_vectors": int(pooled_off[-1]),
            "source": "pooled locally from an int8 index by scripts/26_pool_local_index.py"}
    (a.out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    size = sum(f.stat().st_size for f in a.out.iterdir())
    print(f"{a.out}: {pooled_off[-1]} vectors, {size / 1e9:.2f} GB, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
