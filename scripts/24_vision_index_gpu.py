"""GPU-box job (RunPod): compact vision index variants, scored on the 120 benchmark queries, best one packed for download.

Runs after `embed_pages_gpu.py --only embed` (page and query embeddings, fp16). Builds, from those per-page .npy files:
    full_fp16    every patch vector, fp16 (baseline, ~18.9 GB)
    int8         per-vector symmetric int8 + fp32 scale (~9.5 GB)
    pooled       runs of 4 adjacent vectors, mean then rescaled to the mean norm, fp16 (~4.7 GB)
    pooled_int8  pooled, then int8 (~2.4 GB)
Pooling ceiling (see retriever/vision_local.py pool_adjacent): 1-D runs in token order, not 2x2 spatial blocks.

Each variant is scored with exact MaxSim on the GPU against eval gold, same R@k as eval/runner.py (gold groups,
eval/metrics.py). Chosen = smallest variant whose R@5 is within 0.01 of full_fp16; if none, int8.
Output (--out, default vision_index_dl/): the chosen variant in the LocalVisionIndex layout, vectors.npy split into
parts (parts.json, sha256 per part) so the download can be resumed piecewise, plus vision_index_compare.json.
Locally: put every file into data/embeddings/vision_index/ and run scripts/25_check_vision_index.py (joins the parts).

Standalone on the pod: imports vision_local.py and metrics.py from its own folder (the bundle ships them flat).

    python input/embed_pages_gpu.py --input input --out out --batch 4 --only embed
    python input/24_vision_index_gpu.py --emb out --questions input/questions.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "src")]
try:  # repo checkout
    from pharma_vision_rag.eval.metrics import gold_groups, recall_at_k
    from pharma_vision_rag.retriever.vision_local import MODEL_ID, pool_adjacent, quantize_int8
except ImportError:  # flat RunPod bundle
    from metrics import gold_groups, recall_at_k
    from vision_local import MODEL_ID, pool_adjacent, quantize_int8

POOL = 4
CHUNK_ROWS = 65_536
VARIANTS = ("full_fp16", "int8", "pooled", "pooled_int8")


def page_runs(offsets: np.ndarray, limit: int) -> list[tuple[int, int]]:
    """Page index ranges [a, b) holding at most `limit` rows each (a larger page stands alone)."""
    runs, a = [], 0
    for b in range(1, len(offsets)):
        if offsets[b] - offsets[a] > limit and b - 1 > a:
            runs.append((a, b - 1))
            a = b - 1
    runs.append((a, len(offsets) - 1))
    return runs


# ─── build ────────────────────────────────────────────────────────────────

def build(emb: Path, work: Path, manifest: dict) -> None:
    pages = manifest["pages"]
    dim = manifest.get("dim", 3072)
    offsets = np.concatenate([[0], np.cumsum([p["n_patches"] for p in pages])]).astype(np.int64)
    pooled_off = np.concatenate([[0], np.cumsum([-(-p["n_patches"] // POOL) for p in pages])]).astype(np.int64)
    if all((work / v / "meta.json").exists() for v in VARIANTS):
        print("variants already built, skipping build")
        return
    need = offsets[-1] * dim * (2 + 1) + pooled_off[-1] * dim * (2 + 1)
    free = shutil.disk_usage(work.parent.resolve()).free
    print(f"build needs {need / 1e9:.1f} GB, free {free / 1e9:.1f} GB")
    if need > free:
        raise SystemExit("not enough disk: use a larger volume (100 GB recommended)")

    out = {}
    for v, n, dt in (("full_fp16", offsets[-1], np.float16), ("int8", offsets[-1], np.int8),
                     ("pooled", pooled_off[-1], np.float16), ("pooled_int8", pooled_off[-1], np.int8)):
        (work / v).mkdir(parents=True, exist_ok=True)
        out[v] = np.lib.format.open_memmap(work / v / "vectors.npy", mode="w+", dtype=dt, shape=(int(n), dim))
    scales = {v: np.lib.format.open_memmap(work / v / "scales.npy", mode="w+", dtype=np.float32, shape=(len(out[v]),))
              for v in ("int8", "pooled_int8")}

    t0 = time.time()
    for a, b in page_runs(offsets, CHUNK_ROWS):
        x = np.concatenate([np.load(emb / "pages" / p["source"] / f"{p['page']}.npy") for p in pages[a:b]])
        lo, hi, plo, phi = offsets[a], offsets[b], pooled_off[a], pooled_off[b]
        out["full_fp16"][lo:hi] = x
        out["int8"][lo:hi], scales["int8"][lo:hi] = quantize_int8(x)
        pooled, _ = pool_adjacent(x, offsets[a:b + 1] - lo, POOL)
        out["pooled"][plo:phi] = pooled
        out["pooled_int8"][plo:phi], scales["pooled_int8"][plo:phi] = quantize_int8(pooled)
        print(f"  built pages {b}/{len(pages)}  ({time.time() - t0:.0f}s)", flush=True)

    page_list = [[p["source"], p["page"]] for p in pages]
    for v in VARIANTS:
        out[v].flush()
        off = pooled_off if v.startswith("pooled") else offsets
        np.save(work / v / "offsets.npy", off)
        (work / v / "pages.json").write_text(json.dumps(page_list, ensure_ascii=False), encoding="utf-8")
        is_int8 = v.endswith("int8")
        (work / v / "meta.json").write_text(json.dumps({
            "variant": v, "dim": dim, "dtype": "int8" if is_int8 else "float16",
            "scales": "scales.npy" if is_int8 else None, "pool_factor": POOL if v.startswith("pooled") else 1,
            "model_id": manifest.get("model_id", MODEL_ID), "render_scale": manifest.get("render_scale"),
            "n_pages": len(pages), "n_vectors": int(off[-1])}, indent=1), encoding="utf-8")
    for v in ("full_fp16", "pooled"):
        (work / v / "scales.npy").unlink(missing_ok=True)
    print(f"built {len(VARIANTS)} variants in {time.time() - t0:.0f}s")


# ─── score ────────────────────────────────────────────────────────────────

def load_variant(d: Path, device):
    """Vectors on the GPU when they fit, else left on the host and streamed per chunk."""
    import torch
    vec = np.load(d / "vectors.npy", mmap_mode="r")
    sc = np.load(d / "scales.npy") if (d / "scales.npy").exists() else None
    off = np.load(d / "offsets.npy")
    need = vec.nbytes + 6 * CHUNK_ROWS * vec.shape[1] * 2 + 4e9           # vectors + chunk scratch + headroom
    if need < torch.cuda.mem_get_info(device)[0]:
        gvec = torch.empty(vec.shape, dtype=torch.int8 if vec.dtype == np.int8 else torch.float16, device=device)
        for a in range(0, len(vec), CHUNK_ROWS * 4):
            gvec[a:a + CHUNK_ROWS * 4] = torch.from_numpy(np.ascontiguousarray(vec[a:a + CHUNK_ROWS * 4])).to(device)
        resident = "gpu"
    else:
        gvec, resident = vec, "host (streamed)"
    gsc = torch.from_numpy(sc).to(device) if sc is not None else None
    return gvec, gsc, off, resident


def maxsim(vec, scales, offsets: np.ndarray, q, qid, n_queries: int):
    """[pages, n_queries] exact MaxSim. q: [tokens, dim] fp16 on GPU, qid: query index of every token."""
    import torch
    dev = q.device
    res = torch.empty(len(offsets) - 1, n_queries, dtype=torch.float32, device=dev)
    for a, b in page_runs(offsets, CHUNK_ROWS):
        lo, hi = int(offsets[a]), int(offsets[b])
        rows = vec[lo:hi] if isinstance(vec, torch.Tensor) else torch.from_numpy(np.ascontiguousarray(vec[lo:hi])).to(dev)
        rows = rows.to(torch.float16)
        if scales is not None:                                            # dequantize: int8 * per-vector scale
            rows = rows * scales[lo:hi, None].to(torch.float16)
        sims = (rows @ q.T).float()                                       # [rows, tokens]
        seg = torch.from_numpy(np.repeat(np.arange(b - a), np.diff(offsets[a:b + 1]))).to(dev)
        best = torch.full((b - a, q.shape[0]), -torch.inf, device=dev)
        best.scatter_reduce_(0, seg[:, None].expand_as(sims), sims, "amax")  # best vector per page and token
        res[a:b] = torch.zeros(b - a, n_queries, device=dev).index_add_(1, qid, best)
    return res


def evaluate(work: Path, emb: Path, manifest: dict, questions: Path) -> list[dict]:
    import torch
    dev = torch.device("cuda")
    qmeta = manifest["queries"]
    qs = {q["id"]: q for q in (json.loads(l) for l in questions.read_text(encoding="utf-8").splitlines() if l.strip())}
    mats = [np.load(emb / "queries" / f"{m['id']}.npy") for m in qmeta]
    q_all = torch.from_numpy(np.concatenate(mats)).to(dev, torch.float16)
    qid_all = torch.from_numpy(np.repeat(np.arange(len(mats)), [len(m) for m in mats])).to(dev)
    page_list = json.loads((work / "full_fp16" / "pages.json").read_text(encoding="utf-8"))

    rows, base_top = [], None
    for v in VARIANTS:
        vec, sc, off, resident = load_variant(work / v, dev)
        scores = maxsim(vec, sc, off, q_all, qid_all, len(mats)).cpu().numpy()   # [pages, queries]
        top = np.argsort(-scores, axis=0)[:5].T                                   # [queries, 5]
        r1, r5, r5a = [], [], []
        for j, m in enumerate(qmeta):
            q = qs[m["id"].rsplit("_", 1)[0]]
            ranked = [tuple(page_list[i]) for i in top[j]]
            groups = gold_groups(q)
            r1.append(recall_at_k(ranked, groups, 1))
            r5.append(recall_at_k(ranked, groups, 5))
            if q["type"] == "A":
                r5a.append(r5[-1])
        base_top = top if base_top is None else base_top
        overlap = float(np.mean([len(set(a) & set(b)) / 5 for a, b in zip(top, base_top)]))

        times = []
        n_timed = len(mats) if resident == "gpu" else 5                          # streamed = disk-bound, few samples
        for j in range(n_timed + 1):                                           # first call is warm-up
            qj = torch.from_numpy(mats[j % len(mats)]).to(dev, torch.float16)
            torch.cuda.synchronize()
            t = time.perf_counter()
            maxsim(vec, sc, off, qj, torch.zeros(len(qj), dtype=torch.long, device=dev), 1)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t)
        size = sum(f.stat().st_size for f in (work / v).glob("*.npy"))
        rows.append({"variant": v, "size_gb": round(size / 1e9, 2), "n_vectors": int(off[-1]),
                     "r@1": round(float(np.mean(r1)), 4), "r@5": round(float(np.mean(r5)), 4),
                     "r@5_chart": round(float(np.mean(r5a)), 4), "top5_overlap_vs_fp16": round(overlap, 4),
                     "ms_per_query": round(1000 * float(np.mean(times[1:])), 1), "resident": resident})
        print(rows[-1], flush=True)
        del vec, sc
        torch.cuda.empty_cache()
    return rows


# ─── pack ─────────────────────────────────────────────────────────────────

def choose(rows: list[dict]) -> str:
    base = next(r for r in rows if r["variant"] == "full_fp16")
    ok = [r for r in rows if r["variant"] != "full_fp16" and r["r@5"] >= base["r@5"] - 0.01]
    return min(ok, key=lambda r: r["size_gb"])["variant"] if ok else "int8"


def pack(src: Path, out: Path, part_gb: float) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.name != "vectors.npy":
            shutil.copy2(f, out / f.name)
    parts, step = [], int(part_gb * 1e9)
    with open(src / "vectors.npy", "rb") as fin:
        for i in range(10_000):
            buf = fin.read(step)
            if not buf:
                break
            name = f"vectors.npy.part{i:03d}"
            (out / name).write_bytes(buf)
            parts.append({"name": name, "bytes": len(buf), "sha256": hashlib.sha256(buf).hexdigest()})
    (out / "parts.json").write_text(json.dumps({"file": "vectors.npy", "parts": parts}, indent=1), encoding="utf-8")
    total = sum(f.stat().st_size for f in out.iterdir())
    print(f"packed {src.name} -> {out}: {len(parts)} parts, {len(list(out.iterdir()))} files, {total / 1e9:.2f} GB")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emb", type=Path, default=Path("out"), help="embed_pages_gpu.py output (manifest, pages, queries)")
    ap.add_argument("--questions", type=Path, default=Path("input/questions.jsonl"))
    ap.add_argument("--work", type=Path, default=Path("vidx_work"), help="all four variants (~36 GB)")
    ap.add_argument("--out", type=Path, default=Path("vision_index_dl"), help="what to download")
    ap.add_argument("--part-gb", type=float, default=2.0)
    ap.add_argument("--variant", choices=VARIANTS[1:], help="override the automatic choice")
    a = ap.parse_args()

    manifest = json.loads((a.emb / "manifest.json").read_text(encoding="utf-8"))
    a.work.mkdir(parents=True, exist_ok=True)
    build(a.emb, a.work, manifest)
    rows = evaluate(a.work, a.emb, manifest, a.questions)
    chosen = a.variant or choose(rows)

    print(f"\n{'variant':<13}{'GB':>7}{'R@1':>7}{'R@5':>7}{'R@5 A':>7}{'top5=fp16':>11}{'ms/q':>8}  resident")
    for r in rows:
        print(f"{r['variant']:<13}{r['size_gb']:>7.2f}{r['r@1']:>7.3f}{r['r@5']:>7.3f}{r['r@5_chart']:>7.3f}"
              f"{r['top5_overlap_vs_fp16']:>11.3f}{r['ms_per_query']:>8.1f}  {r['resident']}")
    print(f"chosen: {chosen}")
    if a.out.exists():
        shutil.rmtree(a.out)
    pack(a.work / chosen, a.out, a.part_gb)
    import torch
    report = {"chosen": chosen, "rule": "smallest variant with R@5 >= full_fp16 R@5 - 0.01, else int8",
              "gpu": torch.cuda.get_device_name(0), "n_queries": len(manifest["queries"]), "rows": rows}
    for p in (a.out / "vision_index_compare.json", a.work / "vision_index_compare.json"):
        p.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"download every file in {a.out}/ into data/embeddings/vision_index/, then run scripts/25_check_vision_index.py")


if __name__ == "__main__":
    main()
