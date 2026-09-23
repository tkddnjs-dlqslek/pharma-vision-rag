"""GPU-box job (RunPod): compact vision index variants, scored on the 120 benchmark queries, best one packed for download.

Two ways in. `--embed-now` (recommended) embeds the corpus in this process and fills the variants straight from RAM;
the page vectors never touch the disk. Without it, the script reads the per-page .npy files an earlier
`embed_pages_gpu.py --only embed` wrote. Disk lesson (2026-09-23): a pod whose container disk is network-backed
read those files back at 0.8 MB/s, so the two-step build needed 2+ hours and the pod died first.
Builds four variants:
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

Standalone on the pod: imports vision_local.py, metrics.py and embed_pages_gpu.py from its own folder
(the bundle ships them flat).

    python input/24_vision_index_gpu.py --embed-now --input input --emb out --batch 4   # one pass, recommended
    python input/embed_pages_gpu.py --input input --out out --batch 4 --only embed      # two steps, resumable
    python input/24_vision_index_gpu.py --emb out --questions input/questions.jsonl

`--embed-now` writes nothing but the chosen variant: the four variants are scored straight out of RAM and only the
winner is packed (2.4 to 9.5 GB instead of 36 GB). It still writes the 120 query embeddings and manifest.json to
--emb (a few MB) because scoring needs them. `--save-pages` adds the ~19 GB of per-page .npy files and
`--save-variants` the 36 GB of variants; both make a crash resumable at the cost of the slow disk.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
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


# ─── pod check ────────────────────────────────────────────────────────────

def free_ram() -> float | None:
    try:
        for line in open("/proc/meminfo", encoding="utf-8"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    try:
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, ValueError, OSError):
        return None


def throughput_probe(work: Path, mb: int) -> tuple[float, float]:
    """MB/s written (fsynced) and read back cold. This is what a network-backed container disk fails."""
    buf, f = os.urandom(1 << 20) * 8, work / ".probe.bin"          # 8 MB, incompressible
    fd = os.open(f, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_BINARY", 0))
    t = time.time()
    try:
        for _ in range(max(1, mb // 8)):
            os.write(fd, buf)
        os.fsync(fd)
    finally:
        os.close(fd)
    write = mb / max(time.time() - t, 1e-9)
    fd = os.open(f, os.O_RDONLY | getattr(os, "O_BINARY", 0))
    try:
        if hasattr(os, "posix_fadvise"):        # drop it from the page cache, else we time RAM
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        t = time.time()
        while os.read(fd, 1 << 23):
            pass
    finally:
        os.close(fd)
    f.unlink()
    return write, mb / max(time.time() - t, 1e-9)


def diagnostics(work: Path, probe_mb: int) -> None:
    """Everything that decides whether this pod is worth two hours, printed in the first minute."""
    try:
        import torch
        free, total = torch.cuda.mem_get_info(0)
        print(f"gpu       {torch.cuda.get_device_name(0)}  VRAM {free / 1e9:.1f} / {total / 1e9:.1f} GB free")
    except Exception as e:                                          # noqa: BLE001 - diagnostics must never abort the run
        print(f"gpu       unavailable ({type(e).__name__}: {e})")
    ram = free_ram()
    print(f"ram       {ram / 1e9:.0f} GB available" if ram else "ram       unknown")
    print(f"disk      {shutil.disk_usage(work).free / 1e9:.0f} GB free at {work.resolve()}")
    if probe_mb > 0:
        w, r = throughput_probe(work, probe_mb)
        warn = "  <- SLOW: network-backed disk, move --work to a Volume Disk" if min(w, r) < 100 else ""
        print(f"disk i/o  {w:.0f} MB/s write, {r:.0f} MB/s read cold ({probe_mb} MB){warn}", flush=True)


# ─── one pass: embed straight into RAM ────────────────────────────────────

def embed_now(inp: Path, emb: Path, scale: float, batch: int, save_pages: bool, every: int, model=None):
    """Embed every corpus page and the 120 queries in this process. Returns (manifest, {(doc, page): fp16 array}).

    The page arrays are kept in a dict (~19 GB for the 1,709-page corpus) and handed to `build` as its loader, which
    pops each one as it copies it into the variants. Peak RAM is therefore about the same as the two-step build
    (~36 GB of variants), and no page vector is ever written or read back.
    """
    import embed_pages_gpu as ep                                    # scripts/ next door, or flat in the bundle
    import pypdfium2 as pdfium

    corpus = json.loads((inp / "corpus.json").read_text(encoding="utf-8"))
    docs = [d["id"] for d in corpus]
    missing = [n for n in docs + ["questions.jsonl"] if not (inp / n).exists()]
    if missing:
        raise SystemExit(f"missing in {inp}: {missing}")
    total = sum(d["pages"] for d in corpus)
    emb.mkdir(parents=True, exist_ok=True)
    model = model or ep.load_model()
    queries = ep.embed_queries(model, inp / "questions.jsonl", emb)  # tiny, and fails fast on a broken questions file

    cache, pages, t0, last = {}, [], time.time(), 0
    for doc in docs:
        pdf = pdfium.PdfDocument(str(inp / doc))
        n = len(pdf)
        for i in range(0, n, batch):
            part = list(range(i + 1, min(i + batch, n) + 1))
            for p, arr in zip(part, ep.embed_page_batch(model, pdf, part, scale)):
                cache[(doc, p)] = arr
                pages.append({"source": doc, "page": p, "n_patches": int(arr.shape[0])})
                if save_pages:
                    (emb / "pages" / doc).mkdir(parents=True, exist_ok=True)
                    np.save(emb / "pages" / doc / f"{p}.npy", arr)
            if len(pages) - last >= every:
                last, el = len(pages), max(time.time() - t0, 1e-9)
                rate = len(pages) / el
                print(f"  {len(pages)}/{total} pages  {rate:.2f} p/s  {el / 60:.1f}m elapsed  "
                      f"ETA {max(total - len(pages), 0) / rate / 60:.0f}m", flush=True)
        pdf.close()
        print(f"{doc}: {n} pages  (total {len(pages)}, {time.time() - t0:.0f}s)", flush=True)
    del model
    try:
        import torch
        torch.cuda.empty_cache()                                    # hand the VRAM to the scoring step
    except Exception:                                               # noqa: BLE001 - CPU-only test runs
        pass

    counts = [p["n_patches"] for p in pages]
    print(f"pages {len(counts)}  patches/page min {min(counts)} mean {sum(counts) / len(counts):.0f} max {max(counts)}"
          f"  -> {sum(counts) * 3072 * 2 / 1e9:.1f} GB fp16 in RAM")
    manifest = {"model_id": ep.MODEL_ID, "render_scale": scale, "dim": int(next(iter(cache.values())).shape[1]),
                "pages": pages, "queries": queries}
    (emb / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    return manifest, cache


# ─── build ────────────────────────────────────────────────────────────────

def build(work: Path, manifest: dict, load_page, save: bool = True) -> dict | None:
    """Fill the four variants in RAM. `load_page(page dict) -> [n_patches, dim] fp16`.

    save=True writes all four to `work` and returns None, freeing each array as it is written (two-step path: the
    arrays came off disk anyway). save=False writes nothing and returns {variant: {vectors, scales, offsets, pages,
    meta}} for scoring straight out of RAM; only the chosen variant is then written, by `pack`. That matters on a
    slow pod: writing all four cost about an hour at the ~10 MB/s measured on 2026-09-23.
    """
    pages = manifest["pages"]
    dim = manifest.get("dim", 3072)
    offsets = np.concatenate([[0], np.cumsum([p["n_patches"] for p in pages])]).astype(np.int64)
    pooled_off = np.concatenate([[0], np.cumsum([-(-p["n_patches"] // POOL) for p in pages])]).astype(np.int64)
    if save and all((work / v / "meta.json").exists() for v in VARIANTS):
        print("variants already built, skipping build")
        return None
    need = (offsets[-1] * dim * 3 + pooled_off[-1] * dim * 3) if save else int(offsets[-1] * (dim + 4) * 1.1)
    free = shutil.disk_usage(work.parent.resolve()).free
    print(f"build needs {need / 1e9:.1f} GB on disk ({'all four variants' if save else 'the chosen one only'}), "
          f"free {free / 1e9:.1f} GB")
    if need > free:
        raise SystemExit("not enough disk: use a larger volume (100 GB recommended)")

    # Filled in RAM (~36 GB; the pod has far more). Writing through open_memmap on the RunPod overlay disk stalled:
    # writeback of the dirty mmap pages dropped to ~100 KB/s once the page cache filled.
    out = {}
    for v, n, dt in (("full_fp16", offsets[-1], np.float16), ("int8", offsets[-1], np.int8),
                     ("pooled", pooled_off[-1], np.float16), ("pooled_int8", pooled_off[-1], np.int8)):
        if save:
            (work / v).mkdir(parents=True, exist_ok=True)
        out[v] = np.empty((int(n), dim), dtype=dt)
    scales = {v: np.empty(len(out[v]), dtype=np.float32) for v in ("int8", "pooled_int8")}

    t0 = time.time()
    for a, b in page_runs(offsets, CHUNK_ROWS):
        x = np.concatenate([load_page(p) for p in pages[a:b]])
        lo, hi, plo, phi = offsets[a], offsets[b], pooled_off[a], pooled_off[b]
        out["full_fp16"][lo:hi] = x
        out["int8"][lo:hi], scales["int8"][lo:hi] = quantize_int8(x)
        pooled, _ = pool_adjacent(x, offsets[a:b + 1] - lo, POOL)
        out["pooled"][plo:phi] = pooled
        out["pooled_int8"][plo:phi], scales["pooled_int8"][plo:phi] = quantize_int8(pooled)
        print(f"  built pages {b}/{len(pages)}  ({time.time() - t0:.0f}s)", flush=True)

    page_list = [[p["source"], p["page"]] for p in pages]
    built = {}
    for v in VARIANTS:
        off = pooled_off if v.startswith("pooled") else offsets
        is_int8 = v.endswith("int8")
        meta = {"variant": v, "dim": dim, "dtype": "int8" if is_int8 else "float16",
                "scales": "scales.npy" if is_int8 else None, "pool_factor": POOL if v.startswith("pooled") else 1,
                "model_id": manifest.get("model_id", MODEL_ID), "render_scale": manifest.get("render_scale"),
                "n_pages": len(pages), "n_vectors": int(off[-1])}
        if not save:
            built[v] = {"vectors": out[v], "scales": scales.get(v), "offsets": off, "pages": page_list, "meta": meta}
        else:                                       # pop: each array is freed as soon as it is on disk
            t1 = time.time()
            np.save(work / v / "vectors.npy", out.pop(v))
            if v in scales:
                np.save(work / v / "scales.npy", scales.pop(v))
            print(f"  saved {v} ({time.time() - t1:.0f}s)", flush=True)
            np.save(work / v / "offsets.npy", off)
            (work / v / "pages.json").write_text(json.dumps(page_list, ensure_ascii=False), encoding="utf-8")
            (work / v / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"built {len(VARIANTS)} variants in {time.time() - t0:.0f}s"
          f"{'' if save else ' (kept in RAM, only the chosen one gets written)'}")
    return None if save else built


# ─── score ────────────────────────────────────────────────────────────────

def variant_parts(src) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, list]:
    """(vectors, scales, offsets, page list) from a variant directory or an in-RAM variant dict from build()."""
    if isinstance(src, dict):
        return src["vectors"], src["scales"], src["offsets"], src["pages"]
    return (np.load(src / "vectors.npy", mmap_mode="r"),
            np.load(src / "scales.npy") if (src / "scales.npy").exists() else None,
            np.load(src / "offsets.npy"),
            json.loads((src / "pages.json").read_text(encoding="utf-8")))


def variant_size(src) -> int:
    """Bytes the variant's .npy files take (or would take: np.save adds a 128-byte header to each)."""
    if not isinstance(src, dict):
        return sum(f.stat().st_size for f in src.glob("*.npy"))
    return sum(a.nbytes + 128 for a in (src["vectors"], src["offsets"], src["scales"]) if a is not None)


def load_variant(src, device):
    """Vectors on the GPU when they fit, else left on the host (an mmap or the array build kept) and streamed."""
    import torch
    vec, sc, off, _ = variant_parts(src)
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


def evaluate(sources: dict, emb: Path, manifest: dict, questions: Path) -> list[dict]:
    """`sources`: {variant: directory or in-RAM variant dict}."""
    import torch
    dev = torch.device("cuda")
    qmeta = manifest["queries"]
    qs = {q["id"]: q for q in (json.loads(l) for l in questions.read_text(encoding="utf-8").splitlines() if l.strip())}
    mats = [np.load(emb / "queries" / f"{m['id']}.npy") for m in qmeta]
    q_all = torch.from_numpy(np.concatenate(mats)).to(dev, torch.float16)
    qid_all = torch.from_numpy(np.repeat(np.arange(len(mats)), [len(m) for m in mats])).to(dev)
    page_list = variant_parts(sources["full_fp16"])[3]

    rows, base_top = [], None
    for v in VARIANTS:
        vec, sc, off, resident = load_variant(sources[v], dev)
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
        size = variant_size(sources[v])
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


BLOCK = 1 << 26  # 64 MB: the most that is ever copied while packing


def file_blocks(p: Path):
    with open(p, "rb") as f:
        while buf := f.read(BLOCK):
            yield buf


def array_blocks(arr: np.ndarray):
    """What np.save(arr) would write, in blocks, without a second copy of the array."""
    head = io.BytesIO()
    np.lib.format.write_array_header_1_0(head, np.lib.format.header_data_from_array_1_0(arr))
    yield head.getvalue()
    step = max(1, BLOCK // max(arr.strides[0], 1))              # whole rows: C-contiguous, so slices are copy-free
    for a in range(0, len(arr), step):
        yield arr[a:a + step].tobytes()


def write_parts(blocks, out: Path, step: int) -> list[dict]:
    """Stream `blocks` into vectors.npy.partNNN files of `step` bytes, hashing as they go."""
    parts: list[dict] = []
    fh = h = name = None
    for blk in blocks:
        pos = 0
        while pos < len(blk):
            if fh is None:
                name = f"vectors.npy.part{len(parts):03d}"
                fh, h = open(out / name, "wb"), hashlib.sha256()
            take = blk[pos:pos + step - fh.tell()]
            fh.write(take)
            h.update(take)
            pos += len(take)
            if fh.tell() >= step:
                parts.append({"name": name, "bytes": fh.tell(), "sha256": h.hexdigest()})
                fh.close()
                fh = None
    if fh is not None:
        parts.append({"name": name, "bytes": fh.tell(), "sha256": h.hexdigest()})
        fh.close()
    return parts


def pack(src, out: Path, part_gb: float) -> None:
    """Write one variant in the download layout. `src`: a variant directory or an in-RAM variant dict."""
    out.mkdir(parents=True, exist_ok=True)
    if isinstance(src, dict):
        np.save(out / "offsets.npy", src["offsets"])
        if src["scales"] is not None:
            np.save(out / "scales.npy", src["scales"])
        (out / "pages.json").write_text(json.dumps(src["pages"], ensure_ascii=False), encoding="utf-8")
        (out / "meta.json").write_text(json.dumps(src["meta"], indent=1), encoding="utf-8")
        blocks, label = array_blocks(src["vectors"]), src["meta"]["variant"]
    else:
        for f in src.iterdir():
            if f.name != "vectors.npy":
                shutil.copy2(f, out / f.name)
        blocks, label = file_blocks(src / "vectors.npy"), src.name
    parts = write_parts(blocks, out, int(part_gb * 1e9))
    (out / "parts.json").write_text(json.dumps({"file": "vectors.npy", "parts": parts}, indent=1), encoding="utf-8")
    total = sum(f.stat().st_size for f in out.iterdir())
    print(f"packed {label} -> {out}: {len(parts)} parts, {len(list(out.iterdir()))} files, {total / 1e9:.2f} GB")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emb", type=Path, default=Path("out"), help="embed_pages_gpu.py output (manifest, pages, queries)")
    ap.add_argument("--questions", type=Path, default=Path("input/questions.jsonl"))
    ap.add_argument("--work", type=Path, default=Path("vidx_work"), help="all four variants (~36 GB, --embed-now: unused unless --save-variants)")
    ap.add_argument("--out", type=Path, default=Path("vision_index_dl"), help="what to download")
    ap.add_argument("--part-gb", type=float, default=2.0)
    ap.add_argument("--variant", choices=VARIANTS[1:], help="override the automatic choice")
    ap.add_argument("--embed-now", action="store_true",
                    help="embed the corpus here instead of reading --emb/pages/ (page vectors stay in RAM)")
    ap.add_argument("--input", type=Path, help="--embed-now: bundle dir with the corpus PDFs, corpus.json, questions.jsonl")
    ap.add_argument("--scale", type=float, default=1.5, help="--embed-now: pypdfium2 render scale (1.5 ~ 150 DPI)")
    ap.add_argument("--batch", type=int, default=4, help="--embed-now: images per forward_images call")
    ap.add_argument("--save-pages", action="store_true",
                    help="--embed-now: also write pages/<doc>/<page>.npy (resumable, but pays the slow disk)")
    ap.add_argument("--save-variants", action="store_true",
                    help="--embed-now: write all four variants to --work (~36 GB) instead of only the chosen one")
    ap.add_argument("--progress-every", type=int, default=25, help="--embed-now: pages between progress lines")
    ap.add_argument("--probe-mb", type=int, default=512, help="disk throughput probe size, 0 to skip")
    a = ap.parse_args()

    a.work.mkdir(parents=True, exist_ok=True)
    if a.embed_now and all((a.work / v / "meta.json").exists() for v in VARIANTS) and (a.emb / "manifest.json").exists():
        print("variants already built, skipping the embedding pass")
        a.embed_now = False
    if a.embed_now:
        if a.input is None:
            ap.error("--embed-now needs --input <bundle dir>")
        diagnostics(a.work, a.probe_mb)
        if not a.save_pages:
            print("WARNING: --save-pages is off, so nothing is resumable: a crash before the chosen variant is "
                  "packed loses the whole embedding run (that is the point, the disk is the slow part)", flush=True)
        if not a.questions.exists():
            a.questions = a.input / "questions.jsonl"
        manifest, cache = embed_now(a.input, a.emb, a.scale, a.batch, a.save_pages, a.progress_every)
        variants = build(a.work, manifest, lambda p: cache.pop((p["source"], p["page"])),  # pop: freed as it is copied
                         save=a.save_variants)
    else:
        manifest = json.loads((a.emb / "manifest.json").read_text(encoding="utf-8"))
        variants = build(a.work, manifest, lambda p: np.load(a.emb / "pages" / p["source"] / f"{p['page']}.npy"))
    sources = variants or {v: a.work / v for v in VARIANTS}
    rows = evaluate(sources, a.emb, manifest, a.questions)
    chosen = a.variant or choose(rows)

    print(f"\n{'variant':<13}{'GB':>7}{'R@1':>7}{'R@5':>7}{'R@5 A':>7}{'top5=fp16':>11}{'ms/q':>8}  resident")
    for r in rows:
        print(f"{r['variant']:<13}{r['size_gb']:>7.2f}{r['r@1']:>7.3f}{r['r@5']:>7.3f}{r['r@5_chart']:>7.3f}"
              f"{r['top5_overlap_vs_fp16']:>11.3f}{r['ms_per_query']:>8.1f}  {r['resident']}")
    print(f"chosen: {chosen}")
    if a.out.exists():
        shutil.rmtree(a.out)
    pack(sources[chosen], a.out, a.part_gb)
    import torch
    report = {"chosen": chosen, "rule": "smallest variant with R@5 >= full_fp16 R@5 - 0.01, else int8",
              "gpu": torch.cuda.get_device_name(0), "n_queries": len(manifest["queries"]), "rows": rows}
    (a.out / "vision_index_compare.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    if variants is None:
        (a.work / "vision_index_compare.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"download every file in {a.out}/ into data/embeddings/vision_index/, then run scripts/25_check_vision_index.py")


if __name__ == "__main__":
    main()
