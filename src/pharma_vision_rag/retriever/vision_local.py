"""Local vision page search for arbitrary questions: exact MaxSim over a compact on-disk index, no vector DB.

Index directory (written on the GPU box by scripts/24_vision_index_gpu.py, checked by scripts/25_check_vision_index.py):
    vectors.npy   [N, dim] float16 or int8, every page's patch vectors concatenated (read with mmap)
    scales.npy    [N] float32, int8 only: per-vector symmetric scale, vector ~= int8 * scale
    offsets.npy   [P + 1] int64, page i owns rows offsets[i]:offsets[i+1]
    pages.json    [[doc id, 1-based page], ...] in offsets order
    meta.json     variant, dim, dtype, model_id, render_scale, ...

    score(q, page) = sum over query tokens of max over page vectors of <q_token, v>

RAM stays bounded: vectors are scanned in page-aligned chunks of about CHUNK_ROWS rows, cast into one reused float32 buffer.
By default the int8 index itself is loaded into RAM once (2.4 GB for the pooled corpus, ``resident=False`` keeps the mmap).

Performance (pooled_int8 index, 770,759 x 3072, 1,709 pages, i5-1240P laptop with other load, stored 59-token queries):
    before: 8.9 s per full scan. Profile: int8->float32 copy 5.5 s (numpy copyto is single-threaded, plus mmap page
            faults on a box whose page cache cannot hold the file), matmul 3.2 s, scale 0.1 s, reduceat 0.2 s.
    after:  3.4 s median full scan (120 queries), 0.09 s with a one-document filter (was 0.24 s); top-5 identical on all
            120 queries, scores within 3e-7. Profile: cast 1.1 s, matmul 2.1 s, scale + reduceat 0.2 s. Changes: (1) drop
            all-zero query tokens (batch padding, they add exactly 0 to every page; the stored queries keep 26 of 59 tokens
            at the median), (2) resident index, (3) the cast runs through torch's multi-threaded copy_ when torch is
            importable (4x numpy's single-threaded copyto), (4) 16k-row chunks, one buffer kept per instance.
            The matmul stays in numpy/OpenBLAS (faster than torch's MKL here) and is the floor: this CPU sustains only
            55-90 GFLOPS in fp32, int8 GEMMs (torch._int_mm naive on CPU, fbgemm 2x faster but inexact on the query side)
            do not help, and overlapping the cast with the GEMM in a second thread gained nothing (both saturate the cores).
The query encoder (Nemotron 3B, bf16 on CPU, ~7 GB) is loaded on the first search that needs it.
Pass ``encoder=`` (str -> [tokens, dim] array) to skip the model, e.g. in tests or with a remote encoder.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent.parent  # .parent never raises (flat RunPod bundle)
MODEL_ID = "nvidia/llama-nemotron-colembed-vl-3b-v2"
CHUNK_ROWS = 16_384  # rows cast to float32 at a time (200 MB buffer at dim 3072); 4k and 64k both measured slower
RESIDENT_MAX_BYTES = 3 << 30  # load vectors.npy into RAM when it is at most this big, else mmap
_torch = None  # lazily imported for the multi-threaded cast; False when unavailable
FILES = ("vectors.npy", "offsets.npy", "pages.json", "meta.json")

Encoder = Callable[[str], np.ndarray]


def nemotron_query_encoder(model_id: str = MODEL_ID) -> Encoder:
    """Load the ColEmbed model on CPU (bf16, float params/buffers cast as in scripts/embed_pages_gpu.py)."""
    import torch
    from transformers import AutoModel

    model = AutoModel.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.bfloat16).eval()
    floats = (torch.float32, torch.float16, torch.bfloat16, torch.float64)
    for t in list(model.parameters()) + list(model.buffers()):
        if t.dtype in floats:  # internal ViT is fp32-pinned (CLAUDE.md known issues)
            t.data = t.data.to(torch.bfloat16)

    def encode(query: str) -> np.ndarray:
        with torch.no_grad():
            out = model.forward_queries([query], batch_size=1)
        t = out[0] if isinstance(out, list) or out.dim() == 3 else out
        return t.detach().to(torch.float32).cpu().numpy()
    return encode


def quantize_int8(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-vector symmetric int8: scale = max|v| / 127, v ~= q * scale."""
    x = np.asarray(x, dtype=np.float32)
    scales = np.abs(x).max(axis=1) / 127.0
    scales[scales == 0] = 1.0
    return np.rint(x / scales[:, None]).astype(np.int8), scales.astype(np.float32)


def pool_adjacent(x: np.ndarray, offsets: np.ndarray, factor: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Mean of each run of `factor` consecutive vectors inside a page, rescaled to the run's mean norm.

    ponytail: 1-D runs in the model's token order, not true 2x2 spatial pooling. The page layout (tiles,
    thumbnail, special tokens) is not stored with the embeddings, so runs can straddle row and tile edges and
    mix a non-image token into a patch run. Upgrade path: recover the tile grid from the processor and pool 2x2.
    """
    x = np.asarray(x, dtype=np.float32)
    starts = np.concatenate([np.arange(lo, hi, factor) for lo, hi in zip(offsets[:-1], offsets[1:])])
    counts = np.diff(np.append(starts, offsets[-1])).astype(np.float32)
    mean = np.add.reduceat(x, starts, axis=0) / counts[:, None]
    norms = np.add.reduceat(np.linalg.norm(x, axis=1), starts) / counts
    pooled = mean * (norms / np.maximum(np.linalg.norm(mean, axis=1), 1e-12))[:, None]
    per_page = [len(range(lo, hi, factor)) for lo, hi in zip(offsets[:-1], offsets[1:])]
    return pooled, np.concatenate([[0], np.cumsum(per_page)]).astype(np.int64)


def cast_into(dst: np.ndarray, src: np.ndarray) -> None:
    """int8 / float16 rows -> float32 buffer. torch.copy_ is multi-threaded (measured 4x numpy's copyto); numpy fallback."""
    global _torch
    if _torch is None:
        try:
            import torch as _torch  # noqa: F811
        except ImportError:
            _torch = False
    if _torch:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # a read-only memmap source triggers a "not writable" warning
            _torch.from_numpy(dst).copy_(_torch.from_numpy(np.asarray(src)))
    else:
        np.copyto(dst, src, casting="unsafe")


class LocalVisionIndex:
    @staticmethod
    def available(index_dir: Path) -> bool:
        index_dir = Path(index_dir)
        if not all((index_dir / f).exists() for f in FILES):
            return False
        meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
        return meta.get("dtype") != "int8" or (index_dir / "scales.npy").exists()

    def __init__(self, index_dir: Path = ROOT / "data/embeddings/vision_index", encoder: Encoder | None = None,
                 resident: bool | None = None) -> None:
        index_dir = Path(index_dir)
        if not self.available(index_dir):
            raise FileNotFoundError(f"no complete vision index in {index_dir} (need {', '.join(FILES)})")
        self.meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
        if resident is None:
            resident = (index_dir / "vectors.npy").stat().st_size <= RESIDENT_MAX_BYTES
        self.vectors = np.load(index_dir / "vectors.npy", mmap_mode=None if resident else "r")
        self.offsets = np.load(index_dir / "offsets.npy").astype(np.int64)
        self.pages = [(str(d), int(p)) for d, p in json.loads((index_dir / "pages.json").read_text(encoding="utf-8"))]
        self.scales = np.load(index_dir / "scales.npy").astype(np.float32) if self.vectors.dtype == np.int8 else None
        n = len(self.pages)
        if len(self.offsets) != n + 1 or self.offsets[-1] != len(self.vectors) or np.any(np.diff(self.offsets) <= 0):
            raise ValueError(f"inconsistent index in {index_dir}: {n} pages, {len(self.offsets)} offsets, "
                             f"{len(self.vectors)} vectors (every page needs at least one vector)")
        self._encoder = encoder
        self._buf = None  # float32 chunk buffer, allocated on the first search and reused

    def encode(self, query: str) -> np.ndarray:
        if self._encoder is None:
            self._encoder = nemotron_query_encoder(self.meta.get("model_id", MODEL_ID))
        return np.asarray(self._encoder(query), dtype=np.float32)

    def _chunks(self, page_ids: np.ndarray):
        """Runs of page indices that are contiguous on disk and hold at most CHUNK_ROWS rows (a big page stands alone)."""
        start = 0
        for i in range(1, len(page_ids) + 1):
            if (i == len(page_ids) or page_ids[i] != page_ids[i - 1] + 1
                    or self.offsets[page_ids[i] + 1] - self.offsets[page_ids[start]] > CHUNK_ROWS):
                yield page_ids[start:i]
                start = i

    def search(self, query: str, k: int = 5, document_ids: list[str] | None = None) -> list[tuple[str, int, float]]:
        """Top-k (doc id, page, score) by exact MaxSim over the whole index or the given documents."""
        if document_ids is None:
            page_ids = np.arange(len(self.pages))
        else:
            wanted = set(document_ids)
            page_ids = np.array([i for i, (d, _) in enumerate(self.pages) if d in wanted], dtype=np.int64)
        if len(page_ids) == 0 or k <= 0:
            return []
        q = self.encode(query)
        live = np.abs(q).max(axis=1) > 0
        q = (q[live] if live.any() else q[:1]).T                    # [dim, tokens]; all-zero (padding) tokens add 0 to every page
        scores = np.empty(len(page_ids), dtype=np.float32)
        done = 0
        need = max(CHUNK_ROWS, int(np.diff(self.offsets).max()))
        if self._buf is None or len(self._buf) < need:
            self._buf = np.empty((need, self.vectors.shape[1]), dtype=np.float32)
        buf = self._buf
        for run in self._chunks(page_ids):
            lo, hi = self.offsets[run[0]], self.offsets[run[-1] + 1]
            rows = buf[:hi - lo]
            cast_into(rows, self.vectors[lo:hi])
            sims = rows @ q                                             # [rows, tokens]
            if self.scales is not None:
                sims *= self.scales[lo:hi, None]
            per_page = np.maximum.reduceat(sims, self.offsets[run] - lo, axis=0)  # best vector per token, per page
            scores[done:done + len(run)] = per_page.sum(axis=1)
            done += len(run)
        k = min(k, len(page_ids))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top], kind="stable")]
        return [(*self.pages[page_ids[i]], float(scores[i])) for i in top]
