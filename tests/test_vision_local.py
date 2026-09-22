"""Tests for retriever/vision_local.py on a tiny synthetic index (random vectors, fake encoder, no model).

Run: PYTHONIOENCODING=utf-8 PYTHONPATH=src .venv/Scripts/python.exe tests/test_vision_local.py
 or: PYTHONIOENCODING=utf-8 PYTHONPATH=src .venv/Scripts/python.exe -m pytest tests/test_vision_local.py -q
"""
from __future__ import annotations

import json
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pharma_vision_rag.retriever import vision_local as vl  # noqa: E402

DIM = 16
PAGES = [("a.pdf", 1), ("a.pdf", 2), ("a.pdf", 3), ("b.pdf", 1), ("b.pdf", 2)]
SIZES = [9, 5, 12, 7, 8]


def make_index(d: Path, dtype: str = "float16", seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    mats = [rng.standard_normal((n, DIM)).astype(np.float32) for n in SIZES]
    vecs = np.concatenate(mats)
    if dtype == "int8":
        vecs, scales = vl.quantize_int8(vecs)
        np.save(d / "scales.npy", scales)
    else:
        vecs = vecs.astype(np.float16)
    np.save(d / "vectors.npy", vecs)
    np.save(d / "offsets.npy", np.concatenate([[0], np.cumsum(SIZES)]).astype(np.int64))
    (d / "pages.json").write_text(json.dumps(PAGES))
    (d / "meta.json").write_text(json.dumps({"variant": dtype, "dim": DIM, "dtype": dtype}))
    return mats


def brute(q: np.ndarray, page: np.ndarray) -> float:
    return float(sum(max(float(t @ v) for v in page) for t in q))


def queries(mats):
    """Query i = a few tokens cut from page i (plus noise) -> page i must rank first."""
    rng = np.random.default_rng(1)
    return {f"q{i}": m[:3] + 0.05 * rng.standard_normal((3, DIM)).astype(np.float32) for i, m in enumerate(mats)}


def test_search_order_and_scores():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        mats = make_index(Path(d))
        qs = queries(mats)
        idx = vl.LocalVisionIndex(Path(d), encoder=qs.__getitem__)
        for i, page in enumerate(PAGES):
            hits = idx.search(f"q{i}", k=5)
            assert hits[0][:2] == page, (i, hits)
            want = sorted((brute(qs[f"q{i}"], m.astype(np.float16).astype(np.float32)) for m in mats), reverse=True)
            assert np.allclose([h[2] for h in hits], want, atol=1e-2), (hits, want)


def test_small_chunks_match_one_chunk():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        mats = make_index(Path(d))
        qs = queries(mats)
        idx = vl.LocalVisionIndex(Path(d), encoder=qs.__getitem__)
        full = idx.search("q2", k=5)
        old = vl.CHUNK_ROWS
        try:
            vl.CHUNK_ROWS = 10  # forces one page per chunk, and the 12-row page alone in an oversize chunk
            assert idx.search("q2", k=5) == full
        finally:
            vl.CHUNK_ROWS = old


def test_document_filter():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        mats = make_index(Path(d))
        qs = queries(mats)
        idx = vl.LocalVisionIndex(Path(d), encoder=qs.__getitem__)
        hits = idx.search("q0", k=10, document_ids=["b.pdf"])
        assert [h[:2] for h in hits] and all(h[0] == "b.pdf" for h in hits) and len(hits) == 2
        assert idx.search("q0", k=3, document_ids=["nope.pdf"]) == []
        assert idx.search("q3", k=1, document_ids=["a.pdf", "b.pdf"])[0][:2] == ("b.pdf", 1)


def test_int8_dequantization_path():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        mats = make_index(Path(d), dtype="int8")
        qs = queries(mats)
        idx = vl.LocalVisionIndex(Path(d), encoder=qs.__getitem__)
        assert idx.scales is not None and idx.vectors.dtype == np.int8
        for i, page in enumerate(PAGES):
            hits = idx.search(f"q{i}", k=5)
            assert hits[0][:2] == page
            exact = brute(qs[f"q{i}"], mats[i])
            assert abs(hits[0][2] - exact) < 0.02 * abs(exact) + 0.05, (hits[0][2], exact)
        q, s = vl.quantize_int8(mats[0])
        assert np.abs(q.astype(np.float32) * s[:, None] - mats[0]).max() <= s.max() / 2 + 1e-6


def test_available_and_bad_index():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        assert not vl.LocalVisionIndex.available(Path(d))
        make_index(Path(d), dtype="int8")
        assert vl.LocalVisionIndex.available(Path(d))
        (Path(d) / "scales.npy").unlink()
        assert not vl.LocalVisionIndex.available(Path(d))


def test_pool_adjacent():
    x = np.arange(2 * 7 * 2, dtype=np.float32).reshape(14, 2) + 1
    offsets = np.array([0, 5, 14])
    pooled, off = vl.pool_adjacent(x, offsets, factor=4)
    assert off.tolist() == [0, 2, 5]                    # page 1: 4+1 rows, page 2: 4+4+1
    assert len(pooled) == 5
    grp = x[0:4]
    assert np.allclose(pooled[0] / np.linalg.norm(pooled[0]), grp.mean(0) / np.linalg.norm(grp.mean(0)))
    assert np.isclose(np.linalg.norm(pooled[0]), np.linalg.norm(grp, axis=1).mean())
    assert np.allclose(pooled[1], x[4])                  # singleton run is unchanged


if __name__ == "__main__":
    fails = 0
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception:
            fails += 1
            print(f"FAIL {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - fails}/{len(tests)} passed")
    sys.exit(1 if fails else 0)
