"""Exact late-interaction (MaxSim) vision rankings for the fixed benchmark queries. No vector DB.

v1 upserted every page's patch vectors into a Qdrant multi-vector collection. At 1,709 pages that is ~24 GB
in fp32 and brute-force MaxSim per query. The benchmark has a fixed query set, so instead: stream every
page's patch matrix from disk once, score all queries against it in one matmul, keep the top pages per query.
Exact (no ANN, no pooling), bounded memory (one page at a time), a few minutes on CPU.

    score(q, page) = sum over query tokens of max over page patches of <q_token, patch>

Input : data/embeddings/v2/  (manifest.json, pages/<doc>/<page>.npy, queries/*.npy from scripts/embed_pages_gpu.py)
Output: data/embeddings/v2/vision_rankings.json   {query text: [[doc, page, score], ...top 50]}
        read by pharma_vision_rag.eval.runner (variants: vision, hybrid)

An interactive app needs a first-stage index (mean page vector -> top-N -> MaxSim rerank); these exact
rankings are the reference to validate that approximation against.

Usage:
    PYTHONIOENCODING=utf-8 python scripts/13_score_vision_exact.py [emb_dir]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIR = ROOT / "data" / "embeddings" / "v2"
TOP_K = 50


def score_all(root: Path, top_k: int = TOP_K) -> dict[str, list[list]]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    queries = manifest["queries"]
    q_mats = [np.load(root / "queries" / f"{q['id']}.npy").astype(np.float32) for q in queries]
    bounds = np.cumsum([0] + [m.shape[0] for m in q_mats])
    q_all = np.concatenate(q_mats)                       # [sum_tokens, dim]
    scores = np.empty((len(manifest["pages"]), len(queries)), dtype=np.float32)

    t0 = time.time()
    for i, p in enumerate(manifest["pages"]):
        patches = np.load(root / "pages" / p["source"] / f"{p['page']}.npy").astype(np.float32)
        tok_max = (patches @ q_all.T).max(axis=0)        # best patch for every query token
        scores[i] = np.add.reduceat(tok_max, bounds[:-1])
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(manifest['pages'])} pages  ({time.time() - t0:.0f}s)", flush=True)

    rankings = {}
    for j, q in enumerate(queries):
        order = np.argsort(-scores[:, j])[:top_k]
        rankings[q["text"]] = [[manifest["pages"][i]["source"], manifest["pages"][i]["page"], float(scores[i, j])]
                               for i in order]
    return rankings


def _self_check() -> None:
    import tempfile
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "pages" / "a.pdf").mkdir(parents=True)
        (root / "queries").mkdir()
        pages = [rng.standard_normal((40, 16)).astype(np.float16) for _ in range(3)]
        for k, arr in enumerate(pages, 1):
            np.save(root / "pages" / "a.pdf" / f"{k}.npy", arr)
        q1, q2 = pages[1][:5], pages[2][7:12]            # queries cut from page 2 and page 3
        np.save(root / "queries" / "q1.npy", q1)
        np.save(root / "queries" / "q2.npy", q2)
        (root / "manifest.json").write_text(json.dumps({
            "pages": [{"source": "a.pdf", "page": k} for k in (1, 2, 3)],
            "queries": [{"id": "q1", "text": "one"}, {"id": "q2", "text": "two"}]}))
        r = score_all(root)
        assert r["one"][0][:2] == ["a.pdf", 2] and r["two"][0][:2] == ["a.pdf", 3], r
        brute = sum(max(float(np.dot(t.astype(np.float32), p.astype(np.float32))) for p in pages[1]) for t in q1)
        assert abs(r["one"][0][2] - brute) < 1e-3, "MaxSim differs from the naive double loop"
    print("maxsim self-check ok")


def main() -> None:
    _self_check()
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DIR
    if not (root / "manifest.json").exists():
        raise SystemExit(f"{root}/manifest.json missing: unzip the RunPod zips into {root} first")
    rankings = score_all(root)
    out = root / "vision_rankings.json"
    out.write_text(json.dumps(rankings, ensure_ascii=False), encoding="utf-8")
    print(f"{len(rankings)} queries ranked over the corpus -> {out}")


if __name__ == "__main__":
    main()
