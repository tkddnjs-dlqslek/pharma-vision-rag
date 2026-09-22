"""Check the downloaded local vision index against the reference exact rankings (vision_rankings.json).

1. If vectors.npy is still in parts (scripts/24_vision_index_gpu.py output), verify each part's sha256, join, delete parts.
2. Search the 120 benchmark queries with LocalVisionIndex, using the stored query embeddings (no model), and compare
   the top 5 with data/embeddings/v2/vision_rankings.json: mean top-5 overlap, identical top-1, R@5 against gold.
3. --live N: also encode the first N query texts with the real Nemotron model on CPU (slow, ~7 GB RAM) and report
   encoding time plus top-5 overlap with the stored-embedding result.

Passes when mean top-5 overlap >= 0.9 and R@5 is no more than 0.02 below the reference.

Usage:
    PYTHONIOENCODING=utf-8 PYTHONPATH=src python scripts/25_check_vision_index.py [--index data/embeddings/vision_index] [--live 3]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pharma_vision_rag.eval.metrics import gold_groups, recall_at_k  # noqa: E402
from pharma_vision_rag.retriever.vision_local import LocalVisionIndex  # noqa: E402

V2 = ROOT / "data" / "embeddings" / "v2"


def join_parts(index: Path) -> None:
    spec = json.loads((index / "parts.json").read_text(encoding="utf-8"))
    tmp = index / (spec["file"] + ".tmp")
    with open(tmp, "wb") as out:
        for p in spec["parts"]:
            data = (index / p["name"]).read_bytes()
            if len(data) != p["bytes"] or hashlib.sha256(data).hexdigest() != p["sha256"]:
                tmp.unlink()
                raise SystemExit(f"{p['name']} is corrupt or incomplete: download it again")
            out.write(data)
    tmp.rename(index / spec["file"])
    for p in spec["parts"]:
        (index / p["name"]).unlink()
    print(f"joined {len(spec['parts'])} parts -> {index / spec['file']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, default=ROOT / "data" / "embeddings" / "vision_index")
    ap.add_argument("--live", type=int, default=0, help="also encode N queries with the real model")
    a = ap.parse_args()

    if not (a.index / "vectors.npy").exists() and (a.index / "parts.json").exists():
        join_parts(a.index)
    if not LocalVisionIndex.available(a.index):
        raise SystemExit(f"incomplete index in {a.index}")

    manifest = json.loads((V2 / "manifest.json").read_text(encoding="utf-8"))
    by_text = {q["text"]: q["id"] for q in manifest["queries"]}
    ref = json.loads((V2 / "vision_rankings.json").read_text(encoding="utf-8"))
    qs = {q["id"]: q for q in (json.loads(l) for l in (ROOT / "eval" / "questions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())}
    idx = LocalVisionIndex(a.index, encoder=lambda text: np.load(V2 / "queries" / f"{by_text[text]}.npy"))
    print(f"index {idx.meta.get('variant')}: {len(idx.pages)} pages, {len(idx.vectors):,} vectors {idx.vectors.dtype}")

    overlap, top1, r5_local, r5_ref, times, local_top = [], [], [], [], [], {}
    for text, qid in by_text.items():
        t = time.perf_counter()
        hits = idx.search(text, k=5)
        times.append(time.perf_counter() - t)
        mine = [(d, p) for d, p, _ in hits]
        theirs = [(d, int(p)) for d, p, _ in ref[text][:5]]
        local_top[text] = mine
        overlap.append(len(set(mine) & set(theirs)) / 5)
        top1.append(mine[0] == theirs[0])
        groups = gold_groups(qs[qid.rsplit("_", 1)[0]])
        r5_local.append(recall_at_k(mine, groups, 5))
        r5_ref.append(recall_at_k(theirs, groups, 5))
    m_ov, m_loc, m_ref = float(np.mean(overlap)), float(np.mean(r5_local)), float(np.mean(r5_ref))
    print(f"{len(by_text)} queries: top-5 overlap {m_ov:.3f}, same top-1 {np.mean(top1):.3f}, "
          f"R@5 local {m_loc:.3f} vs reference {m_ref:.3f}, search {np.mean(times):.2f} s/query on CPU (median {np.median(times):.2f})")

    if a.live:
        live = LocalVisionIndex(a.index)
        t = time.perf_counter()
        live.encode("warm-up")
        print(f"live: model load + first encode {time.perf_counter() - t:.1f} s")
        for text in list(by_text)[:a.live]:
            t = time.perf_counter()
            live.encode(text)
            enc = time.perf_counter() - t
            hits = live.search(text, k=5)  # encodes again
            ov = len({(d, p) for d, p, _ in hits} & set(local_top[text])) / 5
            print(f"live: encode {enc:.1f} s, top-5 overlap with stored embedding {ov:.2f}  {text[:40]}")

    ok = m_ov >= 0.9 and m_loc >= m_ref - 0.02
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
