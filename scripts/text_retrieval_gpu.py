"""GPU-box job (RunPod): the whole text retrieval path for the fixed benchmark queries.

The dev PC (16 GB RAM, CPU) needs hours to embed ~15k chunks and thrashes when BGE-M3 and the reranker
are both loaded. On the pod it takes minutes. Standalone apart from chunking.py, which must sit next to
this script (upload src/pharma_vision_rag/retriever/chunking.py unchanged: same chunk rules as local).

Input : <input>/corpus.json, <input>/questions.jsonl, <out>/text_blocks.jsonl (from embed_pages_gpu.py)
Output: <out>/text/
    text_chunks.jsonl        one chunk per line (text, context, source, page, block_type, block_index)
    text_vectors.npy         fp16 [n_chunks, 1024], BGE-M3 dense, L2-normalised, same order as the chunks
    text_candidates.json     {query: top-30 chunks by exact cosine}          -> runner variant "text"
    text_reranked.json       {query: the same 30 re-sorted by bge-reranker}  -> runner variant "text_rerank"
    ../text_results.zip      everything above in one download

Exact cosine over all chunks replaces the Qdrant HNSW search for the benchmark (15k points: same ranking,
no approximation). The vectors are still downloaded so the local Qdrant index can be built without a model.

Usage on the pod:
    pip install -q sentence-transformers
    python text_retrieval_gpu.py --input input --out out --limit 300     # 1 min smoke
    python text_retrieval_gpu.py --input input --out out
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chunking  # noqa: E402  (retriever/chunking.py uploaded next to this file)

EMBED_MODEL = "BAAI/bge-m3"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
TOP_K = 30          # = CHUNK_POOL in eval/runner.py
EMBED_MAX_SEQ = 1024
RERANK_MAX_LEN = 512


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, help="smoke test: first N chunks and 4 queries only, nothing zipped")
    a = ap.parse_args()

    corpus = json.loads((a.input / "corpus.json").read_text(encoding="utf-8"))
    chunking.DOC_LABELS = {d["id"]: d["label"] for d in corpus}  # the repo-relative manifest path does not exist here
    blocks = [json.loads(l) for l in open(a.out / "text_blocks.jsonl", encoding="utf-8") if l.strip()]
    chunks = chunking.build_chunks(blocks)
    qs = [json.loads(l) for l in (a.input / "questions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    queries = [q[k] for q in qs for k in ("q_ko", "q_en")]
    if a.limit:
        chunks, queries = chunks[:a.limit], queries[:4]
    print(f"{len(blocks)} blocks -> {len(chunks)} chunks, {len(queries)} queries", flush=True)
    assert all(c["context"].split(" | ")[0] in chunking.DOC_LABELS.values() for c in chunks[:50]), "document labels not applied"

    import torch
    from sentence_transformers import CrossEncoder, SentenceTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"

    t0 = time.time()
    embedder = SentenceTransformer(EMBED_MODEL, device=device)
    embedder.max_seq_length = EMBED_MAX_SEQ
    vecs = embedder.encode([chunking.embed_text(c) for c in chunks], batch_size=64,
                           normalize_embeddings=True, show_progress_bar=True).astype(np.float32)
    qvecs = embedder.encode(queries, batch_size=64, normalize_embeddings=True).astype(np.float32)
    assert np.isfinite(vecs).all() and abs(float(np.linalg.norm(vecs[0])) - 1) < 1e-3, "bad embeddings"
    print(f"embedded in {time.time() - t0:.0f}s", flush=True)
    del embedder
    torch.cuda.empty_cache()

    sims = qvecs @ vecs.T                                         # exact cosine (both sides normalised)
    candidates = {}
    for qi, query in enumerate(queries):
        order = np.argsort(-sims[qi])[:TOP_K]
        candidates[query] = [{**chunks[i], "score": float(sims[qi, i])} for i in order]

    t0 = time.time()
    reranker = CrossEncoder(RERANK_MODEL, max_length=RERANK_MAX_LEN, device=device)
    reranked = {}
    for query, hits in candidates.items():
        scores = reranker.predict([[query, h["text"]] for h in hits], show_progress_bar=False)
        reranked[query] = sorted(({**h, "rerank_score": float(s)} for h, s in zip(hits, scores)),
                                 key=lambda h: h["rerank_score"], reverse=True)
    print(f"reranked in {time.time() - t0:.0f}s", flush=True)

    top = candidates[queries[0]][0]
    print(f"sample: {queries[0][:50]!r} -> {top['source']} p{top['page']} ({top['score']:.3f})")
    if a.limit:
        print(f"SMOKE OK: {vecs.shape} vectors, {len(candidates)} queries")
        return

    tdir = a.out / "text"
    tdir.mkdir(parents=True, exist_ok=True)
    with open(tdir / "text_chunks.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(c, ensure_ascii=False) + "\n" for c in chunks)
    np.save(tdir / "text_vectors.npy", vecs.astype(np.float16))
    (tdir / "text_candidates.json").write_text(json.dumps(candidates, ensure_ascii=False), encoding="utf-8")
    (tdir / "text_reranked.json").write_text(json.dumps(reranked, ensure_ascii=False), encoding="utf-8")
    zpath = a.out.parent / "text_results.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(tdir.iterdir()):
            z.write(f, f"text/{f.name}")
    print(f"{len(chunks)} chunks, {len(candidates)} queries -> {zpath}  {zpath.stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
