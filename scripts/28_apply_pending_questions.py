"""Apply pending question rewordings (pending_q_ko / pending_q_en in eval/questions.jsonl) and recompute what depends
on the query text, without a GPU box:

    vision   query embedding via the Modal Nemotron encoder (serverless/modal_app.py) -> data/embeddings/v2/queries/<id>_<lang>.npy
             and the top-50 page ranking over the local int8 index (data/embeddings/vision_index, R@5 0.875 vs fp16 0.878)
             -> data/embeddings/v2/vision_rankings.json
    text     BGE-M3 query vector and bge-reranker scores via serverless/modal_text.py, exact cosine over
             data/embeddings/v2/text/text_vectors.npy -> text_candidates.json (top-30) and text_reranked.json
    bm25     nothing stored per query; the runner recomputes it from the question text

Only the changed queries are touched; every other entry keeps its RunPod-computed value. Old keys (old wording) are
removed so the rankings files stay keyed by the current question text. Afterwards run
`python -m pharma_vision_rag.eval.runner --mode all` to refresh the retrieval metrics.

    PYTHONIOENCODING=utf-8 PYTHONPATH=src .venv/Scripts/python.exe scripts/28_apply_pending_questions.py [--dry-run]
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
from pharma_vision_rag.retriever.vision_local import LocalVisionIndex  # noqa: E402
from pharma_vision_rag.retriever.vision_remote import modal_query_encoder  # noqa: E402

QUESTIONS = ROOT / "eval" / "questions.jsonl"
V2 = ROOT / "data" / "embeddings" / "v2"
VISION_INDEX = ROOT / "data" / "embeddings" / "vision_index"   # int8, closest to the fp16 reference
TOP_VISION, TOP_TEXT = 50, 30
NOTE = " | rewording applied {date}: {lang} was {old!r}"


def load_questions() -> list[dict]:
    return [json.loads(l) for l in QUESTIONS.read_text(encoding="utf-8").splitlines() if l.strip()]


def pending_changes(qs: list[dict]) -> list[tuple[dict, str, str, str]]:
    """(question, lang, old text, new text) for every pending rewording that differs from the current text."""
    out = []
    for q in qs:
        for lang in ("ko", "en"):
            new = q.get(f"pending_q_{lang}")
            if new and new != q[f"q_{lang}"]:
                out.append((q, lang, q[f"q_{lang}"], new))
    return out


def vision_update(changes, rankings: dict, dry: bool) -> None:
    enc = modal_query_encoder()
    cache: dict[str, np.ndarray] = {}
    index = LocalVisionIndex(VISION_INDEX, encoder=lambda s: cache[s], resident=False)  # one Modal call per query
    for q, lang, old, new in changes:
        t = time.time()
        emb = cache[new] = enc(new)
        hits = index.search(new, k=TOP_VISION)
        print(f"  vision {q['id']}_{lang}: {emb.shape[0]} tokens, top {hits[0][0]} p{hits[0][1]} ({time.time() - t:.0f}s)")
        if not dry:
            np.save(V2 / "queries" / f"{q['id']}_{lang}.npy", emb.astype(np.float16))
            rankings.pop(old, None)
            rankings[new] = [[d, p, s] for d, p, s in hits]


def text_update(changes, candidates: dict, reranked: dict, dry: bool) -> None:
    import modal
    models = modal.Cls.from_name("pharma-text-models", "TextModels")()
    chunks = [json.loads(l) for l in open(V2 / "text" / "text_chunks.jsonl", encoding="utf-8") if l.strip()]
    vecs = np.load(V2 / "text" / "text_vectors.npy", mmap_mode="r")
    assert len(chunks) == len(vecs), "chunk file and vectors disagree"
    new_texts = [new for _, _, _, new in changes]
    qvecs = np.asarray(models.embed.remote(new_texts), dtype=np.float32)
    for (q, lang, old, new), qv in zip(changes, qvecs):
        sims = np.asarray(vecs, dtype=np.float32) @ qv           # exact cosine, both sides normalised (31 MB)
        order = np.argsort(-sims)[:TOP_TEXT]
        hits = [{**chunks[i], "score": float(sims[i])} for i in order]
        scores = models.rerank.remote(new, [h["text"] for h in hits])
        rr = sorted(({**h, "rerank_score": float(s)} for h, s in zip(hits, scores)),
                    key=lambda h: h["rerank_score"], reverse=True)
        print(f"  text   {q['id']}_{lang}: dense top {hits[0]['source']} p{hits[0]['page']}, "
              f"rerank top {rr[0]['source']} p{rr[0]['page']}")
        if not dry:
            candidates.pop(old, None)
            reranked.pop(old, None)
            candidates[new] = hits
            reranked[new] = rr


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="compute and print, write nothing")
    ap.add_argument("--skip-text", action="store_true")
    ap.add_argument("--skip-vision", action="store_true")
    a = ap.parse_args()

    qs = load_questions()
    changes = pending_changes(qs)
    if not changes:
        print("no pending rewordings")
        return
    print(f"{len(changes)} queries to recompute: " + ", ".join(f"{q['id']}_{lang}" for q, lang, _, _ in changes))

    rankings = json.loads((V2 / "vision_rankings.json").read_text(encoding="utf-8"))
    candidates = json.loads((V2 / "text" / "text_candidates.json").read_text(encoding="utf-8"))
    reranked = json.loads((V2 / "text" / "text_reranked.json").read_text(encoding="utf-8"))
    if not a.skip_vision:
        vision_update(changes, rankings, a.dry_run)
    if not a.skip_text:
        text_update(changes, candidates, reranked, a.dry_run)
    if a.dry_run:
        print("dry run: nothing written")
        return

    today = time.strftime("%Y-%m-%d")
    for q, lang, old, new in changes:
        q[f"q_{lang}"] = new
        del q[f"pending_q_{lang}"]
        q["notes"] = (q.get("notes") or "") + NOTE.format(date=today, lang=lang, old=old)
    QUESTIONS.write_text("\n".join(json.dumps(q, ensure_ascii=False) for q in qs) + "\n", encoding="utf-8")
    if not a.skip_vision:
        (V2 / "vision_rankings.json").write_text(json.dumps(rankings, ensure_ascii=False), encoding="utf-8")
    if not a.skip_text:
        (V2 / "text" / "text_candidates.json").write_text(json.dumps(candidates, ensure_ascii=False), encoding="utf-8")
        (V2 / "text" / "text_reranked.json").write_text(json.dumps(reranked, ensure_ascii=False), encoding="utf-8")
    print(f"applied {len(changes)} rewordings; rankings keyed by the new text. "
          f"Now: PYTHONIOENCODING=utf-8 PYTHONPATH=src python -m pharma_vision_rag.eval.runner --mode all")


if __name__ == "__main__":
    main()
