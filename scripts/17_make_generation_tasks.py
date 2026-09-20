"""Build blind answer-generation tasks for the subagent-based run (no API key: see EXPERIMENT_PLAN section 0).

For each retrieval variant and each benchmark query (60 questions x ko/en), take the top-3 retrieved pages,
render them once to PNG, and write task batches that contain ONLY the question and the page images.
Reference answers, gold pages, question ids and variant names are kept out of the batches (blind generation);
they live in the key file that only the scoring step reads.

Identical (question, page triple) pairs produced by different variants are generated once and shared.

Output (all under eval/results/, gitignored):
    gen_tasks/batch_NN.json   [{task_id, question, images: [abs png paths], pages: ["doc p12", ...]}]
    gen_tasks/key.json        {task_id: {question, lang, id, pages, used_by: [variant, ...]}}
    data/images/eval_pages/   rendered pages

Usage:
    PYTHONIOENCODING=utf-8 python scripts/17_make_generation_tasks.py [--variants text_rerank,vision,hybrid_rerank] [--batch 12]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pypdfium2 as pdfium

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pharma_vision_rag.eval.metrics import ranked_pages  # noqa: E402
from pharma_vision_rag.eval.runner import build_variants  # noqa: E402

TOP_PAGES = 3
IMG_DIR = ROOT / "data" / "images" / "eval_pages"
OUT = ROOT / "eval" / "results" / "gen_tasks"
TARGET_WIDTH = 1400  # px: small chart labels stay legible


def render(doc: str, page: int) -> Path:
    dst = IMG_DIR / f"{doc[:-4]}_p{page}.png"
    if not dst.exists():
        pg = pdfium.PdfDocument(str(ROOT / "data" / "pdf" / "corpus" / doc))[page - 1]
        pg.render(scale=TARGET_WIDTH / pg.get_width()).to_pil().convert("RGB").save(dst)
    return dst


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="text_rerank,vision,hybrid_rerank")
    ap.add_argument("--batch", type=int, default=12, help="tasks per batch file (3 images each)")
    a = ap.parse_args()
    names = a.variants.split(",")
    variants = build_variants(names)
    missing = set(names) - set(variants)
    if missing:
        raise SystemExit(f"variants unavailable: {missing}")

    IMG_DIR.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    qs = [json.loads(l) for l in open(ROOT / "eval" / "questions.jsonl", encoding="utf-8") if l.strip()]
    key: dict[str, dict] = {}
    for name in names:
        for q in qs:
            for lang in ("ko", "en"):
                text = q[f"q_{lang}"]
                pages = ranked_pages(variants[name](text))[:TOP_PAGES]
                tid = hashlib.sha1(json.dumps([text, pages], ensure_ascii=False).encode()).hexdigest()[:10]
                entry = key.setdefault(tid, {"question": text, "lang": lang, "id": q["id"],
                                             "pages": [[d, p] for d, p in pages], "used_by": []})
                entry["used_by"].append(name)

    tasks = []
    for tid, e in sorted(key.items()):
        imgs = [render(d, p) for d, p in e["pages"]]
        tasks.append({"task_id": tid, "question": e["question"], "images": [str(i) for i in imgs],
                      "pages": [f"{d} p{p}" for d, p in e["pages"]]})
    for old in OUT.glob("batch_*.json"):
        old.unlink()
    for n, i in enumerate(range(0, len(tasks), a.batch)):
        (OUT / f"batch_{n:02d}.json").write_text(json.dumps(tasks[i:i + a.batch], ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "key.json").write_text(json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")
    total = len(names) * len(qs) * 2
    print(f"{total} (variant, question, lang) cells -> {len(tasks)} unique tasks, "
          f"{len(list(IMG_DIR.glob('*.png')))} page images, {len(list(OUT.glob('batch_*.json')))} batches of {a.batch}")


if __name__ == "__main__":
    main()
