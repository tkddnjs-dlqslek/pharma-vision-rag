"""Answer generation for the RAG benchmark (EXPERIMENT_PLAN §3/§6, generation stage).

For a retrieval variant, take the fixed top-3 pages per (question, language) from
``eval.runner.build_variants`` (the same precomputed hits the retrieval benchmark uses),
render them as images, and answer with the existing ``ClaudeVisionGenerator`` — every
fixed-pipeline mode (text_only/vision_only/caption/hybrid) already answers this way once
pages are chosen, so generation only differs by *which* pages were retrieved, not by how
the model reads them. Repeated N times because Sonnet output is non-deterministic
(EXPERIMENT_PLAN §6: judge averages 3 repeats).

Resume-safe: (variant, id, lang, repeat) rows already in the output file are skipped, so an
interrupted run restarts cheaply.

Known deviation: this reuses ``ClaudeVisionGenerator``'s default system prompt, which names
"Sanofi" specifically even though the v2 corpus also has Novartis/Roche/AstraZeneca — not
edited here per the "reuse, don't touch generator internals" instruction; flag before reporting.

Cost: see eval/pricing.py — verify those $/1M numbers before reporting.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from pharma_vision_rag.eval.metrics import ranked_pages
from pharma_vision_rag.eval.pricing import cost_usd
from pharma_vision_rag.eval.runner import QUESTIONS, ROOT, build_variants
from pharma_vision_rag.generator.claude_vision import ClaudeVisionGenerator, _image_to_base64
from pharma_vision_rag.utils.pdf import render_page

PDF_DIR = ROOT / "data" / "pdf" / "corpus"
RESULTS = ROOT / "eval" / "results"
TOP_K = 3


def _load_questions(limit: int | None) -> list[dict[str, Any]]:
    qs = [json.loads(l) for l in QUESTIONS.read_text(encoding="utf-8").splitlines() if l.strip()]
    return qs[:limit] if limit else qs


def _existing_keys(path: Path) -> set[tuple[str, str, str, int]]:
    if not path.exists():
        return set()
    keys = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        keys.add((r["variant"], r["id"], r["lang"], r["repeat"]))
    return keys


def top_pages(search, query: str, k: int = TOP_K) -> list[tuple[str, int]]:
    """Rank ``search(query)`` hits to unique (source, page) pairs, keep the top ``k``."""
    return ranked_pages(search(query))[:k]


def render_images(pages: list[tuple[str, int]]) -> list[Any]:
    return [render_page(PDF_DIR / source, page_number=page) for source, page in pages]


def dry_run_summary(query: str, pages: list[tuple[str, int]], images: list[Any]) -> dict[str, Any]:
    """Build the request ClaudeVisionGenerator would send, without a client or a network call.

    "Token-free": reports image count/size and query length, not a token count (that needs
    either the API or a local tokenizer neither of which this is meant to require)."""
    blocks = []
    for i, img in enumerate(images):
        media_type, data = _image_to_base64(img)
        blocks.append({"media_type": media_type, "b64_chars": len(data), "cached": i < 3})
    return {"query_chars": len(query), "n_images": len(images), "images": blocks,
            "pages": [f"{s}:{p}" for s, p in pages]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, help="text | text_rerank | vision | caption | hybrid | hybrid_rerank | ...")
    ap.add_argument("--limit", type=int, default=None, help="first N questions")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--lang", choices=["ko", "en", "both"], default="both")
    ap.add_argument("--dry-run", action="store_true", help="render pages and build the request, never call the API")
    a = ap.parse_args()

    langs = ["ko", "en"] if a.lang == "both" else [a.lang]
    questions = _load_questions(a.limit)

    variants = build_variants([a.variant])
    search = variants.get(a.variant)
    if search is None:
        raise SystemExit(f"variant {a.variant!r} unavailable (missing precomputed hits / index for it)")

    if a.dry_run:
        n = 0
        for q in questions:
            for lang in langs:
                query = q[f"q_{lang}"]
                pages = top_pages(search, query)
                images = render_images(pages)
                summary = dry_run_summary(query, pages, images)
                for repeat in range(a.repeats):
                    print(f"[dry-run] {a.variant} {q['id']} {lang} r{repeat}: "
                          f"{summary['n_images']} images, pages={summary['pages']}, query_chars={summary['query_chars']}")
                    n += 1
        print(f"{n} requests built (dry run, no API call)")
        return

    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / f"generation_{a.variant}.jsonl"
    done = _existing_keys(out_path)
    generator = ClaudeVisionGenerator()

    n_written = n_skipped = 0
    with open(out_path, "a", encoding="utf-8") as f:
        for q in questions:
            for lang in langs:
                query = q[f"q_{lang}"]
                pages = top_pages(search, query)
                images = render_images(pages)
                for repeat in range(a.repeats):
                    key = (a.variant, q["id"], lang, repeat)
                    if key in done:
                        n_skipped += 1
                        continue
                    t0 = time.time()
                    result = generator.generate(query=query, images=images)
                    latency = time.time() - t0
                    record = {
                        "variant": a.variant, "id": q["id"], "lang": lang, "repeat": repeat,
                        "pages": [{"source": s, "page": p} for s, p in pages],
                        # No separate citation-extraction step exists for the fixed-pipeline modes,
                        # so "cited" pages are the pages fed to the model (see module docstring).
                        "cited_pages": [{"source": s, "page": p} for s, p in pages],
                        "answer": result["answer"],
                        "usage": result["usage"],
                        "cost_usd": round(cost_usd(generator.model, result["usage"]), 6),
                        "latency_s": round(latency, 3),
                        "stop_reason": result["stop_reason"],
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()
                    n_written += 1
    print(f"{n_written} rows written, {n_skipped} skipped (already present) -> {out_path}")


if __name__ == "__main__":
    main()
