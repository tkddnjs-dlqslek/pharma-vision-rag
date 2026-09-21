"""Split the 60 x 2 benchmark queries into batches for the E4 agentic subagent run.

Blind, like 17_make_generation_tasks.py: a batch holds only the question text and its language.
Gold pages and reference answers stay in eval/questions.jsonl, which the agent may not open.

    PYTHONIOENCODING=utf-8 python scripts/20_make_agentic_tasks.py [--batch 10] [--name agentic_vision]

--name picks the run: batches go to eval/results/<name>_tasks/ (answers to <name>_answers/). The
INSTRUCTIONS.md in that directory decides which tools the agent gets; it is hand-written per run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--name", default="agentic")
    a = ap.parse_args()
    OUT = ROOT / "eval" / "results" / f"{a.name}_tasks"

    qs = [json.loads(l) for l in open(ROOT / "eval" / "questions.jsonl", encoding="utf-8") if l.strip()]
    # All ko first, then all en, so a question's two language twins never share a batch: an agent
    # that saw both would answer the second from the first's tool calls, which the fixed pipelines
    # (retrieved per language) cannot do, and the ko/en gap would collapse to zero by construction.
    tasks = [{"task_id": f"{q['id']}_{lang}", "question": q[f"q_{lang}"], "lang": lang}
             for lang in ("ko", "en") for q in qs]
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("batch_*.json"):
        old.unlink()
    for n, i in enumerate(range(0, len(tasks), a.batch)):
        (OUT / f"batch_{n:02d}.json").write_text(
            json.dumps(tasks[i:i + a.batch], ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(tasks)} queries -> {len(list(OUT.glob('batch_*.json')))} batches of {a.batch}")


if __name__ == "__main__":
    main()
