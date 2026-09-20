"""Score the subagent-generated answers (companion of 17_make_generation_tasks.py).

Two steps, because the judging itself is also done by subagents (no API key on this machine):
  prepare : join gen_answers/*.jsonl with the reference answers -> judge_tasks/batch_NN.json
            (question, reference answer, answer keys, model answer; no variant names, so the judge is blind
            to which retriever produced the answer). Also records a deterministic keys_matched flag.
  report  : join judge verdicts (judge_verdicts/*.jsonl: {task_id, verdict: correct|partial|wrong, reason})
            back to (variant, question, lang) cells and print accuracy tables + paired sign tests.
            score = correct 1, partial 0.5, wrong 0.

Usage:
    PYTHONIOENCODING=utf-8 python scripts/18_score_generation.py prepare [--batch 40]
    PYTHONIOENCODING=utf-8 python scripts/18_score_generation.py report
"""
from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "eval" / "results"
SCORE = {"correct": 1.0, "partial": 0.5, "wrong": 0.0}


def load_jsonl(pattern: str) -> dict[str, dict]:
    out = {}
    for f in sorted(RES.glob(pattern)):
        for line in open(f, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                out[r["task_id"]] = r
    return out


def questions() -> dict[str, dict]:
    return {q["id"]: q for q in map(json.loads, open(ROOT / "eval" / "questions.jsonl", encoding="utf-8"))}


def prepare(batch: int) -> None:
    key = json.loads((RES / "gen_tasks" / "key.json").read_text(encoding="utf-8"))
    answers, qs = load_jsonl("gen_answers/*.jsonl"), questions()
    missing = sorted(set(key) - set(answers))
    tasks = []
    for tid, a in sorted(answers.items()):
        q = qs[key[tid]["id"]]
        norm = " ".join(a["answer"].split())
        tasks.append({"task_id": tid, "question": key[tid]["question"], "reference_answer": q["answer"],
                      "answer_keys": q["answer_keys"], "model_answer": a["answer"],
                      "keys_matched": bool(q["answer_keys"]) and all(k in norm for k in q["answer_keys"])})
    out = RES / "judge_tasks"
    out.mkdir(exist_ok=True)
    for old in out.glob("batch_*.json"):
        old.unlink()
    for n, i in enumerate(range(0, len(tasks), batch)):
        (out / f"batch_{n:02d}.json").write_text(json.dumps(tasks[i:i + batch], ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(tasks)} answers -> {len(list(out.glob('batch_*.json')))} judge batches; "
          f"{len(missing)} tasks still without an answer; keys_matched {sum(t['keys_matched'] for t in tasks)}")


def sign_test(wins: int, losses: int) -> float:
    n = wins + losses
    return min(1.0, sum(math.comb(n, k) for k in range(max(wins, losses), n + 1)) / 2 ** n * 2) if n else 1.0


def report() -> None:
    key = json.loads((RES / "gen_tasks" / "key.json").read_text(encoding="utf-8"))
    answers, verdicts, qs = load_jsonl("gen_answers/*.jsonl"), load_jsonl("judge_verdicts/*.jsonl"), questions()
    cells, rows = {}, []
    for tid, k in key.items():
        if tid not in verdicts:
            continue
        q, v = qs[k["id"]], verdicts[tid]
        gold = {(g["source"], g["page"]) for g in q["gold_pages"]}
        for variant in k["used_by"]:
            cells[(variant, k["id"], k["lang"])] = SCORE[v["verdict"]]
            rows.append({"variant": variant, "id": k["id"], "type": q["type"], "lang": k["lang"],
                         "period_spec": q["period_spec"], "verdict": v["verdict"], "score": SCORE[v["verdict"]],
                         "found": answers[tid].get("found"), "gold_in_top3": any((d, p) in gold for d, p in k["pages"]),
                         "pages": " ".join(f"{d}:{p}" for d, p in k["pages"]), "answer": answers[tid]["answer"],
                         "reason": v.get("reason", "")})
    if not rows:
        raise SystemExit("no judged answers yet")
    with open(RES / "generation_scored.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    variants = sorted({r["variant"] for r in rows})

    def table(title: str, field: str | None) -> None:
        groups = sorted({r[field] for r in rows}) if field else ["all"]
        print(f"\n{title}\n{'variant':<15}" + "".join(f"{g:>10}" for g in groups))
        for v in variants:
            vals = []
            for g in groups:
                rs = [r for r in rows if r["variant"] == v and (not field or r[field] == g)]
                vals.append(f"{sum(r['score'] for r in rs) / len(rs):.2f} ({len(rs)})" if rs else "-")
            print(f"{v:<15}" + "".join(f"{x:>10}" for x in vals))

    print(f"{len(rows)} scored cells from {len(verdicts)} judged answers")
    table("answer accuracy (correct 1, partial 0.5)", None)
    table("by question type", "type")
    table("by language", "lang")
    table("by period_spec", "period_spec")
    print("\nstrictly correct share / answers given when gold was not in the top 3 (hallucination check)")
    for v in variants:
        rs = [r for r in rows if r["variant"] == v]
        nog = [r for r in rs if not r["gold_in_top3"]]
        print(f"{v:<15} correct {sum(r['verdict'] == 'correct' for r in rs) / len(rs):.2f}   "
              f"no-gold cells {len(nog)}, of which answered instead of 'not found': {sum(bool(r['found']) for r in nog)}, "
              f"of those judged correct/partial: {sum(r['score'] > 0 and bool(r['found']) for r in nog)}")
    print("\npaired sign tests on score")
    for a, b in [(x, y) for i, x in enumerate(variants) for y in variants[i + 1:]]:
        for types in ("ABCD", "A"):
            w = l = 0
            for (v, i, lg), s in cells.items():
                if v == a and (b, i, lg) in cells and qs[i]["type"] in types:
                    w += s > cells[(b, i, lg)]
                    l += s < cells[(b, i, lg)]
            print(f"  {a} vs {b} [{types}]: {w} better, {l} worse, p={sign_test(w, l):.4f}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "prepare":
        prepare(int(sys.argv[sys.argv.index("--batch") + 1]) if "--batch" in sys.argv else 40)
    elif cmd == "report":
        report()
    else:
        sys.exit(__doc__)
