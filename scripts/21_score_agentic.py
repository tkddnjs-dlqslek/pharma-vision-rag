"""Score the E4 agentic run and compare it with the three fixed pipelines.

Same two-step shape as 18_score_generation.py (whose sign_test and jsonl loader this reuses), because
the judging is also done blind: the judge sees the question, the reference answer and the model
answer, and never learns which arm produced it.

    PYTHONIOENCODING=utf-8 python scripts/21_score_agentic.py prepare [--batch 40]
    PYTHONIOENCODING=utf-8 python scripts/21_score_agentic.py report

report also prints how often the agent's own citations land on a gold page, which is the agentic
counterpart of recall@3 for the fixed arms, and paired sign tests against each fixed arm on the
(question, language) cells they share (read from eval/results/generation_scored.csv).
"""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "eval" / "results"
SCORE = {"correct": 1.0, "partial": 0.5, "wrong": 0.0}


def _load_18():
    spec = importlib.util.spec_from_file_location("_score_gen", ROOT / "scripts" / "18_score_generation.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    answers, qs = load_jsonl("agentic_answers/*.jsonl"), questions()
    expected = {f"{q}_{lang}" for q in qs for lang in ("ko", "en")}
    tasks = []
    for tid, a in sorted(answers.items()):
        q = qs[tid.rsplit("_", 1)[0]]
        norm = " ".join(a["answer"].split())
        tasks.append({"task_id": tid, "question": q[f"q_{tid.rsplit('_', 1)[1]}"],
                      "reference_answer": q["answer"], "answer_keys": q["answer_keys"],
                      "model_answer": a["answer"],
                      "keys_matched": bool(q["answer_keys"]) and all(k in norm for k in q["answer_keys"])})
    out = RES / "judge_tasks_agentic"
    out.mkdir(exist_ok=True)
    for old in out.glob("batch_*.json"):
        old.unlink()
    for n, i in enumerate(range(0, len(tasks), batch)):
        (out / f"batch_{n:02d}.json").write_text(json.dumps(tasks[i:i + batch], ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(tasks)} answers -> {len(list(out.glob('batch_*.json')))} judge batches; "
          f"{len(expected - set(answers))} queries still without an answer; "
          f"keys_matched {sum(t['keys_matched'] for t in tasks)}")


def report() -> None:
    sign_test = _load_18().sign_test
    answers, verdicts, qs = load_jsonl("agentic_answers/*.jsonl"), load_jsonl("judge_verdicts_agentic/*.jsonl"), questions()
    rows = []
    for tid, v in verdicts.items():
        qid, lang = tid.rsplit("_", 1)
        q, a = qs[qid], answers[tid]
        gold = {(g["source"], g["page"]) for g in q["gold_pages"]}
        cited = {(c.get("source"), c.get("page")) for c in a.get("cited", [])}
        rows.append({"variant": "agentic", "id": qid, "type": q["type"], "lang": lang,
                     "period_spec": q["period_spec"], "verdict": v["verdict"], "score": SCORE[v["verdict"]],
                     "found": a.get("found"), "gold_in_citations": bool(cited & gold),
                     "tool_calls": a.get("tool_calls"), "answer": a["answer"], "reason": v.get("reason", "")})
    if not rows:
        raise SystemExit("no judged agentic answers yet")
    with open(RES / "agentic_scored.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def table(title: str, field: str | None) -> None:
        groups = sorted({r[field] for r in rows}) if field else ["all"]
        print(f"\n{title}\n{'agentic':<15}" + "".join(f"{g:>10}" for g in groups))
        vals = []
        for g in groups:
            rs = [r for r in rows if not field or r[field] == g]
            vals.append(f"{sum(r['score'] for r in rs) / len(rs):.2f} ({len(rs)})" if rs else "-")
        print(f"{'':<15}" + "".join(f"{x:>10}" for x in vals))

    calls = [r["tool_calls"] for r in rows if isinstance(r["tool_calls"], (int, float))]
    print(f"{len(rows)} judged agentic answers; mean tool calls {sum(calls) / len(calls):.1f}" if calls else f"{len(rows)} judged")
    table("answer accuracy (correct 1, partial 0.5)", None)
    table("by question type", "type")
    table("by language", "lang")
    table("by period_spec", "period_spec")
    print(f"\ncitations landed on a gold page: {sum(r['gold_in_citations'] for r in rows)}/{len(rows)} "
          f"({sum(r['gold_in_citations'] for r in rows) / len(rows):.2f})")
    nog = [r for r in rows if not r["gold_in_citations"]]
    print(f"of the {len(nog)} cells without a gold citation, {sum(bool(r['found']) for r in nog)} answered anyway, "
          f"{sum(r['score'] > 0 and bool(r['found']) for r in nog)} of those judged correct/partial")

    scored = RES / "generation_scored.csv"
    if not scored.exists():
        print("\n(no generation_scored.csv: run 18_score_generation.py report first for the paired tests)")
        return
    fixed: dict[tuple[str, str, str], float] = {}
    for r in csv.DictReader(open(scored, encoding="utf-8")):
        fixed[(r["variant"], r["id"], r["lang"])] = float(r["score"])
    mine = {(r["id"], r["lang"]): r["score"] for r in rows}
    print("\npaired sign tests on score (agentic vs each fixed arm)")
    for other in sorted({k[0] for k in fixed}):
        for types in ("ABCD", "A", "D"):
            w = l = 0
            for (qid, lang), s in mine.items():
                if qs[qid]["type"] in types and (other, qid, lang) in fixed:
                    w += s > fixed[(other, qid, lang)]
                    l += s < fixed[(other, qid, lang)]
            print(f"  agentic vs {other} [{types}]: {w} better, {l} worse, p={sign_test(w, l):.4f}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "prepare":
        prepare(int(sys.argv[sys.argv.index("--batch") + 1]) if "--batch" in sys.argv else 40)
    elif cmd == "report":
        report()
    else:
        sys.exit(__doc__)
