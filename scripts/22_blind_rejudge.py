"""Mixed blind re-judge of all four arms in one pass.

Why: the first judging pass scored the three fixed arms together (blind to arm) but scored the
agentic answers separately, so that judge knew which arm it was grading. Here every answer from
all four arms is pooled, given an anonymous id, shuffled, and judged by fresh subagents with the
same rubric (eval/results/judge_tasks/INSTRUCTIONS.md, copied with new paths). The map from
anonymous id back to arm lives in rejudge_tasks/map.json, which judges may not open.

    PYTHONIOENCODING=utf-8 python scripts/22_blind_rejudge.py prepare [--batch 40]
    PYTHONIOENCODING=utf-8 python scripts/22_blind_rejudge.py report

report prints per-arm accuracy, paired sign tests, and agreement with the first-pass verdicts
(judge_verdicts/ and judge_verdicts_agentic/), which measures how much the first judge drifted.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "eval" / "results"
TASKS, VERDICTS = RES / "rejudge_tasks", RES / "rejudge_verdicts"
SCORE = {"correct": 1.0, "partial": 0.5, "wrong": 0.0}
ARMS = ["text_rerank", "vision", "hybrid_rerank", "agentic"]


def _load_18():
    spec = importlib.util.spec_from_file_location("_score_gen", ROOT / "scripts" / "18_score_generation.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


S18 = _load_18()


def anon(src: str, task_id: str) -> str:
    return hashlib.sha1(f"rejudge:{src}:{task_id}".encode()).hexdigest()[:10]


def pooled() -> list[dict]:
    """Every generated answer once, with the (arm, question, lang) cells it stands for."""
    qs, key = S18.questions(), json.loads((RES / "gen_tasks" / "key.json").read_text(encoding="utf-8"))
    out = []
    for tid, a in S18.load_jsonl("gen_answers/*.jsonl").items():
        k = key[tid]
        out.append({"src": "gen", "task_id": tid, "qid": k["id"], "lang": k["lang"], "question": k["question"],
                    "answer": a["answer"], "cells": [(v, k["id"], k["lang"]) for v in k["used_by"]]})
    for tid, a in S18.load_jsonl("agentic_answers/*.jsonl").items():
        qid, lang = tid.rsplit("_", 1)
        out.append({"src": "agentic", "task_id": tid, "qid": qid, "lang": lang, "question": qs[qid][f"q_{lang}"],
                    "answer": a["answer"], "cells": [("agentic", qid, lang)]})
    return out


def prepare(batch: int) -> None:
    qs = S18.questions()
    rows = sorted(pooled(), key=lambda r: anon(r["src"], r["task_id"]))  # hash order = arm-agnostic shuffle
    tasks, mapping = [], {}
    for r in rows:
        aid = anon(r["src"], r["task_id"])
        q = qs[r["qid"]]
        norm = " ".join(r["answer"].split())
        tasks.append({"task_id": aid, "question": r["question"], "reference_answer": q["answer"],
                      "answer_keys": q["answer_keys"], "model_answer": r["answer"],
                      "keys_matched": bool(q["answer_keys"]) and all(k in norm for k in q["answer_keys"])})
        mapping[aid] = {"src": r["src"], "task_id": r["task_id"]}
    assert len(mapping) == len(rows), "anonymous id collision"
    TASKS.mkdir(exist_ok=True)
    VERDICTS.mkdir(exist_ok=True)
    for old in TASKS.glob("batch_*.json"):
        old.unlink()
    for n, i in enumerate(range(0, len(tasks), batch)):
        (TASKS / f"batch_{n:02d}.json").write_text(json.dumps(tasks[i:i + batch], ensure_ascii=False, indent=1), encoding="utf-8")
    (TASKS / "map.json").write_text(json.dumps(mapping, indent=1), encoding="utf-8")
    rubric = (RES / "judge_tasks" / "INSTRUCTIONS.md").read_text(encoding="utf-8")
    rubric = rubric.replace(r"eval\results\judge_verdicts\batch_NN.jsonl", r"eval\results\rejudge_verdicts\batch_NN.jsonl")
    rubric = rubric.replace("Do not open other batches,", "Do not open other batches, `map.json`,")
    assert "rejudge_verdicts" in rubric and "map.json" in rubric, "rubric path rewrite failed"
    (TASKS / "INSTRUCTIONS.md").write_text(rubric, encoding="utf-8")
    by_src = defaultdict(int)
    for r in rows:
        by_src[r["src"]] += 1
    print(f"{len(tasks)} answers ({dict(by_src)}) -> {len(list(TASKS.glob('batch_*.json')))} batches of {batch}")


def report() -> None:
    qs = S18.questions()
    mapping = json.loads((TASKS / "map.json").read_text(encoding="utf-8"))
    new = {}
    for f in sorted(VERDICTS.glob("batch_*.jsonl")):
        for line in open(f, encoding="utf-8"):
            if line.strip():
                v = json.loads(line)
                new[v["task_id"]] = v["verdict"]
    first = {("gen", t): v["verdict"] for t, v in S18.load_jsonl("judge_verdicts/*.jsonl").items()}
    first.update({("agentic", t): v["verdict"] for t, v in S18.load_jsonl("judge_verdicts_agentic/*.jsonl").items()})

    rows = {(r["src"], r["task_id"]): r for r in pooled()}
    cells: dict[tuple[str, str, str], float] = {}
    agree = defaultdict(lambda: [0, 0, 0.0])  # arm -> [same verdict, n, sum(first - new)]
    for aid, verdict in new.items():
        m = mapping[aid]
        r = rows[(m["src"], m["task_id"])]
        old = first.get((m["src"], m["task_id"]))
        for arm, qid, lang in r["cells"]:
            cells[(arm, qid, lang)] = SCORE[verdict]
            if old is not None:
                a = agree[arm]
                a[0] += old == verdict
                a[1] += 1
                a[2] += SCORE[old] - SCORE[verdict]
    missing = len(mapping) - len(new)
    print(f"{len(new)}/{len(mapping)} answers judged ({missing} missing) -> {len(cells)} cells")

    def mean(arm: str, pred) -> str:
        vals = [s for (a, q, l), s in cells.items() if a == arm and pred(q, l)]
        return f"{sum(vals) / len(vals):.2f}" if vals else "-"

    cols = [("all", lambda q, l: True)] + [(t, lambda q, l, t=t: qs[q]["type"] == t) for t in "ABCD"] + \
           [(lg, lambda q, l, lg=lg: l == lg) for lg in ("en", "ko")] + \
           [(p, lambda q, l, p=p: qs[q]["period_spec"] == p) for p in ("explicit", "ambiguous")]
    print(f"\nblind re-judge accuracy (correct 1, partial 0.5)\n{'arm':<15}" + "".join(f"{c:>10}" for c, _ in cols))
    for arm in ARMS:
        print(f"{arm:<15}" + "".join(f"{mean(arm, p):>10}" for _, p in cols))

    print("\nagreement with the first-pass verdicts (first pass: fixed arms blind, agentic not blind)")
    for arm in ARMS:
        same, n, diff = agree[arm]
        if n:
            print(f"  {arm:<15} same verdict {same / n:.2f} ({n} cells), first pass minus re-judge {diff / n:+.3f} per cell")

    print("\npaired sign tests on score")
    for i, a in enumerate(ARMS):
        for b in ARMS[i + 1:]:
            for types in ("ABCD", "A", "D"):
                w = l = 0
                for (arm, q, lg), s in cells.items():
                    if arm == a and (b, q, lg) in cells and qs[q]["type"] in types:
                        w += s > cells[(b, q, lg)]
                        l += s < cells[(b, q, lg)]
                print(f"  {a} vs {b} [{types}]: {w} better, {l} worse, p={S18.sign_test(w, l):.4f}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "prepare":
        prepare(int(sys.argv[sys.argv.index("--batch") + 1]) if "--batch" in sys.argv else 40)
    elif cmd == "report":
        report()
    else:
        sys.exit(__doc__)
