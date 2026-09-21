"""Mixed blind re-judge of every arm, in rounds.

Why: the first judging pass scored the three fixed arms together (blind to arm) but scored the
agentic answers separately, so that judge knew which arm it was grading. Round r1 pools every answer
from all arms, gives each an anonymous id, shuffles, and has fresh subagents judge them with the same
rubric (eval/results/judge_tasks/INSTRUCTIONS.md, copied with new paths). The map from anonymous id
back to arm lives in <round>_tasks/map.json, which judges may not open.

A later arm (agentic_vision, E4b) is judged in round r2: only the answers r1 has not seen, plus
--anchors answers r1 already judged, re-labelled with fresh ids. The anchors measure how far the r2
judges drift from r1; if drift is small, r1 and r2 verdicts are pooled into one table.

    PYTHONIOENCODING=utf-8 python scripts/22_blind_rejudge.py prepare [--batch 40]                 # r1
    PYTHONIOENCODING=utf-8 python scripts/22_blind_rejudge.py prepare --round r2 --anchors 40      # new arms only
    PYTHONIOENCODING=utf-8 python scripts/22_blind_rejudge.py report

report prints per-arm accuracy, paired sign tests, agreement with the first-pass verdicts
(judge_verdicts/ and judge_verdicts_agentic/), and the r2 anchor agreement when r2 exists.
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
SCORE = {"correct": 1.0, "partial": 0.5, "wrong": 0.0}
ARMS = ["text_rerank", "vision", "hybrid_rerank", "agentic", "agentic_vision"]
AGENT_RUNS = ["agentic", "agentic_vision"]  # <name>_answers/*.jsonl, task_id "<qid>_<lang>"


def _load_18():
    spec = importlib.util.spec_from_file_location("_score_gen", ROOT / "scripts" / "18_score_generation.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


S18 = _load_18()


def dirs(rnd: str) -> tuple[Path, Path]:
    return (RES / "rejudge_tasks", RES / "rejudge_verdicts") if rnd == "r1" else \
           (RES / f"rejudge_{rnd}_tasks", RES / f"rejudge_{rnd}_verdicts")


def anon(salt: str, src: str, task_id: str) -> str:
    return hashlib.sha1(f"{salt}:{src}:{task_id}".encode()).hexdigest()[:10]


def pooled() -> list[dict]:
    """Every generated answer once, with the (arm, question, lang) cells it stands for."""
    qs, key = S18.questions(), json.loads((RES / "gen_tasks" / "key.json").read_text(encoding="utf-8"))
    out = []
    for tid, a in S18.load_jsonl("gen_answers/*.jsonl").items():
        k = key[tid]
        out.append({"src": "gen", "task_id": tid, "qid": k["id"], "lang": k["lang"], "question": k["question"],
                    "answer": a["answer"], "cells": [(v, k["id"], k["lang"]) for v in k["used_by"]]})
    for run in AGENT_RUNS:
        for tid, a in S18.load_jsonl(f"{run}_answers/*.jsonl").items():
            qid, lang = tid.rsplit("_", 1)
            out.append({"src": run, "task_id": tid, "qid": qid, "lang": lang, "question": qs[qid][f"q_{lang}"],
                        "answer": a["answer"], "cells": [(run, qid, lang)], "tool_calls": a.get("tool_calls")})
    return out


def load_map(rnd: str) -> dict[str, dict]:
    f = dirs(rnd)[0] / "map.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


def prepare(batch: int, rnd: str, anchors: int) -> None:
    qs = S18.questions()
    rows = pooled()
    salt = "rejudge" if rnd == "r1" else f"rejudge-{rnd}"
    if rnd == "r1":
        chosen = [(r, False) for r in rows]
    else:
        seen = {(m["src"], m["task_id"]) for m in load_map("r1").values()}
        new = [r for r in rows if (r["src"], r["task_id"]) not in seen]
        old = sorted((r for r in rows if (r["src"], r["task_id"]) in seen), key=lambda r: anon("anchor", r["src"], r["task_id"]))
        chosen = [(r, False) for r in new] + [(r, True) for r in old[:anchors]]
    chosen.sort(key=lambda x: anon(salt, x[0]["src"], x[0]["task_id"]))  # hash order = arm-agnostic shuffle
    tasks, mapping = [], {}
    for r, is_anchor in chosen:
        aid = anon(salt, r["src"], r["task_id"])
        q = qs[r["qid"]]
        norm = " ".join(r["answer"].split())
        tasks.append({"task_id": aid, "question": r["question"], "reference_answer": q["answer"],
                      "answer_keys": q["answer_keys"], "model_answer": r["answer"],
                      "keys_matched": bool(q["answer_keys"]) and all(k in norm for k in q["answer_keys"])})
        mapping[aid] = {"src": r["src"], "task_id": r["task_id"], "anchor": is_anchor}
    assert len(mapping) == len(chosen), "anonymous id collision"
    tdir, vdir = dirs(rnd)
    tdir.mkdir(exist_ok=True)
    vdir.mkdir(exist_ok=True)
    for old in tdir.glob("batch_*.json"):
        old.unlink()
    for n, i in enumerate(range(0, len(tasks), batch)):
        (tdir / f"batch_{n:02d}.json").write_text(json.dumps(tasks[i:i + batch], ensure_ascii=False, indent=1), encoding="utf-8")
    (tdir / "map.json").write_text(json.dumps(mapping, indent=1), encoding="utf-8")
    rubric = (RES / "judge_tasks" / "INSTRUCTIONS.md").read_text(encoding="utf-8")
    out_rel = str(vdir.relative_to(ROOT)).replace("/", "\\")
    rubric = rubric.replace(r"eval\results\judge_verdicts\batch_NN.jsonl", out_rel + r"\batch_NN.jsonl")
    rubric = rubric.replace("Do not open other batches,", "Do not open other batches, `map.json`,")
    assert out_rel in rubric and "map.json" in rubric, "rubric path rewrite failed"
    (tdir / "INSTRUCTIONS.md").write_text(rubric, encoding="utf-8")
    by_src = defaultdict(int)
    for r, is_anchor in chosen:
        by_src["anchor" if is_anchor else r["src"]] += 1
    print(f"{rnd}: {len(tasks)} answers ({dict(by_src)}) -> {len(list(tdir.glob('batch_*.json')))} batches of {batch}")


def verdicts(rnd: str) -> dict[str, str]:
    out = {}
    for f in sorted(dirs(rnd)[1].glob("batch_*.jsonl")):
        for line in open(f, encoding="utf-8"):
            if line.strip():
                v = json.loads(line)
                out[v["task_id"]] = v["verdict"]
    return out


def report() -> None:
    qs = S18.questions()
    rows = {(r["src"], r["task_id"]): r for r in pooled()}
    final: dict[tuple[str, str], str] = {}  # (src, task_id) -> verdict; r1 wins, r2 adds only new answers
    anchor_pairs = []
    for rnd in ("r1", "r2"):
        mapping, got = load_map(rnd), verdicts(rnd)
        if not mapping:
            continue
        print(f"{rnd}: {len(got)}/{len(mapping)} answers judged")
        for aid, verdict in got.items():
            m = mapping[aid]
            k = (m["src"], m["task_id"])
            if m.get("anchor"):
                anchor_pairs.append((final.get(k), verdict))
            elif k not in final:
                final[k] = verdict
    if anchor_pairs:
        pairs = [(a, b) for a, b in anchor_pairs if a is not None]
        same = sum(a == b for a, b in pairs)
        drift = sum(SCORE[b] - SCORE[a] for a, b in pairs) / len(pairs)
        print(f"r2 anchors: same verdict as r1 on {same}/{len(pairs)} ({same / len(pairs):.2f}), r2 minus r1 {drift:+.3f} per answer")

    first = {("gen", t): v["verdict"] for t, v in S18.load_jsonl("judge_verdicts/*.jsonl").items()}
    first.update({("agentic", t): v["verdict"] for t, v in S18.load_jsonl("judge_verdicts_agentic/*.jsonl").items()})
    cells: dict[tuple[str, str, str], float] = {}
    agree = defaultdict(lambda: [0, 0, 0.0])  # arm -> [same verdict, n, sum(first - new)]
    for k, verdict in final.items():
        old = first.get(k)
        for arm, qid, lang in rows[k]["cells"]:
            cells[(arm, qid, lang)] = SCORE[verdict]
            if old is not None:
                a = agree[arm]
                a[0] += old == verdict
                a[1] += 1
                a[2] += SCORE[old] - SCORE[verdict]
    arms = [a for a in ARMS if any(c[0] == a for c in cells)]
    print(f"{len(final)} answers -> {len(cells)} cells")

    def mean(arm: str, pred) -> str:
        vals = [s for (a, q, l), s in cells.items() if a == arm and pred(q, l)]
        return f"{sum(vals) / len(vals):.2f}" if vals else "-"

    cols = [("all", lambda q, l: True)] + [(t, lambda q, l, t=t: qs[q]["type"] == t) for t in "ABCD"] + \
           [(lg, lambda q, l, lg=lg: l == lg) for lg in ("en", "ko")] + \
           [(p, lambda q, l, p=p: qs[q]["period_spec"] == p) for p in ("explicit", "ambiguous")]
    print(f"\nblind re-judge accuracy (correct 1, partial 0.5)\n{'arm':<15}" + "".join(f"{c:>10}" for c, _ in cols))
    for arm in arms:
        print(f"{arm:<15}" + "".join(f"{mean(arm, p):>10}" for _, p in cols))

    # The API loop forces final_answer at MAX_TOOL_CALLS; subagents were only told to stop. Cells that
    # overran are rescored as wrong to show how much the overrun could have bought.
    for run in AGENT_RUNS:
        over = [k for k, r in rows.items() if r["src"] == run and k in final and (r.get("tool_calls") or 0) > 10]
        if over:
            over_cells = {(run, rows[k]["qid"], rows[k]["lang"]) for k in over}
            vals = [0.0 if c in over_cells else s for c, s in cells.items() if c[0] == run]
            print(f"{run}: {len(over)} cells over the 10-call budget ({', '.join(t for _, t in over)}); "
                  f"score if those count as wrong {sum(vals) / len(vals):.2f}")

    print("\nagreement with the first-pass verdicts (first pass: fixed arms blind, agentic not blind)")
    for arm in arms:
        same, n, diff = agree[arm]
        if n:
            print(f"  {arm:<15} same verdict {same / n:.2f} ({n} cells), first pass minus re-judge {diff / n:+.3f} per cell")

    print("\npaired sign tests on score")
    for i, a in enumerate(arms):
        for b in arms[i + 1:]:
            for types in ("ABCD", "A", "C", "D"):
                w = l = 0
                for (arm, q, lg), s in cells.items():
                    if arm == a and (b, q, lg) in cells and qs[q]["type"] in types:
                        w += s > cells[(b, q, lg)]
                        l += s < cells[(b, q, lg)]
                print(f"  {a} vs {b} [{types}]: {w} better, {l} worse, p={S18.sign_test(w, l):.4f}")


if __name__ == "__main__":
    argv = sys.argv
    cmd = argv[1] if len(argv) > 1 else ""
    opt = lambda name, default: argv[argv.index(name) + 1] if name in argv else default  # noqa: E731
    if cmd == "prepare":
        prepare(int(opt("--batch", 40)), opt("--round", "r1"), int(opt("--anchors", 0)))
    elif cmd == "report":
        report()
    else:
        sys.exit(__doc__)
