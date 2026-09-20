"""Haiku judge for the RAG benchmark (EXPERIMENT_PLAN §6): scores each generated answer
against the reference answer on a 3-level scale, averaged over repeats.

Rubric:
    correct  every figure, unit, and period in the model answer matches the reference
    partial  the right figure but a wrong/missing unit or period, OR only some of several
             required values are present
    wrong    otherwise, including refusals ("Not found in the provided pages." /
             "정보를 찾을 수 없습니다.") when the reference has an answer

Deterministic pre-check (``keys_matched``): true iff every ``answer_keys`` string is a
substring of the model answer. Recorded alongside the judge verdict, never used to skip
the judge call — string containment can tell you a figure is present, not whether its unit
or period is right, which is exactly the correct/partial boundary the rubric turns on.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from pharma_vision_rag.eval.runner import QUESTIONS, RESULTS
from pharma_vision_rag.generator.claude_text import ClaudeTextGenerator

SCORES = {"correct": 1.0, "partial": 0.5, "wrong": 0.0}

JUDGE_SYSTEM = """You grade a model's answer against a reference answer for a pharma financial-disclosure Q&A benchmark.

Rubric:
- correct: every figure, unit, and period in the model answer matches the reference answer.
- partial: the right figure appears but with a wrong or missing unit or period, OR only some of several required values are present.
- wrong: otherwise, including refusals ("Not found...", "정보를 찾을 수 없습니다.") when the reference has an answer.

Output ONLY a JSON object, no markdown code fences, no other text:
{"verdict": "correct" | "partial" | "wrong", "reason": "<one short sentence>"}"""

JUDGE_USER_TEMPLATE = """Question: {question}
Reference answer: {reference}
Required values (answer_keys): {answer_keys}
Model answer: {model_answer}"""

_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_verdict_json(text: str) -> dict[str, Any] | None:
    """Strict-then-lenient JSON parse. Returns None (never raises) if nothing usable is found."""
    for candidate in (text, (_JSON_OBJ_RE.search(text) or [None]).group() if _JSON_OBJ_RE.search(text) else None):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, dict) and parsed.get("verdict") in SCORES:
            return parsed
    return None


def judge_one(
    generator: ClaudeTextGenerator,
    question: dict[str, Any],
    model_answer: str,
    max_attempts: int = 2,
) -> dict[str, Any]:
    """One judged record: {verdict, reason, keys_matched, raw, n_attempts}. Retries once on bad JSON."""
    answer_keys = question.get("answer_keys") or []
    keys_matched = bool(answer_keys) and all(k in model_answer for k in answer_keys)
    user = JUDGE_USER_TEMPLATE.format(
        question=question.get("q_en") or question.get("q_ko", ""),
        reference=question["answer"],
        answer_keys=", ".join(answer_keys),
        model_answer=model_answer,
    )

    raw = ""
    for attempt in range(1, max_attempts + 1):
        raw = generator.generate(system=JUDGE_SYSTEM, user=user)["answer"]
        parsed = parse_verdict_json(raw)
        if parsed is not None:
            return {"verdict": parsed["verdict"], "reason": parsed.get("reason", ""),
                    "keys_matched": keys_matched, "raw": raw, "n_attempts": attempt}

    return {"verdict": "wrong", "reason": f"judge output not parseable after {max_attempts} attempts: {raw!r}",
            "keys_matched": keys_matched, "raw": raw, "n_attempts": max_attempts}


def aggregate(records: list[dict[str, Any]], group_key: Callable[[dict[str, Any]], Any]) -> dict[Any, dict[str, Any]]:
    """Mean/min/max score (correct=1, partial=0.5, wrong=0) pooled over every repeat in each group."""
    groups: dict[Any, list[float]] = defaultdict(list)
    for r in records:
        groups[group_key(r)].append(SCORES[r["verdict"]])
    return {k: {"n": len(v), "mean": sum(v) / len(v), "min": min(v), "max": max(v)} for k, v in groups.items()}


def _load_questions_by_id() -> dict[str, dict[str, Any]]:
    return {q["id"]: q for q in (json.loads(l) for l in QUESTIONS.read_text(encoding="utf-8").splitlines() if l.strip())}


def _existing_keys(path: Path) -> set[tuple[str, str, str, int]]:
    if not path.exists():
        return set()
    return {(r["variant"], r["id"], r["lang"], r["repeat"])
            for r in (json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, help="reads eval/results/generation_<variant>.jsonl")
    a = ap.parse_args()

    gen_path = RESULTS / f"generation_{a.variant}.jsonl"
    if not gen_path.exists():
        raise SystemExit(f"no generation file: {gen_path} (run eval.generate --variant {a.variant} first)")
    out_path = RESULTS / f"judged_{a.variant}.jsonl"
    done = _existing_keys(out_path)
    questions = _load_questions_by_id()
    judge = ClaudeTextGenerator()  # default model claude-haiku-4-5, per CLAUDE.md

    rows = [json.loads(l) for l in gen_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    n_written = n_skipped = 0
    with open(out_path, "a", encoding="utf-8") as f:
        for row in rows:
            key = (row["variant"], row["id"], row["lang"], row["repeat"])
            if key in done:
                n_skipped += 1
                continue
            q = questions[row["id"]]
            verdict = judge_one(judge, q, row["answer"])
            record = {**{k: row[k] for k in ("variant", "id", "lang", "repeat")},
                       "type": q["type"], "period_spec": q["period_spec"], **verdict}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            n_written += 1
    print(f"{n_written} judged, {n_skipped} skipped (already judged) -> {out_path}")

    judged_rows = [json.loads(l) for l in out_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    for title, key in (
        ("overall", lambda r: r["variant"]),
        ("by type", lambda r: (r["variant"], r["type"])),
        ("by language", lambda r: (r["variant"], r["lang"])),
        ("by period_spec", lambda r: (r["variant"], r["period_spec"])),
    ):
        print(f"\n{title}")
        for k, s in sorted(aggregate(judged_rows, key).items(), key=lambda kv: str(kv[0])):
            print(f"  {k}: n={s['n']} mean={s['mean']:.3f} min={s['min']:.2f} max={s['max']:.2f}")


if __name__ == "__main__":
    main()
