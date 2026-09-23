"""Faithfulness (groundedness) of the generated answers, plus citation accuracy.

Retrieval R@k (runner) and answer accuracy (18, 21, 22) say whether the right page was found and whether
the answer matched the reference. Neither says whether the answer's claims are actually *on* the pages the
answer was built from: an answer can match the reference and still invent a figure it never saw.

Two independent parts:

  prepare/report  a claim-level groundedness pass graded by subagents (no API key on this box, same pattern
                  as 17/18 and 22). A sample of answers is pooled across arms, given anonymous ids and
                  shuffled; each task carries only the question, the answer text and the evidence page
                  images. The arm, the reference answer and the correctness verdict stay in
                  faith_tasks/map.json, which graders may not open.
                  Evidence pages: for the fixed arms the three pages their generation task contained
                  (gen_tasks/key.json); for the agent arms the pages the answer itself cites.

  report also     citation accuracy for every arm and run, computed here with no subagent: the share of
                  cited pages that are gold, and the share of answers citing at least one gold page.

    PYTHONIOENCODING=utf-8 PYTHONPATH=src python scripts/27_faithfulness.py prepare [--n 20] [--seed 0] [--batch 10]
    PYTHONIOENCODING=utf-8 PYTHONPATH=src python scripts/27_faithfulness.py report

Graders write eval/results/faith_verdicts/batch_NN.jsonl (see faith_tasks/INSTRUCTIONS.md).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import pypdfium2 as pdfium

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
RES = ROOT / "eval" / "results"
TASKS, VERDICTS = RES / "faith_tasks", RES / "faith_verdicts"
IMG_DIR = ROOT / "data" / "images" / "faithfulness"
EVAL_IMG_DIR = ROOT / "data" / "images" / "eval_pages"  # already rendered by 17, reused as-is
TARGET_WIDTH = 1400  # same as 17: small chart labels stay legible
RUN1_ARMS = ["text_rerank", "vision", "hybrid_rerank", "agentic", "agentic_vision"]
GROUNDED = {"grounded": 1.0, "partial": 0.5, "ungrounded": 0.0}

from pharma_vision_rag.eval.metrics import gold_groups  # noqa: E402


def _load(script: str):
    spec = importlib.util.spec_from_file_location(f"_{script}", ROOT / "scripts" / f"{script}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


S22 = _load("22_blind_rejudge")
S18 = S22.S18


def questions(path: Path = ROOT / "eval" / "questions.jsonl") -> dict[str, dict]:
    return {q["id"]: q for q in map(json.loads, open(path, encoding="utf-8")) if q}


def parse_cited(cited) -> list[tuple[str, int]]:
    """Citations in either format -> [(source.pdf, page)], de-duplicated, order kept.

    Fixed arms write "sanofi_2025Q1_pr.pdf p13"; the agents write {"source": ..., "page": 13}.
    """
    out: dict[tuple[str, int], None] = {}
    for c in cited or []:
        if isinstance(c, dict):
            source, page = c.get("source"), c.get("page")
        else:
            m = re.fullmatch(r"\s*(.+?\.pdf)\s*,?\s*p\.?\s*(\d+)\s*", str(c), re.IGNORECASE)
            if not m:
                continue
            source, page = m.group(1), m.group(2)
        try:
            out.setdefault((str(source), int(page)), None)
        except (TypeError, ValueError):
            continue
    return list(out)


def gold_of(question: dict) -> set[tuple[str, int]]:
    return set().union(*gold_groups(question))


def answer_rows(qs: dict[str, dict] | None = None) -> list[dict]:
    """One row per (arm, question, language) cell of every run, with its answer and evidence pages."""
    qs = qs or questions()
    key = json.loads((RES / "gen_tasks" / "key.json").read_text(encoding="utf-8"))
    rows = []
    for src, folder in S22.GEN_RUNS.items():  # rep runs re-answer the *same* tasks, so key.json fits both
        suffix = "@rep2" if src.endswith("_rep2") else ""
        for tid, a in S18.load_jsonl(f"{folder}/*.jsonl").items():
            k = key[tid]
            for variant in k["used_by"]:
                rows.append({"arm": variant + suffix, "src": src, "task_id": tid, "qid": k["id"], "lang": k["lang"],
                             "question": k["question"], "answer": a["answer"], "cited": a.get("cited"),
                             "pages": [(d, int(p)) for d, p in k["pages"]]})
    for run in S22.AGENT_RUNS:
        for tid, a in S18.load_jsonl(f"{run}_answers/*.jsonl").items():
            qid, lang = tid.rsplit("_", 1)
            rows.append({"arm": S22.arm_of(run), "src": run, "task_id": tid, "qid": qid, "lang": lang,
                         "question": qs[qid][f"q_{lang}"], "answer": a["answer"], "cited": a.get("cited"),
                         "pages": parse_cited(a.get("cited"))})
    return rows


def sample_cells(qs: dict[str, dict], n: int, seed: int) -> list[tuple[str, str]]:
    """n (question, language) cells, drawn round-robin over the question types so D (multi-hop) is not lost."""
    rng = random.Random(seed)
    by_type: dict[str, list[str]] = defaultdict(list)
    for qid in sorted(qs):
        by_type[qs[qid]["type"]].append(qid)
    for pool in by_type.values():
        rng.shuffle(pool)
    picked: list[str] = []
    while len(picked) < n and any(by_type.values()):
        for t in sorted(by_type):
            if by_type[t] and len(picked) < n:
                picked.append(by_type[t].pop())
    return [(qid, rng.choice(("ko", "en"))) for qid in sorted(picked)]


def render(doc: str, page: int) -> Path:
    shared = EVAL_IMG_DIR / f"{doc[:-4]}_p{page}.png"
    if shared.exists():
        return shared
    dst = IMG_DIR / f"{doc[:-4]}_p{page}.png"
    if not dst.exists():
        IMG_DIR.mkdir(parents=True, exist_ok=True)
        pg = pdfium.PdfDocument(str(ROOT / "data" / "pdf" / "corpus" / doc))[page - 1]
        pg.render(scale=TARGET_WIDTH / pg.get_width()).to_pil().convert("RGB").save(dst)
    return dst


RUBRIC = """# 근거성(faithfulness) 채점 지침

답변이 기준 정답과 맞았는지가 아니라, 답변의 주장이 함께 주어진 페이지 이미지로 뒷받침되는지를 봅니다.
어느 방식이 만든 답변인지, 채점 결과가 무엇이었는지는 알 수 없고 알아내려 해서도 안 돼요.

작업 파일은 이 폴더의 `batch_NN.json`입니다(NN은 프롬프트로 받음). JSON 리스트이고 항목은
`{task_id, question, answer, images}`예요. `images`는 그 답변이 근거로 삼은 페이지의 PNG 절대 경로입니다.

## 열어도 되는 파일

- 이 지침 파일과 배정받은 `batch_NN.json` 하나
- 그 파일의 `images`에 적힌 PNG

그 외에는 열지 않습니다. 다른 묶음, `map.json`, `eval/questions.jsonl`, PDF 원본, 다른 폴더와 웹은 금지예요.
주어진 이미지만으로 판단합니다.

## 판정 방법

1. 답변을 사실 주장 단위로 쪼갭니다. 수치, 기간, 주체, 비교 서술이 각각 하나의 주장이에요.
   - 인사말, 출처 표기와 "찾을 수 없습니다" 같은 메타 문장은 주장이 아닙니다.
   - 답변이 계산 결과를 제시하면, 계산에 쓴 각 값과 계산 결과를 각각 주장으로 셉니다.
2. 주장마다 다음 중 하나로 표시해요.
   - `supported`: 페이지에 그대로 있거나, 페이지의 값에서 단순 계산(차이, 비율, 합계)으로 바로 나옴
   - `unsupported`: 페이지에서 확인되지 않음. 페이지에 없는 값을 외부 지식이나 추정으로 채운 경우 포함
   - `contradicted`: 페이지의 값과 어긋남
3. 수치는 단위와 기간까지 정확히 맞아야 `supported`입니다. 값은 맞는데 분기나 연도가 다르면 `contradicted`,
   통화나 단위가 빠졌거나 틀리면 `unsupported`로 봅니다. 출처에 인쇄된 자리수로 반올림한 것은 맞는 것으로 쳐요.
4. 표기 형식은 따지지 않습니다. `EUR 3,832 million`과 `€3,832m`와 `3.832 billion euros`는 같은 값이에요.
   한국어 답변과 영어 답변의 기준은 같습니다.
5. 기준 정답을 모르는 상태로 채점합니다. 답변이 질문에 제대로 답했는지는 보지 않아요. 페이지에 없는 말을
   했는지만 봅니다. 회사에 대한 외부 지식으로 페이지를 보충하지 마세요.
6. 답변이 "정보를 찾을 수 없습니다"만 말하고 사실 주장이 없으면 `n_claims`를 0, `verdict`를 `grounded`로 씁니다.

## 판정값

- `grounded`: 모든 주장이 `supported`
- `partial`: 일부만 `supported`이고 `contradicted`는 없음
- `ungrounded`: `contradicted`가 하나라도 있음

## 출력

한 줄에 JSON 한 개씩, 작업 파일과 같은 순서로
`OUT_PATH\\batch_NN.jsonl`(같은 NN)에 씁니다.

`{"task_id": "...", "n_claims": 0, "n_supported": 0, "n_unsupported": 0, "n_contradicted": 0,
"verdict": "grounded|partial|ungrounded", "notes": "<짧은 한 문장>"}`

- `n_supported + n_unsupported + n_contradicted == n_claims`여야 합니다.
- UTF-8, 유효한 JSON, 배정받은 묶음의 **모든** 작업을 빠짐없이 판정해요. 건너뛴 작업이 있으면 안 됩니다.
- 보조 스크립트를 쓴다면 `/tmp` 아래에 두고, 다른 파일을 읽지 않게 하세요.

다 하면 grounded, partial, ungrounded 건수와 이미지를 열지 못한 작업 번호만 답으로 적습니다.
답변 내용은 적지 마세요.
"""


def prepare(n: int, seed: int, batch: int, arms: list[str]) -> None:
    qs = questions()
    cells = sample_cells(qs, n, seed)
    by_cell = {(r["arm"], r["qid"], r["lang"]): r for r in answer_rows(qs)}
    chosen: dict[tuple[str, str], dict] = {}
    missing, no_pages = [], []
    for arm in arms:
        for qid, lang in cells:
            r = by_cell.get((arm, qid, lang))
            if r is None:
                missing.append((arm, qid, lang))
                continue
            if not r["pages"]:
                no_pages.append((arm, qid, lang))
                continue
            entry = chosen.setdefault((r["src"], r["task_id"]), {**r, "arms": []})
            entry["arms"].append(arm)  # one fixed-arm answer often serves several arms

    tasks, mapping = [], {}
    for (src, tid), r in sorted(chosen.items(), key=lambda kv: hashlib.sha1(f"faith:{kv[0][0]}:{kv[0][1]}".encode()).hexdigest()):
        aid = hashlib.sha1(f"faith:{src}:{tid}".encode()).hexdigest()[:10]
        tasks.append({"task_id": aid, "question": r["question"], "answer": r["answer"],
                      "images": [str(render(d, p)) for d, p in r["pages"]]})
        mapping[aid] = {"src": src, "task_id": tid, "arms": r["arms"], "qid": r["qid"], "lang": r["lang"],
                        "pages": [[d, p] for d, p in r["pages"]]}
    assert len(mapping) == len(chosen), "anonymous id collision"

    TASKS.mkdir(parents=True, exist_ok=True)
    VERDICTS.mkdir(parents=True, exist_ok=True)
    for old in TASKS.glob("batch_*.json"):
        old.unlink()
    for i, start in enumerate(range(0, len(tasks), batch)):
        (TASKS / f"batch_{i:02d}.json").write_text(json.dumps(tasks[start:start + batch], ensure_ascii=False, indent=1),
                                                   encoding="utf-8")
    (TASKS / "map.json").write_text(json.dumps(mapping, ensure_ascii=False, indent=1), encoding="utf-8")
    (TASKS / "INSTRUCTIONS.md").write_text(RUBRIC.replace("OUT_PATH", str(VERDICTS)), encoding="utf-8")

    per_arm = defaultdict(int)
    for m in mapping.values():
        for arm in m["arms"]:
            per_arm[arm] += 1
    n_batches = len(list(TASKS.glob("batch_*.json")))
    print(f"{len(cells)} questions x {len(arms)} arms = {len(cells) * len(arms)} cells "
          f"-> {len(tasks)} unique answers to grade (fixed arms share an answer when their top-3 pages match), "
          f"{n_batches} batches of {batch}")
    print(f"  per arm: {dict(sorted(per_arm.items()))}")
    print(f"  images: {sum(len(t['images']) for t in tasks)} page references, "
          f"{len(list(IMG_DIR.glob('*.png'))) if IMG_DIR.exists() else 0} newly rendered here, rest reused from eval_pages")
    if no_pages:
        print(f"  skipped, no evidence pages (agent cited nothing): {len(no_pages)} {no_pages}")
    if missing:
        print(f"  skipped, no answer on file: {len(missing)} {missing}")


def faith_verdicts() -> dict[str, dict]:
    out = {}
    for f in sorted(VERDICTS.glob("batch_*.jsonl")):
        for line in open(f, encoding="utf-8"):
            if line.strip():
                v = json.loads(line)
                out[v["task_id"]] = v
    return out


def correctness() -> dict[tuple[str, str], str]:
    """(src, task_id) -> correct|partial|wrong from script 22's rounds; earliest round that judged it wins."""
    final: dict[tuple[str, str], str] = {}
    for rnd in S22.ROUNDS:
        mapping, got = S22.load_map(rnd), S22.verdicts(rnd)
        for aid, verdict in got.items():
            m = mapping.get(aid)
            if m and not m.get("anchor"):
                final.setdefault((m["src"], m["task_id"]), verdict)
    return final


def aggregate(mapping: dict[str, dict], got: dict[str, dict]) -> dict[str, dict]:
    """Per arm: answers graded, mean per-answer claim support rate, fully grounded share, contradiction rate."""
    out: dict[str, dict] = {}
    for aid, v in got.items():
        m = mapping.get(aid)
        if m is None:
            continue
        claims = v.get("n_claims") or 0
        rate = (v.get("n_supported") or 0) / claims if claims else 1.0  # no claims = nothing to hallucinate
        for arm in m["arms"]:
            s = out.setdefault(arm, {"n": 0, "claims": 0, "supported": 0, "contradicted": 0,
                                     "rate_sum": 0.0, "grounded": 0, "with_contradiction": 0})
            s["n"] += 1
            s["claims"] += claims
            s["supported"] += v.get("n_supported") or 0
            s["contradicted"] += v.get("n_contradicted") or 0
            s["rate_sum"] += rate
            s["grounded"] += v.get("verdict") == "grounded"
            s["with_contradiction"] += (v.get("n_contradicted") or 0) > 0
    return out


def citation_stats(rows: list[dict], qs: dict[str, dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for r in rows:
        gold = gold_of(qs[r["qid"]])
        pages = parse_cited(r["cited"])
        s = out.setdefault(r["arm"], {"answers": 0, "with_citations": 0, "cited": 0, "in_gold": 0, "any_gold": 0})
        s["answers"] += 1
        s["with_citations"] += bool(pages)
        s["cited"] += len(pages)
        s["in_gold"] += sum(p in gold for p in pages)
        s["any_gold"] += any(p in gold for p in pages)
    return out


def report() -> None:
    qs = questions()
    rows = answer_rows(qs)

    mapping = json.loads((TASKS / "map.json").read_text(encoding="utf-8")) if (TASKS / "map.json").exists() else {}
    got = faith_verdicts()
    if mapping and got:
        print(f"faithfulness: {len(got)}/{len(mapping)} graded answers")
        stats = aggregate(mapping, got)
        print(f"\n{'arm':<18}{'answers':>9}{'claims':>8}{'support':>9}{'grounded':>10}{'contradict':>12}")
        for arm in [a for a in RUN1_ARMS if a in stats] + [a for a in sorted(stats) if a not in RUN1_ARMS]:
            s = stats[arm]
            print(f"{arm:<18}{s['n']:>9}{s['claims']:>8}{s['rate_sum'] / s['n']:>9.2f}"
                  f"{s['grounded'] / s['n']:>10.2f}{s['with_contradiction'] / s['n']:>12.2f}")
        print("  support = mean over answers of supported/claims; grounded = every claim supported; "
              "contradict = share of answers with at least one contradicted claim")

        # Cross-tab with the correctness verdicts: correct-but-ungrounded is the case retrieval and accuracy miss.
        correct = correctness()
        cross: dict[str, dict[tuple[str, str], int]] = defaultdict(lambda: defaultdict(int))
        for aid, v in got.items():
            m = mapping.get(aid)
            c = correct.get((m["src"], m["task_id"])) if m else None
            if c:
                for arm in m["arms"]:
                    cross[arm][(c, v.get("verdict"))] += 1
        if cross:
            corr, grnd = ("correct", "partial", "wrong"), ("grounded", "partial", "ungrounded")
            print("\ncorrectness x groundedness (counts of graded answers; cor/par/wr x grounded/part/ungr)")
            print(f"{'arm':<18}" + "".join(f"{c[:3] + '/' + g[:4]:>10}" for c in corr for g in grnd))
            for arm, tab in sorted(cross.items()):
                print(f"{arm:<18}" + "".join(f"{tab[(c, g)]:>10}" for c in corr for g in grnd))
            print("  correct/ungrounded = scored right while contradicting its own pages; "
                  "wrong/grounded = faithful to pages that did not hold the answer")

        # Paired sign tests on the per-cell support rate, same shape as script 22.
        per_cell: dict[tuple[str, str, str], float] = {}
        for aid, v in got.items():
            m = mapping.get(aid)
            if m is None:
                continue
            claims = v.get("n_claims") or 0
            for arm in m["arms"]:
                per_cell[(arm, m["qid"], m["lang"])] = (v.get("n_supported") or 0) / claims if claims else 1.0
        arms = [a for a in RUN1_ARMS if any(c[0] == a for c in per_cell)]
        print("\npaired sign tests on the per-cell claim support rate")
        for i, a in enumerate(arms):
            for b in arms[i + 1:]:
                w = l = 0
                for (arm, qid, lang), s in per_cell.items():
                    if arm == a and (b, qid, lang) in per_cell:
                        w += s > per_cell[(b, qid, lang)]
                        l += s < per_cell[(b, qid, lang)]
                print(f"  {a} vs {b}: {w} better, {l} worse, p={S18.sign_test(w, l):.4f} (n={w + l} discordant cells)")
    else:
        print(f"faithfulness: nothing graded yet ({len(mapping)} tasks prepared, {len(got)} verdicts). "
              f"Run prepare, then the grading subagents.")

    print("\ncitation accuracy (every arm and run, all 120 cells each, no grading needed)")
    print(f"{'arm':<18}{'answers':>9}{'cited':>8}{'in gold':>9}{'>=1 gold':>10}{'no citation':>13}")
    stats = citation_stats(rows, qs)
    for arm in sorted(stats, key=lambda a: (a.split("@")[-1] if "@" in a else "", RUN1_ARMS.index(a.split("@")[0])
                                            if a.split("@")[0] in RUN1_ARMS else 99)):
        s = stats[arm]
        print(f"{arm:<18}{s['answers']:>9}{s['cited']:>8}"
              f"{s['in_gold'] / s['cited'] if s['cited'] else 0:>9.2f}"
              f"{s['any_gold'] / s['answers']:>10.2f}{s['answers'] - s['with_citations']:>13}")
    print("  in gold = share of cited pages that are gold for that question (gold_pages, all hops); "
          "'>=1 gold' counts answers citing at least one. Fixed arms cite from the 3 pages they were given.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--n", type=int, default=20, help="questions to sample (one language each)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch", type=int, default=10, help="tasks per batch file (up to 3 images each)")
    p.add_argument("--arms", default=",".join(RUN1_ARMS))
    sub.add_parser("report")
    a = ap.parse_args()
    if a.cmd == "prepare":
        prepare(a.n, a.seed, a.batch, a.arms.split(","))
    else:
        report()
