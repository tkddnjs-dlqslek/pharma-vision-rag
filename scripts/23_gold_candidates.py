"""List pages that correct answers cited but gold_pages does not contain (gold completeness check).

Every answer judged correct or partial in the blind re-judge (22_blind_rejudge.py) cites the pages it
used. A cited page outside gold_pages is either a missing gold page or a page the answer only leaned
on for context. This script collects those (question, page) pairs and writes one verification task
per pair, with the page rendered to PNG, for subagents to check against the reference answer. Gold is
only changed after that check (and a render by hand for anything surprising), never from citations alone.

    PYTHONIOENCODING=utf-8 python scripts/23_gold_candidates.py [--batch 15]

Output: eval/results/gold_check/batch_NN.json [{check_id, question, reference_answer, image, page}]
        eval/results/gold_check/key.json      {check_id: {qid, source, page, cited_by}}
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
from collections import defaultdict
from pathlib import Path

import pypdfium2 as pdfium

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "eval" / "results"
OUT = RES / "gold_check"
IMG_DIR = ROOT / "data" / "images" / "eval_pages"
TARGET_WIDTH = 1400


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def render(doc: str, page: int) -> Path:
    dst = IMG_DIR / f"{doc[:-4]}_p{page}.png"
    if not dst.exists():
        pg = pdfium.PdfDocument(str(ROOT / "data" / "pdf" / "corpus" / doc))[page - 1]
        pg.render(scale=TARGET_WIDTH / pg.get_width()).to_pil().convert("RGB").save(dst)
    return dst


def cited_pages(answer: dict) -> list[tuple[str, int]]:
    out = []
    for c in answer.get("cited", []):
        if isinstance(c, dict):
            out.append((c.get("source"), int(c.get("page"))))
        else:  # "doc.pdf p12"
            m = re.match(r"(\S+\.pdf) p(\d+)$", str(c).strip())
            if m:
                out.append((m.group(1), int(m.group(2))))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=15)
    a = ap.parse_args()
    rj = _load("_rejudge", "22_blind_rejudge.py")
    s18 = rj.S18
    qs = s18.questions()

    judged: dict[tuple[str, str], str] = {}
    for rnd in ("r1", "r2"):
        mapping, got = rj.load_map(rnd), rj.verdicts(rnd)
        for aid, v in got.items():
            m = mapping[aid]
            if not m.get("anchor"):
                judged.setdefault((m["src"], m["task_id"]), v)

    answers = {("gen", t): r for t, r in s18.load_jsonl("gen_answers/*.jsonl").items()}
    for run in rj.AGENT_RUNS:
        answers.update({(run, t): r for t, r in s18.load_jsonl(f"{run}_answers/*.jsonl").items()})
    key = json.loads((RES / "gen_tasks" / "key.json").read_text(encoding="utf-8"))

    cand: dict[tuple[str, str, int], set[str]] = defaultdict(set)
    for (src, tid), verdict in judged.items():
        if verdict == "wrong":
            continue
        qid = key[tid]["id"] if src == "gen" else tid.rsplit("_", 1)[0]
        gold = {(g["source"], g["page"]) for g in qs[qid]["gold_pages"]}
        for doc, page in cited_pages(answers[(src, tid)]):
            if doc and (doc, page) not in gold:
                cand[(qid, doc, page)].add(f"{src}:{tid}")

    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("batch_*.json"):
        old.unlink()
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    tasks, keymap = [], {}
    for (qid, doc, page), who in sorted(cand.items()):
        cid = hashlib.sha1(f"gold:{qid}:{doc}:{page}".encode()).hexdigest()[:10]
        q = qs[qid]
        tasks.append({"check_id": cid, "question": q["q_en"], "reference_answer": q["answer"],
                      "page": f"{doc} p{page}", "image": str(render(doc, page))})
        keymap[cid] = {"qid": qid, "source": doc, "page": page, "cited_by": sorted(who)}
    for n, i in enumerate(range(0, len(tasks), a.batch)):
        (OUT / f"batch_{n:02d}.json").write_text(json.dumps(tasks[i:i + a.batch], ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "key.json").write_text(json.dumps(keymap, ensure_ascii=False, indent=1), encoding="utf-8")
    per_q = defaultdict(int)
    for qid, _, _ in cand:
        per_q[qid] += 1
    print(f"{len(tasks)} candidate pages over {len(per_q)} questions -> {len(list(OUT.glob('batch_*.json')))} batches")
    print("most candidates:", sorted(per_q.items(), key=lambda x: -x[1])[:10])


if __name__ == "__main__":
    main()
