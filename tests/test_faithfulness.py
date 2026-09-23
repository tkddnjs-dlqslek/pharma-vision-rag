"""Tests for scripts/27_faithfulness.py (no network, no rendering, no subagents).

Run: PYTHONIOENCODING=utf-8 PYTHONPATH=src .venv/Scripts/python.exe tests/test_faithfulness.py
 or: PYTHONIOENCODING=utf-8 PYTHONPATH=src .venv/Scripts/python.exe -m pytest tests/test_faithfulness.py -q
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

_spec = importlib.util.spec_from_file_location("_faith", ROOT / "scripts" / "27_faithfulness.py")
faith = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(faith)  # type: ignore[union-attr]

FAKE_QUESTIONS = [
    {"id": "A01", "type": "A", "q_ko": "A01?", "q_en": "A01?", "answer": "x", "answer_keys": [],
     "gold_pages": [{"source": "a.pdf", "page": 1}, {"source": "a.pdf", "page": 2}], "period_spec": "explicit"},
    {"id": "B01", "type": "B", "q_ko": "B01?", "q_en": "B01?", "answer": "x", "answer_keys": [],
     "gold_pages": [{"source": "b.pdf", "page": 5}], "period_spec": "explicit"},
    {"id": "C01", "type": "C", "q_ko": "C01?", "q_en": "C01?", "answer": "x", "answer_keys": [],
     "gold_pages": [{"source": "c.pdf", "page": 3}], "period_spec": "ambiguous"},
    {"id": "D01", "type": "D", "q_ko": "D01?", "q_en": "D01?", "answer": "x", "answer_keys": [],
     "gold_pages": [{"source": "d.pdf", "page": 1}, {"source": "d.pdf", "page": 9}], "period_spec": "explicit",
     "gold_groups": [[{"source": "d.pdf", "page": 1}], [{"source": "d.pdf", "page": 9}]]},
    {"id": "D02", "type": "D", "q_ko": "D02?", "q_en": "D02?", "answer": "x", "answer_keys": [],
     "gold_pages": [{"source": "e.pdf", "page": 4}], "period_spec": "explicit"},
]


def fake_questions() -> dict[str, dict]:
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "questions.jsonl"
        path.write_text("\n".join(json.dumps(q) for q in FAKE_QUESTIONS), encoding="utf-8")
        return faith.questions(path)


# ─── citation parsing ───────────────────────────────────────────────────


def test_parse_cited_fixed_arm_strings():
    assert faith.parse_cited(["sanofi_2025Q1_pr.pdf p13", "roche_2025Q2_deck.pdf p7"]) == [
        ("sanofi_2025Q1_pr.pdf", 13), ("roche_2025Q2_deck.pdf", 7)]


def test_parse_cited_agent_dicts():
    assert faith.parse_cited([{"source": "a.pdf", "page": 1}, {"source": "b.pdf", "page": "2"}]) == [
        ("a.pdf", 1), ("b.pdf", 2)]


def test_parse_cited_dedupes_and_skips_junk():
    cited = ["a.pdf p1", "a.pdf p1", {"source": "a.pdf", "page": 1}, "no page here", {"source": "b.pdf"}, None]
    assert faith.parse_cited(cited) == [("a.pdf", 1)]
    assert faith.parse_cited(None) == [] and faith.parse_cited([]) == []


# ─── gold matching ──────────────────────────────────────────────────────


def test_gold_of_flattens_alternates_and_hops():
    qs = fake_questions()
    assert faith.gold_of(qs["A01"]) == {("a.pdf", 1), ("a.pdf", 2)}
    assert faith.gold_of(qs["D01"]) == {("d.pdf", 1), ("d.pdf", 9)}, "multi-hop: every hop's pages count as gold"


def test_citation_stats_against_gold():
    qs = fake_questions()
    rows = [
        {"arm": "vision", "qid": "A01", "cited": ["a.pdf p2"]},                       # 1/1 gold
        {"arm": "vision", "qid": "B01", "cited": [{"source": "z.pdf", "page": 1}]},   # 0/1
        {"arm": "vision", "qid": "D01", "cited": ["d.pdf p1", "d.pdf p9", "z.pdf p2"]},  # 2/3, both hops
        {"arm": "agentic", "qid": "C01", "cited": []},                                # no citation at all
    ]
    s = faith.citation_stats(rows, qs)
    assert s["vision"] == {"answers": 3, "with_citations": 3, "cited": 5, "in_gold": 3, "any_gold": 2}
    assert s["agentic"] == {"answers": 1, "with_citations": 0, "cited": 0, "in_gold": 0, "any_gold": 0}


# ─── sampling ───────────────────────────────────────────────────────────


def test_sample_cells_is_deterministic_and_balanced():
    qs = fake_questions()
    first = faith.sample_cells(qs, 4, seed=0)
    assert first == faith.sample_cells(qs, 4, seed=0), "same seed must give the same sample"
    assert len(first) == 4 and len({qid for qid, _ in first}) == 4
    assert {qs[qid]["type"] for qid, _ in first} == {"A", "B", "C", "D"}, "round-robin over types"
    assert all(lang in ("ko", "en") for _, lang in first)
    assert faith.sample_cells(qs, 5, seed=0) != faith.sample_cells(qs, 5, seed=7)  # only 2 D questions differ


def test_sample_cells_caps_at_the_question_count():
    qs = fake_questions()
    assert len(faith.sample_cells(qs, 99, seed=1)) == len(qs)


# ─── report aggregation ─────────────────────────────────────────────────


def _mapping(**arms_by_aid) -> dict[str, dict]:
    return {aid: {"src": "gen", "task_id": aid, "arms": arms, "qid": f"Q{i}", "lang": "ko", "pages": []}
            for i, (aid, arms) in enumerate(arms_by_aid.items())}


def test_aggregate_counts_shared_answers_for_every_arm():
    mapping = _mapping(t1=["text_rerank", "vision"], t2=["vision"], t3=["text_rerank"])
    got = {
        "t1": {"task_id": "t1", "n_claims": 4, "n_supported": 4, "n_unsupported": 0, "n_contradicted": 0, "verdict": "grounded"},
        "t2": {"task_id": "t2", "n_claims": 4, "n_supported": 2, "n_unsupported": 1, "n_contradicted": 1, "verdict": "ungrounded"},
        "t3": {"task_id": "t3", "n_claims": 0, "n_supported": 0, "n_unsupported": 0, "n_contradicted": 0, "verdict": "grounded"},
        "unknown": {"task_id": "unknown", "n_claims": 9, "n_supported": 0, "verdict": "ungrounded"},  # not in map: ignored
    }
    s = faith.aggregate(mapping, got)
    assert s["vision"] == {"n": 2, "claims": 8, "supported": 6, "contradicted": 1,
                           "rate_sum": 1.5, "grounded": 1, "with_contradiction": 1}
    assert abs(s["vision"]["rate_sum"] / s["vision"]["n"] - 0.75) < 1e-9
    assert s["text_rerank"]["n"] == 2 and s["text_rerank"]["grounded"] == 2
    assert abs(s["text_rerank"]["rate_sum"] / 2 - 1.0) < 1e-9, "an answer with no claims counts as fully supported"


if __name__ == "__main__":
    fails = 0
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception:
            fails += 1
            print(f"FAIL {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - fails}/{len(tests)} passed")
    sys.exit(1 if fails else 0)
