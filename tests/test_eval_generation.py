"""Tests for eval/generate.py, eval/judge.py, and modes/agentic.py — no network, no API key.

Run: PYTHONIOENCODING=utf-8 PYTHONPATH=src .venv/Scripts/python.exe tests/test_eval_generation.py
 or: PYTHONIOENCODING=utf-8 PYTHONPATH=src .venv/Scripts/python.exe -m pytest tests/test_eval_generation.py -q

All Anthropic calls go through FakeAnthropic below (scripted responses, records every request).
agentic.py is loaded by file path (see its module docstring): importing it the normal way
(``pharma_vision_rag.modes.agentic``) pulls in docling/sentence-transformers/torch via
``modes/__init__.py`` (measured 76s on this box) before this file's own code even runs.
"""
from __future__ import annotations

import importlib.util
import io
import json
import sys
import traceback
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pharma_vision_rag.eval import generate as generate_mod  # noqa: E402  (light: no torch/docling)
from pharma_vision_rag.eval import judge as judge_mod  # noqa: E402
from pharma_vision_rag.generator.claude_text import ClaudeTextGenerator  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "_test_agentic", ROOT / "src" / "pharma_vision_rag" / "modes" / "agentic.py"
)
agentic = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(agentic)  # type: ignore[union-attr]


# ─── fake Anthropic client ──────────────────────────────────────────────


def text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def tool_use_block(id: str, name: str, input: dict) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=id, name=name, input=input)


def usage(input_tokens: int = 10, output_tokens: int = 10, cache_creation_input_tokens: int = 0,
          cache_read_input_tokens: int = 0) -> SimpleNamespace:
    return SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens,
                            cache_creation_input_tokens=cache_creation_input_tokens,
                            cache_read_input_tokens=cache_read_input_tokens)


class FakeMessages:
    """Replays a scripted list of (content_blocks[, usage[, stop_reason]]) tuples in order.

    Each item may also be a callable(kwargs) -> tuple, for responses that depend on the request
    (e.g. asserting tool_choice was forced). Records every call's kwargs in ``self.calls``."""

    def __init__(self, responses: list) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        # messages is a list the caller keeps appending to after this call returns (it's the
        # same conversation-history object every iteration) — snapshot it so calls[i]["messages"]
        # reflects state at call time, not whatever it grows into later. Individual message dicts
        # are never mutated in place once appended, so a shallow copy of the list is enough.
        if "messages" in kwargs:
            kwargs = {**kwargs, "messages": list(kwargs["messages"])}
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("FakeAnthropic: ran out of scripted responses")
        item = self._responses.pop(0)
        if callable(item):
            item = item(kwargs)
        content = item[0]
        u = item[1] if len(item) > 1 and item[1] is not None else usage()
        stop_reason = item[2] if len(item) > 2 else (
            "tool_use" if any(getattr(b, "type", None) == "tool_use" for b in content) else "end_turn"
        )
        return SimpleNamespace(content=content, usage=u, stop_reason=stop_reason)


class FakeAnthropic:
    def __init__(self, responses: list) -> None:
        self.messages = FakeMessages(responses)


# ─── modes/agentic.py: calculate() ──────────────────────────────────────


def test_calculate_evaluates_percentage_change():
    got = agentic.safe_eval("(3832-3303)/3303*100")
    expected = (3832 - 3303) / 3303 * 100
    assert abs(got - expected) < 1e-9, got


def test_calculate_rejects_import():
    for bad in ("__import__('os')", "__import__('os').system('echo hi')", "open('x')", "[].__class__"):
        try:
            agentic.safe_eval(bad)
        except ValueError:
            continue
        raise AssertionError(f"safe_eval must reject {bad!r}")


# ─── modes/agentic.py: open_page ────────────────────────────────────────


def test_open_page_returns_image_block_for_real_page():
    mode = agentic.AgenticMode(client=FakeAnthropic([]))
    outcome = mode._tool_open_page("sanofi_2025Q1_pr.pdf", 1)
    assert outcome["is_error"] is False, outcome
    assert outcome["content"][0]["type"] == "image"
    assert outcome["content"][0]["source"]["media_type"] == "image/png"
    assert len(outcome["content"][0]["source"]["data"]) > 100


def test_open_page_rejects_unknown_document():
    mode = agentic.AgenticMode(client=FakeAnthropic([]))
    outcome = mode._tool_open_page("does_not_exist.pdf", 1)
    assert outcome["is_error"] is True


def test_open_page_rejects_out_of_range_page():
    mode = agentic.AgenticMode(client=FakeAnthropic([]))
    outcome = mode._tool_open_page("sanofi_2025Q1_pr.pdf", 9999)
    assert outcome["is_error"] is True


def test_search_text_document_filter_applies_before_pool_cut(tmp_path):
    # 60 strong matches in a.pdf outrank the single weak match in b.pdf, so b.pdf sits outside the
    # global top-50 pool; filtering to b.pdf must still find it.
    chunks = [{"chunk_id": f"a{i}", "source": "a.pdf", "page": i, "text": "net debt net debt net debt"} for i in range(60)]
    chunks.append({"chunk_id": "b0", "source": "b.pdf", "page": 1, "text": "net debt and many other unrelated words here"})
    path = tmp_path / "chunks.jsonl"
    path.write_text("\n".join(json.dumps(c) for c in chunks), encoding="utf-8")
    mode = agentic.AgenticMode(client=FakeAnthropic([]), text_chunks_path=path)
    hits = json.loads(mode._tool_search_text("net debt", ["b.pdf"])["content"])
    assert [(h["source"], h["page"]) for h in hits] == [("b.pdf", 1)]
    # second process: index comes from the pickle cache written next to the chunks, same answer
    assert path.with_suffix(".bm25.pkl").exists()
    again = agentic.AgenticMode(client=FakeAnthropic([]), text_chunks_path=path)
    assert json.loads(again._tool_search_text("net debt", ["b.pdf"])["content"]) == hits


def test_search_pages_vision_exact_question_only(tmp_path):
    rankings = tmp_path / "vision_rankings.json"
    rankings.write_text(json.dumps({"Q?": [["a.pdf", 3, 9.0], ["b.pdf", 1, 8.0], ["a.pdf", 7, 7.0]]}), encoding="utf-8")
    mode = agentic.AgenticMode(client=FakeAnthropic([]), page_retriever="vision_precomputed", vision_rankings_path=rankings)
    hits = json.loads(mode._tool_search_pages(" Q? ", None)["content"])
    assert [(h["source"], h["page"]) for h in hits] == [("a.pdf", 3), ("b.pdf", 1), ("a.pdf", 7)]
    filtered = json.loads(mode._tool_search_pages("Q?", ["a.pdf"])["content"])
    assert [h["page"] for h in filtered] == [3, 7]
    assert mode._tool_search_pages("reworded Q", None)["is_error"] is True


# ─── modes/agentic.py: agent loop ───────────────────────────────────────


def test_agent_stops_at_final_answer():
    responses = [
        ([tool_use_block("t1", "calculate", {"expression": "1+1"})],),
        ([tool_use_block("t2", "final_answer", {"answer": "The answer is 2.",
                                                  "citations": [{"document_id": "sanofi_2025Q1_pr.pdf", "page": 1}]})],),
    ]
    fake = FakeAnthropic(responses)
    mode = agentic.AgenticMode(client=fake)
    result = mode.answer("what is 1+1?")
    assert result["answer"] == "The answer is 2."
    assert result["citations"] == [{"document_id": "sanofi_2025Q1_pr.pdf", "page": 1}]
    assert result["tool_calls"] == 1, "final_answer itself must not count toward the tool-call budget"
    assert len(fake.messages.calls) == 2


def test_agent_forces_final_answer_after_max_tool_calls():
    # 10 non-final tool calls, then whatever the 11th (forced) request returns.
    responses = [([tool_use_block(f"t{i}", "calculate", {"expression": "1+1"})],) for i in range(10)]
    responses.append(([tool_use_block("tf", "final_answer", {"answer": "done", "citations": []})],))
    fake = FakeAnthropic(responses)
    mode = agentic.AgenticMode(client=fake, max_tool_calls=10)
    result = mode.answer("do ten things")

    assert result["tool_calls"] == 10
    assert result["answer"] == "done"
    assert len(fake.messages.calls) == 11
    forced_call = fake.messages.calls[10]
    assert forced_call.get("tool_choice") == {"type": "tool", "name": "final_answer"}, forced_call
    # none of the first 10 requests should have been forced
    assert all("tool_choice" not in c for c in fake.messages.calls[:10])


def test_invalid_tool_input_is_error_and_loop_continues():
    responses = [
        ([tool_use_block("t1", "open_page", {"document_id": "nope.pdf", "page": 1})],),
        ([tool_use_block("t2", "final_answer", {"answer": "recovered", "citations": []})],),
    ]
    fake = FakeAnthropic(responses)
    mode = agentic.AgenticMode(client=fake)
    result = mode.answer("open a bad page")

    assert result["answer"] == "recovered"
    assert result["trace"][0]["is_error"] is True
    # the tool_result sent back to the model must carry is_error, not raise
    second_call_messages = fake.messages.calls[1]["messages"]
    tool_result_msg = second_call_messages[-1]["content"][0]
    assert tool_result_msg["type"] == "tool_result"
    assert tool_result_msg["is_error"] is True


def test_agent_falls_back_to_plain_text_if_no_tool_use():
    # model just answers in text without ever calling final_answer
    responses = [([text_block("Not found in the provided pages.")],)]
    fake = FakeAnthropic(responses)
    mode = agentic.AgenticMode(client=fake)
    result = mode.answer("unanswerable question")
    assert result["answer"] == "Not found in the provided pages."
    assert result["citations"] == []
    assert result["tool_calls"] == 0


# ─── eval/judge.py ───────────────────────────────────────────────────────

_QUESTION = {"id": "A01", "type": "A", "q_en": "What were Q1 2024 sales?",
             "answer": "€637 million (Q1 2024).", "answer_keys": ["637"], "period_spec": "explicit"}


def test_judge_parses_good_json():
    fake = FakeAnthropic([([text_block('{"verdict": "correct", "reason": "matches exactly"}')],)])
    gen = ClaudeTextGenerator(client=fake)
    record = judge_mod.judge_one(gen, _QUESTION, "Sales were €637 million in Q1 2024.")
    assert record["verdict"] == "correct"
    assert record["n_attempts"] == 1
    assert record["keys_matched"] is True


def test_judge_retries_once_on_bad_json_then_succeeds():
    fake = FakeAnthropic([
        ([text_block("sure, the verdict is correct I think")],),
        ([text_block('{"verdict": "partial", "reason": "missing unit"}')],),
    ])
    gen = ClaudeTextGenerator(client=fake)
    record = judge_mod.judge_one(gen, _QUESTION, "Sales were 637 (no currency).")
    assert record["verdict"] == "partial"
    assert record["n_attempts"] == 2
    assert len(fake.messages.calls) == 2


def test_judge_gives_up_after_max_attempts():
    fake = FakeAnthropic([([text_block("garbage")],), ([text_block("still garbage")],)])
    gen = ClaudeTextGenerator(client=fake)
    record = judge_mod.judge_one(gen, _QUESTION, "???")
    assert record["verdict"] == "wrong"
    assert "not parseable" in record["reason"]


def test_judge_aggregation_math():
    records = [
        {"variant": "v", "verdict": "correct"},
        {"variant": "v", "verdict": "correct"},
        {"variant": "v", "verdict": "partial"},
        {"variant": "v", "verdict": "wrong"},
    ]
    agg = judge_mod.aggregate(records, lambda r: r["variant"])
    assert agg["v"]["n"] == 4
    assert abs(agg["v"]["mean"] - 0.625) < 1e-9  # (1 + 1 + 0.5 + 0) / 4
    assert agg["v"]["min"] == 0.0
    assert agg["v"]["max"] == 1.0


# ─── eval/generate.py --dry-run (real rendering, no network) ────────────


def test_generate_dry_run_vision_limit_2():
    argv_backup = sys.argv
    sys.argv = ["generate.py", "--dry-run", "--limit", "2", "--variant", "vision"]
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            generate_mod.main()
    finally:
        sys.argv = argv_backup
    out = buf.getvalue()
    assert "requests built (dry run, no API call)" in out
    assert "[dry-run] vision" in out
    # 2 questions x 2 languages x default 3 repeats = 12 printed request lines
    assert out.count("[dry-run] vision") == 12


# ─── plain-python runner (works without pytest too) ─────────────────────

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
