"""Per-model $/1M-token prices used to cost generation, judging, and agentic runs.

CHECK THESE AGAINST CURRENT PRICING BEFORE REPORTING ANY COST FIGURE EXTERNALLY.
Snapshot taken 2026-09-20 while writing eval/generate.py, eval/judge.py, and
modes/agentic.py — Anthropic revises pricing over time and this file is not
fetched live. cache_write/cache_read are the standard ~1.25x / ~0.1x ratios for
5-minute ephemeral caching, not independently confirmed per model here.
"""
from __future__ import annotations

from typing import Any

PRICING: dict[str, dict[str, float]] = {
    # model_id: {input, output, cache_write, cache_read} USD per 1,000,000 tokens
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00, "cache_write": 3.75, "cache_read": 0.30},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00, "cache_write": 1.25, "cache_read": 0.10},
}


def cost_usd(model: str, usage: dict[str, Any]) -> float:
    """Cost in USD for one request's usage dict (input/output/cache_* token counts).

    ``usage["input_tokens"]`` is assumed to already exclude cached tokens, matching
    ``ClaudeVisionGenerator``/``ClaudeTextGenerator``'s usage dicts (see their docstrings
    and the Anthropic docs: cache_read/cache_creation are reported separately).
    """
    p = PRICING.get(model)
    if p is None:
        return float("nan")  # unknown model: never silently under-report cost as 0
    input_tokens = usage.get("input_tokens", 0) or 0
    output_tokens = usage.get("output_tokens", 0) or 0
    cache_write = usage.get("cache_creation_input_tokens", 0) or 0
    cache_read = usage.get("cache_read_input_tokens", 0) or 0
    return (
        input_tokens * p["input"]
        + output_tokens * p["output"]
        + cache_write * p["cache_write"]
        + cache_read * p["cache_read"]
    ) / 1_000_000


def _self_check() -> None:
    usage = {"input_tokens": 1000, "output_tokens": 500, "cache_creation_input_tokens": 2000, "cache_read_input_tokens": 4000}
    got = cost_usd("claude-haiku-4-5", usage)
    expected = (1000 * 1.00 + 500 * 5.00 + 2000 * 1.25 + 4000 * 0.10) / 1_000_000
    assert abs(got - expected) < 1e-12, (got, expected)
    import math
    assert math.isnan(cost_usd("no-such-model", usage))
    print("pricing self-check ok")


if __name__ == "__main__":
    _self_check()
