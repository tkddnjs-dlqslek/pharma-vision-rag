"""Summarise the MCP call log (one JSON line per tool call, written by src/pharma_vision_rag/mcp_server.py) as markdown.

    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/32_mcp_usage_report.py [eval/results/mcp_calls.jsonl]

Default input is eval/results/mcp_calls.jsonl (scripts/31_aws_deploy.py logs puts the box's log there); the local stdio
server writes data/logs/mcp_calls.jsonl. Tables: calls per tool with median and p90 latency and error rate, calls per
day, top documents in the search results.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT = ROOT / "eval" / "results" / "mcp_calls.jsonl"


def load(path: Path) -> list[dict]:
    recs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            recs.append(json.loads(line))
    return recs


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile (round half up); statistics.quantiles needs 2+ points and interpolates."""
    s = sorted(values)
    return s[min(len(s) - 1, int(q * (len(s) - 1) + 0.5))]


def summarise(recs: list[dict]) -> dict:
    by_tool: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        by_tool[r["tool"]].append(r)
    tools = {}
    for tool, rs in sorted(by_tool.items()):
        ms = [float(r.get("ms", 0)) for r in rs]
        errors = sum(not r.get("ok", True) for r in rs)
        tools[tool] = {"calls": len(rs), "median_ms": percentile(ms, 0.5), "p90_ms": percentile(ms, 0.9),
                       "errors": errors, "error_rate": errors / len(rs)}
    days = Counter(r["ts"][:10] for r in recs)
    docs: Counter = Counter()
    for r in recs:
        for hit in r.get("top") or []:
            if hit and hit[0]:
                docs[hit[0]] += 1
    for r in recs:
        if r["tool"] == "open_page" and r.get("ok", True) and r.get("args", {}).get("document_id"):
            docs[r["args"]["document_id"]] += 1
    return {"calls": len(recs), "errors": sum(not r.get("ok", True) for r in recs), "tools": tools,
            "days": dict(sorted(days.items())), "top_docs": docs.most_common(10)}


def render(s: dict) -> str:
    out = [f"MCP calls: {s['calls']} (errors {s['errors']})", "",
           "| tool | calls | median ms | p90 ms | errors | error rate |", "|---|---|---|---|---|---|"]
    for tool, t in s["tools"].items():
        out.append(f"| {tool} | {t['calls']} | {t['median_ms']:.0f} | {t['p90_ms']:.0f} | {t['errors']} | {t['error_rate']:.0%} |")
    out += ["", "| day | calls |", "|---|---|"] + [f"| {d} | {n} |" for d, n in s["days"].items()]
    out += ["", "| document (search top-3 hits + open_page) | mentions |", "|---|---|"] + [f"| {d} | {n} |" for d, n in s["top_docs"]]
    return "\n".join(out)


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    if not path.exists():
        sys.exit(f"{path} not found (scripts/31_aws_deploy.py logs fetches it from the box)")
    print(render(summarise(load(path))))


if __name__ == "__main__":
    main()
