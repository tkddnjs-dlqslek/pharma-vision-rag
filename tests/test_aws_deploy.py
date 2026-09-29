"""Pure functions of scripts/31_aws_deploy.py (no AWS, no network) and the report of scripts/32_mcp_usage_report.py
on a synthetic log, plus the mcp_server call logger with the real tools it wraps (calculate only: no corpus needed).

Run: PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m pytest tests/test_aws_deploy.py -q
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


aws = _load("_aws_deploy", ROOT / "scripts" / "31_aws_deploy.py")
report = _load("_usage_report", ROOT / "scripts" / "32_mcp_usage_report.py")


# ─── 31: remote .env ─────────────────────────────────────────────────────────

LOCAL_ENV = {
    "ANTHROPIC_API_KEY": "sk-ant-secret", "AWS_ACCESS_KEY_ID": "AKIA-secret", "AWS_SECRET_ACCESS_KEY": "aws-secret",
    "ENCODER_TOKEN": "enc-token", "MCP_TOKEN": "mcp-token", "MCP_USER": "pharma", "MCP_PASSWORD": "pw",
    "QDRANT_CLOUD_URL": "https://q.example", "QDRANT_CLOUD_API_KEY": "qkey", "DATA_REPO": "u/pharma-corpus-data",
    "HF_TOKEN": "hf_xxxxxxxxxxxx", "MCP_URL": "http://localhost:8000/mcp", "MCP_PUBLIC_URL": "https://old.example",
    "ENCODER_URL": "https://old-encoders.example", "MODAL_TOKEN_ID": "modal-secret",
}


def test_remote_env_whitelist_and_derived_keys():
    text = aws.remote_env(LOCAL_ENV, "1.2.3.4.sslip.io")
    keys = {line.split("=", 1)[0] for line in text.strip().splitlines()}
    assert keys == {"ENCODER_TOKEN", "MCP_TOKEN", "MCP_USER", "MCP_PASSWORD", "QDRANT_CLOUD_URL", "QDRANT_CLOUD_API_KEY",
                    "DATA_REPO", "PUBLIC_HOST", "MCP_PUBLIC_URL", "ENCODER_URL"}
    assert "PUBLIC_HOST=1.2.3.4.sslip.io\n" in text
    assert "MCP_PUBLIC_URL=https://1.2.3.4.sslip.io\n" in text
    assert "ENCODER_URL=http://encoders:7860\n" in text          # the compose-network address, not the old public one
    for leak in ("sk-ant-secret", "AKIA-secret", "aws-secret", "modal-secret", "old.example", "hf_xxxxxxxxxxxx"):
        assert leak not in text


def test_remote_env_keeps_real_hf_token_and_skips_empty():
    text = aws.remote_env({**LOCAL_ENV, "HF_TOKEN": "hf_realtoken123", "MCP_PASSWORD": ""}, "h")
    assert "HF_TOKEN=hf_realtoken123\n" in text and "MCP_PASSWORD" not in text


def test_update_env_text_rewrites_only_named_keys():
    before = "A=1\nPUBLIC_HOST=old.sslip.io\nB=keep me # comment\n"
    after = aws.update_env_text(before, {"PUBLIC_HOST": "9.9.9.9.sslip.io", "MCP_URL": "https://9.9.9.9.sslip.io/mcp"})
    assert after == "A=1\nPUBLIC_HOST=9.9.9.9.sslip.io\nB=keep me # comment\nMCP_URL=https://9.9.9.9.sslip.io/mcp\n"
    assert aws.update_env_text("", {"X": "1"}) == "X=1\n"
    assert aws.update_env_text("X=0", {"X": "1"}) == "X=1"


# ─── 31: AMI, instance lookup, user-data ─────────────────────────────────────

def test_pick_ami_latest_by_creation_date():
    images = [
        {"ImageId": "ami-old", "CreationDate": "2026-08-01T00:00:00.000Z", "Name": "ubuntu/...-arm64-server-20260801"},
        {"ImageId": "ami-new", "CreationDate": "2026-09-15T00:00:00.000Z", "Name": "ubuntu/...-arm64-server-20260915"},
        {"ImageId": "ami-mid", "CreationDate": "2026-09-01T00:00:00.000Z", "Name": "ubuntu/...-arm64-server-20260901"},
    ]
    assert aws.pick_ami(images)["ImageId"] == "ami-new"
    with pytest.raises(SystemExit):
        aws.pick_ami([])


def _described(*states: str) -> dict:
    return {"Reservations": [{"Instances": [{"InstanceId": f"i-{n}", "State": {"Name": s}} for n, s in enumerate(states)]}]}


def test_find_instance_skips_terminated_and_prefers_running():
    assert aws.find_instance(_described()) is None
    assert aws.find_instance(_described("terminated", "shutting-down")) is None
    assert aws.find_instance(_described("terminated", "stopped"))["InstanceId"] == "i-1"
    assert aws.find_instance(_described("stopped", "running"))["InstanceId"] == "i-1"
    assert aws.find_instance({"Reservations": []}) is None


def test_user_data_installs_docker_for_ubuntu():
    ud = aws.USER_DATA
    assert ud.startswith("#!/bin/bash")
    assert "get.docker.com" in ud and "usermod -aG docker ubuntu" in ud and "git" in ud


def test_constants():
    assert aws.sslip("3.35.10.20") == "3.35.10.20.sslip.io"
    assert aws.AMI_OWNER == "099720109477" and "ubuntu-noble-24.04-arm64-server-*" in aws.AMI_NAME
    assert aws.PEM.name == "pharma-rag.pem" and aws.PEM.parent.name == ".ssh"


# ─── 32: usage report on a synthetic log ─────────────────────────────────────

def _log_lines() -> list[dict]:
    recs = []
    for i in range(10):
        recs.append({"ts": f"2026-10-0{1 + i % 2}T10:00:00.000+00:00", "tool": "search_text",
                     "args": {"query": f"q{i}", "k": 5}, "ms": 100 + i * 10, "ok": True,
                     "top": [["sanofi_2025Q2_pr.pdf", 1], ["sanofi_2025Q2_deck.pdf", 3], ["sanofi_2025Q2_pr.pdf", 5]]})
    recs.append({"ts": "2026-10-02T11:00:00.000+00:00", "tool": "search_pages", "args": {"query": "chart"}, "ms": 40000,
                 "ok": False, "error": "ToolError: HTTP 503"})
    recs.append({"ts": "2026-10-03T11:00:00.000+00:00", "tool": "open_page",
                 "args": {"document_id": "roche_2025Q2_deck.pdf", "page": 7}, "ms": 350.5, "ok": True})
    recs.append({"ts": "2026-10-03T11:00:01.000+00:00", "tool": "calculate", "args": {"expression": "1+1"}, "ms": 0.1, "ok": True})
    return recs


def test_usage_report_summary_and_markdown(tmp_path):
    path = tmp_path / "mcp_calls.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in _log_lines()) + "\n\n", encoding="utf-8")
    s = report.summarise(report.load(path))
    assert s["calls"] == 13 and s["errors"] == 1
    st = s["tools"]["search_text"]
    assert st["calls"] == 10 and st["median_ms"] == 150 and st["p90_ms"] == 180 and st["error_rate"] == 0
    assert s["tools"]["search_pages"] == {"calls": 1, "median_ms": 40000, "p90_ms": 40000, "errors": 1, "error_rate": 1.0}
    assert s["days"] == {"2026-10-01": 5, "2026-10-02": 6, "2026-10-03": 2}
    assert s["top_docs"][0] == ("sanofi_2025Q2_pr.pdf", 20)
    assert ("roche_2025Q2_deck.pdf", 1) in s["top_docs"]
    md = report.render(s)
    assert "| search_text | 10 | 150 | 180 | 0 | 0% |" in md
    assert "| search_pages | 1 | 40000 | 40000 | 1 | 100% |" in md
    assert "| 2026-10-02 | 6 |" in md and "| sanofi_2025Q2_pr.pdf | 20 |" in md


def test_percentile_single_and_nearest_rank():
    assert report.percentile([7.0], 0.9) == 7.0
    assert report.percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.9) == 9


# ─── mcp_server call logger ──────────────────────────────────────────────────

def test_mcp_log_record_truncates_query_and_extracts_top3(tmp_path, monkeypatch):
    monkeypatch.setenv("MCP_LOG_PATH", str(tmp_path / "calls.jsonl"))
    srv = _load("_mcp_server_for_log", ROOT / "src" / "pharma_vision_rag" / "mcp_server.py")
    hits = json.dumps([{"source": "a.pdf", "page": 1, "score": 0.9}, {"source": "b.pdf", "page": 2}, {"source": "c.pdf", "page": 3},
                       {"source": "d.pdf", "page": 4}])
    rec = srv._log_record("search_text", {"query": "x" * 500, "document_ids": None, "k": 5}, 12.34, result=hits)
    assert len(rec["args"]["query"]) == 200
    assert "document_ids" not in rec["args"] and rec["args"]["k"] == 5
    assert rec["ok"] is True and rec["ms"] == 12.3 and rec["top"] == [["a.pdf", 1], ["b.pdf", 2], ["c.pdf", 3]]
    err = srv._log_record("open_page", {"document_id": "x.pdf", "page": 2}, 1.0, error="ToolError: " + "e" * 500)
    assert err["ok"] is False and len(err["error"]) == 200 and "top" not in err
    # open_page returns a list (header + image), not JSON: no top, no crash
    assert "top" not in srv._log_record("open_page", {"document_id": "x.pdf", "page": 2}, 1.0, result=["hdr", object()])
    # the wrapper writes one line per call, errors included, and the tool schema still sees the real signature
    srv.calculate("1+2")
    with pytest.raises(srv.ToolError):
        srv.calculate("__import__('os')")
    lines = [json.loads(line) for line in srv.LOG_PATH.read_text(encoding="utf-8").splitlines()]
    assert [(r["tool"], r["ok"]) for r in lines] == [("calculate", True), ("calculate", False)]
    assert lines[0]["args"] == {"expression": "1+2"} and "error" in lines[1]
    import inspect
    assert list(inspect.signature(srv.search_text).parameters) == ["query", "document_ids", "k"]
