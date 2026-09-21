"""Agentic mode (EXPERIMENT_PLAN §3.3, "E4"): a single Claude agent with tool use,
run on the same 60-question benchmark as the four fixed-pipeline modes.

Why a fifth mode instead of fixing the others: the fixed pipelines retrieve once and
answer. Multi-hop questions (type D), ambiguous periods, and "twin pages" that differ
only by quarter/company need the model to look again after a first read; this mode gives
Claude six tools and up to ``MAX_TOOL_CALLS`` calls to do that itself.

Tools (input -> action):
    list_documents()                   -> eval/corpus.json rows (id, company, period, kind, pages)
    search_text(query, document_ids?)  -> lexical (BM25) search over text_chunks.jsonl. Default
                                           retriever, because a question the agent invents can't be
                                           looked up in eval.runner's precomputed per-question hit
                                           files, and this dev machine (16 GB RAM) can't load BGE-M3.
    search_pages(query, document_ids?) -> caption index (Qdrant `pharma_caption`); a clear tool
                                           error if that collection isn't built yet.
    open_page(document_id, page)       -> page image (for charts search_text can't read)
    calculate(expression)              -> safe arithmetic (ast-based; no names/calls -> no eval())
    final_answer(answer, citations)    -> ends the loop

No LangGraph, no Tool Runner: a manual loop, per the plan ("Claude의 tool use로 직접 구현").
Once ``MAX_TOOL_CALLS`` non-final tool calls have run, the next request forces
``tool_choice: final_answer`` so every run terminates with an answer.

Import path warning (same problem retriever/bm25.py documents for itself): this file lives
under ``pharma_vision_rag.modes``, and ``modes/__init__.py`` imports ``HybridMode``, which
imports the vision retriever, which pulls in docling/sentence-transformers/torch —
``import pharma_vision_rag.modes.agentic`` measured 76s on this dev box before this file's own
code even runs, regardless of what this file itself imports. Do NOT import it that way for
tests or a quick pilot. Instead:
    - tests / any script that just needs ``AgenticMode``: load by file path with
      ``importlib.util.spec_from_file_location`` (see tests/test_eval_generation.py).
    - CLI pilot run: execute this file directly (NOT ``python -m pharma_vision_rag.modes.agentic``,
      which goes through the package and pays the same 76s):
          PYTHONPATH=src PYTHONIOENCODING=utf-8 python src/pharma_vision_rag/modes/agentic.py --limit 10
      Running a file directly does not initialize its parent package, so this stays fast; its own
      ``from pharma_vision_rag.eval... / generator... / utils...`` imports are all light (checked:
      eval/__init__.py is empty, generator/__init__.py and utils/__init__.py only import light modules).
"""
from __future__ import annotations

import ast
import importlib.util
import json
import operator
import time
from pathlib import Path
from typing import Any

import anthropic

from pharma_vision_rag.eval.pricing import cost_usd
from pharma_vision_rag.generator.claude_vision import DEFAULT_MODEL, _image_to_base64
from pharma_vision_rag.utils.pdf import render_page

ROOT = Path(__file__).resolve().parents[3]
CORPUS_PATH = ROOT / "eval" / "corpus.json"
PDF_DIR = ROOT / "data" / "pdf" / "corpus"
TEXT_CHUNKS_PATH = ROOT / "data" / "embeddings" / "v2" / "text" / "text_chunks.jsonl"
BM25_MODULE_PATH = ROOT / "src" / "pharma_vision_rag" / "retriever" / "bm25.py"

MAX_TOOL_CALLS = 10
MAX_TOKENS = 1024
BM25_POOL = 50  # candidates pulled before a document_ids filter, then cut to the tool's k

SYSTEM_PROMPT = """You are a pharma analyst answering questions about pharmaceutical companies' public financial disclosures (press releases, results presentation slides, annual reports), using tools to search the corpus yourself.

Tools: list_documents, search_text (lexical search over parsed text/tables), search_pages (page captions — use for chart-heavy pages), open_page (render a page image — the only way to read a chart), calculate (safe arithmetic — always use this for any increase, decrease, ratio, or percentage change instead of computing it yourself), final_answer (ends the run; always call it to finish).

Rules:
- Answer ONLY from what you retrieved or opened with these tools. If you cannot find the answer, say so in final_answer rather than guessing.
- If the question names a company or a reporting period, call list_documents first and pick the matching document id(s) before searching.
- If the question spans more than one company, decompose it: search/open pages for each company separately, then combine.
- Quote every figure verbatim with its currency/unit (e.g. €3.5 billion, 20.3%) and period (FY2025, Q1 2025).
- Reply in the same language as the question (Korean question -> Korean answer, English question -> English answer).
- You have at most 10 tool calls before you must answer. Call final_answer with your best answer and the pages you used as soon as you are confident."""

TOOLS: list[dict[str, Any]] = [
    {
        "name": "list_documents",
        "description": "List every document in the corpus with its id, company, reporting period, kind (pr/deck/20F), and page count. Call this first when the question names a company or period, to find the right document id before searching.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "search_text",
        "description": "Lexical search over the corpus's parsed text and table chunks. Returns the best-matching passages with their source document id and page. Optionally restrict to specific document ids.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "document_ids": {"type": "array", "items": {"type": "string"}, "description": "Optional document ids to restrict the search to."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_pages",
        "description": "Search page captions (short text descriptions of each page, including chart and table contents) to find candidate pages when the answer is likely on a chart-heavy page. Optionally restrict to specific document ids.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "document_ids": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["query"],
        },
    },
    {
        "name": "open_page",
        "description": "Render one page of a document as an image so you can read it directly, including charts. Use this to verify a figure before answering.",
        "input_schema": {
            "type": "object",
            "properties": {
                "document_id": {"type": "string"},
                "page": {"type": "integer"},
            },
            "required": ["document_id", "page"],
        },
    },
    {
        "name": "calculate",
        "description": "Evaluate a safe arithmetic expression (numbers, + - * / % ** and parentheses only). Use this for any increase, decrease, ratio, or percentage change instead of computing it yourself.",
        "input_schema": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
    {
        "name": "final_answer",
        "description": "Submit your final answer and stop. Always call this to finish, citing every page you used.",
        "input_schema": {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "citations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"document_id": {"type": "string"}, "page": {"type": "integer"}},
                        "required": ["document_id", "page"],
                    },
                },
            },
            "required": ["answer", "citations"],
        },
    },
]


# ─── calculate: ast-based safe arithmetic (no names, no calls -> no eval()) ────────────

_BINOPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Mod: operator.mod, ast.Pow: operator.pow,
    ast.FloorDiv: operator.floordiv,
}
_UNARYOPS = {ast.USub: operator.neg, ast.UAdd: operator.pos}


def safe_eval(expr: str) -> float:
    """Evaluate arithmetic only: numeric literals, +-*/%**, unary +/-, parentheses.
    Anything else (names, calls, attributes, subscripts, comprehensions, ...) raises ValueError."""
    return _eval_node(ast.parse(expr, mode="eval").body)


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError(f"non-numeric constant: {node.value!r}")
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        left, right = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:  # 9**9**9 would hang the loop
            raise ValueError("exponent too large")
        return _BINOPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARYOPS:
        return _UNARYOPS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"disallowed expression node: {type(node).__name__}")


def _load_bm25_module():
    """Load retriever/bm25.py by file path, bypassing pharma_vision_rag.retriever's __init__
    (which imports docling/sentence-transformers/torch and takes minutes on this machine).
    Same trick scripts/15_rebuild_text_index.py uses for chunking.py."""
    spec = importlib.util.spec_from_file_location("_agentic_bm25", BM25_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _load_chunks(path: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


class AgenticMode:
    """Single Claude agent with tool use over the pharma corpus. See module docstring."""

    name = "agentic"

    def __init__(
        self,
        client: anthropic.Anthropic | None = None,
        model: str = DEFAULT_MODEL,
        corpus_path: Path = CORPUS_PATH,
        pdf_dir: Path = PDF_DIR,
        text_chunks_path: Path = TEXT_CHUNKS_PATH,
        qdrant_url: str = "http://localhost:6335",
        text_retriever: str = "lexical",  # "lexical" (default, no model) | "dense" (loads DoclingTextRetriever + Qdrant)
        max_tool_calls: int = MAX_TOOL_CALLS,
    ) -> None:
        self.client = client or anthropic.Anthropic()
        self.model = model
        self.corpus_path = Path(corpus_path)
        self.pdf_dir = Path(pdf_dir)
        self.text_chunks_path = Path(text_chunks_path)
        self.qdrant_url = qdrant_url
        self.text_retriever = text_retriever
        self.max_tool_calls = max_tool_calls
        self._corpus: dict[str, dict[str, Any]] | None = None
        self._bm25 = None  # lazy BM25Index over text_chunks_path
        self._dense = None  # lazy DoclingTextRetriever, only if text_retriever == "dense"

    # ─── corpus ─────────────────────────────────────────────────────────

    def _load_corpus(self) -> dict[str, dict[str, Any]]:
        if self._corpus is None:
            docs = json.loads(self.corpus_path.read_text(encoding="utf-8"))
            self._corpus = {d["id"]: d for d in docs}
        return self._corpus

    # ─── tool implementations ──────────────────────────────────────────

    def _tool_list_documents(self) -> dict[str, Any]:
        return {"content": json.dumps(list(self._load_corpus().values()), ensure_ascii=False), "is_error": False}

    def _tool_search_text(self, query: str, document_ids: list[str] | None) -> dict[str, Any]:
        if self.text_retriever == "dense":
            hits = self._search_text_dense(query, document_ids)
        else:
            if self._bm25 is None:
                bm25_mod = _load_bm25_module()
                self._bm25 = bm25_mod.BM25Index(_load_chunks(self.text_chunks_path))
            raw = self._bm25.search(query, k=BM25_POOL)
            if document_ids:
                allowed = set(document_ids)
                raw = [h for h in raw if h.get("source") in allowed]
            hits = [{"source": h["source"], "page": h["page"], "block_type": h.get("block_type"),
                     "text": h["text"][:400], "score": round(float(h["score"]), 4)} for h in raw[:5]]
        return {"content": json.dumps(hits, ensure_ascii=False), "is_error": False}

    def _search_text_dense(self, query: str, document_ids: list[str] | None) -> list[dict[str, Any]]:
        # Heavy path: pulls docling/sentence-transformers/torch via retriever/__init__.py.
        # Not exercised on this dev machine (16 GB RAM); kept for a GPU box with those deps installed.
        if self._dense is None:
            from pharma_vision_rag.retriever.docling_text import DoclingTextRetriever
            self._dense = DoclingTextRetriever(qdrant_url=self.qdrant_url)
        hits = self._dense.search(query, k=5)
        if document_ids:
            allowed = set(document_ids)
            hits = [h for h in hits if h.get("source") in allowed]
        return hits

    def _tool_search_pages(self, query: str, document_ids: list[str] | None) -> dict[str, Any]:
        from urllib.parse import urlparse

        host = urlparse(self.qdrant_url).hostname or "localhost"
        try:
            from qdrant_client import QdrantClient  # not installed on every box -> tool error, not a crash
            qc = QdrantClient(host=host, grpc_port=6336, prefer_grpc=True, check_compatibility=False, timeout=10)
            exists = qc.collection_exists("pharma_caption")
        except Exception as e:  # noqa: BLE001 - Qdrant unreachable, client missing, etc.
            return {"content": f"caption index not built (Qdrant unreachable: {e})", "is_error": True}
        if not exists:
            return {"content": "caption index not built (pharma_caption collection is empty; "
                                "needs ANTHROPIC_API_KEY to index — see CLAUDE.md dependency table).", "is_error": True}
        from pharma_vision_rag.retriever.caption import CaptionRetriever  # heavy: only reached if the collection exists
        cap = CaptionRetriever(qdrant_url=self.qdrant_url)
        hits = cap.search(query, k=5)
        if document_ids:
            allowed = set(document_ids)
            hits = [h for h in hits if h.get("source") in allowed]
        return {"content": json.dumps(hits, ensure_ascii=False), "is_error": False}

    def _tool_open_page(self, document_id: Any, page: Any) -> dict[str, Any]:
        corpus = self._load_corpus()
        doc = corpus.get(document_id)
        if doc is None:
            return {"content": f"unknown document_id {document_id!r}. Call list_documents first.", "is_error": True}
        try:
            page = int(page)
        except (TypeError, ValueError):
            return {"content": f"page must be an integer, got {page!r}", "is_error": True}
        if not (1 <= page <= doc["pages"]):
            return {"content": f"{document_id} has {doc['pages']} pages; page {page} is out of range.", "is_error": True}
        pdf_path = self.pdf_dir / document_id
        if not pdf_path.exists():
            return {"content": f"PDF file missing on disk: {pdf_path}", "is_error": True}
        image = render_page(pdf_path, page_number=page)
        media_type, data = _image_to_base64(image)
        return {
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}},
                {"type": "text", "text": f"{document_id} p.{page}"},
            ],
            "is_error": False,
        }

    def _tool_calculate(self, expression: str) -> dict[str, Any]:
        try:
            return {"content": str(safe_eval(expression)), "is_error": False}
        except Exception as e:  # noqa: BLE001 - report as a tool error, don't raise into the loop
            return {"content": f"calculate error: {e}", "is_error": True}

    def _run_tool(self, name: str, inp: dict[str, Any]) -> dict[str, Any]:
        try:
            if name == "list_documents":
                return self._tool_list_documents()
            if name == "search_text":
                return self._tool_search_text(inp.get("query", ""), inp.get("document_ids"))
            if name == "search_pages":
                return self._tool_search_pages(inp.get("query", ""), inp.get("document_ids"))
            if name == "open_page":
                return self._tool_open_page(inp.get("document_id"), inp.get("page"))
            if name == "calculate":
                return self._tool_calculate(inp.get("expression", ""))
            return {"content": f"unknown tool {name!r}", "is_error": True}
        except Exception as e:  # noqa: BLE001 - a tool bug must not crash the agent loop
            return {"content": f"tool error: {e}", "is_error": True}

    # ─── agent loop ─────────────────────────────────────────────────────

    @staticmethod
    def _accumulate(total: dict[str, int], usage: Any) -> None:
        total["input_tokens"] += usage.input_tokens
        total["output_tokens"] += usage.output_tokens
        total["cache_creation_input_tokens"] += getattr(usage, "cache_creation_input_tokens", 0) or 0
        total["cache_read_input_tokens"] += getattr(usage, "cache_read_input_tokens", 0) or 0

    def answer(self, query: str) -> dict[str, Any]:
        messages: list[dict[str, Any]] = [{"role": "user", "content": query}]
        trace: list[dict[str, Any]] = []
        tool_calls_made = 0
        usage_total = {"input_tokens": 0, "output_tokens": 0, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
        final: dict[str, Any] | None = None

        # Belt-and-braces bound on turns; once tool_calls_made >= max_tool_calls the next
        # request forces final_answer, so this should never actually bind.
        for _ in range(self.max_tool_calls + 3):
            kwargs: dict[str, Any] = {}
            if tool_calls_made >= self.max_tool_calls:
                kwargs["tool_choice"] = {"type": "tool", "name": "final_answer"}

            response = self.client.messages.create(
                model=self.model, max_tokens=MAX_TOKENS, system=SYSTEM_PROMPT,
                tools=TOOLS, messages=messages, **kwargs,
            )
            self._accumulate(usage_total, response.usage)
            messages.append({"role": "assistant", "content": response.content})

            tool_uses = [b for b in response.content if getattr(b, "type", None) == "tool_use"]
            if not tool_uses:
                text = next((b.text for b in response.content if getattr(b, "type", None) == "text"), "")
                final = {"answer": text, "citations": []}
                trace.append({"tool": None, "note": "model stopped without calling final_answer"})
                break

            tool_results: list[dict[str, Any]] = []
            stop = False
            for tu in tool_uses:
                if tu.name == "final_answer":
                    final = {"answer": tu.input.get("answer", ""), "citations": tu.input.get("citations", [])}
                    trace.append({"tool": "final_answer", "input": tu.input, "is_error": False})
                    tool_results.append({"type": "tool_result", "tool_use_id": tu.id, "content": "received"})
                    stop = True
                    continue
                outcome = self._run_tool(tu.name, tu.input)
                tool_calls_made += 1
                trace.append({"tool": tu.name, "input": tu.input, "is_error": outcome["is_error"]})
                tool_results.append({
                    "type": "tool_result", "tool_use_id": tu.id,
                    "content": outcome["content"], "is_error": outcome["is_error"],
                })

            if stop:
                break
            messages.append({"role": "user", "content": tool_results})

        if final is None:
            final = {"answer": "", "citations": []}
            trace.append({"tool": None, "note": "loop exhausted without a final answer"})

        return {
            "mode": self.name, "answer": final["answer"], "citations": final["citations"],
            "trace": trace, "tool_calls": tool_calls_made, "usage": usage_total,
            "cost_usd": round(cost_usd(self.model, usage_total), 6),
        }


# ─── pilot / batch CLI ──────────────────────────────────────────────────


def _cli() -> None:
    import argparse

    from pharma_vision_rag.eval.runner import QUESTIONS, RESULTS

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="first N questions")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--lang", choices=["ko", "en", "both"], default="both")
    ap.add_argument("--text-retriever", choices=["lexical", "dense"], default="lexical")
    a = ap.parse_args()

    langs = ["ko", "en"] if a.lang == "both" else [a.lang]
    questions = [json.loads(l) for l in QUESTIONS.read_text(encoding="utf-8").splitlines() if l.strip()]
    if a.limit:
        questions = questions[: a.limit]

    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / "generation_agentic.jsonl"
    done: set[tuple[str, str, str, int]] = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done.add(("agentic", r["id"], r["lang"], r["repeat"]))

    mode = AgenticMode(text_retriever=a.text_retriever)
    n_written = n_skipped = 0
    with open(out_path, "a", encoding="utf-8") as f:
        for q in questions:
            for lang in langs:
                query = q[f"q_{lang}"]
                for repeat in range(a.repeats):
                    key = ("agentic", q["id"], lang, repeat)
                    if key in done:
                        n_skipped += 1
                        continue
                    t0 = time.time()
                    result = mode.answer(query)
                    latency = time.time() - t0
                    record = {
                        "variant": "agentic", "id": q["id"], "lang": lang, "repeat": repeat,
                        "answer": result["answer"], "citations": result["citations"],
                        "cited_pages": [{"source": c.get("document_id"), "page": c.get("page")} for c in result["citations"]],
                        "tool_calls": result["tool_calls"], "trace": result["trace"],
                        "usage": result["usage"], "cost_usd": result["cost_usd"],
                        "latency_s": round(latency, 3),
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()
                    n_written += 1
    print(f"{n_written} rows written, {n_skipped} skipped (already present) -> {out_path}")


if __name__ == "__main__":
    _cli()
