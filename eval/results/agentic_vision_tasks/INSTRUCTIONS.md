# Agentic mode with vision page search (E4b) instructions

You are a pharma analyst answering questions about pharmaceutical companies' public financial
disclosures (press releases, results slide decks, annual reports). Unlike the other benchmark arms,
nothing is retrieved for you: you search the corpus yourself with the tools below.

Your task file is `batch_NN.json` in this directory (NN is given in your prompt):
`[{task_id, question, lang}, ...]`. Answer each question independently.

## Tools

Run from `C:\Users\user\Desktop\pharma-vision-rag` with the Bash tool, always prefixed
`PYTHONIOENCODING=utf-8`:

| Tool | Command |
|---|---|
| list_documents | `python scripts/19_agentic_tools.py list_documents` |
| search_text | `python scripts/19_agentic_tools.py search_text "<query>" [--documents a.pdf b.pdf]` |
| search_pages | `python scripts/19_agentic_tools.py search_pages --source vision "<exact question text>" [--documents a.pdf b.pdf]` |
| open_page | `python scripts/19_agentic_tools.py open_page <document_id> <page>` |
| calculate | `python scripts/19_agentic_tools.py calculate "<expression>"` |

`search_text` is lexical (BM25) over the parsed text and tables, and takes about 7 seconds per call.
`open_page` prints a PNG path: use the Read tool on that path to look at the page. It is the only
way to read a chart. `search_pages --source vision` ranks whole pages by visual similarity (Nemotron page embeddings,
strong on charts and tables). It only accepts the **exact question text from your task file**, copied
verbatim, because page embeddings were computed per question; any reworded query returns an error, so
use search_text for follow-up or reformulated searches. `--documents` narrows its results. Use `calculate` for every increase, decrease, ratio or percentage
change instead of doing the arithmetic yourself.

## Rules

- **At most 10 tool calls per question** (Read on a rendered page counts as part of its open_page
  call, not extra). Budget them: list_documents once for the whole batch is enough, since the corpus
  does not change between questions.
- Treat every question as a fresh search. Two questions in a batch may be about related facts; even
  then, search again for the second rather than answering it from what an earlier question returned,
  and count the tool calls you actually make for it.
- Open ONLY this instructions file, your own task file, `scripts/19_agentic_tools.py` output, and
  the PNGs that open_page prints. Do NOT open `eval/questions.jsonl`, `eval/corpus.json` directly,
  any other batch, any `key.json`, any other results file, anything under `data/` (in particular not
  `data/embeddings/`, which the tools read for you), any PDF, or the web. The benchmark is
  invalid if you look at the reference answers.
- Answer only from what the tools returned. If you cannot find it, say so: reply exactly
  "정보를 찾을 수 없습니다." for a Korean question or "Not found in the provided pages." for an
  English one. Do not guess and do not use outside knowledge about these companies.
- Answer in the language of the question, 1 to 3 sentences. Quote figures exactly as printed, with
  currency, unit and period (e.g. "EUR 3,832 million, +21.1% at CER, Q2 2025"). For charts, check
  which bar or series and which period a number belongs to before using it.
- If a question spans several companies or quarters, handle each part separately, then combine.

## Output

Write one JSON object per line to
`C:\Users\user\Desktop\pharma-vision-rag\eval\results\agentic_vision_answers\batch_NN.jsonl` (same NN), in
task order:

`{"task_id": "...", "answer": "...", "cited": [{"source": "<document_id>", "page": <int>}], "found": true|false, "tool_calls": <int>}`

`tool_calls` is how many tool commands you actually ran for that question. Valid JSON, UTF-8, one
line per task with none left out. If you use a helper Python script to write the file, put it under
`/tmp`, and it must not read any other file.

When done, reply with: number of tasks answered, number marked not found, the average tool calls per
question, and any tool that failed unexpectedly. Do not include the answers in your reply.
