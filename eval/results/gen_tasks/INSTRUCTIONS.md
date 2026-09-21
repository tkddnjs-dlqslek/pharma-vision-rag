# Answer generation instructions (blind)

You are the answer generator in a retrieval-augmented QA benchmark. For each task you get a question and exactly
three page images that a retriever returned. Answer from those three pages only.

Your task file is `batch_NN.json` in this directory (NN is given in your prompt). It is a JSON list of tasks:
`{task_id, question, images: [3 absolute PNG paths], pages: ["<document> p<page>", ...]}` where `pages[i]` names `images[i]`.

## Strict rules (the benchmark is invalid if you break them)

- Open ONLY this instructions file, your task file, and the PNG paths listed inside your task file (use the Read tool
  on each PNG to look at the page). Do not open, list, grep or search any other file or directory, do not read PDFs,
  do not use the web, do not run scripts that read other data. Never open anything else under `eval/` or `data/`
  (in particular not `key.json`, not other batches, not `gen_answers/` of other batches).
- Treat every task independently. Base the answer for a task only on that task's three pages, even if you remember
  a page from an earlier task that seems relevant. Cite only pages from the task's own `pages` list.
- Do not use outside knowledge about the companies. If the three pages do not contain the answer, say so: reply
  exactly "정보를 찾을 수 없습니다." for a Korean question or "Not found in the provided pages." for an English
  question (you may add one short sentence on what the pages do contain). Do not guess. A wrong page set is an
  expected, legitimate outcome.
- Answer in the language of the question. 1 to 3 sentences. Quote figures exactly as printed, with currency, unit
  and period (e.g. "EUR 3,832 million, +21.1% at CER, Q2 2025"). For charts read the labels carefully: check which
  bar / segment / series and which period a number belongs to before using it. If a multi-part question is only
  partly answerable from the pages, answer the part you can and state what is missing.

## Output

Write one JSON object per line to `C:\Users\user\Desktop\pharma-vision-rag\eval\results\gen_answers\batch_NN.jsonl`
(same NN), in task order:

`{"task_id": "...", "answer": "...", "cited": ["<document> p<page>", ...], "found": true|false}`

`cited` lists the page(s) among the three that actually support the answer (empty list when not found). Valid JSON,
UTF-8. If you use a helper Python script to write the file, put it under `/tmp`, and it must not read any other file.

Work through all tasks in the file. When done, reply with: number of tasks answered, number marked not found, and any
task where an image failed to load (task_id and which image). Do not include the answers in your reply.
