# Gold page check instructions

You check whether a document page carries the answer to a benchmark question. Each task gives a question,
its reference answer, and one rendered page image. Your verdicts decide whether the page is added to the
benchmark's gold set, so be strict: a page only "carries" the answer if a reader could take the figure
straight off this page.

Your task file is `batch_NN.json` in this directory (NN is given in your prompt):
`[{check_id, question, reference_answer, page, image}]`.

## Rules

- Open ONLY this file, your task file, and the PNG paths listed in it (use the Read tool on each image).
  Do not open anything else under `eval/` or `data/`, any PDF, or the web.
- For each task, look at the page and list which specific figures or facts **from the reference answer**
  appear on it, verbatim as printed on the page (for example "Dupixent Q2 2025 3,832"). Check that the
  period, company and metric match the reference, not just the number: 3,832 for a different quarter,
  a different product or a different company does not count.
- Charts count if the value is printed as a label. A bar you would have to estimate from the axis does not.
- A restated or rounded figure that the reference itself accepts counts (the reference says so when it does).
- verdict:
  - `carries`: at least one figure or fact the reference answer relies on is printed on this page for the
    right period, entity and metric.
  - `no`: none is, including when the page only discusses the topic, gives a different period, or gives
    a total from which the figure would have to be derived.

## Output

Write one JSON object per line to
`C:\Users\user\Desktop\pharma-vision-rag\eval\results\gold_check_verdicts\batch_NN.jsonl` (same NN), in task order:

`{"check_id": "...", "verdict": "carries|no", "facts": ["<figure as printed, with period and label>", ...], "note": "<one short clause>"}`

`facts` is empty for `no`. Valid JSON, UTF-8, one line per task. If you use a helper Python script to write
the file, put it under `/tmp`, and it must not read any other file.

When done, reply with the counts of carries / no and any image that failed to load.
