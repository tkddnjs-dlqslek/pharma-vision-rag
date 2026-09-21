# Judging instructions (blind)

You grade answers produced by a retrieval-augmented QA system against a reference answer. You do NOT know which
retriever produced each answer, and you must not try to find out.

Your task file is `batch_NN.json` in this directory (NN is given in your prompt). It is a JSON list:
`{task_id, question, reference_answer, answer_keys, model_answer, keys_matched}`.
`answer_keys` are the substantive figures the reference answer hinges on; `keys_matched` is a naive substring check
(informative only, it is often wrong about formatting, so judge the text yourself).

## Rules

- Open ONLY this instructions file and your own task file. Do not open other batches, `map.json`, other directories, PDFs,
  images, `key.json`, the questions file, or the web. Judge from the text in front of you.
- Verdict for each task, exactly one of:
  - `correct`  : every substantive figure/fact the question asked for is present and matches the reference.
  - `partial`  : part of a multi-part question is answered correctly and the rest is missing, OR the right figure is
                 given with a wrong or missing unit/period/label, OR the answer states it cannot answer one hop of a
                 multi-hop question but gets the other right.
  - `wrong`    : a substantive figure contradicts the reference, OR the answer is about a different period, entity or
                 metric than asked, OR the model said it could not find the answer.
- "Not found" counts as `wrong`. That is intended: the retriever failed to supply the page. Do not give credit for
  honest abstention, and do not penalise it extra either.
- Formatting never matters. `EUR 3,832 million`, `€3,832m` and `3.832 billion euros` are the same figure. Rounding to
  the precision printed in the source is fine. Korean and English answers are graded the same way.
- Extra correct context beyond what was asked is not penalised. Extra WRONG claims are: a confident wrong figure
  alongside the right one is `partial` at best, and `wrong` if the wrong figure is the one presented as the answer.
- Some documents restate figures (for example Sanofi's 2024 quarters were restated to exclude Opella). If the model
  reports a different vintage of the same metric and says so explicitly, grade `partial`. If it reports it as the
  plain answer with no note, grade by whether it matches the reference.
- Judge only against the reference answer. Do not use outside knowledge about these companies to overrule it.

## Output

Write one JSON object per line to
`C:\Users\user\Desktop\pharma-vision-rag\eval\results\rejudge_verdicts\batch_NN.jsonl` (same NN), in task order:

`{"task_id": "...", "verdict": "correct|partial|wrong", "reason": "<one short clause>"}`

Valid JSON, UTF-8, one line per task in the task file, no line left out. If you use a helper Python script to write
the file, put it under `/tmp`, and it must not read any other file.

When done, reply with: counts of correct / partial / wrong, and any task where the reference answer itself looked
defective (for example it asks for a period the reference does not cover). Do not include the answers in your reply.
