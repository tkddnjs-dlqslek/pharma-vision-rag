# pharma-vision-rag

> Multimodal RAG benchmark on chart- and table-dense pharma disclosures, queried in Korean and English.
> Does vision retrieval still beat a heavily optimized text pipeline? And what does an agent add on top?

**Status**: Phase 3 complete (2026-09-21). Retrieval and answer-level benchmarks done for five arms on a
1,709-page corpus. Current plan and full results: [docs/EXPERIMENT_PLAN.md](docs/EXPERIMENT_PLAN.md) section 0.

---

## Architecture

**1. Overall layout**

![Overall layout](아키텍처%20구조도%201.svg)

**2. How one question is answered**

![Question flow](아키텍처%20구조도%202.svg)

**3. Building the indexes (one-off)**

![Index build](아키텍처%20구조도%203.svg)

The host LLM (claude.ai web, Claude Desktop or Claude Code) is the agent; the MCP server on Modal supplies text search (Qdrant Cloud + BGE-M3 + reranker), vision page search (Nemotron ColEmbed over a pooled int8 index) and page rendering. Every runtime component sits on a free tier; only the one-off index build ran on a paid GPU (about $1).

## Results in one table

120 queries (60 questions x KO/EN), answer accuracy with correct = 1, partial = 0.5. Every answer was generated
**independently several times** (fixed arms twice, agents three times), and all answers from every arm were pooled,
anonymized and judged blind with one rubric. The headline number is the mean over all runs of each arm.

| Arm | Retrieval R@5 | Answer (mean) | Run 1 | Run 2 | Run 3 | Charts | Tables | Prose | Multi-hop | EN / KO |
|---|---|---|---|---|---|---|---|---|---|---|
| text_rerank (BGE-M3 + reranker) | 0.72 | 0.59 | 0.59 | 0.59 | | 0.64 | 0.49 | 0.91 | 0.36 | 0.65 / 0.53 |
| vision (Nemotron ColEmbed, MaxSim) | 0.88 | 0.79 | 0.79 | 0.79 | | 0.88 | 0.85 | 0.80 | 0.49 | 0.81 / 0.77 |
| hybrid_rerank (RRF of both) | 0.89 | 0.77 | 0.75 | 0.78 | | 0.82 | 0.83 | 0.90 | 0.39 | 0.78 / 0.75 |
| agentic, BM25 only (E4) | n/a | 0.84 | 0.87 | 0.81 | 0.86 | 0.82 | 0.89 | 0.77 | 0.87 | 0.84 / 0.85 |
| **agentic, BM25 + vision (E4b)** | n/a | **0.89** | 0.93 | 0.90 | 0.85 | **0.89** | **0.90** | 0.82 | **0.93** | 0.91 / 0.87 |

- **Vision beats the best text pipeline** at both stages, identically in both runs: chart retrieval R@5 0.97 vs 0.72
  (p=0.006), answer score 0.79 vs 0.59 (43 wins, 18 losses, p=0.002). Chunking fixes, a reranker and BM25 fusion did not close the gap.
- **Agents win on multi-hop** and nowhere else reliably: neither agent lost a single multi-hop cell to any fixed arm
  (0-14 to 0-18). Single-shot retrieval fills its top pages with the prior-year "twin" page or with one company;
  decomposing the question and filtering by document fixes both.
- **With a vision tool, the agent beats vision alone** over three runs (34 wins, 17 losses, p=0.024), but less clearly than
  after two runs (p=0.002): its third run scored 0.85. **Without it (BM25 only), it does not** (31-22, p=0.27).
  Run 1 alone suggested it did (p=0.03); later runs did not reproduce that, so the claim was withdrawn.
- **Agents are much noisier than fixed pipelines**: the same cell got the same score across runs 79-83% of the time for
  agents vs 89-94% for fixed arms. E4b run 1 vs run 3, same setup, differ significantly (20-5, p=0.004). The main cause
  is tool choice: agents sometimes skip the vision tool, or run it and never open the page it found.
- **On charts, neither agent beats vision alone** (6-7): single-shot vision already reaches R@5 0.97 there.
- **Korean costs the text path 12 points** and the vision path 4.
- **The agent-over-vision margin depends on how open-period questions are scored.** Ten questions name no period.
  Re-judging their 230 answers under an explicit rule (one correctly labelled period counts as correct) keeps the
  ranking (E4b 0.93, E4 0.90, vision 0.82, hybrid 0.81, text 0.64) but E4b vs vision is no longer significant
  (28-15, p=0.07). Vision scores lowest on those questions (0.72): it tends to retrieve the prior-year twin page.
- **Answers are faithful to the pages they saw**: on a blind 95-answer sample, zero claims contradicted the evidence
  pages and only one answer had an unsupported claim. Every wrong answer in that sample was faithful to a page that
  did not hold what the question asked, so the remaining errors are retrieval and period selection, not hallucination.
- **Citation accuracy** (share of cited pages that are gold): agents 0.86-0.93, vision and hybrid 0.77-0.79,
  text_rerank 0.73-0.74, which cites no page at all in 27 of 120 answers.
- Fixed-weight hybrid fusion did not beat vision alone (p=0.75); its R@1 is lower (0.58 vs 0.64).

**Cost per answer** (measured on the subagent runs): fixed arms 13.6k tokens and 15 s (generation only, always
three full-page images); agents 11.2-11.4k tokens and 36-37 s with 2.6-2.7 tool calls. The agents use *fewer*
tokens because they narrow down with text snippets before opening a page, but take about 2.5 times as long,
partly because each tool call was a fresh process that re-imported the SDK and rebuilt the BM25 index. That was fixed
before the third agent run (calls now 0.5-1 s).

## How it was run

| Stage | What ran | Where |
|---|---|---|
| Page and query embedding | `nvidia/llama-nemotron-colembed-vl-3b-v2` (1,802 patch vectors per page), exact MaxSim over all pages | RunPod GPU, `scripts/embed_pages_gpu.py`, `scripts/13_score_vision_exact.py` |
| Text parsing and retrieval | Docling (OCR off) -> chunking rules -> `BAAI/bge-m3` dense 1024-d -> `BAAI/bge-reranker-v2-m3` | RunPod GPU, `scripts/text_retrieval_gpu.py` |
| Retrieval metrics | group-based Recall@k and NDCG@k from precomputed rankings (no models on the dev box) | local, `eval/runner.py` |
| Answer generation | top-3 pages as images, blind to arm and gold; generated by Claude Sonnet 5 subagents | `scripts/17_make_generation_tasks.py` |
| Agentic arms | same Sonnet 5 subagents calling the tool bodies of `modes/agentic.py` as shell commands (list documents, BM25 search, open page, calculator; E4b adds vision page search), max 10 calls | `scripts/19_agentic_tools.py`, `20_make_agentic_tasks.py` |
| Judging | 1,382 pooled answers (two runs of fixed arms, three of agents), anonymized, blind judge subagents in four rounds, one rubric; each later round mixes in 40 already-judged anchors to measure judge drift (-0.013 to +0.025 per answer) | `scripts/22_blind_rejudge.py` |

Generation and judging went through Claude Code subagents rather than the Anthropic API (no API key during
this phase). The API paths (`eval/generate.py`, `eval/judge.py`, the `modes/agentic.py` loop) are written and
unit-tested with a fake client, but **have not been run against the live API**.

## Corpus and questions

- **27 documents, 1,709 pages** (`eval/corpus.json`): Sanofi 2024 and 2025 press releases, results decks and
  both 20-Fs, plus Novartis, Roche and AstraZeneca Q1-Q3 2025 decks. Chosen so the text alone (~1M tokens) does
  not fit in a long context, and so that quarter-, year- and company-level "twin pages" make retrieval hard.
- **60 questions** with Korean and English versions (`eval/questions.jsonl`): 20 chart, 20 table, 10 prose,
  10 multi-hop; 10 have an ambiguous period. `gold_pages` lists every page carrying the answer, checked by
  rendering; multi-hop questions use `gold_groups` (one group per hop).
- PDFs are not redistributed (`data/pdf/` is gitignored).

## Known limitations

- Two generation runs per fixed arm, three per agent. Enough to show the agents are noisy (one agent's runs span
  0.85 to 0.93), not enough to pin their scores down.
- Until 2026-09-21 `CLAUDE.md`, which every subagent loads, held reference values for three questions (A01, A02, B08).
  Scores with those questions and the defective D03 held out are unchanged (within 0.01).
- The agents' BM25 document filter had a bug during runs 1 and 2 (it filtered after cutting to the global top 50,
  so documents outside that pool returned nothing). Fixed before run 3, with a regression test; the two
  cross-company questions that showed the symptom (D06, D08) went from 0.56 to 1.0 for E4 in run 3.
- The agents' vision tool only accepts the original question text (page embeddings are precomputed per
  question on a GPU box), so reformulated follow-up searches fall back to BM25. A live vision endpoint would lift this.
- Each agent run had one cell over the 10-call budget (the API loop enforces it; subagents were only told to).
  Scoring those as wrong moves run 1 of E4 to 0.86 and of E4b to 0.92.
- One question (D03) omits the company name. The question set has not had an external review.
- Gold completeness: pages that correct answers cited outside the gold set were checked (92 candidates, 38 added
  after a render check). This raised vision's multi-hop R@5 from 0.42 to 0.62, more than it raised the text arms.
- The caption arm, query transformation and HyDE are implemented but not run (they need an API key).

## Repository layout

```
src/pharma_vision_rag/
  retriever/   docling_text, chunking, bm25, text_baseline/qt/hyde, nemotron, caption
  rerank/      zerank2.py  (= BAAI/bge-reranker-v2-m3; the file keeps its PRD-era name)
  generator/   claude_vision, claude_text
  modes/       text_only, vision_only, caption, hybrid (keyword-weighted RRF), agentic
  router/      LangGraph graph for hybrid (a deterministic DAG, not an agent)
  eval/        metrics, runner, generate, judge, pricing
scripts/       11-23: corpus build, validation, GPU jobs, task builders, scorers (00-10 are Phase 1 history)
eval/          corpus.json, questions.jsonl, page_inventory.csv
docs/          EXPERIMENT_PLAN.md (current plan and results), CORPUS.md, FAILURE_ANALYSIS.md
```

## Reproducing

```bash
cp .env.example .env                      # QDRANT_URL=http://localhost:6335; never commit .env
docker compose up -d qdrant               # ports 6335 (REST) / 6336 (gRPC)
pip install -r requirements.txt           # Windows: set PYTHONUTF8=1 first; see CLAUDE.md for pinned torch

python scripts/11_build_corpus.py         # raw PDFs -> data/pdf/corpus + eval/corpus.json
python scripts/16_make_runpod_bundle.py   # then run embed_pages_gpu.py, 13 and text_retrieval_gpu.py on a GPU box
PYTHONPATH=src python -m pharma_vision_rag.eval.runner --mode all   # retrieval metrics from the GPU outputs
```

The GPU outputs the local runner needs are small (rankings, text vectors, query embeddings); the 18.9 GB of page
patch embeddings are regenerated on demand (about $2 and one hour on RunPod). Windows setup notes and known
pitfalls are in [CLAUDE.md](CLAUDE.md).

## Use it from Claude Desktop

`src/pharma_vision_rag/mcp_server.py` is a stdio MCP server exposing the agentic benchmark's tools
(`list_documents`, `search_text`, `search_pages`, `open_page`, `calculate`) over the 27-document corpus, so you can
ask new questions (Korean or English). It needs the corpus PDFs and `data/embeddings/v2/text/text_chunks.jsonl`
locally; `search_pages` answers only when the local vision index is installed, otherwise it tells Claude to use
`search_text`. Run it as a file, not with `-m` (the package import pulls torch).

Add to `claude_desktop_config.json` (Windows: `%APPDATA%\Claude\claude_desktop_config.json`), adjusting the paths:

```json
{
  "mcpServers": {
    "pharma-corpus": {
      "command": "C:\\Users\\user\\Desktop\\pharma-vision-rag\\.venv\\Scripts\\python.exe",
      "args": ["C:\\Users\\user\\Desktop\\pharma-vision-rag\\src\\pharma_vision_rag\\mcp_server.py"],
      "env": {"PYTHONIOENCODING": "utf-8"}
    }
  }
}
```

Claude Code equivalent:

```bash
claude mcp add pharma-corpus --env PYTHONIOENCODING=utf-8 -- C:\Users\user\Desktop\pharma-vision-rag\.venv\Scripts\python.exe C:\Users\user\Desktop\pharma-vision-rag\src\pharma_vision_rag\mcp_server.py
```

### Remote (claude.ai web, any MCP client)

`serverless/modal_mcp.py` serves the same `MCPServer` over streamable HTTP on Modal (CPU, scales to zero, no card
needed), behind OAuth (claude.ai, Claude Desktop) or a static bearer token (Claude Code). The corpus, `text_chunks.jsonl` and the 2.4 GB pooled vision index live in the
Modal Volume `pharma-corpus-data` at their repo-relative paths; `search_pages` calls the deployed `pharma-vision-encoder`
app and `search_text` the `pharma-text-models` app plus Qdrant Cloud from inside Modal, so nothing runs on your machine.

```bash
modal volume create pharma-corpus-data
modal volume put pharma-corpus-data data/pdf/corpus /data/pdf/corpus              # Git Bash: MSYS_NO_PATHCONV=1
modal volume put pharma-corpus-data eval/corpus.json /eval/corpus.json
modal volume put pharma-corpus-data data/embeddings/v2/text/text_chunks.jsonl /data/embeddings/v2/text/text_chunks.jsonl
modal volume put pharma-corpus-data data/embeddings/vision_index_pooled /data/embeddings/vision_index_pooled
# one secret with MCP_TOKEN (secrets.token_hex(16)), MCP_USER, MCP_PASSWORD (secrets.token_urlsafe(16)),
# QDRANT_CLOUD_URL, QDRANT_CLOUD_API_KEY; keep the same values in .env
modal secret create pharma-mcp-token --from-dotenv <file-with-those-five-lines> --force
modal deploy serverless/modal_mcp.py       # prints https://<workspace>--pharma-mcp-web.modal.run
```

The MCP endpoint is `<that URL>/mcp`; put it in `.env` as `MCP_URL` and the token as `MCP_TOKEN`, then
`python serverless/modal_mcp.py` runs a smoke test (initialize, list tools, one call per tool, timings). A request
without `Authorization: Bearer <token>` gets 401. Measured from the dev box (2026-09-28): cold `initialize` 8 s
(container start, BM25 index, 2.4 GB vision index read from the volume), warm 1 s; `search_text` 0.5 s; `open_page`
1.4~2 s; `search_pages` 26 s when the encoder app is cold, 3.2 s warm. The container stays warm 5 minutes.

Register it:

```bash
# Claude Code
claude mcp add --transport http pharma-corpus-remote https://tkddnjs-dlqslek--pharma-mcp-web.modal.run/mcp \
  --header "Authorization: Bearer <token>"
```

**claude.ai web and Claude Desktop** (custom connectors speak OAuth only, so the Modal app is also a small OAuth 2.1
authorization server: `/.well-known/oauth-authorization-server`, `/register`, `/authorize`, `/token`, and a `/login`
form; tokens and codes live in the Modal Dict `pharma-mcp-auth`):

1. claude.ai > Settings > Connectors > Add custom connector. Name: `pharma-corpus`. Remote MCP server URL:
   `https://tkddnjs-dlqslek--pharma-mcp-web.modal.run/mcp`. Leave Advanced settings (OAuth client id/secret) empty:
   the connector registers itself (dynamic client registration). Click Add.
2. Click Connect. A browser tab opens the server's login page: enter `MCP_USER` and `MCP_PASSWORD` from `.env`
   (the same values are in the Modal secret `pharma-mcp-token`). You are sent back to claude.ai with the connector
   enabled. The access token lasts 30 days, the refresh token 90; a Dict entry unused for 7 days expires (Modal's
   rule), after which the connector asks you to log in again.
3. Claude Desktop shows the same connector list (Settings > Connectors), so it needs no separate setup.

Verified with the `mcp` SDK's own OAuth client (`OAuthClientProvider`: metadata discovery, `/register`, PKCE
`/authorize`, the login form with a wrong then the right password, `/token`, then `list_tools` and `search_text`
with the issued token; refresh and revocation are covered by `tests/test_modal_mcp.py`). Not verified: the claude.ai
connector UI itself (it cannot be driven from here); it uses the same standard flow, but if it refuses, check the
Modal app logs (`modal app logs pharma-mcp`) for the request that failed.

### Vision search for new questions

`search_pages` needs two things: the page index and a query encoder.

1. **Page index**: built once on a GPU box by `scripts/24_vision_index_gpu.py --embed-now` (about an hour on a
   RunPod RTX 4090, roughly $1), downloaded into `data/embeddings/vision_index/`, then shrunk on the dev box with
   `scripts/26_pool_local_index.py` to the 2.4 GB `vision_index_pooled/` the server uses by default
   (`PHARMA_VISION_INDEX` overrides). A full scan takes about 3.4 s on a laptop CPU, 0.1 s with a document filter.
2. **Query encoder**: the same Nemotron 3B model must embed the question. The dev box cannot hold it, so it runs
   remotely. Cheapest path, no card required:

   ```bash
   pip install modal && modal token new                      # once, browser login
   MODAL_GPU=none modal deploy serverless/modal_app.py       # CPU container inside Modal's free credits
   MODAL_GPU=none modal run serverless/modal_app.py          # smoke test: prints the embedding shape and time
   ```

   The MCP server picks the Modal encoder automatically when `~/.modal.toml` exists (`VISION_ENCODER=modal|runpod|none`
   forces it). A cold start loads the 7 GB weights from a Modal Volume (about a minute or two); after that a query takes
   seconds, and the container scales to zero when idle. With a payment method on file, `modal deploy` without
   `MODAL_GPU=none` puts it on an L4 instead. `serverless/handler.py` + `Dockerfile` are the RunPod Serverless
   equivalent (`RUNPOD_API_KEY`, `RUNPOD_ENDPOINT_ID`).

## License

This project's own code, question set and evaluation scripts are **MIT-licensed**.

Third-party models and data keep their own terms:
- **Nemotron ColEmbed VL 3B v2**: check the Hugging Face model card before any commercial use. This project is non-commercial research.
- **BGE-M3, bge-reranker-v2-m3, Docling**: see their repositories.
- **Claude**: Anthropic terms.
- **Company PDFs**: public disclosures used as inputs, not redistributed here.

## Author

김상원 (Sangwon Kim) — Sanofi pharma intern, AI engineering portfolio project.
