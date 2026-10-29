# Ignite Demo: Retrieval vs Harness, Live

**Status:** draft for a Claude Code session
**Date:** 2026-10-29
**Owner:** AJ
**Context:** Egen Ignite talk, "Will the Real Agentic Search Please Stand Up?" (~25 min). The demo fills about 3–4 minutes in Part 5 and supplies real numbers for the "why it works" and "reading the researcher's notes" slides.

## 1. Goal

Show the room the difference between a single planned pass and an agentic loop, on the same question, the same data and the same model, with the agent's decisions visible as they happen.

The audience should see three things, which the talk tells them to watch for:

1. **Tool choice**: different tools and sources for different parts of the question.
2. **The gap**: the moment the loop decides it doesn't have enough, and what it does next (refine, broaden, switch source).
3. **The bill**: turns, tool calls, seconds and cost to get the right answer.

## 2. Non-goals

- No changes to the library under `src/`. The demo is a consumer of the harness. If a library change turns out to be needed, stop and raise it as a separate PR.
- No production deployment, auth hardening or multi-user concerns. The demo runs on a laptop.
- No real client or Egen data. Everything is synthetic.

## 3. Deliverables

All new files live under `demo/ignite/`.

| Path | Purpose |
|---|---|
| `demo/ignite/data/*.jsonl` | Synthetic corpus, one file per source |
| `demo/ignite/data/questions.yaml` | Demo questions with gold document ids |
| `demo/ignite/check_data.py` | Validates the corpus: ids unique, every gold id exists, the planted chain is intact |
| `demo/ignite/build.py` | Builds the harness (backends, driver, analyzer, controller, budget) from a small config |
| `demo/ignite/run_side_by_side.py` | Runs one question in `retrieval` and `harness` mode; prints a compact trace and summary; saves every event to JSONL |
| `demo/ignite/serve.py` | Serves the demo profile over HTTP/SSE for the viewer (localhost only) |
| `demo/ignite/viewer/index.html` | Single-file trace viewer: live (SSE) or replay (saved JSONL) |
| `demo/ignite/beir/` | BEIR-format export of the corpus and questions, for the eval script |
| `demo/ignite/runs/` | Saved event logs from rehearsals; the best ones are the backup |
| `demo/ignite/README.md` | How to run, rehearse and fall back |

## 4. The scenario and corpus

A fictional company, **Lumen Retail**, with a product search service. All names, dates and numbers are invented.

### 4.1 The hero question

> "Did last month's latency spike come from the new reranker, and is it fixed?"

The answer needs four hops, each in a different source:

| Hop | Source | Gold document | What it gives |
|---|---|---|---|
| 1 | `incidents` | `INC-4821` | p95 latency spike on `search-api` starting Sept 14. No root cause recorded. |
| 2 | `deploys` | `DEP-0913-07` | `search-api` v4.12 on Sept 13 enabled reranker v2 (links PR 1187). |
| 3 | `pull_requests` | `PR-1203` | Rollback of reranker v2's batch size plus a cache fix, merged Sept 18; references `INC-4821`. |
| 4 | `dashboards` | `DASH-W38-search` | Week of Sept 18: `search-api` p95 back to baseline. |

The chain must only be discoverable by following it: hop 3 is best found by searching for something learned in hop 2 (the PR number or "reranker v2"), and hop 4 by the date learned in hop 3.

### 4.2 Sources

Each source is its own backend, so tool calls visibly target different sources.

| Source | Docs | Shape | Notes |
|---|---|---|---|
| `incidents` | ~30 | ticket text + metadata (`id`, `opened`, `service`, `severity`, `status`) | Include an August latency incident caused by a CDN change (distractor) |
| `deploys` | ~60 | short structured records + metadata (`date`, `service`, `version`, `pr`, `author`) | Several `search-api` deploys around Sept 13–14, only one relevant; a same-day unrelated feature-flag deploy |
| `pull_requests` | ~25 | PR title + description + metadata (`number`, `merged`, `service`) | Include PR 1187 (reranker v2 rollout), PR 1203 (the fix), and an older reranker v1 PR (distractor) |
| `dashboards` | ~12 | weekly metric summaries per service | Include a week where a *different* service had a p95 spike (distractor) |
| `runbooks` | ~10 | operational docs | Pure distractors with overlapping vocabulary ("latency", "rollback") |

### 4.3 Supporting questions (for the eval)

Write 8–12 more questions with gold ids, mixing:

- **Single-hop** (3–4): answerable from one document, such as "Who authored the reranker v2 PR?" Retrieval mode should do about as well here. This backs the talk's "least agentic setting that works" point.
- **Multi-hop** (5–8): two to four hops, like the hero question. The harness should win here.

### 4.4 Data rules

- Write the corpus by hand (or have Claude draft it), commit it as JSONL, and keep it deterministic. Don't generate it at runtime.
- Keep documents realistic in length and tone: tickets read like tickets, PRs like PRs.
- `check_data.py` must fail if a gold id is missing or a chain link (incident → deploy → PR → dashboard) is broken.

## 5. Harness setup

### 5.1 Configuration

| Role | Default | Option |
|---|---|---|
| Driver | Vertex `gemini-*-flash` (our stack) | Anthropic Sonnet, if Vertex access is a problem |
| Analyzer | `LLMJudge` on the same client | — |
| Controller | built-in heuristic | `typesafe` (Jev), as an optional second run |
| Embedder | `sentence_transformers` small model if it installs cleanly; otherwise `hash` | — |
| Budget | `max_turns: 4`, `max_tool_calls: 24`, `max_seconds: 60`, `max_cost_usd: 0.50` | Tune after rehearsal |

Set `price_per_mtok` for Vertex so cost reads correctly; the "bill" is part of the demo.

### 5.2 First investigation step

Before building: check how `FilesBackend` exposes metadata when loading from a directory versus `from_documents`. The deploy log needs filterable fields (`date`, `service`) so the agent can use `filter_search`.

- If YAML-configured file backends can carry metadata, use `search.yaml` and `load_harness`.
- If not, build the backends in Python with `FilesBackend.from_documents(...)` in `build.py`, and have `serve.py` construct the service from those `Harness` objects instead of from YAML.

Record which path was taken in the README.

## 6. Side-by-side runner

`run_side_by_side.py --question hero [--controller heuristic|typesafe] [--save]`

- Runs the question in `retrieval` mode, then `harness` mode, using `Harness.stream()`.
- Prints a compact live trace per run: phase changes, each tool call (`source`, tool, short args), decisions with their notes.
- Ends with a summary table per mode: turns, tool calls, wall-clock seconds, cost, stop reason, top 5 hits, and **gold recall** (how many of the four gold docs are in the top 10).
- `--save` writes every event to `runs/<timestamp>-<question>-<mode>.jsonl`, one event per line (`ev.model_dump_json()`).

## 7. Trace viewer

A single static HTML file, no build step, readable on a projector.

**Two modes**

- **Live:** connects to `serve.py` and streams both modes for the chosen question.
- **Replay:** loads two saved JSONL files and plays them back with timing preserved, with 1×/2×/4× speed and pause. This is the backup if the network or a model API fails on the day.

**Layout**

- Two columns: **Retrieval mode (one pass)** on the left, **Harness mode (full loop)** on the right.
- Each column has a swimlane: one row per source (`incidents`, `deploys`, `pull_requests`, `dashboards`, `runbooks`), with time or turn across. Each tool call is a dot in its source's row, labelled with the tool name.
- A phase strip above each swimlane lights up `plan → query → judge → decide` as events arrive.
- Decision badges between turns: `continue`, `refine`, `broaden`, `switch_source`, `stop`, each showing the controller's note.
- A results list under each swimlane that grows as `results_updated` arrives. Gold docs get a subtle marker in replay mode only (so it isn't a giveaway live).
- A metrics footer per column: turns, tool calls, seconds, cost, stop reason.

**Implementation notes**

- Parse SSE from `fetch` with a small reader, since `EventSource` can't send a POST body. The TypeScript client in `clients/typescript` is the reference for event shapes; don't add a build step just to use it.
- Large type (minimum 18px body), high contrast, no dependence on colour alone.
- No external network calls from the page.

## 8. Eval

- Export the corpus and questions to BEIR format under `demo/ignite/beir/` (`corpus.jsonl`, `queries.jsonl`, `qrels/test.tsv`).
- Check whether `scripts/eval_beir.py` accepts a local dataset via `--dataset ignite --data-dir demo/ignite`. If it only downloads public datasets, add a thin `demo/ignite/eval.py` that calls the existing eval runner on the local export, without changing `scripts/`.
- Run `--modes bm25,retrieval,harness` and save the table to `demo/ignite/results.md` with nDCG@10, recall@10 and MRR per mode, split into single-hop and multi-hop questions, plus average turns, tool calls and cost.

These numbers replace the borrowed figures on the talk's "why it works" slide.

## 9. Acceptance criteria

- `check_data.py` passes.
- Hero question, harness mode: all four gold docs in the top 10 in at least 4 of 5 runs, finishing in under 45 seconds.
- Hero question, retrieval mode: misses at least one gold doc in most runs. If retrieval mode reliably wins, make the corpus harder (better distractors, a less obvious link between hops). Don't weaken the retrieval mode.
- Single-hop questions: retrieval and harness modes score within a few points of each other.
- The viewer runs live against `serve.py` and in replay from saved files, with no internet access needed for replay.
- At least two clean rehearsal runs saved in `runs/` as the backup.
- `uv run pytest -q` and `uv run ruff check` stay clean; nothing under `src/` changed.

## 10. Rehearsal and fallback

- Rehearse three times on the presentation laptop and network.
- Before the talk: warm up once (model connections, embedder load), then reload the viewer.
- If the live run stalls past 60 seconds or errors: switch the viewer to replay and say so plainly.
- Keep `results.md` and one saved run open in a tab.

## 11. Open questions

1. Vertex or Anthropic as the driver on the day? Vertex fits "our stack"; Anthropic may be faster to set up.
2. Include a Jev (TypeSafe) controller run live, or only mention it? Live adds a dependency and a key.
3. Egen branding in the viewer, or neutral?
4. Should the viewer also show a third column for `model` mode (SID-1), if access exists? Default: no.

## 12. Suggested task order

1. Investigate `FilesBackend` metadata and the eval script's dataset loading (5.2, 8). Decide the build path.
2. Write the corpus, questions and `check_data.py`.
3. `build.py` and `run_side_by_side.py`; iterate on the corpus until the acceptance criteria hold in the terminal.
4. `serve.py` and the viewer, live mode.
5. Replay mode and saved runs.
6. BEIR export and eval results.
7. README and rehearsal notes.
