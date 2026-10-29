# Ignite demo: retrieval vs harness, live

One question, one corpus, one pair of models, two modes. **Retrieval** plans one pass and ranks
what it finds. **Harness** runs the full loop: plan → query → judge → decide, and goes again when
it does not have enough. The viewer shows both side by side as they run: which tools hit which
sources, when the loop decides to keep going, and the bill.

Talk: "Will the Real Agentic Search Please Stand Up?" (Egen Ignite). Spec:
`docs/superpowers/specs/2026-10-29-ignite-demo-design.md`.

The corpus is synthetic: **Lumen Retail**, a made-up company. The hero question needs four hops,
each in a different source:

> Did last month's latency spike come from the new reranker, and is it fixed?

`INC-4821` (incident, no root cause) → `DEP-0913-07` (the deploy that enabled reranker v2) →
`PR-1203` (the fix) → `DASH-W38-search` (p95 back to baseline).

## Setup

```bash
uv sync --extra server --extra local --extra openai --extra gcp
gcloud auth application-default login
export IGNITE_GCP_PROJECT=<your project>        # rehearsals used gen-ai-accel-dev
```

Models and prices are in `config.yaml` (Vertex, `global` endpoint):

| role | model | price per 1M tokens (in / out) |
|---|---|---|
| planner (driver) | `gemini-3.5-flash` | $1.50 / $9.00 |
| judge | `gemini-3.5-flash-lite` | $0.30 / $2.50 |

Both modes use the same pair. Why these and not Gemini 3.8 Flash: see `tuning-log.md`. In short,
3.7/3.8 Flash stalled for 35–60 s on about one call in eight, which kills a live run.

To use Anthropic instead: comment out `driver:` and `judge:` and uncomment the `anthropic` line;
set `ANTHROPIC_API_KEY` and `uv sync --extra anthropic`.

### How the harness is built

The YAML `files` backend loads a folder of files and keeps only file metadata (path, size, …).
The deploy log needs filterable `date`, `service` and `pr`, so `build.py` builds one
`FilesBackend.from_documents(...)` per source from `data/*.jsonl` in Python, and `serve.py`
passes those `Harness` objects to the library's `create_app`. Nothing under `src/` changed.

`build.py` also works around three library gaps (each worth its own library PR):

- `SerialEmbedder`: `Harness.setup()` indexes all five sources at once; the shared
  sentence-transformers model segfaults when two threads load or run it.
- `RequestDefaults` / `make_client`: the library's Vertex client cannot send `reasoning_effort`
  or a per-request timeout. The demo sends `reasoning_effort: low`, cuts a request at 15 s and
  retries twice.
- The library's `decide` events leave out the controller's note. The viewer reads the notes from
  the trace in the final event, so in live mode the notes appear when each run ends.

## Commands

```bash
uv run python -m demo.ignite.check_data                      # corpus checks: ids, gold ids, the hero chain
uv run python -m demo.ignite.run_side_by_side --question hero         # terminal trace + summary
uv run python -m demo.ignite.run_side_by_side --question hero --save  # also writes runs/<stamp>-hero-*.jsonl
uv run python -m demo.ignite.serve                           # http://127.0.0.1:8765/ (viewer + API)
uv run python -m demo.ignite.eval --modes bm25,retrieval,harness      # writes beir/ and results.md
uv run pytest demo/ignite/tests -q                           # demo tests (offline; needs node)
```

Other questions: `--question <id>`; ids are in `data/questions.yaml`.

**Jev (TypeSafe) controller**, optional: set `TYPESAFE_API_KEY`, then
`run_side_by_side --controller typesafe`. `serve` adds an `ignite-typesafe` profile when the key
is set (the viewer itself runs the `ignite` profile).

## The viewer

`viewer/index.html`, one file, no build step, no network calls except to `serve.py`.

- **Live:** open `http://127.0.0.1:8765/`, pick a question, press **Run**. Both modes start at
  the same time.
- **Replay:** press **Switch to replay** (or open `viewer/index.html` straight from Finder; no
  server or internet needed). Pick the three files of one saved run:
  `<stamp>-hero-retrieval.jsonl`, `<stamp>-hero-harness.jsonl` and `<stamp>-hero.meta.json`.
  Press **Play**. Speed 1×/2×/4×, **Pause**, **Replay again**. Gold docs get ★ gold in replay
  only.

## Numbers for the slides

`results.md` (11 questions, one run each):

| | retrieval recall@10 | harness recall@10 | harness cost vs retrieval |
|---|---|---|---|
| multi-hop (7) | 0.738 | 0.952 | ~2.3× |
| single-hop (4) | 1.000 | 1.000 | ~2.3× |

Hero question, round 8 of `tuning-log.md`: harness found all four hops in 4 of 5 runs in
13–16 s for about $0.026; retrieval missed at least one hop in 5 of 5 for about $0.009.

Say on the slide: both modes run with `max_limit: 8` and `max_calls_per_turn: 6`
(`config.yaml`). Small result pages make a single pass miss more than the loop does. On
single-hop questions both modes score the same (nDCG@10 1.000) and the harness costs about 2.3×
as much.

## Rehearsal checklist

1. Rehearse three times on the presentation laptop, on the venue network if you can.
2. Before the talk: `gcloud auth application-default login` (tokens expire), start `serve`, run
   the hero question once to warm up (model connections, embedder load), then reload the page.
3. Keep a second tab open on `viewer/index.html` from disk with the backup run loaded in replay.
4. Keep `results.md` open in a third tab.

## Fallback

If a live run stalls past 60 seconds or shows **✗ Search failed**: press **Switch to replay**
(or go to the replay tab) and say so plainly: "the live API is slow, here is the same run from
this morning."

Backup runs in `runs/` (round 8 settings):

- `20261029-115838-hero-*`: harness 4/4 in 13.5 s, retrieval 1/4.
- `20261029-120008-hero-*`: harness 4/4 in 15.5 s, retrieval 2/4.

Known risk: Vertex latency spikes. A planner call that times out three times in a row ends the
harness run with `APITimeoutError`. It happened in 1 of 5 runs on Gemini 3.8 Flash and 0 of 5 on
3.5 Flash. Since the review fix, one model call waits at most about 30 s (15 s, one retry),
and the viewer shows "⚠ Past 60 s. Switch to replay?" in a live column that runs long.
