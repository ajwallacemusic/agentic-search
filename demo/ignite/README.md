# Ignite demo: retrieval vs harness, live

One question, one corpus, one pair of models, two modes. **Retrieval** plans one pass and ranks
what it finds. **Harness** runs the full loop: plan → query → judge → decide, and goes again when
it does not have enough. The viewer shows both side by side as they run: which tools the agent
uses on each turn and what it searches for, when the loop decides to keep going, and the bill.

Talk: "Will the Real Agentic Search Please Stand Up?" (Egen Ignite). Spec:
`docs/superpowers/specs/2026-10-29-ignite-demo-design.md`.

The corpus is public data with human labels: a seeded sample of **MuSiQue** (multi-hop questions
over Wikipedia paragraphs). It is the same sample the public eval scores: 60 questions (20 each
2-, 3- and 4-hop) and 5,000 paragraphs, in one source, `wiki`. Nothing in it was written by us.
The demo questions are picked from the sample by MuSiQue id in `questions.yaml`.

Be open about the pick: the hero was chosen from a scan of all 20 sampled 4-hop questions
because it shows the gap clearly (`tuning-log.md`, round 9). The fair numbers are the
60-question averages in `results-musique.md`.

The hero question is a 4-hop question (`4hop1__777217_32392_823060_610794`):

> What county is the city that shares a border with the state capital of the state where Andrew
> Deveaux was born located in?

Each hop needs a fact from the hop before it, so one planned pass cannot know what to search for
next. Its four supporting paragraphs are the gold.

| Hop | Gold paragraph | What it gives |
|---|---|---|
| 1 | *Andrew Deveaux* | born in South Carolina |
| 2 | *Charleston, South Carolina* | the state capital moved to Columbia in 1786 |
| 3 | *WWNQ* | a radio station in Forest Acres, serving the Columbia market |
| 4 | *Forest Acres, South Carolina* | Forest Acres is in Richland County (the answer) |

Hops 2 and 3 sit in paragraphs about other things (Charleston, a radio station), so a search for
the question's own words does not find them.

## Setup

```bash
uv sync --extra server --extra local --extra openai --extra gcp
gcloud auth application-default login
export IGNITE_GCP_PROJECT=ajw-sandbox-497707     # all demo runs use this project (Gemini 4 Argon works here; gen-ai-accel-dev refuses it)
```

The first run downloads MuSiQue (about 30 MB) into `public/.cache/` (git-ignored) and checks it
against a pinned SHA-256. After that the demo needs no download. Do the first run before the
talk.

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
Each paragraph needs its `title` as metadata, so `corpus.py` loads the MuSiQue sample,
`build.py` builds a `FilesBackend.from_documents(...)` from it in Python, and `serve.py` passes
that `Harness` to the library's `create_app`. Nothing under `src/` changed.

`build.py` also works around three library gaps (each worth its own library PR):

- `SerialEmbedder`: `Harness.setup()` indexes all sources at once; the shared
  sentence-transformers model segfaults when two threads load or run it.
- `RequestDefaults` / `make_client`: the library's Vertex client cannot send `reasoning_effort`
  or a per-request timeout. The demo sends `reasoning_effort: low`, cuts a request at 15 s and
  retries twice.
- The library's `decide` events leave out the controller's note. The viewer reads the notes from
  the trace in the final event, so in live mode the notes appear when each run ends.

## Commands

```bash
uv run python -m demo.ignite.run_side_by_side --question hero         # terminal trace + summary
uv run python -m demo.ignite.run_side_by_side --question hero --save  # also writes runs/<stamp>-hero-*.jsonl
uv run python -m demo.ignite.serve                           # http://127.0.0.1:8765/ (viewer + API)
uv run python -m demo.ignite.public.eval_public --dataset musique     # writes results-musique.md
uv run pytest demo/ignite/tests -q                           # demo tests (offline; needs node)
```

Other questions: `--question <id>`; ids are in `questions.yaml`. To add one, add a line with a
MuSiQue id from the seed-7 sample; `load_demo` fails if the id is not in the sample. Setup embeds
5,000 paragraphs, so `serve` and the runner take a minute or two to start.

**Jev (TypeSafe) controller**, optional: set `TYPESAFE_API_KEY`, then
`run_side_by_side --controller typesafe`. `serve` adds an `ignite-typesafe` profile when the key
is set (the viewer itself runs the `ignite` profile).

## The viewer

`viewer/index.html`, one file, no build step, no network calls except to `serve.py`.

- **Live:** open `http://127.0.0.1:8765/`, pick a question, press **Run**. Both modes start at
  the same time.
- **Swimlanes:** one row per tool (`lexical`, `vector`, `filter`, `fetch`), one column per turn.
  Each dot shows what that call searched for; hover for the full arguments. There is one source,
  so the story is tool choice and follow-up searches, not source switching.
- **Replay:** press **Switch to replay** (or open `viewer/index.html` straight from Finder; no
  server or internet needed). Pick the three files of one saved run:
  `<stamp>-hero-retrieval.jsonl`, `<stamp>-hero-harness.jsonl` and `<stamp>-hero.meta.json`.
  Press **Play**. Speed 1×/2×/4×, **Pause**, **Replay again**. Gold docs get ★ gold in replay
  only.

## Numbers for the slides

Hero question, round 9 of `tuning-log.md` (5 runs): the harness found all four gold paragraphs
in 4 of 5 runs, in 14–19 s, for about $0.019. Retrieval found 1 of 4 in all 5 runs, for about
$0.005.

MuSiQue, all 60 sampled questions (`results-musique.md`): recall@10 is 0.792 for harness and
0.647 for retrieval. On 4-hop questions it is 0.775 vs 0.550. The harness costs about 2.6×.

Say on the slide: both modes run with `max_limit: 8` and `max_calls_per_turn: 6`
(`config.yaml`). Small result pages make a single pass miss more than the loop does.

## Public datasets (real labels, not written by us)

Both sets have human labels; nothing was written or tuned by us. MuSiQue is also the live demo's
corpus. WANDS checks the "least agentic setting that works" point. Same planner, judge and
settings as the demo; one source each; one run per query; seed 7.

| dataset | what it tests | sample | bm25 | retrieval | harness | harness cost vs retrieval |
|---|---|---|---|---|---|---|
| MuSiQue (multi-hop Wikipedia) | does the loop chase hops? | 60 questions (20 each 2/3/4-hop), 5,000 paragraphs | recall@10 0.468 | 0.647 | **0.792** | ~2.6× |
| WANDS (Wayfair product search) | does the loop help plain ranking? | 60 queries, 42,994 products | nDCG@10 0.715 | **0.796** | 0.776 | ~3.5× |

- **MuSiQue:** the loop gains most on 4-hop questions (recall@10 0.775 vs 0.550).
- **WANDS:** the loop does not improve ranking (nDCG@10 0.776 vs 0.796, within noise for one run
  of 60 queries) and costs about 3.5×. Use the least agentic setting that works.
- Per-hop gaps of a few points (e.g. MuSiQue 3-hop, 0.750 vs 0.717) are within noise too; quote
  the "All" rows and 4-hop.
- Full tables: `results-musique.md`, `results-wands.md`. Live runs took about 12 and 21
  minutes and cost about $1.50 and $2.10.

Why MuSiQue and not HotpotQA: MuSiQue was built to drop questions a single-hop shortcut can answer,
a known weakness of HotpotQA, and each question ships with its own 20 paragraphs, so a sample is
self-contained.

Re-run (downloads into `public/.cache/`, git-ignored; each file is checked against a pinned
SHA-256):

```bash
uv run python -m demo.ignite.public.eval_public --dataset musique   # --n 20 per hop, --corpus-size 5000
uv run python -m demo.ignite.public.eval_public --dataset wands     # --n 60
```

Credits: MuSiQue, Trivedi et al., TACL 2022 (CC BY 4.0), github.com/StonyBrookNLP/musique.
WANDS, Chen et al., ECIR 2022 (MIT), github.com/wayfair/WANDS.

## Rehearsal checklist

1. Rehearse three times on the presentation laptop, on the venue network if you can.
2. Before the talk: `gcloud auth application-default login` (tokens expire), start `serve`, run
   the hero question once to warm up (model connections, embedder load), then reload the page.
3. Keep a second tab open on `viewer/index.html` from disk with the backup run loaded in replay.
4. Keep `results-musique.md` open in a third tab.

## Fallback

If a live run stalls past 60 seconds or shows **✗ Search failed**: press **Switch to replay**
(or go to the replay tab) and say so plainly: "the live API is slow, here is the same run from
this morning."

Backup runs in `runs/`:

- `20261029-163918-hero-*`: harness 4/4 in 11.1 s, retrieval 1/4.
- `20261029-163955-hero-*`: harness 4/4 in 11.9 s, retrieval 1/4.
- `20261029-163707-hero-*`: harness 4/4 in 15.2 s, retrieval 1/4.

The harness found all four in 9 of 12 hero runs so far, so do expect a 3/4 now and then.

Known risk: Vertex latency spikes. A planner call that times out twice in a row ends the
harness run with `APITimeoutError`. One model call waits at most about 30 s (15 s, one retry),
and the viewer shows "⚠ Past 60 s. Switch to replay?" in a live column that runs long.
