# Ignite demo: retrieval vs harness, live

One question, one corpus, one pair of models, two modes. **Retrieval** plans one pass and ranks
what it finds. **Harness** runs the full loop: plan → query → judge → decide, and goes again when
it does not have enough. The viewer shows both side by side as they run: which tools the agent
uses on each turn and what it searches for, the step it is on, when the loop decides to keep
going, and the bill.

Talk: "Will the Real Agentic Search Please Stand Up?" (Egen Ignite). Specs:
`docs/superpowers/specs/2026-10-29-ignite-demo-design.md`,
`docs/superpowers/specs/2026-10-30-ignite-wands-design.md`.

## The story in three parts

1. **MuSiQue: the loop wins on multi-hop questions.** Each hop needs a fact from the hop before
   it. One planned pass cannot know what to search for next.
2. **WANDS: the loop does not help plain product search.** Same models, same settings. Ranking
   is about the same (within noise) for about 3× the cost. Use the least agentic setting that
   works.
3. **WANDS match: the loop wins again when the question has steps.** Same WANDS catalogue, our own
   rule-labeled questions: "I have product X. Find a Y with the same wood tone." First find X and
   read its wood tone, then search for Y. One pass cannot do the second step.

Three question sets, two corpora, all in `questions.yaml`:

| set | corpus | questions | gold |
|---|---|---|---|
| `musique` | MuSiQue sample, 5,000 Wikipedia paragraphs (`wiki`) | MuSiQue ids | the supporting paragraphs (human labels) |
| `wands` | WANDS catalogue, 42,994 products (`products`) | WANDS query ids | "Exact" labels (human labels) |
| `wands-match` | the same WANDS catalogue | rule ids from `wands_match.yaml` | made by a rule from WANDS product features (ours) |

Be open about the picks: every demo question was picked from a full scan because it shows the
point clearly (`tuning-log.md`, rounds 9 and 10). The fair numbers are the averages in
`results-musique.md`, `results-wands.md` and `results-wands-match.md`.

### MuSiQue hero

`hero` (`4hop1__777217_32392_823060_610794`):

> What county is the city that shares a border with the state capital of the state where Andrew
> Deveaux was born located in?

| Hop | Gold paragraph | What it gives |
|---|---|---|
| 1 | *Andrew Deveaux* | born in South Carolina |
| 2 | *Charleston, South Carolina* | the state capital moved to Columbia in 1786 |
| 3 | *WWNQ* | a radio station in Forest Acres, serving the Columbia market |
| 4 | *Forest Acres, South Carolina* | Forest Acres is in Richland County (the answer) |

Hops 2 and 3 sit in paragraphs about other things (Charleston, a radio station), so a search for
the question's own words does not find them.

### WANDS picks

- `awning` (query 260, "retractable side awning"): a tie. Both modes nDCG@10 1.0; the harness
  costs about 4× more.
- `trays` (query 346, "nesting tray set"): the harness costs about 4.5× more and scores a little
  lower (nDCG@10 0.93 vs 0.96).

### WANDS match hero

`match-hero` (rule `match-19`):

> I have the Alaniz star single rocking chair. Find a patio sofa or patio table with the same wood
> tone.

The chair's features say `dswoodtone : light wood`. Gold: the 40 patio sofas and patio tables with
that wood tone. The question never says "light wood", so a single pass searches for the chair or
for "patio sofa" and finds no gold. The harness makes a 2-step plan, opens the chair, notes "wood
tone: light wood" in its notebook, then searches patio sofas and tables for it.

How the questions are made (`public/wands_match.py`):

- The anchor product has exactly one value for the attribute (`wood_tone` from `dswoodtone`,
  `color`, or `style`). Its name is unique in the catalogue, and the value is not in its name.
- Gold: every product in the target classes whose feature has the same value. Not the anchor.
- `select` picks the rules, deterministically. We wrote only the question text. `labels`
  rebuilds `wands_match_labels.json` from the rules; a test checks the committed file matches.

```bash
uv run python -m demo.ignite.public.wands_match select   # candidate rules (prints YAML)
uv run python -m demo.ignite.public.wands_match labels   # rewrites wands_match_labels.json
```

## Setup

```bash
uv sync --extra server --extra local --extra openai --extra gcp
gcloud auth application-default login
export IGNITE_GCP_PROJECT=ajw-sandbox-497707     # all demo runs use this project (Gemini 4 Argon works here; gen-ai-accel-dev refuses it)
```

The first run downloads MuSiQue (about 30 MB) and WANDS into `public/.cache/` (git-ignored) and
checks each file against a pinned SHA-256. After that the demo needs no download. Do the first run
before the talk.

**Start-up time.** Setup embeds every document. The first WANDS start embeds 42,994 products and
takes about 4.5 minutes. The vectors are then kept in
`public/.cache/embeddings/document-vectors.npz` (`embedding_cache` in `config.yaml`), and later
starts take about 10 s. Start `serve` once before the talk to fill the cache.

Models and prices are in `config.yaml` (Vertex, `global` endpoint, project `ajw-sandbox-497707`):

| role | model | price per 1M tokens (in / out) | per call |
|---|---|---|---|
| planner (driver) | `gemini-4-argon` | $2.00 / $10.00 | `reasoning_effort: low`, cut at 45 s, 1 retry |
| judge | `gemini-3.5-flash-lite` | $0.30 / $2.50 | cut at 15 s, 1 retry |

Both modes use the same pair. Argon prices are the introductory prices; they are reported to go
to $4 / $20 on 2026-12-01. Why Argon: with `gemini-3.5-flash` as the planner, the WANDS match
harness scored P@10 0.055, because Flash did not write facts to its notebook. Argon does
(`tuning-log.md`, round 10).

To use Anthropic instead: comment out `driver:` and `judge:` and uncomment the `anthropic` line;
set `ANTHROPIC_API_KEY` and `uv sync --extra anthropic`.

### How the harness is built

The YAML `files` backend loads a folder of files and keeps only file metadata (path, size, …).
Each document needs its title as metadata, so `corpus.py` loads the samples, `build.py` builds
one `FilesBackend.from_documents(...)` per corpus in Python, and `serve.py` passes those to the
library's `create_app` (profile `ignite` for MuSiQue, `ignite-wands` for WANDS).

**Step tracking** (`settings.steps: true` in `config.yaml`) is a library feature (library PR #3,
merged). The planner keeps a short step plan and a notebook of
facts it read in documents, with the document each fact came from. The loop does not stop while a
step is open. The viewer shows the steps and facts as they happen. The demo itself changes nothing under
`src/`.

The library fixes the demo once worked around are merged (library PRs #4, #5 and #6): the
sentence-transformers embedder is thread-safe, `vertex` clients take `reasoning_effort`,
`request_timeout_s` and `max_retries` from config, and `decide` events carry the controller's
note. The viewer still reads notes from the final trace as a fallback, for replays saved before
the fix.

## Commands

```bash
uv run python -m demo.ignite.run_side_by_side --question hero         # terminal trace + summary
uv run python -m demo.ignite.run_side_by_side --question match-hero   # WANDS match hero
uv run python -m demo.ignite.run_side_by_side --question hero --save  # also writes runs/<stamp>-hero-*.jsonl
uv run python -m demo.ignite.serve                           # http://127.0.0.1:8765/ (viewer + API)
uv run pytest demo/ignite/tests -q                           # demo tests (offline; needs node)
```

Other questions: `--question <id>`; ids are in `questions.yaml`. To add one, add a line with its
set and ref (a MuSiQue id, a WANDS query id from the seed-7 sample, or a rule id);
`load_demo` fails if the ref is not in the sample.

**Jev (TypeSafe) controller**, optional: set `TYPESAFE_API_KEY`, then
`run_side_by_side --controller typesafe`. `serve` adds an `ignite-typesafe` profile when the key
is set (the viewer itself runs the `ignite` and `ignite-wands` profiles).

## The viewer

`viewer/index.html`, one file, no build step, no network calls except to `serve.py`.

- **Live:** open `http://127.0.0.1:8765/`, pick a question (grouped by set), press **Run**. Both modes
  start at the same time.
- **Swimlanes:** one row per tool (`lexical`, `vector`, `hybrid`, `filter`, `fetch`, …), one
  column per turn. Each dot shows what that call searched for; hover for the full arguments.
- **Steps:** under the harness lanes, the step plan with the current step marked **NOW**, and
  each fact the planner noted, with its document.
- **Results and answer:** each lane lists its top 10 documents (title and first lines). When a
  mode finishes, an answer box shows what a reader model (the judge model, `answer.py`) answers
  from that mode's top 10 only; for MuSiQue it says ✓ correct or ✗ wrong against the dataset's
  answer. The ★ gold marks and the "★ n of m gold in top 10" count appear only once that mode has
  finished, so nothing is given away mid-run. Needs `serve`: a replay opened from Finder has no
  answer box.
- **Replay:** press **Switch to replay** (or open `viewer/index.html` straight from Finder; no
  server or internet needed). Pick the three files of one saved run:
  `<stamp>-<id>-retrieval.jsonl`, `<stamp>-<id>-harness.jsonl` and `<stamp>-<id>.meta.json`.
  Press **Play**. Speed 1×/2×/4×, **Pause**, **Replay again**. Gold docs get ★ gold in replay
  only.

## Numbers for the slides

All numbers: Argon planner, step tracking on, project `ajw-sandbox-497707`, one run per
question, seed 7, 2026-11-09 (judge sees noted facts; `max_limit: 20`; `tuning-log.md` round 11).

| set | what it tests | sample | metric | bm25 | retrieval | harness | harness cost vs retrieval |
|---|---|---|---|---|---|---|---|
| MuSiQue (multi-hop Wikipedia) | does the loop chase hops? | 60 questions (20 each 2/3/4-hop), 5,000 paragraphs | recall@10 | 0.468 | 0.697 | **0.875** | ~3.3× |
| WANDS (Wayfair product search) | does the loop help plain ranking? | 60 queries, 42,994 products | nDCG@10 | 0.715 | 0.787 | 0.766 | ~3.1× |
| WANDS match (ours) | does the loop help when a step needs a fact read on the way? | 20 questions, same catalogue | P@10 | 0.015 | 0.050 | **0.515** | ~4.6× |

- **MuSiQue:** 4-hop recall@10 0.800 vs 0.575.
- **WANDS:** -0.02 nDCG@10 for about 3× the cost and about 3× the time (47 s vs 15 s). The sign
  has flipped between runs (+0.03 on 2026-10-30); call it a tie.
- **WANDS match:** wood tone 0.714 vs 0.086, color 0.414 vs 0.014, style 0.400 vs 0.050 (P@10).
  Quote the average, not only the hero.
- The harness is slow: about 40–55 s per question, against about 10–15 s for retrieval.

Hero runs (`tuning-log.md`, round 10):

- MuSiQue `hero`: harness 4/4 gold in 3 of 3 runs, 31–35 s, about $0.043. Retrieval 1/4, about
  $0.010.
- WANDS match `match-hero` (2026-11-09): harness 10 of 10 gold in 3 of 3 runs, 46–47 s, about
  $0.06. Retrieval 0 in all 3.

Say on the slide: both modes run with `max_limit: 20` and `max_calls_per_turn: 6`
(`config.yaml`), and a budget of 4 turns.

## Public datasets and re-runs

MuSiQue and WANDS have human labels; nothing in them was written or tuned by us. WANDS match
uses the WANDS catalogue with our rule labels.

```bash
uv run python -m demo.ignite.public.eval_public --dataset musique       # --n 20 per hop, --corpus-size 5000
uv run python -m demo.ignite.public.eval_public --dataset wands         # --n 60
uv run python -m demo.ignite.public.eval_public --dataset wands-match   # all 20 rules
```

The three runs took about 25, 25 and 10 minutes and cost about $2.70, $3.80 and $1.30.

Why MuSiQue and not HotpotQA: MuSiQue was built to drop questions a single-hop shortcut can answer,
a known weakness of HotpotQA, and each question ships with its own 20 paragraphs, so a sample is
self-contained.

Credits: MuSiQue, Trivedi et al., TACL 2022 (CC BY 4.0), github.com/StonyBrookNLP/musique.
WANDS, Chen et al., ECIR 2022 (MIT), github.com/wayfair/WANDS.

## Rehearsal checklist

1. Rehearse three times on the presentation laptop, on the venue network if you can.
2. Before the talk: `gcloud auth application-default login` (tokens expire), start `serve` (fills
   the embedding cache on the first start), run `hero` and `match-hero` once to warm up, then
   reload the page.
3. Keep a second tab open on `viewer/index.html` from disk with the backup runs loaded in replay.
4. Keep the three `results-*.md` files open in a third tab.

## Fallback

If a live run stalls past 60 seconds or shows **✗ Search failed**: press **Switch to replay**
(or go to the replay tab) and say so plainly: "the live API is slow, here is the same run from
this morning."

Backup runs in `runs/` (Argon):

- `20261030-133739-hero-*`, `20261030-133833-hero-*`, `20261030-133927-hero-*`: MuSiQue hero,
  harness 4/4, retrieval 1/4.
- `20261109-132250-hero-*` (retrieval 2/4), `20261109-132357-hero-*` (retrieval 1/4): MuSiQue hero,
  harness 4/4, current code; decide events carry the controller's notes.
- `20261109-121547-match-hero-*`, `20261109-121658-match-hero-*`, `20261109-121804-match-hero-*`:
  harness 10 gold in top 10, retrieval 0 (current code).
- `20261030-*-match-hero-*`: older (judge without facts); harness 10, 10 and 8 gold.
- `20261030-133448-trays-*`: WANDS, both modes 9 of 19 gold in top 10; harness 4× the cost.

The `20261029-*-hero-*` runs are older, from the Flash planner.

**Known risk: live time.** A harness run takes 30–60 s with Argon, and the viewer shows "⚠ Past
60 s. Switch to replay?" in a live column that runs long. A planner call that times out twice in a
row ends the harness run with `APITimeoutError` (2 of 60 MuSiQue runs in the eval). One planner
call waits at most about 90 s (45 s, one retry). For the WANDS match hero, expect a weak run (2/10
once in 5); keep the replay ready.
