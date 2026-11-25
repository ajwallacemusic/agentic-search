# Ignite demo: retrieval vs harness vs swarm, live

One question, one corpus, one pair of models, three modes. **Retrieval** plans one pass and ranks
what it finds. **Harness** runs the full loop: plan → query → judge → decide, and goes again when
it does not have enough. **Swarm** splits the question into parts with one planner call, runs one
harness agent per part at the same time (up to 4, the harness budget divided between them), and
merges their lists (RRF). The viewer shows all three side by side as they run: which tools the
agent uses on each turn and what it searches for, the step it is on, when the loop decides to
keep going, and the bill.

Talk: "Will the Real Agentic Search Please Stand Up?" (Egen Ignite). Specs:
`docs/superpowers/specs/2026-10-29-ignite-demo-design.md`,
`docs/superpowers/specs/2026-10-30-ignite-wands-design.md`,
`docs/superpowers/specs/2026-11-19-ignite-swarm-design.md`.

## The story in four parts

1. **MuSiQue: the loop wins on multi-hop questions.** Each hop needs a fact from the hop before
   it. One planned pass cannot know what to search for next.
2. **WANDS: the loop does not help plain product search.** Same models, same settings. Ranking
   is about the same (within noise; nDCG@10 0.797 vs 0.775) for about 1.4× the cost: the loop
   usually stops after one turn. Use the least agentic setting that works.
3. **WANDS match: the loop wins again when the question has steps.** Same WANDS catalogue, our own
   rule-labeled questions: "I have product X. Find a Y with the same wood tone." First find X and
   read its wood tone, then search for Y. One pass cannot do the second step.
4. **Swarm: splitting does not help when the work is a chain.** A swarm splits the work into
   parts that run side by side. Our questions are chains: hop 2 needs hop 1's answer, and "same
   wood tone" needs the anchor's wood tone read first. So the swarm loses to the single harness
   on all three sets, at a higher cost, and on WANDS match it is no better than one pass. The
   exception is when the model already knows the chain: on `hero` the split named the state
   capital and its neighbours from the model's own memory, and the swarm found 4 of 4.

Three question sets, two corpora, all in `questions.yaml`:

| set | corpus | questions | gold |
|---|---|---|---|
| `musique` | MuSiQue sample, 5,000 Wikipedia paragraphs (`wiki`) | MuSiQue ids | the supporting paragraphs (human labels) |
| `wands` | WANDS catalogue, 42,994 products (`products`) | WANDS query ids | "Exact" labels (human labels) |
| `wands-match` | the same WANDS catalogue | rule ids from `wands_match.yaml` | made by a rule from WANDS product features (ours) |

Be open about the picks: every demo question was picked from a full scan because it shows the
point clearly (`tuning-log.md`, rounds 9 and 10). The fair numbers are the averages in
`results-musique.md`, `results-wands.md` and `results-wands-match.md`.

Swarm check (`wands-multi`, eval only, not on stage): 20 shopping tasks whose parts do not depend
on each other, built from real WANDS queries and human labels (`wands_multi.yaml`). There the
swarm beats the harness on part coverage@10 (0.800 vs 0.546). See `results-wands-multi.md` and
`tuning-log.md`, round 16.

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

- `awning` (query 260, "retractable side awning"): a tie. Retrieval and harness both nDCG@10 1.0
  (2026-11-10). Since the stop-when-done rule (round 14) the harness stops after 1 turn: same
  cost as one pass. Rehearsal 2026-11-19, 2 runs: all three modes 10 of 26 gold in top 10 (the
  whole top 10); retrieval and harness about $0.015 and 12–15 s, swarm $0.045–0.057 and
  17–25 s (3–4× the cost for the same list).
- `trays` (query 346, "nesting tray set"): a tie at the same cost since round 14 (it was 4.5× at
  60 s before the rule). Rehearsal 2026-11-19, 2 runs: all three modes 9 of 19 gold in top 10;
  retrieval and harness about $0.015–0.017, 13–15 s; swarm $0.053–0.062, 27–28 s.

### WANDS match hero

`match-hero` (rule `match-19`):

> I have the Alaniz star single rocking chair. Find a patio sofa or patio table with the same wood
> tone.

The chair's features say `dswoodtone : light wood`. Gold: the 40 patio sofas and patio tables with
that wood tone. The question never says "light wood", so a single pass searches for the chair or
for "patio sofa" and finds no gold. The harness makes a 2-step plan, opens the chair, notes "wood
tone: light wood" in its notebook, then searches patio sofas and tables for it.

The swarm cannot do this: its split runs before anyone has read the chair, so the agents get
"find the chair's wood tone", "find patio sofas and their wood tones", "find patio tables and
their wood tones" at the same time. The sofa and table agents do not know which wood tone to look
for. They return patio pieces that list some wood tone, and about a third of those are light
wood (40 of the 122 patio sofas and tables that list one), so on `match-hero` the swarm lands 3–5 of
10 by chance. This explains that one question only; the set-wide wood-tone swarm P@10 is 0.029.

**Rehearsal warning (2026-11-19):** in both rehearsal runs the harness took until turn 2 to note
the wood tone, leaving 2 turns for step 2, and scored only 2 and 6 of 10 (swarm 3 and 5,
retrieval 0). On 2026-11-09 it was 10 of 10 in 3 of 3 runs. Across all 20 match questions the
harness still averages P@10 0.515 against the swarm's 0.025, so quote the average. Have the
2026-11-09 replay ready in case the live run comes in weak.

More match examples, one per attribute (picked from a scan of all 20 rules on 2026-11-09, then
rehearsed twice each; gold in top 10, retrieval / harness):

| id | rule | question | retrieval | harness |
|---|---|---|---|---|
| `match-bookcase` | `match-07` | Schatzi rainbow wave coffee table → bookcase, same wood tone | 0, 0 | 10, 8 |
| `match-vase` | `match-17` | Bruster metal rock climber decor → fountain or vase, same color | 0, 0 | 8, 7 |
| `match-faucet` | `match-06` | Nuala toilet paper holder → sink faucet and towel bar, same style | 0, 0 | 9, 8 |

Harness time 38–51 s (2026-11-09, 60 s budget, before the swarm; not re-run). The answer box
names matching products for the harness and says "not found" for retrieval.

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

## SID-2 Swarm (mode `sid-swarm`)

SID-2 Swarm is SID AI's trained multi-agent search model. Unlike our `swarm`, nothing of ours
plans or splits: SID's server runs up to 6 agents that fork and message each other. Our harness
runs the searches they ask for (`search` → `vector_search`, `text_search` → `lexical_search`,
`read` from the agent's cache) over the same documents, and fuses their reports with RRF, as
SID's reference implementation does (`sid_swarm.py`). It runs in the library's model mode, with
no judge and its own budget (`sid_swarm` in `config.yaml`). It is an eval and
`run_side_by_side` mode; it has no viewer lane yet.

```bash
uv sync --extra server --extra local --extra openai --extra gcp --extra sid   # sid-sdk, Python 3.12+
security add-generic-password -a "$USER" -s sid-api-key -w   # or export SID_API_KEY
uv run python -m demo.ignite.run_side_by_side --question hero --modes harness,sid-swarm
uv run python -m demo.ignite.public.eval_public --dataset musique --modes harness,sid-swarm
```

SID does not document which event carries token counts; the driver adds any `usage` it sees,
and if none arrives the run note says the cost reads 0.

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

Models and prices are in `config.yaml` (planner on Vertex, `global` endpoint, project
`ajw-sandbox-497707`; judge on the Gemini Developer API):

| role | model | price per 1M tokens (in / out) | per call |
|---|---|---|---|
| planner (driver) | `gemini-4-hydrogen` | $1.50 / $7.50 | `reasoning_effort: low`, cut at 45 s, 1 retry |
| judge | `gemini-decision-1` | $0.10 / not billed | one yes/no probability per hit, cut at 15 s, 1 retry |

All three modes use the same pair (since 2026-11-25, `tuning-log.md` round 17). Why Hydrogen: as the
planner on WANDS match it scored P@10 0.650 and 0.600 in two runs, against Argon 0.445, Carbon 0.520
and Neon 0.485, at about 30 s and $0.04 per question (Argon 66 s, $0.06). Why Decision 1: on 720
human-labelled WANDS pairs it was right 0.929 of the time against flash-lite's 0.900, and found more
of the right products, at half the cost.

**Decision 1 needs an AI Studio key.** On 2026-11-25 it is not on Vertex (404). Make a key at
aistudio.google.com (project `ajw-sandbox-497707`) and store it in the macOS Keychain, or set
`GEMINI_API_KEY`:

```bash
security add-generic-password -a "$USER" -s gemini-api-key -w   # paste the key twice
```

If AI Studio is down, copy the `judge_baseline:` block (flash-lite on Vertex) into `judge:` in `config.yaml`.
The eval numbers in `results-*.md` are still from Argon + flash-lite until the next full run.

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
step is open. Once no step is open (no plan, or every step done), it stops when at least 8 of
the top 10 are judged relevant or a turn adds no judged-relevant result (library setting
`stop_when_done`, tuning-log round 14). Simple WANDS questions now take 1.4 turns and about
21 s instead of 3.2 turns and 47 s, at the same quality. The viewer shows the steps and facts as they happen. The demo itself changes nothing under
`src/`.

The library fixes the demo once worked around are merged (library PRs #4, #5 and #6): the
sentence-transformers embedder is thread-safe, `vertex` clients take `reasoning_effort`,
`request_timeout_s` and `max_retries` from config, and `decide` events carry the controller's
note. The viewer still reads notes from the final trace as a fallback, for replays saved before
the fix.

### How the swarm is built

`swarm.py` (demo code, nothing under `src/`): one planner call splits the question into parts
(falls back to the whole question if the split fails), then one harness agent per part runs at
the same time, up to 4 (`swarm:` in `config.yaml`: `max_agents: 4`, `agent_max_turns: 2`,
`rrf_k: 60`). RRF merges their top lists. `serve.py` streams it at `/ignite/swarm`.

**Fair spend.** The whole swarm gets the same budget as one harness run (`budget` in
`config.yaml`: 24 tool calls, $0.50, 120 s). The split call's cost comes off the top; each agent
gets an even share of what is left (tool calls rounded down, at least 1) and 2 turns. The
whole run is cut at 120 s; agents still running are cancelled and their hits so far are kept.
This caps the swarm at one harness budget. It does **not** make actual spend equal: a single
harness usually stops early (stop-when-done), while each swarm agent tends to use its whole
share, so the swarm spends about 20–24 tool calls and about $0.07–0.08 per question against the
harness's 8–18 calls and $0.02–0.06.

## Commands

```bash
uv run python -m demo.ignite.run_side_by_side --question hero         # terminal trace + summary (all three modes)
uv run python -m demo.ignite.run_side_by_side --question match-hero   # WANDS match hero
uv run python -m demo.ignite.run_side_by_side --question hero --save  # also writes runs/<stamp>-hero-*.jsonl
uv run python -m demo.ignite.run_side_by_side --question hero --modes retrieval,harness  # skip the swarm
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

- **Live:** open `http://127.0.0.1:8765/`, pick a question (grouped by set), press **Run**. All three
  modes start at the same time. At 1920×1080 the three columns sit side by side (about 610 px
  each); below 1500 px the swarm column goes under the other two, below 1100 px all three stack.
- **Swarm column:** the split's parts, then one row per agent: its part, one dot per turn, hits
  so far and status (running, ✓ done, ✗ failed, ✂ cut at 120 s). Then the merged top 10.
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
- **Check it yourself** (`check.py`): once a mode finishes, a box under the question states the
  rule behind ★ (MuSiQue: the paragraphs its authors list as needed; WANDS: Wayfair raters' Exact
  label; WANDS match: product type plus one catalogue field, no model). Click any result to open
  its full text, with the deciding field highlighted, and the reason it counts or not. WANDS match
  also shows right results per target type, so "9 faucets, 0 towel bars" is visible. Needs `serve`.
- **Replay:** press **Switch to replay** (or open `viewer/index.html` straight from Finder; no
  server or internet needed). Pick the files of one saved run:
  `<stamp>-<id>-retrieval.jsonl`, `<stamp>-<id>-harness.jsonl`, `<stamp>-<id>-swarm.jsonl` and
  `<stamp>-<id>.meta.json`. Runs saved before 2026-11-19 have no swarm file; the swarm column
  then has nothing to play.
  Press **Play**. Speed 1×/2×/4×, **Pause**, **Replay again**. Gold docs get ★ gold in replay
  only.

## Numbers for the slides

All numbers: Argon planner, step tracking on, stop-when-done on, project `ajw-sandbox-497707`,
one run per question, seed 7, a 120 s budget for every mode, evals run one at a time,
2026-11-19 (`tuning-log.md` round 15).

| set | what it tests | sample | metric | bm25 | retrieval | harness | swarm |
|---|---|---|---|---|---|---|---|
| MuSiQue (multi-hop Wikipedia) | does the loop chase hops? | 60 questions (20 each 2/3/4-hop), 5,000 paragraphs | recall@10 | 0.468 | 0.757 | **0.885** | 0.749 |
| WANDS (Wayfair product search) | does the loop help plain ranking? | 60 queries, 42,994 products | nDCG@10 | 0.715 | 0.775 | 0.797 | 0.758 |
| WANDS match (ours) | does the loop help when a step needs a fact read on the way? | 20 questions, same catalogue | P@10 | 0.015 | 0.010 | **0.515** | 0.025 |

Average cost and time per question (retrieval / harness / swarm):

| set | cost | seconds | tool calls |
|---|---|---|---|
| MuSiQue | $0.014 / $0.042 / $0.080 | 15 / 40 / 33 | 4.6 / 15.3 / 23.8 |
| WANDS | $0.018 / $0.024 / $0.074 | 14 / 21 / 32 | 5.9 / 8.1 / 22.1 |
| WANDS match | $0.013 / $0.064 / $0.074 | 10 / 46 / 31 | 3.5 / 18.4 / 20.0 |

- **MuSiQue:** 4-hop recall@10 harness 0.812, retrieval 0.588, swarm 0.637. Harness about 3.1×
  the cost of retrieval.
- **WANDS:** harness +0.02 nDCG@10 over retrieval for about 1.4× the cost (1.4 turns, 21 s vs
  14 s). The sign has flipped between runs (-0.02 on 2026-11-09); call it a tie.
- **WANDS match:** harness P@10 by attribute: wood tone 0.700, color 0.371, style 0.467;
  retrieval 0.000, 0.029, 0.000. Harness about 5× the cost of retrieval. Quote the average, not
  only the hero.
- **Swarm: it loses everywhere.** Below the harness on all three sets, about level with one pass
  on MuSiQue and WANDS, and no better than one pass on WANDS match (P@10 0.025: wood tone 0.029,
  color 0.029, style 0.017). It costs the most: about 1.9× the harness on MuSiQue, 3× on WANDS,
  1.2× on WANDS match. The split cannot know what step 1 will find (the anchor's wood tone, the
  next hop's name), so the agents search for the wrong thing at the same time.
- The harness is slow: about 40–46 s per chain question, 21 s on plain WANDS, against about
  10–15 s for retrieval.

Hero runs (rehearsal 2026-11-19, 120 s budget, `tuning-log.md` round 15):

- MuSiQue `hero`, 2 runs: harness 4/4 gold both times, 41–42 s, about $0.049. Retrieval 1/4,
  about $0.009–0.015. Swarm 4/4 both times, 24–28 s, $0.073–0.087: it ties the harness on gold
  and is faster, as it is on MuSiQue overall and on `match-hero` (the swarm eval ran one question
  at a time and the others two, so the seconds are only roughly comparable). The split already
  names the chain from the model's own memory ("What is the
  state capital of South Carolina?", "Which cities border Columbia, South Carolina?"). Say so on
  stage: the swarm works here because the model knows the answer path, not because it searched
  for it.
- WANDS match `match-hero`, 2 runs: harness 2 and 6 of 10, 46–52 s, $0.061–0.069; swarm 3 and 5,
  24–32 s, $0.062–0.083; retrieval 0. Weaker than on 2026-11-09 (harness 10 of 10 in 3 of 3,
  46–47 s); see the warning above.

Say on the slide: all modes run with `max_limit: 20` and `max_calls_per_turn: 6`
(`config.yaml`) and a budget of 4 turns, 24 tool calls, $0.50 and 120 s. The swarm gets that same
budget in total, split between its agents (2 turns each); it is a cap, and the swarm spends more
of it than the harness does.

## Public datasets and re-runs

MuSiQue and WANDS have human labels; nothing in them was written or tuned by us. WANDS match
uses the WANDS catalogue with our rule labels.

```bash
uv run python -m demo.ignite.public.eval_public --dataset musique --modes bm25,retrieval,harness,harness-flashlite,swarm      # --n 20 per hop, --corpus-size 5000
uv run python -m demo.ignite.public.eval_public --dataset wands --modes bm25,retrieval,harness,harness-flashlite,swarm        # --n 60
uv run python -m demo.ignite.public.eval_public --dataset wands-match --modes bm25,retrieval,harness,harness-flashlite,swarm  # all 20 rules
```

`harness-flashlite` is an eval-only judge control: the same harness with the earlier judge
(gemini-3.5-flash-lite, config `judge_baseline`) in place of Gemini Decision 1.

Run them one at a time: run together they share the model and run slower, and hit the time
budget (tuning-log round 14). With all four modes the three runs cost about $8.10, $6.90 and
$3.00 (about $18 in all; the swarm is more than half of it).

Why MuSiQue and not HotpotQA: MuSiQue was built to drop questions a single-hop shortcut can answer,
a known weakness of HotpotQA, and each question ships with its own 20 paragraphs, so a sample is
self-contained.

Credits: MuSiQue, Trivedi et al., TACL 2022 (CC BY 4.0), github.com/StonyBrookNLP/musique.
WANDS, Chen et al., ECIR 2022 (MIT), github.com/wayfair/WANDS.

## Rehearsal checklist

1. Rehearse three times on the presentation laptop, on the venue network if you can.
2. Before the talk: `gcloud auth application-default login` (tokens expire), start `serve` (fills
   the embedding cache on the first start), run `hero` and `match-hero` once to warm up, then
   reload the page. Check all three lanes at 1920×1080 on the projector.
3. Keep a second tab open on `viewer/index.html` from disk with the backup runs loaded in replay.
4. Keep the three `results-*.md` files open in a third tab.

## Fallback

If a live run stalls past 120 seconds or shows **✗ Search failed**: press **Switch to replay**
(or go to the replay tab) and say so plainly: "the live API is slow, here is the same run from
this morning."

Backup runs in `runs/` (Argon):

- 2026-11-19, all three modes, 120 s budget (rehearsal; gold in top 10 retrieval / harness /
  swarm): `20261119-201201-hero-*` (1/4, 4/4, 4/4), `20261119-201330-hero-*` (1/4, 4/4, 4/4),
  `20261119-201503-awning-*` and `20261119-201558-awning-*` (10, 10, 10),
  `20261119-201703-trays-*` and `20261119-201808-trays-*` (9, 9, 9),
  `20261119-201917-match-hero-*` (0, 2, 3), `20261119-202051-match-hero-*` (0, 6, 5).
  `20261119-175800-hero-swarm.jsonl`: a swarm-only smoke run (4/4). For a strong match-hero
  replay use the 2026-11-09 runs below (no swarm file).

- `20261030-133739-hero-*`, `20261030-133833-hero-*`, `20261030-133927-hero-*`: MuSiQue hero,
  harness 4/4, retrieval 1/4.
- `20261109-132250-hero-*` (retrieval 2/4), `20261109-132357-hero-*` (retrieval 1/4): MuSiQue hero,
  harness 4/4, 60 s budget; decide events carry the controller's notes.
- `20261109-121547-match-hero-*`, `20261109-121658-match-hero-*`, `20261109-121804-match-hero-*`:
  harness 10 gold in top 10, retrieval 0 (60 s budget, no swarm file).
- `20261109-163827-*` and `20261109-1639*-*` (`match-bookcase`, `match-vase`, `match-faucet`):
  2 runs each, harness 7–10 gold in top 10, retrieval 0.
- `20261030-*-match-hero-*`: older (judge without facts); harness 10, 10 and 8 gold.
- `20261030-133448-trays-*`: WANDS, both modes 9 of 19 gold in top 10; harness 4× the cost.

The `20261029-*-hero-*` runs are older, from the Flash planner.

**Known risk: live time.** A harness run takes 30–60 s with Argon, a swarm run 25–40 s, and the
viewer shows "⚠ Past 120 s. Switch to replay?" in a live column that runs long. The swarm is cut
at 120 s by itself (agents still running show ✂ cut, their hits so far are kept). A planner call that times out twice in a
row ends the harness run with `APITimeoutError` (2 of 60 MuSiQue runs in an earlier eval;
none in the 2026-11-19 evals). One planner call waits at most about 90 s (45 s, one retry).
For the WANDS match hero, expect a weak run: 2/10 once in 5 before 2026-11-19, then 2/10 and
6/10 in the two 2026-11-19 rehearsal runs; keep the replay ready.
