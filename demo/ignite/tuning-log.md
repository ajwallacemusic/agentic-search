# Harness tuning log (hero question)

Rounds 1–8 ran on an earlier hand-written corpus (a made-up company, "Lumen Retail"). That corpus
was removed on 2026-10-29 in favour of the MuSiQue sample; it stays in git history. The model
and settings findings still hold: they are why `config.yaml` uses gemini-3.5-flash, a flash-lite
judge, `reasoning_effort: low` and 15 s timeouts.

Acceptance (spec §9): harness 4/4 gold in the top 10 in ≥4 of 5 runs, each under 45 s; retrieval
misses at least one gold doc in most runs.

| Round | Change | Retrieval gold@10 | Harness gold@10 | Harness time | Notes |
|---|---|---|---|---|---|
| 1 | first corpus, 3.8-flash for planning and judging | 4/4 | 4/4 | 63.6 s (time budget) | Corpus too easy: one "reranker" search per source found every hop. Judging took ~60 s. |
| 2 | 8 reranker decoy deploys and 4 decoy PRs; hero deploy and fix PR worded less obviously; settings `max_limit: 8`, `max_calls_per_turn: 6`, `judge_batch_size: 48` | 1/4, 2/4, 2/4, 3/4 | 3/4, 4/4, 3/4, 2/4 | 60–75 s | Retrieval now misses. Three tool errors per run: the model named the collection after the source, so collections are now named after their sources. Judge calls on 3.8-flash took 6–55 s. |
| 3 | `reasoning_effort: low`, 30 s request timeout | — | — | — | Judge probe: ~6 s typical, but stalls of 20–50 s remain (Vertex latency). |
| 4 | everything on gemini-3.5-flash-lite | 1/4 ×3 | 3/4, 3/4, 2/4 | 8–10 s | Fast, but flash-lite never filtered the deploy log by date or PR; it repeated "reranker". |
| 5 | planner 3.8-flash, judge 3.5-flash-lite | 3/4, 2/4, 2/4, 2/4, 2/4 | 4/4, 3/4, failed, 4/4, 3/4 | 20–67 s | The 3/4 runs found the *fix* deploy (DEP-0918-05), not the *cause*: PR-1203 did not name the PR that caused the regression. One planner call timed out 4× (fatal). |
| 6 | PR-1203 says "a regression from PR 1187" (new check rule); request timeout 15 s, 2 retries | 2/4, 2/4, 1/4, 2/4, 2/4 | 4/4, 4/4, 4/4, failed, 4/4 | 39.4, 42.9, 40.6, —, 22.3 s | **Passes.** The failure was three 15 s Vertex stalls in a row on one planner call: `APITimeoutError`. That is what replay is for. |
| 7 | planner gemini-3.5-flash (judge unchanged). A latency probe of 8 parallel planning calls: 3.8-flash 4–60 s, 3.7-flash 3 s–timeout, 3.5-flash 2.7–3.9 s | 3/4, 2/4, 1/4, 2/4, 2/4 | 3/4, 4/4, 4/4, 4/4, 4/4 | 16.1, 22.3, 14.8, 17.4, 14.8 s | **Passes, and no stalls.** Costs more per run (~$0.027 vs ~$0.015) because 3.5-flash is priced higher than the 3.8 introductory rate. |
| 8 | after review: source descriptions give the year (2026); 1 retry per call; `call_timeout` and `decider_timeout` 15 s | 1/4, 2/4, 2/4, 2/4, 2/4 | 4/4, 2/4, 4/4, 4/4, 4/4 | 13.5, 34.0, 15.5, 16.0, 16.2 s | **Passes.** Round 7's backup had filtered incidents by 2020 dates; round 8's date filters all use 2026. |

Round 6's live viewer check also failed twice on 3.8-flash planner timeouts with both modes running at once; that is why round 7 changed the planner.

## Single-hop check (round 6 settings, one run each; not re-run on round 7)

| Question | Retrieval | Harness |
|---|---|---|
| reranker-author | 1/1, 5 calls, $0.0045 | 1/1, 8 calls, $0.0078 |
| eu-slowdown | 1/1, 6 calls, $0.0059 | 1/1, 12 calls, $0.0105 |
| apple-pay | 1/1, 6 calls, $0.0044 | 1/1, 11 calls, $0.0084 |
| key-rotation | 1/1, 3 calls, $0.0037 | 1/1, 11 calls, $0.0086 |

Same answers; the harness costs about twice as much and takes a second turn to decide it is done.

## Backup runs kept in `runs/` (round 8 settings)

- `20261029-115838-hero-*`: harness 4/4 in 13.5 s, retrieval 1/4.
- `20261029-120008-hero-*`: harness 4/4 in 15.5 s, retrieval 2/4.

## Round 9: MuSiQue hero (2026-10-29)

Same models and settings as round 8. Scan: each of the 20 sampled 4-hop questions once per mode.
Picked `4hop1__777217_32392_823060_610794` (Andrew Deveaux → South Carolina → Columbia → Forest
Acres → Richland County): retrieval 1/4, harness 4/4 in the scan.

| Question | Retrieval gold@10 | Harness gold@10 | Harness time | Cost per run (retrieval / harness) |
|---|---|---|---|---|
| hero | 1/4 ×5 | 3/4, 4/4, 4/4, 4/4, 4/4 | 13.6–18.5 s | ~$0.005 / ~$0.019 |
| wluj (spare 4-hop) | 2/4 ×4, 3/4 | 4/4, 3/4, 4/4, 4/4, 3/4 | 11.1–12.9 s | ~$0.006 / ~$0.014 |

**Passes** spec §9: harness 4/4 in 4 of 5 runs, all under 45 s; retrieval misses in 5 of 5.

## Round 10: step tracking and Gemini 4 Argon (2026-10-30)

Changes: library step tracking on (`settings.steps: true`, branch `feat/step-tracking`); planner
`gemini-3.5-flash` → `gemini-4-argon` (`reasoning_effort: low`, cut at 45 s, 1 retry); all runs in
project `ajw-sandbox-497707` (`gen-ai-accel-dev` refuses Argon with a 403 terms-of-use error).
Judge unchanged.

WANDS match, harness P@10 (20 questions, one run each):

| Planner | Steps | P@10 | Note |
|---|---|---|---|
| 3.5 Flash | off | 0.050 | stepping stone judged not relevant; loop loses its place |
| 3.5 Flash | on | 0.055 | Flash rarely writes facts to the notebook |
| Argon, cut at 15 s | on | 0.180 | 3 time-outs |
| Argon, cut at 45 s | on | 0.260 | no errors (scratch run) |
| Argon, cut at 45 s | on | 0.230 | committed `results-wands-match.md`; retrieval 0.020 |

Full re-run with Argon (committed results files), harness vs retrieval:

| Set | Metric | Retrieval | Harness | Flash harness (round 9 settings + steps) |
|---|---|---|---|---|
| MuSiQue (60) | recall@10 | 0.715 | 0.886 | 0.828 |
| WANDS (60) | nDCG@10 | 0.772 | 0.803 | 0.794 |
| WANDS match (20) | P@10 | 0.020 | 0.230 | 0.055 |

Harness cost per question with Argon: about $0.035 (MuSiQue), $0.050 (WANDS), $0.052 (WANDS
match). Time: about 39–40 s on average. Step-tracking spec §6 goal for WANDS match was 0.30:
**not met** (0.230). MuSiQue and WANDS quality goals are met; the WANDS cost bound (within 10% of
$0.0273) is not, because Argon costs more per token than Flash.

Picks, from the per-question scores of that re-run:

- `match-hero` = `match-19` (P@10 harness 0.9, retrieval 0.0). Next best: `match-04` (0.8 vs 0.0).
- `awning` = WANDS 260 (tie at nDCG 1.0; harness ~4× cost).
- `trays` = WANDS 346 (harness 0.93 vs 0.96; ~4.5× cost).

Rehearsals:

| Question | Retrieval gold@10 | Harness gold@10 | Harness time | Cost per run (retrieval / harness) |
|---|---|---|---|---|
| match-hero (5 runs) | 0/40 ×5 | 10, 8, 2, 7, 10 (of 40) | 34.9–60.3 s | ~$0.010 / ~$0.052 |
| hero (3 runs) | 1/4 ×3 | 4/4 ×3 | 31.5–34.9 s | ~$0.010 / ~$0.043 |
| trays (1 run) | 9/19 | 9/19 | 44.4 s | $0.013 / $0.054 |

Kept as backups: match-hero runs 1, 2 and 5; all 3 hero runs; trays. Runs 3 and 4 of match-hero
were deleted. Replay checked in a headless browser for one run per set: no page errors, steps and
facts show.

## Round 11: WANDS match to 0.515 (2026-11-09)

Goal: WANDS match harness P@10 at least 0.30 (spec §6; round 10 had 0.230). Traces showed the
harness running the right search (`style : modern & contemporary` in class Reception Benches)
and the judge marking every hit not relevant: it saw only "same style as the Tenafly end table",
not the table's style, which the planner had noted.

| Test | Change (cumulative) | Harness P@10 | Note |
|---|---|---|---|
| 0 | round 10 | 0.230 | |
| 1 | `max_limit` 8 → 20 | 0.260 | the model asked for 50–100 hits per call and got 8 |
| 2 | source description gives the `key : value` feature format | 0.260 | wood tone 0.53, color and style down; runs hit the 60 s budget |
| 3 | judge gets the noted facts with the question (library, `roles/analyzer.py`); WANDS documents put features before the description (17% of gold had the attribute past the judge's 1200-char snippet) | 0.495 | scratch run |
| 4 | same, all modes | **0.515** | committed `results-wands-match.md`; retrieval 0.050 |

Re-run of the other sets with the same code (committed results files), retrieval vs harness:
MuSiQue recall@10 0.697 vs 0.875 (round 10: 0.715 vs 0.886); WANDS nDCG@10 0.787 vs 0.766
(round 10: 0.772 vs 0.803; the WANDS documents changed order, and the gap is within noise both
times). Harness cost per question: $0.043 (MuSiQue), $0.055 (WANDS), $0.064 (WANDS match). Harness
time: 42–54 s on average, up from 39–40 s; WANDS match is close to the 60 s budget.

Rehearsals, `match-hero`: harness 10/10 gold in 3 of 3 runs (46.6, 45.6, 46.8 s; $0.048–0.068),
retrieval 0/10 in all 3. Saved as `20261109-121547-*`, `20261109-121658-*`, `20261109-121804-*`.


## Round 12: the viewer's answer box (2026-11-09)

The viewer shows an answer read from each mode's top 10 (`answer.py`, `POST /ignite/answer`). A
live hero run showed "not found" for a harness run that had all 4 gold paragraphs. Cause: the
reader, not the search. MuSiQue states two hero hops loosely: Deveaux was "from" South Carolina
(not "born in"), and Forest Acres is "part of the Columbia metro area" (not "shares a border with
Columbia"). A strict reader rejects the chain.

Answer step on saved hero runs (each harness run had 4/4 gold), 3 tries per run:

| Reader | Prompt | Harness right | Retrieval (1/4 gold) |
|---|---|---|---|
| 3.5 Flash-Lite (judge) | answer in one sentence | 9 of 12 runs-tries (2 runs mostly "not found") | not found |
| 3.5 Flash-Lite | steps, then `Answer:` | 12 of 18 (some replies cut before `Answer:`) | not found |
| 3.5 Flash | steps, then `Answer:` | 16 of 18 | not found |
| Argon | steps, then `Answer:` | 4 of 18 ("no document says it shares a border") | not found |
| 3.5 Flash | steps; accept loosely stated hops | **18 of 18** | not found |

`match-hero`: the anchor chair is rarely in the harness's top 10, so the reader could not check
its wood tone and said "not found". The viewer now also sends the mode's noted facts (the
harness notebook; retrieval has none). Result: harness names 3 matching products in 9 of 9 tries;
retrieval "not found" in 9 of 9. Hero with notes: harness 18 of 18.

Config: `answer` block (gemini-3.5-flash, ~4 s per answer). Not used by search or the evals.

## Round 13: more match examples (2026-11-09)

Scan of all 20 WANDS match rules, one run each, P@10 retrieval / harness: 01 0.0/0.3, 02 0.0/0.4,
03 0.0/0.2, 04 0.0/1.0 (harness 704 s: an API stall), 05 0.4/0.0, 06 0.3/0.8, 07 0.0/1.0,
08 0.5/0.7, 09 0.0/0.0, 10 0.0/1.0, 11 0.0/0.0, 12 0.0/0.5, 13 0.0/1.0, 14 0.0/0.5, 15 0.0/0.5,
16 0.1/0.9, 17 0.0/1.0, 18 0.0/0.4, 19 0.0/0.8, 20 0.0/0.0.

Added one per attribute: `match-bookcase` (07, wood tone), `match-vase` (17, color),
`match-faucet` (06, style). Rehearsals, 2 runs each, gold in top 10: harness 10 and 8, 8 and 7,
9 and 8; retrieval 0 in all 6. Harness 38–51 s. Kept as backups.


## Round 14: stop when done (2026-11-10)

Problem: on simple questions the harness ran to the 4-turn cap. WANDS: 3.2 turns, 47 s, $0.055
per question, for the same P@10 as one pass (0.845 vs 0.842). The heuristic controller had no
rule for "nothing left to do, results good": with step tracking on it kept going while results
changed.

Method (no live tuning): run the harness once with a raised cap (6 turns, 100 s) on all 140
public questions and save every event (`public/turnscan.py`). A later turn never changes an
earlier one, so the top 10 after each turn is exactly what any stop rule would have returned.
`public/turnsim.py` replays candidate rules on those runs, on all questions and on two fixed
halves.

Simulated, 4-turn cap (P@10 / avg turns / avg s):

| Rule | MuSiQue | WANDS | WANDS match |
|---|---|---|---|
| none (today) | 0.262 / 3.33 / 42.7 | 0.850 / 3.36 / 54.5 | 0.521 / 4.00 / 55.4 |
| top 10 unchanged | 0.265 / 3.48* | 0.850 / 3.19 / 51.9 | 0.495* |
| ≥ 8 of top 10 judged relevant (ignore steps) | 0.262 / 3.30 | 0.857 / 1.47 / 23.7 | 0.505 / 2.95 |
| **no step open + (≥ 8 judged relevant or no gain)** | **0.262 / 3.33** | **0.859 / 1.41 / 23.0** | **0.521 / 3.95** |

(* 6-turn cap.) The step gate matters: without it the rule stops match questions before their
second step (P@10 0.505). The simple WANDS questions never set a plan (60 of 60), so "no step
open" is true from turn 1. Both halves agree for the chosen rule. Raising the cap to 6 helps only
WANDS match (P@10 0.521 → 0.563, at 87 s); not taken.

Rule, in the library (`Controller._done_stop`; settings `stop_when_done`, `done_min_relevant`
= 8; only with `steps: true`): after a turn, stop if no step is open and either at least 8 of the
top 10 are judged relevant or the turn added no judged-relevant result to the top 10.

Live re-run with the rule (harness rows of the results files):

| Set | Before | After |
|---|---|---|
| WANDS | nDCG 0.766, P@10 0.845, 3.2 turns, 46.9 s, $0.055 | nDCG 0.788, P@10 0.855, 1.5 turns, 26.6 s, $0.026 |
| WANDS match | P@10 0.515, 3.9 turns, 54.3 s, $0.064 | P@10 0.525, 3.5 turns, 63.0 s, $0.057 |
| MuSiQue | recall 0.875, 3.3 turns, 42.5 s | recall 0.865, 3.2 turns, 45.2 s |

A first live run of all three sets at once scored lower (WANDS P@10 0.818, match 0.490): the
shared model was slower, match runs hit the 60 s budget, and retrieval dropped as well (MuSiQue
recall 0.697 → 0.661). WANDS and match were re-run one at a time (rows above). MuSiQue's row is
from the shared run; the rule almost never fires there (simulation: same turns), so its change is
noise.

## Viewer: third lane for the swarm (2026-11-19)

The viewer now has three columns. At 1920×1080 all three fit side by side (each about 610 px
wide); no column is stacked. Below 1500 px wide the swarm column goes under the other two, and
below 1100 px all three stack. Checked on a synthetic swarm replay; the real hero replays come
in the live rehearsal.

## Round 15: swarm mode, 120 s budget for every mode (2026-11-19)

**What changed.**

- Time budget 60 s → 120 s for **all** modes (retrieval, harness, swarm; `budget.max_seconds`
  in `config.yaml`), so no mode is cut short where another is not. The viewer's stall line moved
  to 120 s ("⚠ Past 120 s").
- New mode **swarm** (`swarm.py`): one planner call splits the question into up to 4 parts, one
  harness agent per part runs at the same time (2 turns each), RRF (k = 60) merges their lists.
- **Fair-spend rule:** the whole swarm gets one harness budget (24 tool calls, $0.50, 120 s).
  The split call's cost comes off the top; each agent gets an even share of the rest. This is a
  **cap**, not equal spend. The single harness usually stops early (stop-when-done, round 14),
  while swarm agents use close to their whole share, so the swarm spends about 20–24 tool calls
  and $0.07–0.08 per question against the harness's 8–18 calls and $0.02–0.06.

**Evals**, all four modes, run one at a time (not in parallel; round 14 showed shared runs score
lower), 120 s budget. The new numbers replace the 60 s numbers in the results files.

| Set (metric) | bm25 | retrieval | harness | swarm |
|---|---|---|---|---|
| MuSiQue (recall@10) | 0.468 | 0.757 | **0.885** | 0.749 |
| WANDS (nDCG@10) | 0.715 | 0.775 | **0.797** | 0.758 |
| WANDS match (P@10) | 0.015 | 0.010 | **0.515** | 0.025 |

Cost and time per question, retrieval / harness / swarm:

| Set | Cost | Seconds | Tool calls |
|---|---|---|---|
| MuSiQue | $0.014 / $0.042 / $0.080 | 15 / 40 / 33 | 4.6 / 15.3 / 23.8 |
| WANDS | $0.018 / $0.024 / $0.074 | 14 / 21 / 32 | 5.9 / 8.1 / 22.1 |
| WANDS match | $0.013 / $0.064 / $0.074 | 10 / 46 / 31 | 3.5 / 18.4 / 20.0 |

Cost of the eval runs (avg cost × n, all modes): MuSiQue about $8.1, WANDS about $6.9, WANDS
match about $3.0; about $18 in all, of which the swarm is about $10.7. No errors in any mode.

**The swarm lost on every set.** Plainly:

- **MuSiQue:** recall@10 0.749, below the harness (0.885) and level with one pass (0.757), at
  1.9× the harness's cost. By hops: 2-hop 0.825 (harness 0.925, retrieval 0.800), 3-hop 0.783
  (0.917, 0.883), 4-hop 0.637 (0.812, 0.588). Its MRR (0.885) is higher than the harness's
  (0.856) but below retrieval (0.941). It is faster than the harness (33 s vs 40 s) because its
  agents run at once and get only 2 turns.
- **WANDS:** nDCG@10 0.758, below both retrieval (0.775) and harness (0.797), at 3× the
  harness's cost and the slowest of the three (32 s). Plain product search has nothing to split;
  the parts are rewordings of the same query.
- **WANDS match:** P@10 0.025 (wood tone 0.029, color 0.029, style 0.017), no better than one
  pass (0.010) and far below the harness (0.515), at 1.2× the harness's cost. The split runs
  before anyone has read the anchor, so the agents for the target types do not know which wood
  tone, color or style to look for.

Why: our questions are chains. Hop 2 needs hop 1's answer; "same wood tone" needs the anchor's
wood tone first. A swarm runs the parts side by side, so a later part cannot use an earlier
part's answer. That is the talk point.

**The exception: the model already knows the chain.** Live smoke run on `hero` (swarm only):
4/4 gold, 25.9 s, $0.0898. The split already named the chain from the model's own memory: "Where
was Andrew Deveaux born?", "What is the state capital of South Carolina?", "Which cities share a
border with Columbia...". One rehearsal split even guessed a city ("What county is West Columbia,
South Carolina located in?"; West Columbia is not the gold city, Forest Acres is), and the
"cities that border Columbia" agent still found Forest Acres. So the swarm wins on `hero` because
the model knows the path, not because it searched for it. The eval shows this does not hold in
general: on MuSiQue's 4-hop questions the swarm's recall is 0.637.

**Rehearsal** (CLI, `run_side_by_side --question <id> --save`, all three modes, one run at a
time, 2× each; gold in top 10, seconds, cost):

| Question | Run | Retrieval | Harness | Swarm |
|---|---|---|---|---|
| `hero` | 1 | 1/4, 15.3 s, $0.0148 | 4/4, 41.1 s, $0.0503 | 4/4, 24.1 s, $0.0866 (4 agents) |
| `hero` | 2 | 1/4, 12.7 s, $0.0093 | 4/4, 41.6 s, $0.0480 | 4/4, 28.4 s, $0.0734 (3 agents) |
| `awning` | 1 | 10/26, 12.3 s, $0.0142 | 10/26, 12.9 s, $0.0147 | 10/26, 17.1 s, $0.0454 |
| `awning` | 2 | 10/26, 14.6 s, $0.0144 | 10/26, 12.0 s, $0.0154 | 10/26, 25.4 s, $0.0571 |
| `trays` | 1 | 9/19, 12.6 s, $0.0167 | 9/19, 12.7 s, $0.0154 | 9/19, 26.8 s, $0.0616 |
| `trays` | 2 | 9/19, 13.3 s, $0.0153 | 9/19, 15.3 s, $0.0152 | 9/19, 27.8 s, $0.0532 |
| `match-hero` | 1 | 0/40, 12.4 s, $0.0145 | 2/40, 45.7 s, $0.0688 | 3/40, 23.5 s, $0.0622 |
| `match-hero` | 2 | 0/40, 13.8 s, $0.0122 | 6/40, 52.4 s, $0.0610 | 5/40, 32.4 s, $0.0825 |

Every lane finished inside 120 s (longest 52.4 s); no swarm agent was cut; no errors. Saved as
`runs/20261119-2012*`–`20261119-2020*`. Each saved `-swarm.jsonl` was loaded through the
viewer's own replay functions (`assignFiles` → `parseJsonl` → `swarmTimeline` → `schedule` →
`reduceSwarm`, run in node like the viewer tests): all 9 (with the smoke run) parse every line,
play in order, end done, and give the same gold count as the CLI. The browser rehearsal (live
and replay, at 1920×1080 on the presentation laptop) is still to do (AJ).

**Surprises.**

- **`match-hero` came in weak for the harness: 2 and 6 of 10** (it was 10 of 10 in 3 of 3 on
  2026-11-09). In both runs the planner fetched the chair on turn 1 but noted "dswoodtone :
  light wood" only on turn 2, leaving 2 of its 4 turns for step 2. The 120 s budget is not the
  limit here (46–52 s); the 4-turn cap is. The match-set average held (P@10 0.515, same as
  2026-11-09), so this looks like run-to-run variance on one question, but it is a stage risk:
  keep the 2026-11-09 replay ready.
- **The swarm scored 3 and 5 on `match-hero`, level with or above the harness in these runs.**
  Not a real win: its split told the sofa and table agents to "check their wood tone", so they
  return patio sofas and tables that list a wood tone. Of the 420 patio sofas and tables, 122
  list a wood tone and 40 are light wood, so about a third of such a list is right by chance
  (this explains `match-hero` only; the set-wide wood-tone swarm P@10 is 0.029).
  On the 20-question set the swarm's P@10 is 0.025.
- On `awning` and `trays` all three modes return the same top 10. The swarm costs 3–4× for it.

**Layout.** All three lanes side by side at 1920×1080 (each about 610 px); below 1500 px the swarm
lane goes under the other two, below 1100 px all three stack. Four explainer cards (retrieval,
harness, swarm, why), with the flow-diagram notes at 13 px in the CSS. Concurrency: the swarm eval rows ran one
question at a time (concurrency 1) while retrieval and harness ran two at a time, so the swarm's
seconds are not directly comparable; the results files say so. Rehearsal runs were one at a time.

## Round 16: does the swarm work when the parts are independent? (2026-11-23)

**Why.** Round 15's swarm lost on all three sets, and all three are chain questions. SID's report
says swarms win on wide "find every item" searches and gain little on chain sets (HotpotQA
+0.05). AJ: "I want to validate that swarm can work." Design:
`docs/superpowers/specs/2026-11-23-swarm-wide-recall-design.md`.

**What changed.**

- New eval set `wands-multi` (`public/wands_multi.py`, frozen in `wands_multi.yaml`, seed 7,
  built before any model ran): 20 shopping tasks, each joining 3 or 4 real WANDS queries from
  different product classes. Text is a fixed template ("I'm shopping for a few things: a; b; c.").
  Gold per part = that query's human Exact labels.
- New scores: part coverage@10 (share of parts with an Exact product in the top 10; headline) and
  part recall@30 (per part, capped so a huge part does not swamp a small one).
- New control mode `swarm-4x`: 4× tool calls and cost (96, $2.00), 4 turns per agent (AJ: with 2
  turns × 6 calls an agent can never use more than 12 calls), same 120 s. Eval only.
- Pass mark set before the run: the swarm works if its part coverage beats the harness by ≥ 0.10
  and its part recall is not lower.
- No change to the swarm algorithm, the viewer or the library.

**Result: the swarm works here.**

| mode | part coverage@10 | part recall@30 | nDCG@10 | cost | seconds | tool calls |
|---|---|---|---|---|---|---|
| retrieval | 0.642 | 0.552 | 0.894 | $0.019 | 17.2 | 6.0 |
| harness | 0.546 | 0.586 | 0.809 | $0.027 | 26.2 | 8.4 |
| swarm | **0.800** | 0.642 | 0.756 | $0.071 | 22.6 | 21.8 |
| swarm-4x | **0.863** | 0.646 | 0.767 | $0.082 | 27.3 | 26.6 |

- Gap +0.254 on part coverage at the fair budget, and higher part recall: the bar is met with
  room. It is not "only spend": the fair-budget swarm passes on its own. The 4× control adds a
  little (+0.063 coverage) at little extra real spend, because agents stop when done.
- The harness covers fewer parts than one plain pass (0.546 vs 0.642). Its loop goes deep on
  one or two parts and fills the top 10 with them; P@10 and nDCG stay high, but whole parts go
  missing. The swarm gives each part its own agent, so each part gets a share of the list.
- nDCG@10 is lowest for the swarm. It pools all parts' labels, so it rewards a top 10 drawn
  from one big part. That is why part coverage is the headline here (see the results file).
- Cost: the swarm costs 2.7× the harness per question. 20 questions is a small set; treat the
  numbers as rough. No errors in any mode.

**Spend:** full run about $3.96 (avg cost × 20, all four modes), smoke run (2 questions) about
$0.35.

**What it means.** The swarm is a tool for wide questions whose parts stand alone; the loop is
the tool for chains. Same models, same budget cap: the shape of the question picks the winner.
That is the talk point, and it matches SID's report. Next: part B (agents that talk) tests
whether a shared board can help the swarm on chains without losing this win.

**Set notes.** m01 and m10 hold near-duplicate desk queries (different WANDS ids); m08 has 6
products Exact for two parts. Neither favours a mode. Not re-rolled.
