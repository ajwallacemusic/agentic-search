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
