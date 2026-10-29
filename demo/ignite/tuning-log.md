# Corpus and harness tuning log (hero question)

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

Round 6's live viewer check also failed twice on 3.8-flash planner timeouts with both modes running at once; that is why round 7 changed the planner.

## Single-hop check (round 6 settings, one run each; not re-run on round 7)

| Question | Retrieval | Harness |
|---|---|---|
| reranker-author | 1/1, 5 calls, $0.0045 | 1/1, 8 calls, $0.0078 |
| eu-slowdown | 1/1, 6 calls, $0.0059 | 1/1, 12 calls, $0.0105 |
| apple-pay | 1/1, 6 calls, $0.0044 | 1/1, 11 calls, $0.0084 |
| key-rotation | 1/1, 3 calls, $0.0037 | 1/1, 11 calls, $0.0086 |

Same answers; the harness costs about twice as much and takes a second turn to decide it is done.

## Backup runs kept in `runs/` (round 7 settings)

- `20261029-114713-hero-*`: harness 4/4 in 14.8 s, retrieval 1/4.
- `20261029-114741-hero-*`: harness 4/4 in 17.4 s, retrieval 2/4.
