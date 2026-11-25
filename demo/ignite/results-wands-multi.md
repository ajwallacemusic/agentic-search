# Public eval: wands-multi

WANDS multi: our own shopping tasks over the WANDS catalogue (Wayfair, MIT), built to test the swarm on questions whose parts do not depend on each other. Each question joins 3 or 4 real WANDS shopper queries from different product classes (picked once by a seeded rule before any model ran, `wands_multi.yaml`). Gold for a part is that query's human-labelled Exact products. Part coverage@10 = share of parts with at least one Exact product in the top 10 (headline). Part recall@30 = per part, Exact products found in the top 30 out of at most 30 / parts, averaged over parts. Pass mark, set before the run: the swarm works if its part coverage beats the harness by at least 0.10 and its part recall is not lower.

Sample: 20 queries, 42994 documents in one source (`products`). Settings: `questions=wands_multi.yaml`.

Source files (SHA-256): `product.csv` d993926254572e6eba96c8fd87cc549a17fb91ad3748308036eee4cf92b10ac6, `label.csv` c11fe81ad62f17f56f316b0ec9630ebe8fbe1393578cb0ca4f05c17253a180ef, `wands_multi.yaml` 74293d7da5c77d074981b57daf82390c327a962c451380fef63289730fbc836f

## Part scores

| mode | part coverage@10 | part recall@30 | nDCG@10 | avg cost | avg seconds | avg tool calls | n | errors |
|---|---|---|---|---|---|---|---|---|
| retrieval | 0.600 | 0.582 | 0.871 | $0.0100 | 5.4 | 5.9 | 20 | 0 |
| harness | 0.600 | 0.602 | 0.873 | $0.0101 | 5.4 | 5.9 | 20 | 0 |
| harness-flashlite | 0.625 | 0.573 | 0.845 | $0.0261 | 21.4 | 10.5 | 20 | 0 |
| swarm | 0.804 | 0.678 | 0.814 | $0.0331 | 8.6 | 17.1 | 20 | 0 |
| swarm-4x | 0.821 | 0.687 | 0.809 | $0.0338 | 9.5 | 18.4 | 20 | 0 |
| sid-swarm | 0.667 | 0.620 | 0.812 | $0.0615 | 13.6 | 42.3 | 20 | 0 |

## Verdict

The swarm works here: swarm part coverage@10 0.804 vs harness 0.600 (gap +0.204); part recall@30 0.678 vs 0.602; swarm-4x 0.821 / 0.687.

20 questions is a small set: treat the bar as rough.

## All questions

| mode | nDCG@10 | P@10 | recall@10 | MRR | avg turns | avg tool calls | avg cost | avg seconds | n | errors |
|---|---|---|---|---|---|---|---|---|---|---|
| retrieval | 0.871 | 0.935 | 0.011 | 1.000 | 1.0 | 5.9 | $0.0100 | 5.4 | 20 | 0 |
| harness | 0.873 | 0.940 | 0.011 | 0.950 | 1.0 | 5.9 | $0.0101 | 5.4 | 20 | 0 |
| harness-flashlite | 0.845 | 0.945 | 0.011 | 1.000 | 1.8 | 10.5 | $0.0261 | 21.4 | 20 | 0 |
| swarm | 0.814 | 0.920 | 0.011 | 0.950 | 1.4 | 17.1 | $0.0331 | 8.6 | 20 | 0 |
| swarm-4x | 0.809 | 0.925 | 0.011 | 0.950 | 1.4 | 18.4 | $0.0338 | 9.5 | 20 | 0 |
| sid-swarm | 0.812 | 0.915 | 0.011 | 0.942 | 0.0 | 42.3 | $0.0615 | 13.6 | 20 | 0 |

`bm25`: one BM25 search with the question text, no model. `retrieval` and `harness` use the same planner, judge and settings as the Ignite demo (`config.yaml`). `swarm`: one planner call splits the question; up to 4 harness agents run at the same time on the parts with the harness budget divided between them; RRF merges their lists.

`swarm` ran one query at a time (concurrency 1; its agents already run in parallel) while the other modes ran two at a time, so its seconds are not directly comparable.

`harness-flashlite` (eval-only judge control): the same harness, planner and budget with the earlier judge, gemini-3.5-flash-lite on Vertex, in place of Gemini Decision 1. It shows what the judge change does to scores, time and cost.

`swarm-4x` (eval-only control): the same swarm with 4x the tool calls and cost (96 calls, $2.00), 4 turns per agent, the same 120 s; also concurrency 1. It shows whether extra spend alone explains a swarm win.

`sid-swarm`: SID-2 Swarm (SID AI's trained multi-agent search model). SID's server runs up to 6 agents that fork and message each other; our harness runs their searches over the same documents and fuses their reports with RRF. No judge; its own budget (200 tool calls, $1.00, the same 120 s); concurrency 1.

One run per query. Model runs vary, so treat differences of a few points as noise.
