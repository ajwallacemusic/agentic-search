# Public eval: wands-multi

WANDS multi: our own shopping tasks over the WANDS catalogue (Wayfair, MIT), built to test the swarm on questions whose parts do not depend on each other. Each question joins 3 or 4 real WANDS shopper queries from different product classes (picked once by a seeded rule before any model ran, `wands_multi.yaml`). Gold for a part is that query's human-labelled Exact products. Part coverage@10 = share of parts with at least one Exact product in the top 10 (headline). Part recall@30 = per part, Exact products found in the top 30 out of at most 30 / parts, averaged over parts. Pass mark, set before the run: the swarm works if its part coverage beats the harness by at least 0.10 and its part recall is not lower.

Sample: 20 queries, 42994 documents in one source (`products`). Settings: `questions=wands_multi.yaml`.

Source files (SHA-256): `product.csv` d993926254572e6eba96c8fd87cc549a17fb91ad3748308036eee4cf92b10ac6, `label.csv` c11fe81ad62f17f56f316b0ec9630ebe8fbe1393578cb0ca4f05c17253a180ef, `wands_multi.yaml` 74293d7da5c77d074981b57daf82390c327a962c451380fef63289730fbc836f

## Part scores

| mode | part coverage@10 | part recall@30 | nDCG@10 | avg cost | avg seconds | avg tool calls | n | errors |
|---|---|---|---|---|---|---|---|---|
| retrieval | 0.642 | 0.552 | 0.894 | $0.0189 | 17.2 | 6.0 | 20 | 0 |
| harness | 0.546 | 0.586 | 0.809 | $0.0266 | 26.2 | 8.4 | 20 | 0 |
| swarm | 0.800 | 0.642 | 0.756 | $0.0706 | 22.6 | 21.8 | 20 | 0 |
| swarm-4x | 0.863 | 0.646 | 0.767 | $0.0818 | 27.3 | 26.6 | 20 | 0 |

## Verdict

The swarm works here: swarm part coverage@10 0.800 vs harness 0.546 (gap +0.254); part recall@30 0.642 vs 0.586; swarm-4x 0.863 / 0.646.

20 questions is a small set: treat the bar as rough.

## All questions

| mode | nDCG@10 | P@10 | recall@10 | MRR | avg turns | avg tool calls | avg cost | avg seconds | n | errors |
|---|---|---|---|---|---|---|---|---|---|---|
| retrieval | 0.894 | 0.965 | 0.012 | 0.975 | 1.0 | 6.0 | $0.0189 | 17.2 | 20 | 0 |
| harness | 0.809 | 0.890 | 0.011 | 1.000 | 1.4 | 8.4 | $0.0266 | 26.2 | 20 | 0 |
| swarm | 0.756 | 0.885 | 0.011 | 0.942 | 1.4 | 21.8 | $0.0706 | 22.6 | 20 | 0 |
| swarm-4x | 0.767 | 0.875 | 0.010 | 0.892 | 1.9 | 26.6 | $0.0818 | 27.3 | 20 | 0 |

`bm25`: one BM25 search with the question text, no model. `retrieval` and `harness` use the same planner, judge and settings as the Ignite demo (`config.yaml`). `swarm`: one planner call splits the question; up to 4 harness agents run at the same time on the parts with the harness budget divided between them; RRF merges their lists.

`swarm` ran one query at a time (concurrency 1; its agents already run in parallel) while the other modes ran two at a time, so its seconds are not directly comparable.

`swarm-4x` (eval-only control): the same swarm with 4x the tool calls and cost (96 calls, $2.00), 4 turns per agent, the same 120 s; also concurrency 1. It shows whether extra spend alone explains a swarm win.

One run per query. Model runs vary, so treat differences of a few points as noise.

## Notes on the set

- Two questions hold near-duplicate WANDS queries with different ids: m01 "orren ellis l shape
  desk" and m10 "orren ellis l shaped desk". The rules allow it; the set was not re-rolled.
- m08 has 6 products that are Exact for two of its parts (Weather Instruments and Wall Clocks).
  They count for each part, as the design says.
- nDCG@10 and P@10 pool all parts' labels, so a mode that fills its top 10 from one large part
  scores well on them while it misses other parts. Part coverage is the headline for this set.
