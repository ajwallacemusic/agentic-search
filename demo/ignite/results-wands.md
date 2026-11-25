# Public eval: wands

WANDS (Wayfair, Chen et al., ECIR 2022, MIT): product search with human labels. Gold: Exact = 2, Partial = 1. Many queries have dozens of Exact products, so recall@10 stays low for every mode; nDCG@10 is the headline metric.

Sample: 60 queries, 42994 documents in one source (`products`). Settings: `n=60`, `seed=7`.

Source files (SHA-256): `product.csv` d993926254572e6eba96c8fd87cc549a17fb91ad3748308036eee4cf92b10ac6, `query.csv` 63b61660560fecc33ec490804c7e2b81402ee3e7c31a9cbb5e03736639f68e95, `label.csv` c11fe81ad62f17f56f316b0ec9630ebe8fbe1393578cb0ca4f05c17253a180ef

## All questions

| mode | nDCG@10 | P@10 | recall@10 | MRR | avg turns | avg tool calls | avg cost | avg seconds | n | errors |
|---|---|---|---|---|---|---|---|---|---|---|
| bm25 | 0.715 | 0.780 | 0.058 | 0.879 | 1.0 | 1.0 | $0.0000 | 0.1 | 60 | 0 |
| retrieval | 0.825 | 0.895 | 0.062 | 0.964 | 1.0 | 4.7 | $0.0082 | 4.3 | 60 | 0 |
| harness | 0.832 | 0.890 | 0.062 | 0.968 | 1.2 | 5.8 | $0.0100 | 5.5 | 60 | 0 |
| harness-flashlite | 0.806 | 0.878 | 0.063 | 0.952 | 1.4 | 6.5 | $0.0188 | 11.9 | 60 | 0 |
| swarm | 0.805 | 0.863 | 0.061 | 0.967 | 1.3 | 19.4 | $0.0390 | 9.2 | 60 | 0 |
| sid-swarm | 0.804 | 0.872 | 0.063 | 0.946 | 0.0 | 32.3 | $0.0533 | 11.3 | 60 | 0 |

`bm25`: one BM25 search with the question text, no model. `retrieval` and `harness` use the same planner, judge and settings as the Ignite demo (`config.yaml`). `swarm`: one planner call splits the question; up to 4 harness agents run at the same time on the parts with the harness budget divided between them; RRF merges their lists.

`swarm` ran one query at a time (concurrency 1; its agents already run in parallel) while the other modes ran two at a time, so its seconds are not directly comparable.

`harness-flashlite` (eval-only judge control): the same harness, planner and budget with the earlier judge, gemini-3.5-flash-lite on Vertex, in place of Gemini Decision 1. It shows what the judge change does to scores, time and cost.

`sid-swarm`: SID-2 Swarm (SID AI's trained multi-agent search model). SID's server runs up to 6 agents that fork and message each other; our harness runs their searches over the same documents and fuses their reports with RRF. No judge; its own budget (200 tool calls, $1.00, the same 120 s); concurrency 1.

One run per query. Model runs vary, so treat differences of a few points as noise.
