# Public eval: wands-match

WANDS match: our own questions over the WANDS catalogue (Wayfair, MIT). Each names a product the shopper owns and asks for other product types that match it on wood tone, color or style. Questions and rules are ours (picked by a seeded rule before any model ran, `wands_match.yaml`); labels are built from the rules, not by a model: gold = a target product type with the anchor's value. Most questions have more than 10 right answers, so P@10 (share of the top 10 that is right) is the headline metric.

Sample: 20 queries, 42994 documents in one source (`products`). Settings: `questions=wands_match.yaml`, `labels=wands_match_labels.json`.

Source files (SHA-256): `product.csv` d993926254572e6eba96c8fd87cc549a17fb91ad3748308036eee4cf92b10ac6, `wands_match.yaml` 5eb81376ec150348c8c70e37db5d91c0d0debfc85300b1710a84b76b6c90f2f4, `wands_match_labels.json` 239aef4d5ba94f9c7a258d11e25a05bdbea0553d1d53d8110f517fd3ad4d2a98

## wood_tone questions

| mode | nDCG@10 | P@10 | recall@10 | MRR | avg turns | avg tool calls | avg cost | avg seconds | n | errors |
|---|---|---|---|---|---|---|---|---|---|---|
| bm25 | 0.032 | 0.043 | 0.002 | 0.040 | 1.0 | 1.0 | $0.0000 | 0.3 | 7 | 0 |
| retrieval | 0.000 | 0.000 | 0.000 | 0.000 | 1.0 | 2.1 | $0.0063 | 3.4 | 7 | 0 |
| harness | 0.753 | 0.714 | 0.271 | 0.893 | 3.9 | 11.3 | $0.0313 | 15.7 | 7 | 0 |
| harness-flashlite | 0.866 | 0.814 | 0.329 | 0.929 | 4.0 | 11.0 | $0.0421 | 25.2 | 7 | 0 |
| swarm | 0.216 | 0.214 | 0.114 | 0.281 | 1.7 | 13.7 | $0.0298 | 9.5 | 7 | 0 |
| sid-swarm | 0.824 | 0.771 | 0.332 | 0.857 | 0.0 | 30.1 | $0.0571 | 14.0 | 7 | 0 |

## color questions

| mode | nDCG@10 | P@10 | recall@10 | MRR | avg turns | avg tool calls | avg cost | avg seconds | n | errors |
|---|---|---|---|---|---|---|---|---|---|---|
| bm25 | 0.000 | 0.000 | 0.000 | 0.000 | 1.0 | 1.0 | $0.0000 | 0.2 | 7 | 0 |
| retrieval | 0.000 | 0.000 | 0.000 | 0.000 | 1.0 | 2.4 | $0.0064 | 2.9 | 7 | 0 |
| harness | 0.567 | 0.514 | 0.449 | 0.627 | 3.9 | 11.6 | $0.0298 | 14.1 | 7 | 0 |
| harness-flashlite | 0.369 | 0.329 | 0.295 | 0.579 | 4.0 | 13.7 | $0.0459 | 28.8 | 7 | 0 |
| swarm | 0.195 | 0.157 | 0.142 | 0.495 | 1.4 | 12.0 | $0.0278 | 7.7 | 7 | 0 |
| sid-swarm | 0.550 | 0.414 | 0.433 | 0.929 | 0.0 | 33.1 | $0.0611 | 12.7 | 7 | 0 |

## style questions

| mode | nDCG@10 | P@10 | recall@10 | MRR | avg turns | avg tool calls | avg cost | avg seconds | n | errors |
|---|---|---|---|---|---|---|---|---|---|---|
| bm25 | 0.000 | 0.000 | 0.000 | 0.022 | 1.0 | 1.0 | $0.0000 | 0.3 | 6 | 0 |
| retrieval | 0.000 | 0.000 | 0.000 | 0.000 | 1.0 | 2.2 | $0.0060 | 2.9 | 6 | 0 |
| harness | 0.687 | 0.633 | 0.257 | 0.852 | 4.0 | 12.3 | $0.0317 | 38.3 | 6 | 0 |
| harness-flashlite | 0.529 | 0.450 | 0.251 | 0.778 | 4.0 | 11.0 | $0.0398 | 26.4 | 6 | 0 |
| swarm | 0.026 | 0.033 | 0.026 | 0.069 | 1.8 | 14.7 | $0.0334 | 9.9 | 6 | 0 |
| sid-swarm | 0.531 | 0.517 | 0.284 | 0.576 | 0.0 | 36.8 | $0.0616 | 14.5 | 6 | 0 |

## All questions

| mode | nDCG@10 | P@10 | recall@10 | MRR | avg turns | avg tool calls | avg cost | avg seconds | n | errors |
|---|---|---|---|---|---|---|---|---|---|---|
| bm25 | 0.011 | 0.015 | 0.001 | 0.021 | 1.0 | 1.0 | $0.0000 | 0.3 | 20 | 0 |
| retrieval | 0.000 | 0.000 | 0.000 | 0.000 | 1.0 | 2.2 | $0.0062 | 3.1 | 20 | 0 |
| harness | 0.668 | 0.620 | 0.329 | 0.788 | 3.9 | 11.7 | $0.0309 | 21.9 | 20 | 0 |
| harness-flashlite | 0.591 | 0.535 | 0.294 | 0.761 | 4.0 | 11.9 | $0.0427 | 26.8 | 20 | 0 |
| swarm | 0.152 | 0.140 | 0.097 | 0.292 | 1.6 | 13.4 | $0.0302 | 9.0 | 20 | 0 |
| sid-swarm | 0.640 | 0.570 | 0.353 | 0.798 | 0.0 | 33.2 | $0.0598 | 13.7 | 20 | 0 |

`bm25`: one BM25 search with the question text, no model. `retrieval` and `harness` use the same planner, judge and settings as the Ignite demo (`config.yaml`). `swarm`: one planner call splits the question; up to 4 harness agents run at the same time on the parts with the harness budget divided between them; RRF merges their lists.

`swarm` ran one query at a time (concurrency 1; its agents already run in parallel) while the other modes ran two at a time, so its seconds are not directly comparable.

`harness-flashlite` (eval-only judge control): the same harness, planner and budget with the earlier judge, gemini-3.5-flash-lite on Vertex, in place of Gemini Decision 1. It shows what the judge change does to scores, time and cost.

`sid-swarm`: SID-2 Swarm (SID AI's trained multi-agent search model). SID's server runs up to 6 agents that fork and message each other; our harness runs their searches over the same documents and fuses their reports with RRF. No judge; its own budget (200 tool calls, $1.00, the same 120 s); concurrency 1.

One run per query. Model runs vary, so treat differences of a few points as noise.
