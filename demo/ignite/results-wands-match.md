# Public eval: wands-match

WANDS match: our own questions over the WANDS catalogue (Wayfair, MIT). Each names a product the shopper owns and asks for other product types that match it on wood tone, color or style. Questions and rules are ours (picked by a seeded rule before any model ran, `wands_match.yaml`); labels are built from the rules, not by a model: gold = a target product type with the anchor's value. Most questions have more than 10 right answers, so P@10 (share of the top 10 that is right) is the headline metric.

Sample: 20 queries, 42994 documents in one source (`products`). Settings: `questions=wands_match.yaml`, `labels=wands_match_labels.json`.

Source files (SHA-256): `product.csv` d993926254572e6eba96c8fd87cc549a17fb91ad3748308036eee4cf92b10ac6, `wands_match.yaml` 5eb81376ec150348c8c70e37db5d91c0d0debfc85300b1710a84b76b6c90f2f4, `wands_match_labels.json` 239aef4d5ba94f9c7a258d11e25a05bdbea0553d1d53d8110f517fd3ad4d2a98

## wood_tone questions

| mode | nDCG@10 | P@10 | recall@10 | MRR | avg turns | avg tool calls | avg cost | avg seconds | n | errors |
|---|---|---|---|---|---|---|---|---|---|---|
| bm25 | 0.032 | 0.043 | 0.002 | 0.040 | 1.0 | 1.0 | $0.0000 | 0.3 | 7 | 0 |
| retrieval | 0.000 | 0.000 | 0.000 | 0.000 | 1.0 | 3.3 | $0.0118 | 9.4 | 7 | 0 |
| harness | 0.687 | 0.700 | 0.205 | 0.671 | 3.7 | 17.4 | $0.0635 | 45.3 | 7 | 0 |
| swarm | 0.024 | 0.029 | 0.033 | 0.080 | 2.0 | 19.1 | $0.0682 | 28.8 | 7 | 0 |

## color questions

| mode | nDCG@10 | P@10 | recall@10 | MRR | avg turns | avg tool calls | avg cost | avg seconds | n | errors |
|---|---|---|---|---|---|---|---|---|---|---|
| bm25 | 0.000 | 0.000 | 0.000 | 0.000 | 1.0 | 1.0 | $0.0000 | 0.2 | 7 | 0 |
| retrieval | 0.058 | 0.029 | 0.036 | 0.214 | 1.0 | 4.0 | $0.0141 | 9.8 | 7 | 0 |
| harness | 0.447 | 0.371 | 0.396 | 0.652 | 4.0 | 19.9 | $0.0639 | 47.9 | 7 | 0 |
| swarm | 0.023 | 0.029 | 0.029 | 0.066 | 2.0 | 20.3 | $0.0765 | 32.9 | 7 | 0 |

## style questions

| mode | nDCG@10 | P@10 | recall@10 | MRR | avg turns | avg tool calls | avg cost | avg seconds | n | errors |
|---|---|---|---|---|---|---|---|---|---|---|
| bm25 | 0.000 | 0.000 | 0.000 | 0.022 | 1.0 | 1.0 | $0.0000 | 0.3 | 6 | 0 |
| retrieval | 0.000 | 0.000 | 0.000 | 0.003 | 1.0 | 3.2 | $0.0124 | 9.2 | 6 | 0 |
| harness | 0.484 | 0.467 | 0.253 | 0.685 | 4.0 | 18.0 | $0.0632 | 45.5 | 6 | 0 |
| swarm | 0.011 | 0.017 | 0.013 | 0.034 | 2.0 | 20.7 | $0.0781 | 31.7 | 6 | 0 |

## All questions

| mode | nDCG@10 | P@10 | recall@10 | MRR | avg turns | avg tool calls | avg cost | avg seconds | n | errors |
|---|---|---|---|---|---|---|---|---|---|---|
| bm25 | 0.011 | 0.015 | 0.001 | 0.021 | 1.0 | 1.0 | $0.0000 | 0.3 | 20 | 0 |
| retrieval | 0.020 | 0.010 | 0.013 | 0.076 | 1.0 | 3.5 | $0.0128 | 9.5 | 20 | 0 |
| harness | 0.542 | 0.515 | 0.286 | 0.669 | 3.9 | 18.4 | $0.0636 | 46.3 | 20 | 0 |
| swarm | 0.020 | 0.025 | 0.025 | 0.062 | 2.0 | 20.0 | $0.0741 | 31.1 | 20 | 0 |

`bm25`: one BM25 search with the question text, no model. `retrieval` and `harness` use the same planner, judge and settings as the Ignite demo (`config.yaml`). `swarm`: one planner call splits the question; up to 4 harness agents run at the same time on the parts with the harness budget divided between them; RRF merges their lists.

`swarm` ran one query at a time (concurrency 1; its agents already run in parallel) while the other modes ran two at a time, so its seconds are not directly comparable.

One run per query. Model runs vary, so treat differences of a few points as noise.
