# Public eval: wands-match

WANDS match: our own questions over the WANDS catalogue (Wayfair, MIT). Each names a product the shopper owns and asks for other product types that match it on wood tone, color or style. Questions and rules are ours (picked by a seeded rule before any model ran, `wands_match.yaml`); labels are built from the rules, not by a model: gold = a target product type with the anchor's value. Most questions have more than 10 right answers, so P@10 (share of the top 10 that is right) is the headline metric.

Sample: 20 queries, 42994 documents in one source (`products`). Settings: `questions=wands_match.yaml`, `labels=wands_match_labels.json`.

Source files (SHA-256): `product.csv` d993926254572e6eba96c8fd87cc549a17fb91ad3748308036eee4cf92b10ac6, `wands_match.yaml` 5eb81376ec150348c8c70e37db5d91c0d0debfc85300b1710a84b76b6c90f2f4, `wands_match_labels.json` 239aef4d5ba94f9c7a258d11e25a05bdbea0553d1d53d8110f517fd3ad4d2a98

## wood_tone questions

| mode | nDCG@10 | P@10 | recall@10 | MRR | avg turns | avg tool calls | avg cost | avg seconds | n | errors |
|---|---|---|---|---|---|---|---|---|---|---|
| bm25 | 0.032 | 0.043 | 0.002 | 0.040 | 1.0 | 1.0 | $0.0000 | 0.3 | 7 | 0 |
| retrieval | 0.143 | 0.086 | 0.105 | 0.214 | 1.0 | 3.7 | $0.0132 | 14.4 | 7 | 0 |
| harness | 0.695 | 0.714 | 0.226 | 0.650 | 3.6 | 17.0 | $0.0599 | 64.0 | 7 | 0 |

## color questions

| mode | nDCG@10 | P@10 | recall@10 | MRR | avg turns | avg tool calls | avg cost | avg seconds | n | errors |
|---|---|---|---|---|---|---|---|---|---|---|
| bm25 | 0.000 | 0.000 | 0.000 | 0.000 | 1.0 | 1.0 | $0.0000 | 0.3 | 7 | 0 |
| retrieval | 0.021 | 0.014 | 0.016 | 0.071 | 1.0 | 3.7 | $0.0139 | 9.7 | 7 | 0 |
| harness | 0.472 | 0.429 | 0.409 | 0.632 | 3.6 | 17.6 | $0.0581 | 57.1 | 7 | 0 |

## style questions

| mode | nDCG@10 | P@10 | recall@10 | MRR | avg turns | avg tool calls | avg cost | avg seconds | n | errors |
|---|---|---|---|---|---|---|---|---|---|---|
| bm25 | 0.000 | 0.000 | 0.000 | 0.022 | 1.0 | 1.0 | $0.0000 | 0.3 | 6 | 0 |
| retrieval | 0.048 | 0.050 | 0.051 | 0.067 | 1.0 | 4.0 | $0.0154 | 11.2 | 6 | 0 |
| harness | 0.453 | 0.417 | 0.259 | 0.562 | 3.3 | 14.8 | $0.0515 | 68.6 | 6 | 0 |

## All questions

| mode | nDCG@10 | P@10 | recall@10 | MRR | avg turns | avg tool calls | avg cost | avg seconds | n | errors |
|---|---|---|---|---|---|---|---|---|---|---|
| bm25 | 0.011 | 0.015 | 0.001 | 0.021 | 1.0 | 1.0 | $0.0000 | 0.3 | 20 | 0 |
| retrieval | 0.072 | 0.050 | 0.058 | 0.120 | 1.0 | 3.8 | $0.0141 | 11.8 | 20 | 0 |
| harness | 0.544 | 0.525 | 0.300 | 0.617 | 3.5 | 16.6 | $0.0567 | 63.0 | 20 | 0 |

`bm25`: one BM25 search with the question text, no model. `retrieval` and `harness` use the same planner, judge and settings as the Ignite demo (`config.yaml`).

Harness rows: 2026-11-10, with the stop-when-done rule (tuning-log round 14); bm25 and retrieval rows: 2026-11-09 (the rule does not touch them: retrieval never asks the controller). One run per query. Model runs vary, so treat differences of a few points as noise.
