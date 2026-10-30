# Public eval: wands-match

WANDS match: our own questions over the WANDS catalogue (Wayfair, MIT). Each names a product the shopper owns and asks for other product types that match it on wood tone, color or style. Questions and rules are ours (picked by a seeded rule before any model ran, `wands_match.yaml`); labels are built from the rules, not by a model: gold = a target product type with the anchor's value. Most questions have more than 10 right answers, so P@10 (share of the top 10 that is right) is the headline metric.

Sample: 20 queries, 42994 documents in one source (`products`). Settings: `questions=wands_match.yaml`, `labels=wands_match_labels.json`.

Source files (SHA-256): `product.csv` d993926254572e6eba96c8fd87cc549a17fb91ad3748308036eee4cf92b10ac6, `wands_match.yaml` 5eb81376ec150348c8c70e37db5d91c0d0debfc85300b1710a84b76b6c90f2f4, `wands_match_labels.json` 239aef4d5ba94f9c7a258d11e25a05bdbea0553d1d53d8110f517fd3ad4d2a98

## wood_tone questions

| mode | nDCG@10 | P@10 | recall@10 | MRR | avg turns | avg tool calls | avg cost | avg seconds | n | errors |
|---|---|---|---|---|---|---|---|---|---|---|
| bm25 | 0.032 | 0.043 | 0.002 | 0.040 | 1.0 | 1.0 | $0.0000 | 0.3 | 7 | 0 |
| retrieval | 0.031 | 0.014 | 0.029 | 0.071 | 1.0 | 4.1 | $0.0107 | 9.2 | 7 | 0 |
| harness | 0.292 | 0.271 | 0.157 | 0.347 | 4.0 | 19.9 | $0.0549 | 40.1 | 7 | 0 |

## color questions

| mode | nDCG@10 | P@10 | recall@10 | MRR | avg turns | avg tool calls | avg cost | avg seconds | n | errors |
|---|---|---|---|---|---|---|---|---|---|---|
| bm25 | 0.000 | 0.000 | 0.000 | 0.000 | 1.0 | 1.0 | $0.0000 | 0.2 | 7 | 0 |
| retrieval | 0.000 | 0.000 | 0.000 | 0.000 | 1.0 | 4.0 | $0.0104 | 10.0 | 7 | 0 |
| harness | 0.153 | 0.157 | 0.156 | 0.185 | 3.9 | 18.7 | $0.0504 | 39.2 | 7 | 0 |

## style questions

| mode | nDCG@10 | P@10 | recall@10 | MRR | avg turns | avg tool calls | avg cost | avg seconds | n | errors |
|---|---|---|---|---|---|---|---|---|---|---|
| bm25 | 0.000 | 0.000 | 0.000 | 0.022 | 1.0 | 1.0 | $0.0000 | 0.3 | 6 | 0 |
| retrieval | 0.078 | 0.050 | 0.038 | 0.167 | 1.0 | 4.0 | $0.0109 | 13.5 | 6 | 0 |
| harness | 0.251 | 0.267 | 0.223 | 0.290 | 3.8 | 18.7 | $0.0512 | 41.9 | 6 | 0 |

## All questions

| mode | nDCG@10 | P@10 | recall@10 | MRR | avg turns | avg tool calls | avg cost | avg seconds | n | errors |
|---|---|---|---|---|---|---|---|---|---|---|
| bm25 | 0.011 | 0.015 | 0.001 | 0.021 | 1.0 | 1.0 | $0.0000 | 0.3 | 20 | 0 |
| retrieval | 0.034 | 0.020 | 0.022 | 0.075 | 1.0 | 4.0 | $0.0106 | 10.7 | 20 | 0 |
| harness | 0.231 | 0.230 | 0.177 | 0.273 | 3.9 | 19.1 | $0.0522 | 40.3 | 20 | 0 |

`bm25`: one BM25 search with the question text, no model. `retrieval` and `harness` use the same planner, judge and settings as the Ignite demo (`config.yaml`).

One run per query. Model runs vary, so treat differences of a few points as noise.
