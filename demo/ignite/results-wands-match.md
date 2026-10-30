# Public eval: wands-match

WANDS match: our own questions over the WANDS catalogue (Wayfair, MIT). Each names a product the shopper owns and asks for other product types that match it on wood tone, color or style. Questions and rules are ours (picked by a seeded rule before any model ran, `wands_match.yaml`); labels are built from the rules, not by a model: gold = a target product type with the anchor's value. Most questions have more than 10 right answers, so P@10 (share of the top 10 that is right) is the headline metric.

Sample: 20 queries, 42994 documents in one source (`products`). Settings: `questions=wands_match.yaml`, `labels=wands_match_labels.json`.

Source files (SHA-256): `product.csv` d993926254572e6eba96c8fd87cc549a17fb91ad3748308036eee4cf92b10ac6, `wands_match.yaml` 5eb81376ec150348c8c70e37db5d91c0d0debfc85300b1710a84b76b6c90f2f4, `wands_match_labels.json` 239aef4d5ba94f9c7a258d11e25a05bdbea0553d1d53d8110f517fd3ad4d2a98

## wood_tone questions

| mode | nDCG@10 | P@10 | recall@10 | MRR | avg turns | avg tool calls | avg cost | avg seconds | n | errors |
|---|---|---|---|---|---|---|---|---|---|---|
| bm25 | 0.032 | 0.043 | 0.002 | 0.040 | 1.0 | 1.0 | $0.0000 | 0.3 | 7 | 0 |
| retrieval | 0.031 | 0.014 | 0.001 | 0.143 | 1.0 | 2.6 | $0.0063 | 5.4 | 7 | 0 |
| harness | 0.093 | 0.086 | 0.034 | 0.183 | 2.9 | 9.4 | $0.0208 | 22.0 | 7 | 0 |

## color questions

| mode | nDCG@10 | P@10 | recall@10 | MRR | avg turns | avg tool calls | avg cost | avg seconds | n | errors |
|---|---|---|---|---|---|---|---|---|---|---|
| bm25 | 0.000 | 0.000 | 0.000 | 0.000 | 1.0 | 1.0 | $0.0000 | 0.2 | 7 | 0 |
| retrieval | 0.000 | 0.000 | 0.000 | 0.000 | 1.0 | 2.9 | $0.0061 | 8.2 | 7 | 0 |
| harness | 0.044 | 0.043 | 0.052 | 0.090 | 3.0 | 11.4 | $0.0221 | 20.6 | 7 | 0 |

## style questions

| mode | nDCG@10 | P@10 | recall@10 | MRR | avg turns | avg tool calls | avg cost | avg seconds | n | errors |
|---|---|---|---|---|---|---|---|---|---|---|
| bm25 | 0.000 | 0.000 | 0.000 | 0.022 | 1.0 | 1.0 | $0.0000 | 0.3 | 6 | 0 |
| retrieval | 0.000 | 0.000 | 0.000 | 0.000 | 1.0 | 3.7 | $0.0071 | 6.4 | 6 | 0 |
| harness | 0.011 | 0.017 | 0.013 | 0.042 | 3.0 | 10.3 | $0.0211 | 23.0 | 6 | 0 |

## All questions

| mode | nDCG@10 | P@10 | recall@10 | MRR | avg turns | avg tool calls | avg cost | avg seconds | n | errors |
|---|---|---|---|---|---|---|---|---|---|---|
| bm25 | 0.011 | 0.015 | 0.001 | 0.021 | 1.0 | 1.0 | $0.0000 | 0.3 | 20 | 0 |
| retrieval | 0.011 | 0.005 | 0.000 | 0.050 | 1.0 | 3.0 | $0.0065 | 6.7 | 20 | 0 |
| harness | 0.051 | 0.050 | 0.034 | 0.108 | 3.0 | 10.4 | $0.0213 | 21.8 | 20 | 0 |

`bm25`: one BM25 search with the question text, no model. `retrieval` and `harness` use the same planner, judge and settings as the Ignite demo (`config.yaml`).

One run per query. Model runs vary, so treat differences of a few points as noise.

