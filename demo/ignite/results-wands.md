# Public eval: wands

WANDS (Wayfair, Chen et al., ECIR 2022, MIT): product search with human labels. Gold: Exact = 2, Partial = 1. Many queries have dozens of Exact products, so recall@10 stays low for every mode; nDCG@10 is the headline metric.

Sample: 60 queries, 42994 documents in one source (`products`). Settings: `n=60`, `seed=7`.

Source files (SHA-256): `product.csv` d993926254572e6eba96c8fd87cc549a17fb91ad3748308036eee4cf92b10ac6, `query.csv` 63b61660560fecc33ec490804c7e2b81402ee3e7c31a9cbb5e03736639f68e95, `label.csv` c11fe81ad62f17f56f316b0ec9630ebe8fbe1393578cb0ca4f05c17253a180ef

## All questions

| mode | nDCG@10 | P@10 | recall@10 | MRR | avg turns | avg tool calls | avg cost | avg seconds | n | errors |
|---|---|---|---|---|---|---|---|---|---|---|
| bm25 | 0.715 | 0.780 | 0.058 | 0.879 | 1.0 | 1.0 | $0.0000 | 0.1 | 60 | 0 |
| retrieval | 0.787 | 0.842 | 0.059 | 0.943 | 1.0 | 5.9 | $0.0178 | 14.9 | 60 | 0 |
| harness | 0.788 | 0.855 | 0.061 | 0.933 | 1.5 | 8.8 | $0.0261 | 26.6 | 60 | 0 |

`bm25`: one BM25 search with the question text, no model. `retrieval` and `harness` use the same planner, judge and settings as the Ignite demo (`config.yaml`).

Harness rows: 2026-11-10, with the stop-when-done rule (tuning-log round 14); bm25 and retrieval rows: 2026-11-09 (the rule does not touch them: retrieval never asks the controller). One run per query. Model runs vary, so treat differences of a few points as noise.
