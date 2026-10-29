# Public eval: wands

WANDS (Wayfair, Chen et al., ECIR 2022, MIT): product search with human labels. Gold: Exact = 2, Partial = 1. Many queries have dozens of Exact products, so recall@10 stays low for every mode; nDCG@10 is the headline metric.

Sample: 60 queries, 42994 documents in one source (`products`). Settings: `n=60`, `seed=7`.

Source files (SHA-256): `product.csv` d993926254572e6eba96c8fd87cc549a17fb91ad3748308036eee4cf92b10ac6, `query.csv` 63b61660560fecc33ec490804c7e2b81402ee3e7c31a9cbb5e03736639f68e95, `label.csv` c11fe81ad62f17f56f316b0ec9630ebe8fbe1393578cb0ca4f05c17253a180ef

## All questions

| mode | nDCG@10 | recall@10 | MRR | avg turns | avg tool calls | avg cost | avg seconds | n | errors |
|---|---|---|---|---|---|---|---|---|---|
| bm25 | 0.715 | 0.058 | 0.879 | 1.0 | 1.0 | $0.0000 | 0.1 | 60 | 0 |
| retrieval | 0.796 | 0.062 | 0.958 | 1.0 | 4.2 | $0.0077 | 5.5 | 60 | 0 |
| harness | 0.776 | 0.062 | 0.917 | 3.3 | 14.7 | $0.0273 | 18.1 | 60 | 0 |

`bm25`: one BM25 search with the question text, no model. `retrieval` and `harness` use the same planner, judge and settings as the Ignite demo (`config.yaml`).

One run per query. Model runs vary, so treat differences of a few points as noise.

Label note: 14 (query, product) pairs in `label.csv` carry two labels. This run kept the file's last label; the sampler now keeps the higher grade. One pair is in this sample (query 425, product 40107: Exact and Partial), so at most one grade in one of 60 queries differs from a re-run.
