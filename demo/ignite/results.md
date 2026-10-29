# Ignite eval results

## Single-hop questions

| mode | nDCG@10 | recall@10 | MRR | avg turns | avg tool calls | avg cost | avg seconds | n | errors |
|---|---|---|---|---|---|---|---|---|---|
| bm25 | 0.487 | 1.000 | 0.321 | 1.0 | 5.0 | $0.0000 | 0.0 | 4 | 0 |
| retrieval | 1.000 | 1.000 | 1.000 | 1.0 | 4.8 | $0.0083 | 5.0 | 4 | 0 |
| harness | 0.908 | 1.000 | 0.875 | 2.5 | 10.0 | $0.0193 | 11.3 | 4 | 0 |

## Multi-hop questions

| mode | nDCG@10 | recall@10 | MRR | avg turns | avg tool calls | avg cost | avg seconds | n | errors |
|---|---|---|---|---|---|---|---|---|---|
| bm25 | 0.404 | 0.619 | 0.409 | 1.0 | 5.0 | $0.0000 | 0.0 | 7 | 0 |
| retrieval | 0.678 | 0.690 | 0.786 | 1.0 | 5.3 | $0.0088 | 5.1 | 7 | 0 |
| harness | 0.870 | 0.905 | 1.000 | 2.4 | 10.6 | $0.0200 | 10.9 | 7 | 0 |

`bm25`: one BM25 search per source with the question text, no model. `retrieval` and `harness` use the same planner, judge and settings (`config.yaml`); retrieval runs one planned pass, harness runs the full loop.

Small samples: one run per question, 4 single-hop and 7 multi-hop questions. Model runs vary, so treat differences of a few points as noise.
