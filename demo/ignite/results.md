# Ignite eval results

## Single-hop questions

| mode | nDCG@10 | recall@10 | MRR | avg turns | avg tool calls | avg cost | avg seconds | n | errors |
|---|---|---|---|---|---|---|---|---|---|
| bm25 | 0.487 | 1.000 | 0.321 | 1.0 | 5.0 | $0.0000 | 0.0 | 4 | 0 |
| retrieval | 1.000 | 1.000 | 1.000 | 1.0 | 4.8 | $0.0083 | 5.4 | 4 | 0 |
| harness | 1.000 | 1.000 | 1.000 | 2.2 | 10.0 | $0.0189 | 9.9 | 4 | 0 |

## Multi-hop questions

| mode | nDCG@10 | recall@10 | MRR | avg turns | avg tool calls | avg cost | avg seconds | n | errors |
|---|---|---|---|---|---|---|---|---|---|
| bm25 | 0.404 | 0.619 | 0.409 | 1.0 | 5.0 | $0.0000 | 0.0 | 7 | 0 |
| retrieval | 0.744 | 0.738 | 0.929 | 1.0 | 5.4 | $0.0091 | 6.4 | 7 | 0 |
| harness | 0.914 | 0.952 | 1.000 | 2.6 | 10.4 | $0.0209 | 11.4 | 7 | 0 |

`bm25`: one BM25 search per source with the question text, no model. `retrieval` and `harness` use the same planner, judge and settings (`config.yaml`); retrieval runs one planned pass, harness runs the full loop.

Small samples: one run per question, 4 single-hop and 7 multi-hop questions. Model runs vary, so treat differences of a few points as noise.
