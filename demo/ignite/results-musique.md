# Public eval: musique

MuSiQue v1.0 (Trivedi et al., TACL 2022, CC BY 4.0): multi-hop questions over Wikipedia paragraphs, built so a single-hop shortcut cannot answer them. Gold: the supporting paragraphs. recall@10 = share of supporting paragraphs in the top 10.

Sample: 60 queries, 5000 documents in one source (`wiki`). Settings: `per_hop=20`, `corpus_size=5000`, `seed=7`.

Source files (SHA-256): `musique_ans_v1.0_dev.jsonl` 15fa63794d18a94ce12411aca6e2327e65b6e83b0b1490efab3f1962e48abf3b

## 2hop questions

| mode | nDCG@10 | recall@10 | MRR | avg turns | avg tool calls | avg cost | avg seconds | n | errors |
|---|---|---|---|---|---|---|---|---|---|
| bm25 | 0.477 | 0.500 | 0.633 | 1.0 | 1.0 | $0.0000 | 0.0 | 20 | 0 |
| retrieval | 0.652 | 0.675 | 0.819 | 1.0 | 4.0 | $0.0066 | 5.7 | 20 | 0 |
| harness | 0.805 | 0.850 | 0.887 | 2.6 | 11.4 | $0.0177 | 11.7 | 20 | 0 |

## 3hop questions

| mode | nDCG@10 | recall@10 | MRR | avg turns | avg tool calls | avg cost | avg seconds | n | errors |
|---|---|---|---|---|---|---|---|---|---|
| bm25 | 0.476 | 0.517 | 0.708 | 1.0 | 1.0 | $0.0000 | 0.0 | 20 | 0 |
| retrieval | 0.686 | 0.717 | 0.877 | 1.0 | 4.5 | $0.0070 | 5.5 | 20 | 0 |
| harness | 0.715 | 0.750 | 0.877 | 2.5 | 11.4 | $0.0176 | 14.8 | 20 | 0 |

## 4hop questions

| mode | nDCG@10 | recall@10 | MRR | avg turns | avg tool calls | avg cost | avg seconds | n | errors |
|---|---|---|---|---|---|---|---|---|---|
| bm25 | 0.408 | 0.388 | 0.725 | 1.0 | 1.0 | $0.0000 | 0.0 | 20 | 0 |
| retrieval | 0.610 | 0.550 | 0.975 | 1.0 | 4.5 | $0.0071 | 6.1 | 20 | 0 |
| harness | 0.764 | 0.775 | 0.904 | 2.7 | 13.0 | $0.0192 | 19.1 | 20 | 0 |

## All questions

| mode | nDCG@10 | recall@10 | MRR | avg turns | avg tool calls | avg cost | avg seconds | n | errors |
|---|---|---|---|---|---|---|---|---|---|
| bm25 | 0.454 | 0.468 | 0.689 | 1.0 | 1.0 | $0.0000 | 0.0 | 60 | 0 |
| retrieval | 0.649 | 0.647 | 0.890 | 1.0 | 4.3 | $0.0069 | 5.8 | 60 | 0 |
| harness | 0.761 | 0.792 | 0.889 | 2.6 | 12.0 | $0.0182 | 15.2 | 60 | 0 |

`bm25`: one BM25 search with the question text, no model. `retrieval` and `harness` use the same planner, judge and settings as the Ignite demo (`config.yaml`).

One run per query. Model runs vary, so treat differences of a few points as noise.
