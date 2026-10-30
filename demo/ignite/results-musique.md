# Public eval: musique

MuSiQue v1.0 (Trivedi et al., TACL 2022, CC BY 4.0): multi-hop questions over Wikipedia paragraphs, built so a single-hop shortcut cannot answer them. Gold: the supporting paragraphs. recall@10 = share of supporting paragraphs in the top 10. Distractors come from other questions and are unlabeled, so a near-duplicate of a gold paragraph counts as a miss for every mode alike.

Sample: 60 queries, 5000 documents in one source (`wiki`). Settings: `per_hop=20`, `corpus_size=5000`, `seed=7`.

Source files (SHA-256): `musique_ans_v1.0_dev.jsonl` 15fa63794d18a94ce12411aca6e2327e65b6e83b0b1490efab3f1962e48abf3b

## 2hop questions

| mode | nDCG@10 | P@10 | recall@10 | MRR | avg turns | avg tool calls | avg cost | avg seconds | n | errors |
|---|---|---|---|---|---|---|---|---|---|---|
| bm25 | 0.477 | 0.100 | 0.500 | 0.633 | 1.0 | 1.0 | $0.0000 | 0.0 | 20 | 0 |
| retrieval | 0.782 | 0.155 | 0.775 | 0.950 | 1.0 | 4.3 | $0.0099 | 8.0 | 20 | 0 |
| harness | 0.866 | 0.185 | 0.925 | 0.900 | 3.1 | 14.3 | $0.0342 | 29.9 | 20 | 1 |

## 3hop questions

| mode | nDCG@10 | P@10 | recall@10 | MRR | avg turns | avg tool calls | avg cost | avg seconds | n | errors |
|---|---|---|---|---|---|---|---|---|---|---|
| bm25 | 0.476 | 0.155 | 0.517 | 0.708 | 1.0 | 1.0 | $0.0000 | 0.0 | 20 | 0 |
| retrieval | 0.724 | 0.220 | 0.733 | 0.917 | 1.0 | 4.7 | $0.0108 | 8.7 | 20 | 0 |
| harness | 0.865 | 0.280 | 0.933 | 0.917 | 3.0 | 13.0 | $0.0332 | 31.4 | 20 | 0 |

## 4hop questions

| mode | nDCG@10 | P@10 | recall@10 | MRR | avg turns | avg tool calls | avg cost | avg seconds | n | errors |
|---|---|---|---|---|---|---|---|---|---|---|
| bm25 | 0.408 | 0.155 | 0.388 | 0.725 | 1.0 | 1.0 | $0.0000 | 0.0 | 20 | 0 |
| retrieval | 0.682 | 0.255 | 0.637 | 0.975 | 1.0 | 4.5 | $0.0106 | 9.8 | 20 | 0 |
| harness | 0.731 | 0.320 | 0.800 | 0.850 | 3.4 | 17.0 | $0.0383 | 55.1 | 20 | 1 |

## All questions

| mode | nDCG@10 | P@10 | recall@10 | MRR | avg turns | avg tool calls | avg cost | avg seconds | n | errors |
|---|---|---|---|---|---|---|---|---|---|---|
| bm25 | 0.454 | 0.137 | 0.468 | 0.689 | 1.0 | 1.0 | $0.0000 | 0.0 | 60 | 0 |
| retrieval | 0.729 | 0.210 | 0.715 | 0.947 | 1.0 | 4.5 | $0.0104 | 8.8 | 60 | 0 |
| harness | 0.821 | 0.262 | 0.886 | 0.889 | 3.1 | 14.8 | $0.0352 | 38.8 | 60 | 2 |

`bm25`: one BM25 search with the question text, no model. `retrieval` and `harness` use the same planner, judge and settings as the Ignite demo (`config.yaml`).

One run per query. Model runs vary, so treat differences of a few points as noise.
