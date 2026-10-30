# Public eval: musique

MuSiQue v1.0 (Trivedi et al., TACL 2022, CC BY 4.0): multi-hop questions over Wikipedia paragraphs, built so a single-hop shortcut cannot answer them. Gold: the supporting paragraphs. recall@10 = share of supporting paragraphs in the top 10. Distractors come from other questions and are unlabeled, so a near-duplicate of a gold paragraph counts as a miss for every mode alike.

Sample: 60 queries, 5000 documents in one source (`wiki`). Settings: `per_hop=20`, `corpus_size=5000`, `seed=7`.

Source files (SHA-256): `musique_ans_v1.0_dev.jsonl` 15fa63794d18a94ce12411aca6e2327e65b6e83b0b1490efab3f1962e48abf3b

## 2hop questions

| mode | nDCG@10 | P@10 | recall@10 | MRR | avg turns | avg tool calls | avg cost | avg seconds | n | errors |
|---|---|---|---|---|---|---|---|---|---|---|
| bm25 | 0.477 | 0.100 | 0.500 | 0.633 | 1.0 | 1.0 | $0.0000 | 0.0 | 20 | 0 |
| retrieval | 0.648 | 0.130 | 0.650 | 0.861 | 1.0 | 3.3 | $0.0067 | 4.8 | 20 | 0 |
| harness | 0.854 | 0.185 | 0.925 | 0.892 | 3.2 | 11.2 | $0.0216 | 13.2 | 20 | 0 |

## 3hop questions

| mode | nDCG@10 | P@10 | recall@10 | MRR | avg turns | avg tool calls | avg cost | avg seconds | n | errors |
|---|---|---|---|---|---|---|---|---|---|---|
| bm25 | 0.476 | 0.155 | 0.517 | 0.708 | 1.0 | 1.0 | $0.0000 | 0.0 | 20 | 0 |
| retrieval | 0.653 | 0.195 | 0.650 | 0.908 | 1.0 | 4.0 | $0.0076 | 5.3 | 20 | 0 |
| harness | 0.783 | 0.250 | 0.833 | 0.938 | 3.1 | 12.3 | $0.0228 | 15.4 | 20 | 0 |

## 4hop questions

| mode | nDCG@10 | P@10 | recall@10 | MRR | avg turns | avg tool calls | avg cost | avg seconds | n | errors |
|---|---|---|---|---|---|---|---|---|---|---|
| bm25 | 0.408 | 0.155 | 0.388 | 0.725 | 1.0 | 1.0 | $0.0000 | 0.0 | 20 | 0 |
| retrieval | 0.561 | 0.205 | 0.512 | 0.893 | 1.0 | 3.5 | $0.0071 | 6.0 | 20 | 0 |
| harness | 0.729 | 0.290 | 0.725 | 0.942 | 3.4 | 14.6 | $0.0249 | 18.7 | 20 | 0 |

## All questions

| mode | nDCG@10 | P@10 | recall@10 | MRR | avg turns | avg tool calls | avg cost | avg seconds | n | errors |
|---|---|---|---|---|---|---|---|---|---|---|
| bm25 | 0.454 | 0.137 | 0.468 | 0.689 | 1.0 | 1.0 | $0.0000 | 0.0 | 60 | 0 |
| retrieval | 0.621 | 0.177 | 0.604 | 0.888 | 1.0 | 3.6 | $0.0071 | 5.4 | 60 | 0 |
| harness | 0.789 | 0.242 | 0.828 | 0.924 | 3.2 | 12.7 | $0.0231 | 15.7 | 60 | 0 |

`bm25`: one BM25 search with the question text, no model. `retrieval` and `harness` use the same planner, judge and settings as the Ignite demo (`config.yaml`).

One run per query. Model runs vary, so treat differences of a few points as noise.
