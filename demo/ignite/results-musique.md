# Public eval: musique

MuSiQue v1.0 (Trivedi et al., TACL 2022, CC BY 4.0): multi-hop questions over Wikipedia paragraphs, built so a single-hop shortcut cannot answer them. Gold: the supporting paragraphs. recall@10 = share of supporting paragraphs in the top 10. Distractors come from other questions and are unlabeled, so a near-duplicate of a gold paragraph counts as a miss for every mode alike.

Sample: 60 queries, 5000 documents in one source (`wiki`). Settings: `per_hop=20`, `corpus_size=5000`, `seed=7`.

Source files (SHA-256): `musique_ans_v1.0_dev.jsonl` 15fa63794d18a94ce12411aca6e2327e65b6e83b0b1490efab3f1962e48abf3b

## 2hop questions

| mode | nDCG@10 | P@10 | recall@10 | MRR | avg turns | avg tool calls | avg cost | avg seconds | n | errors |
|---|---|---|---|---|---|---|---|---|---|---|
| bm25 | 0.477 | 0.100 | 0.500 | 0.633 | 1.0 | 1.0 | $0.0000 | 0.0 | 20 | 0 |
| retrieval | 0.783 | 0.160 | 0.800 | 0.931 | 1.0 | 4.4 | $0.0130 | 14.7 | 20 | 0 |
| harness | 0.832 | 0.185 | 0.925 | 0.833 | 3.0 | 13.4 | $0.0380 | 39.2 | 20 | 0 |
| swarm | 0.775 | 0.165 | 0.825 | 0.904 | 2.0 | 23.9 | $0.0778 | 34.1 | 20 | 0 |

## 3hop questions

| mode | nDCG@10 | P@10 | recall@10 | MRR | avg turns | avg tool calls | avg cost | avg seconds | n | errors |
|---|---|---|---|---|---|---|---|---|---|---|
| bm25 | 0.476 | 0.155 | 0.517 | 0.708 | 1.0 | 1.0 | $0.0000 | 0.0 | 20 | 0 |
| retrieval | 0.811 | 0.265 | 0.883 | 0.917 | 1.0 | 4.7 | $0.0137 | 13.1 | 20 | 0 |
| harness | 0.882 | 0.275 | 0.917 | 0.967 | 3.1 | 14.1 | $0.0400 | 39.7 | 20 | 0 |
| swarm | 0.719 | 0.235 | 0.783 | 0.900 | 2.0 | 23.4 | $0.0751 | 36.6 | 20 | 0 |

## 4hop questions

| mode | nDCG@10 | P@10 | recall@10 | MRR | avg turns | avg tool calls | avg cost | avg seconds | n | errors |
|---|---|---|---|---|---|---|---|---|---|---|
| bm25 | 0.408 | 0.155 | 0.388 | 0.725 | 1.0 | 1.0 | $0.0000 | 0.0 | 20 | 0 |
| retrieval | 0.644 | 0.235 | 0.588 | 0.975 | 1.0 | 4.6 | $0.0136 | 18.0 | 20 | 0 |
| harness | 0.715 | 0.325 | 0.812 | 0.768 | 3.9 | 18.6 | $0.0490 | 42.3 | 20 | 0 |
| swarm | 0.628 | 0.255 | 0.637 | 0.850 | 2.0 | 24.0 | $0.0859 | 28.1 | 20 | 0 |

## All questions

| mode | nDCG@10 | P@10 | recall@10 | MRR | avg turns | avg tool calls | avg cost | avg seconds | n | errors |
|---|---|---|---|---|---|---|---|---|---|---|
| bm25 | 0.454 | 0.137 | 0.468 | 0.689 | 1.0 | 1.0 | $0.0000 | 0.0 | 60 | 0 |
| retrieval | 0.746 | 0.220 | 0.757 | 0.941 | 1.0 | 4.6 | $0.0135 | 15.3 | 60 | 0 |
| harness | 0.810 | 0.262 | 0.885 | 0.856 | 3.3 | 15.3 | $0.0423 | 40.4 | 60 | 0 |
| swarm | 0.707 | 0.218 | 0.749 | 0.885 | 2.0 | 23.8 | $0.0796 | 32.9 | 60 | 0 |

`bm25`: one BM25 search with the question text, no model. `retrieval` and `harness` use the same planner, judge and settings as the Ignite demo (`config.yaml`). `swarm`: one planner call splits the question; up to 4 harness agents run at the same time on the parts with the harness budget divided between them; RRF merges their lists.

`swarm` ran one query at a time (concurrency 1; its agents already run in parallel) while the other modes ran two at a time, so its seconds are not directly comparable.

One run per query. Model runs vary, so treat differences of a few points as noise.
