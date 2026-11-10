# Public eval: musique

MuSiQue v1.0 (Trivedi et al., TACL 2022, CC BY 4.0): multi-hop questions over Wikipedia paragraphs, built so a single-hop shortcut cannot answer them. Gold: the supporting paragraphs. recall@10 = share of supporting paragraphs in the top 10. Distractors come from other questions and are unlabeled, so a near-duplicate of a gold paragraph counts as a miss for every mode alike.

Sample: 60 queries, 5000 documents in one source (`wiki`). Settings: `per_hop=20`, `corpus_size=5000`, `seed=7`.

Source files (SHA-256): `musique_ans_v1.0_dev.jsonl` 15fa63794d18a94ce12411aca6e2327e65b6e83b0b1490efab3f1962e48abf3b

## 2hop questions

| mode | nDCG@10 | P@10 | recall@10 | MRR | avg turns | avg tool calls | avg cost | avg seconds | n | errors |
|---|---|---|---|---|---|---|---|---|---|---|
| bm25 | 0.477 | 0.100 | 0.500 | 0.633 | 1.0 | 1.0 | $0.0000 | 0.0 | 20 | 0 |
| retrieval | 0.752 | 0.150 | 0.750 | 0.912 | 1.0 | 4.2 | $0.0125 | 17.5 | 20 | 0 |
| harness | 0.851 | 0.180 | 0.900 | 0.875 | 3.0 | 13.6 | $0.0388 | 44.3 | 20 | 0 |

## 3hop questions

| mode | nDCG@10 | P@10 | recall@10 | MRR | avg turns | avg tool calls | avg cost | avg seconds | n | errors |
|---|---|---|---|---|---|---|---|---|---|---|
| bm25 | 0.476 | 0.155 | 0.517 | 0.708 | 1.0 | 1.0 | $0.0000 | 0.0 | 20 | 0 |
| retrieval | 0.749 | 0.230 | 0.767 | 0.942 | 1.0 | 4.6 | $0.0134 | 14.3 | 20 | 0 |
| harness | 0.833 | 0.280 | 0.933 | 0.863 | 3.0 | 13.3 | $0.0390 | 42.2 | 20 | 0 |

## 4hop questions

| mode | nDCG@10 | P@10 | recall@10 | MRR | avg turns | avg tool calls | avg cost | avg seconds | n | errors |
|---|---|---|---|---|---|---|---|---|---|---|
| bm25 | 0.408 | 0.155 | 0.388 | 0.725 | 1.0 | 1.0 | $0.0000 | 0.0 | 20 | 0 |
| retrieval | 0.598 | 0.230 | 0.575 | 0.912 | 1.0 | 4.3 | $0.0134 | 12.6 | 20 | 0 |
| harness | 0.743 | 0.305 | 0.762 | 0.887 | 3.6 | 18.1 | $0.0479 | 49.2 | 20 | 0 |

## All questions

| mode | nDCG@10 | P@10 | recall@10 | MRR | avg turns | avg tool calls | avg cost | avg seconds | n | errors |
|---|---|---|---|---|---|---|---|---|---|---|
| bm25 | 0.454 | 0.137 | 0.468 | 0.689 | 1.0 | 1.0 | $0.0000 | 0.0 | 60 | 0 |
| retrieval | 0.700 | 0.203 | 0.697 | 0.922 | 1.0 | 4.4 | $0.0131 | 14.8 | 60 | 0 |
| harness | 0.809 | 0.255 | 0.865 | 0.875 | 3.2 | 15.0 | $0.0419 | 45.2 | 60 | 0 |

`bm25`: one BM25 search with the question text, no model. `retrieval` and `harness` use the same planner, judge and settings as the Ignite demo (`config.yaml`).

Harness rows: 2026-11-10, with the stop-when-done rule (tuning-log round 14); bm25 and retrieval rows: 2026-11-09 (the rule does not touch them: retrieval never asks the controller). One run per query. Model runs vary, so treat differences of a few points as noise.
