# Public eval: musique

MuSiQue v1.0 (Trivedi et al., TACL 2022, CC BY 4.0): multi-hop questions over Wikipedia paragraphs, built so a single-hop shortcut cannot answer them. Gold: the supporting paragraphs. recall@10 = share of supporting paragraphs in the top 10. Distractors come from other questions and are unlabeled, so a near-duplicate of a gold paragraph counts as a miss for every mode alike.

Sample: 60 queries, 5000 documents in one source (`wiki`). Settings: `per_hop=20`, `corpus_size=5000`, `seed=7`.

Source files (SHA-256): `musique_ans_v1.0_dev.jsonl` 15fa63794d18a94ce12411aca6e2327e65b6e83b0b1490efab3f1962e48abf3b

## 2hop questions

| mode | nDCG@10 | P@10 | recall@10 | MRR | avg turns | avg tool calls | avg cost | avg seconds | n | errors |
|---|---|---|---|---|---|---|---|---|---|---|
| bm25 | 0.477 | 0.100 | 0.500 | 0.633 | 1.0 | 1.0 | $0.0000 | 0.0 | 20 | 0 |
| retrieval | 0.789 | 0.155 | 0.775 | 0.975 | 1.0 | 3.5 | $0.0064 | 4.0 | 20 | 0 |
| harness | 0.873 | 0.195 | 0.975 | 0.867 | 2.7 | 9.5 | $0.0206 | 9.8 | 20 | 0 |
| harness-flashlite | 0.870 | 0.185 | 0.925 | 0.906 | 2.5 | 8.9 | $0.0241 | 14.3 | 20 | 0 |
| swarm | 0.759 | 0.155 | 0.775 | 0.884 | 2.0 | 20.3 | $0.0425 | 10.4 | 20 | 0 |
| sid-swarm | 0.879 | 0.190 | 0.950 | 0.917 | 0.0 | 41.0 | $0.0642 | 11.4 | 20 | 0 |

## 3hop questions

| mode | nDCG@10 | P@10 | recall@10 | MRR | avg turns | avg tool calls | avg cost | avg seconds | n | errors |
|---|---|---|---|---|---|---|---|---|---|---|
| bm25 | 0.476 | 0.155 | 0.517 | 0.708 | 1.0 | 1.0 | $0.0000 | 0.0 | 20 | 0 |
| retrieval | 0.730 | 0.230 | 0.767 | 0.904 | 1.0 | 4.0 | $0.0067 | 3.7 | 20 | 0 |
| harness | 0.896 | 0.290 | 0.967 | 0.950 | 2.9 | 11.3 | $0.0227 | 11.7 | 20 | 0 |
| harness-flashlite | 0.795 | 0.250 | 0.833 | 0.900 | 2.9 | 10.6 | $0.0255 | 16.2 | 20 | 0 |
| swarm | 0.555 | 0.190 | 0.633 | 0.655 | 2.0 | 20.7 | $0.0437 | 11.7 | 20 | 0 |
| sid-swarm | 0.877 | 0.275 | 0.917 | 0.925 | 0.0 | 35.4 | $0.0621 | 11.6 | 20 | 1 |

## 4hop questions

| mode | nDCG@10 | P@10 | recall@10 | MRR | avg turns | avg tool calls | avg cost | avg seconds | n | errors |
|---|---|---|---|---|---|---|---|---|---|---|
| bm25 | 0.408 | 0.155 | 0.388 | 0.725 | 1.0 | 1.0 | $0.0000 | 0.0 | 20 | 0 |
| retrieval | 0.658 | 0.255 | 0.637 | 0.942 | 1.0 | 3.8 | $0.0067 | 4.0 | 20 | 0 |
| harness | 0.778 | 0.330 | 0.825 | 0.867 | 3.8 | 13.8 | $0.0277 | 14.4 | 20 | 0 |
| harness-flashlite | 0.724 | 0.290 | 0.725 | 0.892 | 3.5 | 13.4 | $0.0315 | 21.4 | 20 | 0 |
| swarm | 0.600 | 0.265 | 0.662 | 0.778 | 2.0 | 21.6 | $0.0481 | 11.3 | 20 | 0 |
| sid-swarm | 0.819 | 0.335 | 0.838 | 0.942 | 0.0 | 36.0 | $0.0569 | 13.4 | 20 | 0 |

## All questions

| mode | nDCG@10 | P@10 | recall@10 | MRR | avg turns | avg tool calls | avg cost | avg seconds | n | errors |
|---|---|---|---|---|---|---|---|---|---|---|
| bm25 | 0.454 | 0.137 | 0.468 | 0.689 | 1.0 | 1.0 | $0.0000 | 0.0 | 60 | 0 |
| retrieval | 0.726 | 0.213 | 0.726 | 0.940 | 1.0 | 3.8 | $0.0066 | 3.9 | 60 | 0 |
| harness | 0.849 | 0.272 | 0.922 | 0.894 | 3.1 | 11.5 | $0.0236 | 12.0 | 60 | 0 |
| harness-flashlite | 0.796 | 0.242 | 0.828 | 0.899 | 3.0 | 10.9 | $0.0270 | 17.3 | 60 | 0 |
| swarm | 0.638 | 0.203 | 0.690 | 0.772 | 2.0 | 20.9 | $0.0448 | 11.1 | 60 | 0 |
| sid-swarm | 0.858 | 0.267 | 0.901 | 0.928 | 0.0 | 37.4 | $0.0611 | 12.1 | 60 | 1 |

`bm25`: one BM25 search with the question text, no model. `retrieval` and `harness` use the same planner, judge and settings as the Ignite demo (`config.yaml`). `swarm`: one planner call splits the question; up to 4 harness agents run at the same time on the parts with the harness budget divided between them; RRF merges their lists.

`swarm` ran one query at a time (concurrency 1; its agents already run in parallel) while the other modes ran two at a time, so its seconds are not directly comparable.

`harness-flashlite` (eval-only judge control): the same harness, planner and budget with the earlier judge, gemini-3.5-flash-lite on Vertex, in place of Gemini Decision 1. It shows what the judge change does to scores, time and cost.

`sid-swarm`: SID-2 Swarm (SID AI's trained multi-agent search model). SID's server runs up to 6 agents that fork and message each other; our harness runs their searches over the same documents and fuses their reports with RRF. No judge; its own budget (200 tool calls, $1.00, the same 120 s); concurrency 1.

One run per query. Model runs vary, so treat differences of a few points as noise.
