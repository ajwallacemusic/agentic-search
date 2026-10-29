# Ignite Public-Dataset Eval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Score bm25, retrieval and harness modes on two public datasets with human labels: a MuSiQue sample (multi-hop, Wikipedia) and a WANDS sample (product search), next to the synthetic Ignite numbers.

**Architecture:** `demo/ignite/public/` fetches each dataset into a git-ignored cache and checks it against pinned SHA-256 hashes, draws a seeded sample, builds a one-source harness with the same planner, judge and settings as the demo, and runs the library's `run_eval`. Results go to `demo/ignite/results-<dataset>.md`. Downloaded data is never committed.

**Tech Stack:** as the Ignite demo (Python 3.11, uv, pytest, Vertex via `config.yaml`).

**Spec:** the Ignite spec (`docs/superpowers/specs/2026-10-29-ignite-demo-design.md` §8 "Eval") plus AJ's request on 2026-10-29: "a mix: WANDS and HotpotQA or MuSiQue, whichever is more reliable and can get a good sample of".

## Decisions (made while writing this plan)

- **MuSiQue, not HotpotQA.** MuSiQue (Trivedi et al., 2022) was built to remove questions answerable by a single-hop shortcut, which is a known weakness of HotpotQA. Each question ships with 20 paragraphs (supporting ones flagged), so a sample is self-contained. License CC BY 4.0. File: `musique_ans_v1.0_dev.jsonl` (2,417 answerable questions: 1,252 2-hop, 760 3-hop, 405 4-hop).
  - Source: the Hugging Face mirror `bdsaglam/musique`, verified byte-identical to the official Google Drive zip. SHA-256 `15fa63794d18a94ce12411aca6e2327e65b6e83b0b1490efab3f1962e48abf3b`.
- **WANDS** (Wayfair, MIT), pinned to commit `3b74dcf4ba29ab8ff3e6a50b5b09fc627cb882b5`, 42,994 products, 480 queries, 233,448 labels.
  - `product.csv` d993926254572e6eba96c8fd87cc549a17fb91ad3748308036eee4cf92b10ac6
  - `query.csv` 63b61660560fecc33ec490804c7e2b81402ee3e7c31a9cbb5e03736639f68e95
  - `label.csv` c11fe81ad62f17f56f316b0ec9630ebe8fbe1393578cb0ca4f05c17253a180ef
- **Samples** (seed 7, configurable):
  - MuSiQue: 60 questions, 20 each from 2-hop, 3-hop and 4-hop. The corpus is the deduplicated paragraphs of the sampled questions, plus paragraphs from other dev questions as distractors, up to 5,000 paragraphs. Gold: supporting paragraphs, relevance 1. Groups: `2hop`, `3hop`, `4hop`.
  - WANDS: 60 queries that have at least one Exact label. Corpus: the full catalogue. Gold: Exact = 2, Partial = 1 (graded, as WANDS is normally scored). Group: `all`. Headline metric: nDCG@10, because a median query has 28 Exact products, so recall@10 is capped well below 1.
- **Same models and settings as the demo** (`config.yaml`), one source per dataset.

## Global Constraints

- Nothing under `src/` or `scripts/` changes.
- Downloaded files live in `demo/ignite/public/.cache/` (git-ignored) and are verified by SHA-256 before use; a mismatch is an error, never a silent re-download.
- Tests are offline: tiny fixture files in the datasets' real formats.
- `uv run pytest -q`, `uv run pytest demo/ignite/tests -q`, `uv run ruff check` stay clean.

## Review Focus

1. **A hash mismatch** (the upstream file changed, or a download was cut short): fetch fails with the expected and actual hash and names the file. Test: Task 1 `test_hash_mismatch_fails`.
2. **A sampled question whose supporting paragraph was deduplicated into another id**: the gold must point at the id that is actually in the corpus. Test: Task 2 `test_shared_paragraph_keeps_one_id_and_gold_follows`.
3. **The corpus cap dropping a gold paragraph**: the cap only ever drops distractors. Test: Task 2 `test_cap_never_drops_sampled_paragraphs`.
4. **Same seed, different sample** (e.g. dict order): samples are deterministic. Tests: Task 2 and 3 `..._is_deterministic`.
5. **Product rows with odd fields** (empty description, a description with tabs or newlines): loading must not crash or lose the product. Test: Task 3 `test_products_with_empty_fields_load`.

---

### Task 1: Pinned fetch

**Files:** Create `demo/ignite/public/__init__.py`, `demo/ignite/public/fetch.py`; modify `.gitignore`; test `demo/ignite/tests/test_public_fetch.py`.

**Produces:** `CACHE_DIR: Path`; `FILES: dict[str, dict[str, tuple[str, str]]]` (dataset → file name → (url, sha256)); `sha256_of(path) -> str`; `fetch(dataset: str, cache_dir: Path = CACHE_DIR, download: Callable[[str, Path], None] = _download) -> Path` (returns the dataset's folder; downloads only missing files; checks every file's hash; raises `FetchError` on a mismatch, naming the file and both hashes).

- [ ] Tests: `test_cached_file_with_right_hash_is_not_downloaded`, `test_missing_file_is_downloaded_and_checked`, `test_hash_mismatch_fails`, `test_unknown_dataset_fails`. Run, watch fail, implement, run, commit.

### Task 2: MuSiQue sample

**Files:** Create `demo/ignite/public/datasets.py`; test `demo/ignite/tests/test_public_datasets.py`.

**Produces:** `PublicSample` dataclass: `name: str`, `source: str`, `description: str`, `documents: list[Document]`, `queries: dict[str, str]`, `qrels: dict[str, dict[str, int]]`, `groups: dict[str, list[str]]`, plus `to_beir() -> BeirDataset`. `sample_musique(path: Path, *, per_hop: int = 20, corpus_size: int = 5000, seed: int = 7) -> PublicSample`. Paragraph ids: `msq-<first 12 hex of sha1(title + "\n" + text)>`. Document text: `title\n\ntext`, metadata `{"title": title}`.

- [ ] Tests: `test_samples_per_hop_group`, `test_gold_is_the_supporting_paragraphs`, `test_shared_paragraph_keeps_one_id_and_gold_follows`, `test_cap_never_drops_sampled_paragraphs`, `test_distractors_fill_to_the_cap`, `test_musique_sample_is_deterministic`. Fixture: a few hand-written records in the real MuSiQue shape. Run, fail, implement, pass, commit.

### Task 3: WANDS sample

**Files:** Modify `demo/ignite/public/datasets.py`; test `demo/ignite/tests/test_public_datasets.py`.

**Produces:** `sample_wands(folder: Path, *, n: int = 60, seed: int = 7) -> PublicSample`. Doc id `wands-<product_id>`; text `product_name\n\nproduct_class · category hierarchy\n\nproduct_description\n\nproduct_features`; metadata `product_class`, `category`, `rating_count`, `average_rating` (numbers when parseable). Group `all`.

- [ ] Tests: `test_only_queries_with_an_exact_label`, `test_labels_are_graded`, `test_products_with_empty_fields_load`, `test_wands_sample_is_deterministic`. Run, fail, implement, pass, commit.

### Task 4: One-source harness and eval CLI

**Files:** Modify `demo/ignite/build.py` (extract `build_harness_for`), `demo/ignite/eval.py` (extract `evaluate_groups`); create `demo/ignite/public/eval_public.py`; tests in `test_build.py`, `test_eval.py`, `demo/ignite/tests/test_public_eval.py`.

**Produces:**
- `build_harness_for(cfg, docs: dict[str, list[Document]], descriptions: dict[str, str], *, controller="heuristic", driver=None, analyzer=None, embedder=None) -> Harness`; `build_demo_harness` calls it with the Ignite sources.
- `evaluate_groups(modes, harness_for, dataset: BeirDataset, groups: dict[str, list[str]], *, name: str) -> dict[tuple[str, str], KindReport]`; `evaluate` calls it with the Ignite kinds.
- `bm25_harness_for(docs, descriptions) -> Harness`; `bm25_harness` calls it.
- `results_table(reports, groups: dict[str, str]) -> list[str]` (markdown lines), reused by `results_markdown`.
- CLI: `uv run python -m demo.ignite.public.eval_public --dataset musique|wands [--n …] [--seed 7] [--modes bm25,retrieval,harness]` writes `demo/ignite/results-<dataset>.md` with the sample settings, source hashes and per-group tables.

- [ ] Tests: the existing Ignite eval tests stay green; `test_build_harness_for_one_source`; `test_eval_public_bm25_on_fixture_writes_tables` (fixture MuSiQue sample, bm25 only, offline). Run, fail, implement, pass, commit.

### Task 5: Run it and write it up

- [ ] Run `eval_public --dataset musique` and `--dataset wands` live (about $2 each). Record cost and time.
- [ ] Add a "Public datasets" section to `demo/ignite/README.md`: what each set tests, sample sizes, headline numbers, licenses and citations, and how to re-run.
- [ ] Full checks; commit; merge into `main` (AJ chose local merges).
