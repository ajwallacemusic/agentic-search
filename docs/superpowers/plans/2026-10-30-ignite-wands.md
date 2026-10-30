# Ignite WANDS Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the live Ignite demo on three question sets (MuSiQue, WANDS as published, and our own rule-labeled "WANDS match" questions), and score WANDS match in the public eval.

**Architecture:** `public/wands_match.py` parses WANDS product features, picks "look it up, then match it" questions by fixed rules, and builds their labels from those rules. `corpus.py` loads one `DemoData` per corpus (`musique`, `wands`); the server gets one harness profile per corpus, and the viewer posts each question to its corpus's profile. A disk cache for document embeddings keeps the 42,994-product start-up short.

**Tech Stack:** as the Ignite demo (Python 3.12, uv, pytest, node for viewer tests, Vertex via `config.yaml`).

**Spec:** `docs/superpowers/specs/2026-10-30-ignite-wands-design.md` (builds on `docs/superpowers/specs/2026-10-29-ignite-demo-design.md`).

## Global Constraints

- Nothing under `src/` or `scripts/` changes.
- WANDS documents stay exactly as `_product_doc` builds them today (spec §7, AJ's decision). No new metadata.
- Labels come from rules over WANDS fields only; never from a model; never hand-edited (spec §3, §4).
- The WANDS match question set is fixed by rule and committed **before any model runs on it** (spec §4).
- Every WANDS match question is scored and reported, including losses. If the harness does not win, stop and ask AJ before changing the set (spec §10).
- Anchor: exactly one value for the attribute; name unique in the catalogue; value not in the anchor's name; target classes differ from the anchor's class (spec §5).
- Gold: product in a target class **and** any of its attribute values equals the anchor's value, after normalizing case and spaces; relevance 1; anchor never gold (spec §5.3).
- Headline metric for WANDS match: precision@10 (spec §5.3).
- `serve` ready in under 5 minutes on this laptop (spec §8). Measured 2026-10-30: about 4 min to embed WANDS with no cache.
- `uv run pytest -q`, `uv run pytest demo/ignite/tests -q`, `uv run ruff check` stay clean.
- Commits on a branch, merged to local `main` only when AJ says so. Commit messages end with the attribution lines from the session.

## Review Focus

1. **A feature value with odd spacing or case** ("Light  Wood" vs "light wood"): must match. Test: Task 1 `test_values_normalize_case_and_spaces`.
2. **A target that lists the anchor's value as its second value** (multi-valued target): must be gold. Test: Task 1 `test_target_matches_on_any_value`.
3. **A question whose text names the anchor wrong or leaks the value**: caught before any model runs. Test: Task 2 `test_question_text_names_anchor_and_hides_value`.
4. **A question id used in two sets** (e.g. `hero` in MuSiQue and WANDS match): load fails, naming the id. Test: Task 5 `test_duplicate_question_id_fails`.
5. **A cache file from a different embedding model** (config changed): never used. Test: Task 3 `test_cache_is_keyed_by_model`.

---

### Task 1: Features, rules and labels

**Files:** Create `demo/ignite/public/wands_match.py`; test `demo/ignite/tests/test_wands_match.py`.

**Interfaces:**
- Consumes: `demo.ignite.public.datasets._tsv`, `PublicSample`, `_product_doc`.
- Produces:
  - `ATTRIBUTES: dict[str, str] = {"wood_tone": "dswoodtone", "color": "color", "style": "style"}` (attribute → feature key).
  - `norm(value: str) -> str`: lower case, spaces collapsed, stripped.
  - `parse_features(text: str) -> dict[str, list[str]]`: split on `|`, then on the first `:`; key = `norm(key)` with spaces removed; values normalized, de-duplicated, in order.
  - `class MatchRule(BaseModel)`: `id: str`, `anchor_id: str`, `attribute: Literal["wood_tone", "color", "style"]`, `targets: list[str]` (product classes), `text: str = ""`.
  - `class Product` (dataclass): `product_id: str`, `name: str`, `product_class: str`, `features: dict[str, list[str]]`. `load_products(folder: Path) -> dict[str, Product]`.
  - `anchor_value(rule: MatchRule, products: dict[str, Product]) -> str` (raises `ValueError` if the anchor has zero or several values).
  - `gold_for(rule, products) -> list[str]`: sorted doc ids `wands-<product_id>`.
  - `select_rules(products, *, n: int = 20, seed: int = 7, min_gold: int = 5, max_share: float = 0.5) -> list[MatchRule]`: seeded; rules with `text=""`; ids `match-01`…; mixed across the three attributes (round-robin by attribute); 1 or 2 target classes each.

- [ ] **Step 1: Failing tests** with a fixture `product.csv` (tab-separated, the real columns) of ~12 hand-written products:
  - `test_parse_features_splits_keys_and_values`: `"dswoodtone : Light Wood|color:gray|color : Gray|style : rustic"` → `{"dswoodtone": ["light wood"], "color": ["gray"], "style": ["rustic"]}`.
  - `test_values_normalize_case_and_spaces`: anchor `"Light  Wood"`, target `"light wood"` → target is gold.
  - `test_target_matches_on_any_value`: target with `style : modern|style : rustic`, anchor `rustic` → gold.
  - `test_anchor_with_two_values_is_rejected`: `anchor_value` raises `ValueError`.
  - `test_gold_is_target_class_and_same_value_and_never_the_anchor`.
  - `test_select_rules_follows_the_rules`: every rule's anchor has one value, a name unique in the fixture, a value not in its name; targets differ from the anchor class; each gold set has ≥ `min_gold` and ≤ `max_share` of the target classes' products.
  - `test_select_rules_is_deterministic`: same seed → same rules.
- [ ] **Step 2: Run** `uv run pytest demo/ignite/tests/test_wands_match.py -q`. Expected: FAIL (module missing).
- [ ] **Step 3: Implement** the module as specified.
- [ ] **Step 4: Run** the same command. Expected: PASS.
- [ ] **Step 5: Commit** `feat(demo): WANDS match rules and rule-based labels`.

### Task 2: Fix the WANDS match question set (no model runs)

**Files:** Create `demo/ignite/wands_match.yaml` (rules + text), `demo/ignite/wands_match_labels.json` (generated); modify `demo/ignite/public/wands_match.py` (CLI); test `demo/ignite/tests/test_wands_match.py`.

**Interfaces:**
- Produces: `MATCH_PATH = demo/ignite/wands_match.yaml`; `LABELS_PATH = demo/ignite/wands_match_labels.json` (`{rule_id: [doc_id, ...]}`); `load_rules(path=MATCH_PATH) -> list[MatchRule]`; `build_labels(rules, products) -> dict[str, list[str]]`.
- CLI: `uv run python -m demo.ignite.public.wands_match select` prints 20 rules as YAML (anchor name, class, value and gold count as comments); `... labels` rewrites `LABELS_PATH` from `MATCH_PATH`.

- [ ] **Step 1: Failing tests:**
  - `test_question_text_names_anchor_and_hides_value`: for every rule in `MATCH_PATH`, `text` contains the anchor's name (case-insensitive) and does not contain its attribute value. Needs the cache: `pytest.skip` if `public/.cache/wands/product.csv` is missing.
  - `test_committed_labels_match_the_rules`: `build_labels(load_rules(), load_products(...)) == json.load(LABELS_PATH)`. Same skip.
  - `test_rule_ids_are_unique`.
- [ ] **Step 2: Run.** Expected: FAIL (files missing).
- [ ] **Step 3:** Add the CLI. Run `select`. Paste the 20 rules into `wands_match.yaml` with a header comment giving the selection settings (seed 7, `min_gold`, `max_share`). Write each `text` by hand as a shopper would ask: name the anchor and the target types, never the value (spec §5.1 example). Run `labels`.
- [ ] **Step 4: Run** `uv run pytest demo/ignite/tests/test_wands_match.py -q`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(demo): 20 WANDS match questions and their labels` **before Task 4 runs any model.**

### Task 3: Embedding cache

**Files:** Modify `demo/ignite/build.py`, `demo/ignite/config.yaml`; test `test_build.py`.

**Interfaces:** `CachedEmbedder(inner: Embedder, path: Path)`: wraps `embed`. For `purpose == "document"` and text items, looks each item up by `sha1(f"{inner.id}\n{text}")`; embeds only misses; saves new vectors to `path` (an `.npz`: keys + float32 matrix). Query embeddings and non-text items pass through. `build_harness_for` wraps the embedder as `SerialEmbedder(CachedEmbedder(...))` when `cfg["embedding_cache"]` is set; `config.yaml` sets `embedding_cache: public/.cache/embeddings/bge-small-en-v1.5.npz` (relative to `demo/ignite/`, git-ignored with the rest of `.cache`); one file holds every corpus, since keys are text hashes.

- [ ] **Step 1: Failing tests:** `test_cache_hits_skip_the_inner_embedder` (second call embeds 0 items); `test_cache_survives_a_new_instance` (reload from file); `test_cache_is_keyed_by_model` (inner id changes → all miss); `test_queries_are_never_cached`.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `uv run pytest demo/ignite/tests -q`. Expected: PASS.
- [ ] **Step 5: Measure** WANDS harness `setup()` time cold and warm; ledger both. Expected: warm well under 5 min.
- [ ] **Step 6: Commit** `feat(demo): disk cache for document embeddings`.

### Task 4: Score WANDS match

**Files:** Modify `demo/ignite/eval.py`, `demo/ignite/public/wands_match.py`, `demo/ignite/public/eval_public.py`; tests in `test_wands_match.py`, `test_public_eval.py`.

**Interfaces:**
- `TurnRecorder` also records `self.top10: dict[str, list[str]]` (question text → first 10 hit doc ids).
- `KindReport` gains `precision: list[float] = field(default_factory=list)`; `evaluate_groups` fills it as |top 10 ∩ gold| / 10 per run (an errored run counts 0).
- `results_table` adds a `P@10` column after `nDCG@10`.
- `sample_wands_match(folder: Path, rules_path: Path = MATCH_PATH, labels_path: Path = LABELS_PATH) -> PublicSample`: name `"wands-match"`, source `"products"`, the same description and documents as `sample_wands`, queries = rule texts, qrels from the labels file (relevance 1), groups by attribute.
- `eval_public --dataset wands-match` uses `fetch("wands")` and writes `results-wands-match.md`; `ABOUT["wands-match"]` says the questions and rules are ours, the labels are rule-built, and P@10 is the headline.

- [ ] **Step 1: Failing tests:** `test_precision_at_10_counts_gold_in_top_10` (bm25 on the fixture: known value); `test_results_table_has_p_at_10`; `test_sample_wands_match_groups_by_attribute` (fixture rules + labels in tmp_path).
- [ ] **Step 2: Run** `uv run pytest demo/ignite/tests -q`. Expected: the new tests FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `uv run pytest demo/ignite/tests -q`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(demo): score WANDS match with precision@10`.
- [ ] **Step 6: Live run** `IGNITE_GCP_PROJECT=gen-ai-accel-dev uv run python -m demo.ignite.public.eval_public --dataset wands-match`. Record cost and time. Commit `results-wands-match.md`.
- [ ] **Step 7: Gate.** If harness P@10 is not above retrieval on "All", **stop and report to AJ** (spec §10). Do not edit the question set.

### Task 5: One `DemoData` per corpus

**Files:** Modify `demo/ignite/corpus.py`, `demo/ignite/questions.yaml`, `demo/ignite/build.py`, `demo/ignite/run_side_by_side.py`; tests `test_corpus.py`, `helpers.py`, `test_build.py`, `test_runner.py`.

**Interfaces:**
- `Question` gains `set: Literal["musique", "wands", "wands-match"]` and `corpus: Literal["musique", "wands"]`; `musique_id` becomes `ref: str` (MuSiQue id, WANDS query id, or match rule id).
- `questions.yaml` entries: `{id, set, ref}`. Existing MuSiQue entries get `set: musique`.
- `load_demo(*, corpora: set[str] | None = None, musique_path: Path | None = None, wands_folder: Path | None = None, questions_path: Path = QUESTIONS_PATH, match_path: Path = MATCH_PATH, labels_path: Path = LABELS_PATH) -> dict[str, DemoData]`. Loads only the corpora asked for (default: both). WANDS questions use `sample_wands` (the seeded 60; ref must be in it) with gold = Exact (gain 2) ids; match questions use `sample_wands_match`.
- `build_demo_harness(cfg, *, data: DemoData, ...)` (now required).
- Runner: `--question <id>` loads the question's corpus only.

- [ ] **Step 1: Failing tests:** `test_questions_come_from_the_musique_sample` (updated to `ref`), `test_wands_question_gold_is_exact_labels`, `test_match_question_gold_is_the_rule_labels`, `test_only_requested_corpora_load` (no WANDS read when `corpora={"musique"}`), `test_duplicate_question_id_fails`. Fixtures: the existing MuSiQue fixture plus a tiny WANDS folder (`product.csv`, `query.csv`, `label.csv`).
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement**; update helpers and the existing tests to the new shapes.
- [ ] **Step 4: Run** `uv run pytest demo/ignite/tests -q`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(demo): load the demo per corpus; questions from three sets`.

### Task 6: Server and viewer for three sets

**Files:** Modify `demo/ignite/serve.py`, `demo/ignite/viewer/index.html`; tests `test_serve.py`, `viewer.test.mjs`.

**Interfaces:**
- Profiles: `ignite` (musique), `ignite-wands` (wands); `<profile>-typesafe` for each when `TYPESAFE_API_KEY` is set.
- `PROFILE_OF = {"musique": "ignite", "wands": "ignite-wands"}`. `/ignite/questions` rows: `{id, text, set, profile}` (no gold).
- Viewer core: `SETS = ["musique", "wands", "wands-match"]` with labels `MuSiQue (multi-hop Wikipedia)`, `WANDS (product search)`, `WANDS match (our questions)`; `optionsHtml(questions) -> string` with one `<optgroup>` per set, in `SETS` order; live runs post `q.profile`.

- [ ] **Step 1: Failing tests:** `test_questions_endpoint_gives_set_and_profile`; viewer `optionsHtml groups questions by set in order` and `live run posts the question's profile` (pure function `requestBody(q, mode)`).
- [ ] **Step 2: Run** `uv run pytest demo/ignite/tests -q`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** `feat(demo): serve and view all three question sets`.

### Task 7: Pick, rehearse, write up

- [ ] **Step 1: Scan** (one run per mode) every WANDS match question, and the first 10 of the seeded 60 WANDS queries. Pick: one WANDS match hero (harness clearly ahead), 2 WANDS queries (a tie, and one where harness costs more for nothing). Add them to `questions.yaml` with a comment saying they were picked from a scan.
- [ ] **Step 2: Rehearse** the WANDS match hero 5 times; save at least 2 clean backups and 1 WANDS backup with `run_side_by_side --save`. Check one replay per set in a browser (no page errors).
- [ ] **Step 3: README:** three-part story for the talk; a short section per set; WANDS match rules and how to re-run (`select`, `labels`, `eval_public --dataset wands-match`); the pick disclosure; backups; start-up time. `tuning-log.md`: round 10 with the rehearsal numbers.
- [ ] **Step 4: Full checks:** `uv run pytest -q`, `uv run pytest demo/ignite/tests -q`, `uv run ruff check`. Commit `docs(demo): WANDS parts of the Ignite demo`.
