# Swarm Validation Part A (wands-multi) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A frozen 20-question `wands-multi` eval set whose parts do not depend on each other,
two part-level scores, a `swarm-4x` control mode, and one live run of all four modes with a
written verdict.

**Architecture:** `demo/ignite/public/wands_multi.py` picks WANDS queries by a seeded rule and
writes `demo/ignite/wands_multi.yaml`; its loader returns a `PublicSample` with a new `parts`
field. `demo/ignite/eval.py` keeps each run's top 30 and scores part coverage@10 and part
recall@30 when a sample has parts; it also prints a parts table and the verdict.
`eval_public.py` wires the dataset and the `swarm-4x` mode.

**Tech Stack:** Python 3.12, pytest, pydantic, PyYAML, the existing `agentic_search` eval runner
(`run_eval`), `uv run`.

**Spec:** `docs/superpowers/specs/2026-11-23-swarm-wide-recall-design.md`

## Global Constraints

- Gold comes from WANDS human labels only (`label.csv`): part gold = that query's **Exact**
  products; nDCG qrels = union of the parts' graded labels (Exact = 2, Partial = 1, as
  `WANDS_GAIN`). No model touches gold or question text.
- The set is built once with seed 7 and committed before any mode runs. Do not re-roll after
  results.
- Picking rules: parts in a question have distinct, non-empty `query_class`; no class in
  `NOT_PRODUCT_TYPES` (`wands_match.py`); each part has ≥ 3 Exact products; no WANDS query is
  used in two questions; 3 or 4 parts per question; 20 questions.
- Question text template, exactly: `I'm shopping for a few things: {p1}; {p2}; {p3}.` (parts
  joined by `"; "`, WANDS query text unchanged, trailing period).
- Part coverage@10 = share of parts with ≥ 1 of their Exact products in the top 10.
- Part recall@30 = mean over parts of min(1, |part Exact ∩ top 30| ÷ min(|part Exact|,
  ⌊30 ÷ number of parts⌋)). A product that is Exact for several parts counts for each.
- `swarm-4x` = the swarm with total budget `max_tool_calls` 96 and `max_cost_usd` 2.00;
  and `agent_max_turns` 4 (AJ, 2026-11-23); `max_seconds` stays 120. Eval only.
- Swarm modes (`swarm`, `swarm-4x`) run at concurrency 1; others at 2 (as round 15).
- Verdict, checked in this order, gap = swarm coverage − harness coverage (absolute):
  1. `works`: gap ≥ 0.10 and swarm part recall@30 ≥ harness part recall@30.
  2. `spend`: rule 1 fails for `swarm` but passes for `swarm-4x`.
  3. `unclear`: gap ≥ 0.05.
  4. `no`: otherwise.
- No change to `src/agentic_search/`, the viewer, or existing datasets' results.
- Commit trailer: `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_01PLBefd2n2rWWiFSuokjSAN`.

## Review Focus

1. A WANDS query with an empty `query_class` would make "distinct classes" pass wrongly → it is
   never picked (test in Task 1).
2. A run that errored has no hits → its coverage and part recall are 0, not skipped (Task 2).
3. A question with 4 parts uses ⌊30 ÷ 4⌋ = 7 as the per-part cap; a part with 2 Exact uses 2
   (Task 2).
4. Old datasets (no `parts`) must produce byte-identical tables to today: the parts table and
   verdict only appear when parts exist (Task 2).
5. `swarm-4x` must not change `config.yaml` or the shared cfg dict other modes use (Task 3).

---

### Task 1: Build and freeze the wands-multi set

**Files:**
- Create: `demo/ignite/public/wands_multi.py`
- Modify: `demo/ignite/public/datasets.py` (`PublicSample`: add field)
- Create (generated, committed): `demo/ignite/wands_multi.yaml`
- Test: `demo/ignite/tests/test_wands_multi.py`

**Interfaces:**
- Produces:
  - `PublicSample.parts: dict[str, dict[str, list[str]]] = field(default_factory=dict)` —
    question id → part id (the WANDS query id, as str) → Exact doc ids (`"wands-<product_id>"`).
  - `MULTI_PATH = DEMO_DIR / "wands_multi.yaml"`
  - `class MultiQuestion(BaseModel): id: str; parts: list[str]; text: str; gold: dict[str, list[str]]`
  - `select_questions(queries: dict[str, tuple[str, str]], exact: dict[str, list[str]], *, n: int = 20, seed: int = 7, min_exact: int = 3) -> list[MultiQuestion]`
    — `queries`: query id → (text, query_class); `exact`: query id → Exact doc ids (sorted).
    Ids `m01`…`m20`. Part count per question: `rng.choice([3, 4])`.
  - `question_text(part_texts: list[str]) -> str`
  - `sample_wands_multi(folder: Path, path: Path = MULTI_PATH, *, n: int | None = None) -> PublicSample`
    — `n` keeps the first `n` questions (for a smoke run); name
    `"wands-multi"`, source `"products"`, description `WANDS_DESCRIPTION`, whole catalogue as
    documents, qrels = union of parts' graded labels (max grade per doc), `groups={"all": ids}`,
    `parts` from the YAML gold.
  - CLI: `uv run python -m demo.ignite.public.wands_multi --seed 7 --n 20 > demo/ignite/wands_multi.yaml`

- [ ] **Step 1: Write the failing tests** in `test_wands_multi.py` with a hand-made toy
  `queries`/`exact` (about 12 queries, a few sharing a class, one with class `""`, one with class
  `"Licensed Products"`, one with only 2 Exact):

```python
def test_question_text_template():
    assert question_text(["teak chair", "patio umbrella", "outdoor rug"]) == \
        "I'm shopping for a few things: teak chair; patio umbrella; outdoor rug."

def test_select_rules(toy):
    qs = select_questions(*toy, n=3, seed=7)
    used = [p for q in qs for p in q.parts]
    assert len(used) == len(set(used))                       # no reuse
    for q in qs:
        classes = [toy[0][p][1] for p in q.parts]
        assert len(set(classes)) == len(classes)             # distinct classes
        assert "" not in classes and "Licensed Products" not in classes
        assert 3 <= len(q.parts) <= 4
        assert all(len(toy[1][p]) >= 3 for p in q.parts)
        assert q.gold == {p: toy[1][p] for p in q.parts}
        assert q.text == question_text([toy[0][p][0] for p in q.parts])

def test_select_is_deterministic(toy):
    assert select_questions(*toy, n=3, seed=7) == select_questions(*toy, n=3, seed=7)

def test_sample_loader_unions_graded_qrels(tmp_path):
    # tiny label.csv / query.csv / product.csv in tmp_path + a 1-question YAML with 2 parts,
    # where product P is Exact for part A and Partial for part B
    s = sample_wands_multi(tmp_path, tmp_path / "multi.yaml")
    assert s.qrels["m01"]["wands-P"] == 2
    assert s.parts["m01"] == {"A": [...Exact of A...], "B": [...Exact of B...]}
    assert s.groups == {"all": ["m01"]}

def test_frozen_yaml_matches_rules():
    # reads the committed demo/ignite/wands_multi.yaml (skip if the WANDS cache is missing)
    qs = [MultiQuestion(**q) for q in yaml.safe_load(MULTI_PATH.read_text())]
    assert len(qs) == 20 and [q.id for q in qs] == [f"m{i:02d}" for i in range(1, 21)]
```

- [ ] **Step 2: Run** `uv run pytest demo/ignite/tests/test_wands_multi.py -v` — expect FAIL
  (module missing).
- [ ] **Step 3: Implement** `wands_multi.py` and the `PublicSample.parts` field. Pick by
  shuffling eligible query ids with `random.Random(seed)` and filling each question greedily with
  the next ids whose class is not yet in it; stop at `n` questions. Read files with `_tsv`, build
  docs with `_product_doc`, as `sample_wands` does. YAML: one mapping per question, gold lists in
  flow style.
- [ ] **Step 4: Generate the set**: run the CLI above on the cached WANDS files
  (`demo/ignite/public/.cache/wands/`). Check by eye: 20 questions, readable text.
- [ ] **Step 5: Run** `uv run pytest demo/ignite/tests -q` — expect all pass.
- [ ] **Step 6: Commit** `wands_multi.py`, `datasets.py`, `wands_multi.yaml`, the test:
  `feat(ignite): wands-multi eval set (frozen, seed 7)`.

---

### Task 2: Part scores, parts table and verdict

**Files:**
- Modify: `demo/ignite/eval.py` (`TurnRecorder`, `KindReport`, `evaluate_groups`; new functions)
- Test: `demo/ignite/tests/test_eval.py`

**Interfaces:**
- Consumes: `PublicSample.parts` shape from Task 1.
- Produces:
  - `TurnRecorder.ranked: dict[str, list[str]]` — question text → first 30 doc ids. Keep
    `top10` as is.
  - `KindReport.coverage: list[float]`, `KindReport.part_recall: list[float]` (default empty).
  - `part_coverage(ranked: list[str], parts: dict[str, list[str]], k: int = 10) -> float`
  - `part_recall(ranked: list[str], parts: dict[str, list[str]], k: int = 30) -> float`
  - `evaluate_groups(..., parts: dict[str, dict[str, list[str]]] | None = None)` — when given,
    fills the two lists per run (an errored or missing run scores 0). Swarm concurrency 1 when
    `mode.startswith("swarm")`.
  - `parts_table(reports) -> list[str]` — markdown: `| mode | part coverage@10 | part recall@30 | nDCG@10 | avg cost | avg seconds | avg tool calls | n | errors |`.
  - `verdict(reports, group: str = "all") -> tuple[str, str]` — (code, one plain sentence with
    the numbers); code ∈ `works | spend | unclear | no`. Needs `harness` and `swarm` rows;
    `swarm-4x` optional.

- [ ] **Step 1: Write the failing tests:**

```python
P = {"a": ["a1", "a2", "a3"], "b": ["b1", "b2"], "c": ["c1"] * 1, "d": [f"d{i}" for i in range(20)]}

def test_part_coverage():
    assert part_coverage(["x", "a2", "b1"] + ["x"] * 7 + ["c1"], P) == 0.5   # c1 is rank 11
    assert part_coverage([], P) == 0.0

def test_part_recall_caps():
    # 4 parts -> per-part cap 7; part d has 20 Exact, 7 found -> 1.0; b has 2 Exact, 1 found -> 0.5
    ranked = ["b1"] + [f"d{i}" for i in range(7)]
    assert part_recall(ranked, P) == pytest.approx((0 + 0.5 + 0 + 1.0) / 4)

def test_shared_product_counts_for_each_part():
    assert part_coverage(["s"], {"a": ["s"], "b": ["s"]}) == 1.0

def test_verdict_order():
    # build KindReports with given coverage/part_recall lists
    assert verdict(rep(harness=(0.5, 0.4), swarm=(0.62, 0.4)))[0] == "works"
    assert verdict(rep(harness=(0.5, 0.4), swarm=(0.62, 0.3), **{"swarm-4x": (0.7, 0.5)}))[0] == "spend"
    assert verdict(rep(harness=(0.5, 0.4), swarm=(0.62, 0.3)))[0] == "unclear"
    assert verdict(rep(harness=(0.5, 0.4), swarm=(0.56, 0.5)))[0] == "unclear"
    assert verdict(rep(harness=(0.5, 0.4), swarm=(0.54, 0.9)))[0] == "no"

async def test_evaluate_groups_scores_parts_and_errors():
    # fake harness (tests/helpers.py style) returning fixed hits for q1 and raising for q2
    # -> coverage == [expected, 0.0], part_recall == [expected, 0.0]

async def test_old_datasets_unchanged():
    # evaluate_groups without parts -> coverage == part_recall == []; results_table output
    # identical to a fixture captured before the change
```

- [ ] **Step 2: Run** `uv run pytest demo/ignite/tests/test_eval.py -v` — expect FAIL.
- [ ] **Step 3: Implement** in `eval.py`. Use floating-point math as written in Global
  Constraints; compare thresholds with a 1e-9 tolerance so 0.10 exactly passes.
- [ ] **Step 4: Run** `uv run pytest demo/ignite/tests -q` — expect all pass.
- [ ] **Step 5: Commit** `feat(ignite): part coverage, part recall, verdict`.

---

### Task 3: Wire wands-multi and swarm-4x into eval_public

**Files:**
- Modify: `demo/ignite/public/eval_public.py`
- Test: `demo/ignite/tests/test_public_eval.py`

**Interfaces:**
- Consumes: `sample_wands_multi`, `MULTI_PATH` (Task 1); `parts_table`, `verdict`,
  `evaluate_groups(parts=...)` (Task 2); `build_demo_swarm`, `build_harness_for`, `load_config`.
- Produces:
  - `a deep copy with
    `budget.max_tool_calls` and `budget.max_cost_usd` times `factor`; signature `scaled_budget_cfg(cfg: dict, factor: int, agent_max_turns: int) -> dict` sets
    `swarm.agent_max_turns`; called with `(cfg, 4, 4)`. `max_seconds` and `budget.max_turns`
    unchanged.
  - `_harness_for` handles `"swarm-4x"`: `build_harness_for(scaled_budget_cfg(cfg, 4, 4), ...)`
    then `build_demo_swarm`.
  - `--dataset wands-multi` passes `--n` through to `sample_wands_multi` (settings `{"questions": "wands_multi.yaml"}`; hashes
    `product.csv`, `label.csv`, `wands_multi.yaml`); `ABOUT["wands-multi"]` explains the set,
    the two scores and the pass mark in plain words.
  - For wands-multi, `write_results` passes `sample.parts`, appends `parts_table`, then a
    `## Verdict` section with `verdict(...)`'s sentence. Footer names `swarm-4x` and says both
    swarm modes ran at concurrency 1.

- [ ] **Step 1: Write the failing tests:**

```python
def test_scaled_budget_cfg_does_not_mutate():
    cfg = {"budget": {"max_turns": 4, "max_tool_calls": 24, "max_seconds": 120,
                      "max_cost_usd": 0.50}, "swarm": {"agent_max_turns": 2}}
    out = scaled_budget_cfg(cfg, 4, 4)
    assert out["budget"] == {"max_turns": 4, "max_tool_calls": 96, "max_seconds": 120,
                             "max_cost_usd": 2.00}
    assert out["swarm"] == {"agent_max_turns": 4}
    assert cfg["budget"]["max_tool_calls"] == 24

def test_parser_accepts_wands_multi():
    assert build_parser().parse_args(["--dataset", "wands-multi"]).dataset == "wands-multi"

async def test_write_results_wands_multi_has_parts_and_verdict(tmp_path):
    # tiny PublicSample with parts + fake harness_for for modes harness, swarm, swarm-4x
    text = await write_results(sample, ["harness", "swarm", "swarm-4x"], tmp_path / "r.md",
                               settings={}, hashes={}, harness_for=fake)
    assert "part coverage@10" in text and "## Verdict" in text and "swarm-4x" in text
```

- [ ] **Step 2: Run** `uv run pytest demo/ignite/tests/test_public_eval.py -v` — expect FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `uv run pytest -q` (whole repo) and `node --test demo/ignite/tests/viewer.test.mjs`
  — expect all pass (pytest was 691 passed, 5 skipped before this plan, plus the new tests).
- [ ] **Step 5: Commit** `feat(ignite): wands-multi dataset and swarm-4x control in evals`.

---

### Task 4: Live run, results and tuning log (controller runs this, not a subagent)

**Files:**
- Create: `demo/ignite/results-wands-multi.md` (written by the run)
- Modify: `demo/ignite/tuning-log.md` (round 16), `demo/ignite/README.md` (one line + link)

- [ ] **Step 1: Smoke run** 2 questions, all four modes, to check cost and errors:
  `uv run python -m demo.ignite.public.eval_public --dataset wands-multi --modes retrieval,harness,swarm,swarm-4x`
  plus `--n 2` and `--out` pointing to the scratchpad (do not overwrite the real results file;
  add `--out` to the parser in this step if Task 3 did not). Expected: no errors; cost per question in line with $0.02–0.30.
- [ ] **Step 2: Full run**, in the background, nothing else running against the model:
  `uv run python -m demo.ignite.public.eval_public --dataset wands-multi --modes retrieval,harness,swarm,swarm-4x`
  Expected: `results-wands-multi.md` written, 0 errors per mode (if errors > 0, report them;
  do not re-run selected questions).
- [ ] **Step 3: Write tuning-log round 16**: what was built, the table, the verdict sentence,
  real spend (avg cost × n per mode), and what it means for part B. Say plainly if the swarm lost.
- [ ] **Step 4: README**: one line under the eval section pointing to `results-wands-multi.md`.
- [ ] **Step 5: Commit** `eval(ignite): wands-multi results and round 16`.
