# Step Tracking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the harness loop work in steps for questions that need them: a step plan, a notebook of facts with sources, opened documents shown to the planner, and a stop rule that counts step progress. Off by default.

**Architecture:** Three "workspace" tools (`plan_steps`, `note`, `step_done`) are offered to the planner alongside the search tools when `HarnessSettings.steps` is on. The `Planner` role applies them to new `SearchState.steps` / `SearchState.notebook` fields and passes only the search calls on to the executor, so no driver or decider protocol changes beyond optional fields, and no extra model call. The planner view, the digest and the controller read the steps and notebook.

**Tech Stack:** Python 3.12, pydantic, pytest (offline: `agentic_search.testing.ScriptedDriver`, `KeywordJudge`), TypeScript client (vitest).

**Spec:** `docs/superpowers/specs/2026-10-30-step-tracking-design.md`

## Global Constraints

- `HarnessSettings.steps` defaults to `False`; with it off, behaviour, prompts, tools and events are unchanged (spec §5).
- Loop mode only; delegate mode ignores the setting (spec §9.2).
- `Decider.judge` unchanged; relevance still means "relevant to the whole question" and alone decides returned results (spec §9.1).
- Protocol changes are optional fields with defaults (`PlannerView`, `ControllerView`, `TurnSummary`) (spec §9.3).
- No extra model call: the plan comes back as tool calls in the normal planning call (spec §5, §8).
- Every notebook fact cites a document key present in the pool; otherwise it is dropped (spec §5).
- No prompt or rule mentions a dataset, wood tone or "features" (spec §5).
- Snippet length is a setting, default 240 (today's value) (spec §3.8).
- `uv run pytest -q`, `uv run ruff check`, `npm --prefix clients/typescript run typecheck` and `npm --prefix clients/typescript test` stay clean.
- Library work on `feat/step-tracking`; demo work (Task 7) on `demo/ignite-wands` rebased onto it; merge to `main` only after Task 8's numbers and AJ's go (spec §9.4).

## Review Focus

1. **A plan with no steps or too many** (`plan_steps` with `[]`, or 20 steps): `[]` clears nothing and adds no plan; more than 6 are cut to 6. Test: Task 2 `test_plan_steps_is_capped_and_empty_is_ignored`.
2. **`step_done` with a bad index** (-1, 99, a string): ignored and traced, never raises. Test: Task 2 `test_bad_step_done_is_ignored`.
3. **A huge or empty fact**: empty dropped; longer than 300 characters cut to 300. Test: Task 2 `test_fact_text_is_trimmed_and_empty_dropped`.
4. **Steps on, but the model never uses the workspace tools**: same calls, same stop reason as steps off. Test: Task 5 `test_steps_on_without_a_plan_behaves_like_off`.
5. **Budget runs out with a step open**: the budget stop wins. Test: Task 5 `test_budget_beats_an_open_step`.

---

### Task 1: Settings, types and state

**Files:** Modify `src/agentic_search/core/harness.py` (`HarnessSettings`, `Analyzer(...)` construction), `src/agentic_search/models/base.py`, `src/agentic_search/core/state.py`; tests `tests/core/test_harness.py`, `tests/roles/test_analyzer.py`.

**Interfaces (Produces):**
- `HarnessSettings`: `steps: bool = False`, `snippet_chars: int = Field(240, ge=40)`, `fetch_chars: int = Field(1500, ge=100)`.
- `models/base.py`: `class Step(BaseModel): text: str; done: bool = False`; `class Fact(BaseModel): text: str; source_key: str; turn: int`.
- `PlannerView`: `steps: list[Step] = []`, `notebook: list[Fact] = []`. `ControllerView`: same two fields. `TurnSummary`: `n_facts: int = 0`, `n_steps_done: int = 0`.
- `SearchState`: `steps: list[Step]`, `notebook: list[Fact]` (empty defaults); `current_step(self) -> int | None` (index of the first step not done, or None).
- `Analyzer(..., snippet_chars=s.snippet_chars)` from settings.

- [ ] **Step 1: Failing tests:** `test_step_settings_default_off` (`HarnessSettings().steps is False`, `snippet_chars == 240`, `fetch_chars == 1500`); `test_snippet_chars_comes_from_settings` (harness with `snippet_chars=80` → digest lines ≤ 80 chars of body); `test_current_step_is_first_open` (`[done, open, open]` → 1; all done → None; empty → None).
- [ ] **Step 2: Run** `uv run pytest tests/core tests/roles -q`. Expected: the new tests FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `uv run pytest -q`. Expected: all PASS.
- [ ] **Step 5: Commit** `feat: step-tracking settings, types and state`.

### Task 2: Workspace tools

**Files:** Create `src/agentic_search/roles/workspace.py`; modify `src/agentic_search/roles/planner.py`, `src/agentic_search/core/harness.py` (`_run_loop`); test `tests/roles/test_workspace.py`.

**Interfaces (Produces):**
- `WORKSPACE_TOOLS: list[ToolSpec]`:
  - `plan_steps(steps: list[str])`: "Set the plan for a question that needs several steps, where a later search depends on something read earlier. Call once, early. Skip it for one-step questions."
  - `note(fact: str, source_key: str)`: "Record a fact read in a document, for use in later searches. source_key is the document's `source:doc_id`."
  - `step_done(step: int)`: "Mark step N (1-based) as done."
- `MAX_STEPS = 6`, `MAX_FACT_CHARS = 300`, `WORKSPACE_NAMES = {t.name for t in WORKSPACE_TOOLS}`.
- `@dataclass class WorkspaceUpdate: n_facts: int = 0; n_steps_done: int = 0; changed: bool = False`.
- `apply_workspace(state: SearchState, calls: list[ToolCall]) -> tuple[list[ToolCall], WorkspaceUpdate]`: returns the non-workspace calls unchanged and in order; applies workspace calls; drops bad ones with a `state.trace.add("workspace_dropped", ...)` entry (reason); `plan_steps` is accepted only while `state.steps` is empty.
- `Planner(..., steps: bool = False)`: when on, adds `WORKSPACE_TOOLS` to the tools sent to the driver and fills `view.steps` / `view.notebook`; returns `(PlanResult with search calls only, WorkspaceUpdate)` via a new method `plan_with_workspace(state, tools)`; `plan()` unchanged for callers.
- `_run_loop`: uses `plan_with_workspace` when `settings.steps`; workspace calls never count toward `usage.tool_calls` or `max_calls_per_turn`; if a turn yields no search calls but `update.changed` and a step is open, plan once more in the same turn before returning `NO_PLAN`; `TurnSummary` gets `n_facts` / `n_steps_done` from the update.

- [ ] **Step 1: Failing tests** (ScriptedDriver turns that mix `call("plan_steps", ...)`, `call("note", ...)`, `call("step_done", ...)` with `lexical_search`):
  - `test_steps_off_offers_no_workspace_tools` (driver's `tools_seen` has none of `WORKSPACE_NAMES`).
  - `test_plan_steps_sets_the_plan` (`state.steps` texts equal the call's list; a second `plan_steps` later is dropped).
  - `test_plan_steps_is_capped_and_empty_is_ignored`.
  - `test_note_needs_a_known_source` (key not in pool → dropped; key in pool → `Fact` with that turn).
  - `test_fact_text_is_trimmed_and_empty_dropped`.
  - `test_bad_step_done_is_ignored`.
  - `test_workspace_calls_do_not_count_as_tool_calls` (usage.tool_calls equals search calls only).
  - `test_turn_with_only_workspace_calls_replans_once` (turn script: `[plan_steps]`, then `[lexical_search]` → the search runs in turn 0).
  - `test_delegate_mode_ignores_steps`.
- [ ] **Step 2: Run** `uv run pytest tests/roles/test_workspace.py -q`. Expected: FAIL (module missing).
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `uv run pytest -q`. Expected: all PASS.
- [ ] **Step 5: Commit** `feat: workspace tools for step plans and notes`.

### Task 3: The planner sees the plan and the notebook

**Files:** Modify `src/agentic_search/models/driver.py`; test `tests/models/test_driver.py`.

**Interfaces:** `render_planner_view` adds, only when non-empty, `## Steps` (one line each: `1. [done] …` / `2. [current] …` / `3. …`) and `## Notebook` (`- fact (from source:doc_id)`), placed before `## Results so far`. `STEPS_GUIDE: str` appended to the system prompt only when a tool named `plan_steps` is in `tools`:

> Some questions need steps: first find one thing, read it, then search using what you read. For those, call plan_steps first, note each fact you read that a later step needs (with its document key), and mark steps done. Search for the current step; facts in the notebook are not answers unless they also answer the question. Documents you open with fetch are shown to you in full.

- [ ] **Step 1: Failing tests:** `test_view_renders_steps_and_notebook`; `test_view_without_steps_is_unchanged` (string equal to today's rendering for an empty plan); `test_steps_guide_only_with_workspace_tools`; `test_steps_guide_names_no_dataset` (no "wands", "wood", "feature", "musique", case-insensitive).
- [ ] **Step 2: Run** `uv run pytest tests/models -q`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `uv run pytest -q`. Expected: PASS.
- [ ] **Step 5: Commit** `feat: planner prompt and view show steps and notebook`.

### Task 4: Opened documents in the digest

**Files:** Modify `src/agentic_search/roles/analyzer.py`, `src/agentic_search/core/harness.py` (pass `steps`, `fetch_chars`); test `tests/roles/test_analyzer.py`.

**Interfaces:** `Analyzer(..., show_opened: bool = False, fetch_chars: int = 1500)`. When `show_opened`, `render_digest` appends `Opened documents:` with `- source:doc_id: <text, whitespace collapsed, cut to fetch_chars>` for every key returned by a `fetch` call this turn, new or not, relevant or not, with the source policy applied (withheld text stays withheld).

- [ ] **Step 1: Failing tests:** `test_fetched_known_document_is_shown_in_full_when_steps_on`; `test_opened_section_absent_when_steps_off`; `test_opened_text_respects_source_policy`; `test_opened_text_is_cut_to_fetch_chars`.
- [ ] **Step 2: Run** `uv run pytest tests/roles/test_analyzer.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `uv run pytest -q`. Expected: PASS.
- [ ] **Step 5: Commit** `feat: show opened documents to the planner when steps are on`.

### Task 5: Stop rule counts step progress

**Files:** Modify `src/agentic_search/roles/controller.py`, `src/agentic_search/core/harness.py`; test `tests/roles/test_controller.py`.

**Interfaces:** `Controller.heuristic(state)`: when `state.steps` has an open step, before the existing rules:
1. last turn `n_facts or n_steps_done` → `CONTINUE`, note `step {k}: {text}`;
2. the last two turns both had no new relevant results, no facts and no step done → `STOP`, note `stuck on step {k}`;
3. otherwise run the existing rules, but a `STOP` or `BROADEN` becomes `CONTINUE` with note `step {k}: {text}`.
With no plan or all steps done, the existing rules apply unchanged. `ControllerView` gets `steps` and `notebook` for model deciders. Budgets are checked before the controller, as today.

- [ ] **Step 1: Failing tests:** `test_fact_turn_continues_with_the_current_step`; `test_open_step_turns_stop_into_continue`; `test_two_idle_turns_stop_as_stuck`; `test_all_steps_done_uses_the_old_rules`; `test_steps_on_without_a_plan_behaves_like_off` (full harness run, same calls and stop reason as steps off); `test_budget_beats_an_open_step` (max_turns 2, plan of 3 steps → `BUDGET_TURNS`); `test_controller_view_carries_steps`.
- [ ] **Step 2: Run** `uv run pytest tests/roles/test_controller.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `uv run pytest -q`. Expected: PASS.
- [ ] **Step 5: Commit** `feat: the stop rule counts step progress`.

### Task 6: Events and TypeScript client

**Files:** Modify `src/agentic_search/events.py`, the workspace module (emit), `clients/typescript/src/events.ts`; tests `tests/test_events.py`, `clients/typescript/test/`.

**Interfaces:** `StepsUpdated(_Event)`: `type: Literal["steps_updated"]`, `steps: list[Step]`, `current: int | None` (0-based). `FactNoted(_Event)`: `type: Literal["fact_noted"]`, `fact: str`, `source_key: str`. Both added to the event union; emitted by `apply_workspace` (one `StepsUpdated` per turn in which steps changed, one `FactNoted` per kept fact). Never emitted with steps off. TS: matching interfaces in the `SearchEvent` union.

- [ ] **Step 1: Failing tests:** `test_steps_and_facts_stream_as_events` (SSE round trip through `parse_event`); `test_no_step_events_when_off`; a vitest case parsing both types.
- [ ] **Step 2: Run** `uv run pytest tests/test_events.py -q` and `npm --prefix clients/typescript test`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `uv run pytest -q`, `npm --prefix clients/typescript run typecheck`, `npm --prefix clients/typescript test`. Expected: PASS.
- [ ] **Step 5: Commit** `feat: steps_updated and fact_noted events`. Then `uv run ruff check`.

### Task 7: Demo picks it up (branch `demo/ignite-wands`)

**Files:** `demo/ignite/config.yaml` (`settings.steps: true`), `demo/ignite/run_side_by_side.py` (`format_event` prints `▸ step k: …` and `✎ fact (from key)`), `demo/ignite/viewer/index.html` (core `reduce` handles `steps_updated` / `fact_noted`; a steps strip with the current step marked and a notebook list under the decisions); tests in `demo/ignite/tests/`.

- [ ] **Step 1:** `git checkout demo/ignite-wands && git rebase feat/step-tracking`. Expected: clean (the branches touch different files). Leave the stash alone.
- [ ] **Step 2: Failing tests:** `test_config_turns_steps_on`; `test_format_event_shows_steps_and_facts`; viewer `reduce keeps steps and facts` and `stepsHtml marks the current step`.
- [ ] **Step 3: Implement.** **Step 4: Run** `uv run pytest demo/ignite/tests -q`. Expected: PASS. **Step 5: Commit** `feat(demo): show steps and facts`.

### Task 8: Measure (live, about $4) and stop for AJ

- [ ] Run `eval_public --dataset wands-match`, `--dataset musique`, `--dataset wands` with steps on (`IGNITE_GCP_PROJECT=gen-ai-accel-dev`). Record cost and time.
- [ ] Compare against spec §6: WANDS match P@10 goal ≥ 0.30 (was 0.050); MuSiQue recall@10 not below ~0.762 (was 0.792); WANDS nDCG@10 not below ~0.746 (was 0.776) and cost per query ≤ $0.0300 (was $0.0273).
- [ ] Commit the results files on the demo branch. **Stop and report to AJ.** No merge to `main`, and no change to the question sets, without AJ.
