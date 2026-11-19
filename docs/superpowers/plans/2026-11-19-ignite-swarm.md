# Ignite Swarm Lane Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a third `swarm` mode to the Ignite demo (split → parallel harness agents → RRF), shown live as a third viewer lane and scored in the public evals.

**Architecture:** `demo/ignite/swarm.py` wraps the existing shared `Harness`: one planner call splits the question, N concurrent `harness.stream(...)` calls run with a per-call divided `Budget`, and RRF merges their ranked keys. `serve.py` exposes it as SSE at `POST /ignite/swarm`; the eval runner treats a `Swarm` like a harness (`search` + async context manager). The viewer gets a third column fed by its own reducer. No change under `src/`.

**Tech Stack:** Python 3.12, asyncio, pydantic, FastAPI/Starlette SSE, pytest (async), plain JS viewer tested with `node --test`.

**Spec:** `docs/superpowers/specs/2026-11-19-ignite-swarm-design.md`

## Global Constraints

- No change under `src/agentic_search/` (spec, Non-goals).
- No external URLs in the viewer; `test_no_external_urls` must keep passing (demo runs offline).
- Same `budget` for all modes: `{max_turns: 4, max_tool_calls: 24, max_seconds: 120, max_cost_usd: 0.50}`.
- Swarm total spend = one harness budget: split cost first, then per agent `max_tool_calls = max(1, 24 // N)`, `max_cost_usd = (0.50 − split_cost) / N`, `max_turns = 2`, `max_seconds = 120` (wall clock for the whole swarm).
- `swarm:` config defaults: `max_agents: 4`, `agent_max_turns: 2`, `rrf_k: 60`. Split returns 3–6 sub-questions, capped at `max_agents`; fewer than 2 → fall back to one agent on the original question.
- Agent query text exactly: `"Main question: {question}. Your part: {part}."`
- RRF: `score(key) = Σ 1 / (rrf_k + rank)` with rank starting at 1; ties keep first-seen order.
- Gold marks hidden until a lane finishes (existing rule, applies to the swarm lane).
- Overtime warning at 120 s for all lanes.
- Existing retrieval/harness lanes and old two-lane replays must keep working.
- Run live evals one set at a time, never in parallel. GCP project `ajw-sandbox-497707` (`IGNITE_GCP_PROJECT`).
- Commits end with the attribution lines from the session reminder.

## Review Focus

1. **One agent fails or stalls** (Vertex timeout, `search_failed`): the swarm should still merge the other agents' hits and finish; the lane marks that agent failed. → test in Task 2.
2. **The 120 s cut hits mid-run**: agents still running are cancelled, their last `results_updated` hits are kept, and `swarm_finished` still arrives. → test in Task 2.
3. **Split model returns junk** (prose, fenced JSON, empty list, 9 items, duplicates): fenced JSON parses, junk falls back to one agent, more than `max_agents` are cut, duplicates removed. → tests in Task 1.
4. **Viewer client disconnects mid-run** (presenter switches question): all agent tasks are cancelled, no orphan Vertex calls. → test in Task 3.
5. **Replay of an old run with no `-swarm.jsonl`**: swarm column shows "No saved file for this mode." and the verdict still renders for the two modes present. → test in Task 5.

---

## File Structure

| File | Role |
|---|---|
| `demo/ignite/swarm.py` (new) | Split, budget division, RRF, `Swarm` (stream + search) |
| `demo/ignite/tests/test_swarm.py` (new) | Offline tests for `swarm.py` |
| `demo/ignite/config.yaml` | `max_seconds: 120`, `swarm:` block |
| `demo/ignite/build.py` | `build_demo_swarm(...)` |
| `demo/ignite/serve.py` | `POST /ignite/swarm` SSE endpoint |
| `demo/ignite/eval.py`, `demo/ignite/public/eval_public.py` | `swarm` mode |
| `demo/ignite/run_side_by_side.py` | run + `--save` swarm |
| `demo/ignite/viewer/index.html`, `demo/ignite/tests/viewer.test.mjs` | Third lane |
| `demo/ignite/README.md`, `tuning-log.md`, `results-*.md` | New numbers |

## Wire format (shared by Tasks 2, 3, 4, 5)

Swarm events are JSON objects with `type`. On SSE they are framed like the library's:
`event: <type>\ndata: <json>\n\n`. In replay files, one JSON object per line in `runs/<stamp>-<id>-swarm.jsonl`.

| type | fields |
|---|---|
| `swarm_started` | `question: str`, `max_agents: int`, `budget: {...}` (Budget JSON), `at_ms: float` |
| `swarm_split` | `sub_questions: list[str]`, `fallback: bool`, `usage: {input_tokens, output_tokens, cost_usd}`, `at_ms` |
| `agent_event` | `agent: int` (0-based), `event: {...}` (library event, rendered with `render_event(..., include_content=False, include_trace=<req>)` parsed back to a dict) |
| `swarm_finished` | `hits: list[{key, source, doc_id, title, snippet, score, agents: list[int]}]` (top `snapshot_k`, RRF score), `notes: list[str]`, `usage: {input_tokens, output_tokens, cost_usd, tool_calls, turns}`, `seconds: float`, `agent_status: list["done"|"failed"|"cut"]`, `at_ms` |
| `swarm_failed` | `error_type: str`, `message: str`, `at_ms` |

`at_ms` = ms since `swarm_started`. `usage.turns` = max over agents; `tool_calls`, tokens, cost = sums (cost includes split).

---

### Task 1: Pure swarm parts — split, budget, RRF

**Files:**
- Create: `demo/ignite/swarm.py`
- Test: `demo/ignite/tests/test_swarm.py`

**Interfaces:**
- Produces:
  - `class SwarmConfig(BaseModel)`: `max_agents: int = 4`, `agent_max_turns: int = 2`, `rrf_k: int = 60`
  - `class Split(BaseModel)`: `sub_questions: list[str]`, `fallback: bool`, `usage: ModelUsage`
  - `def parse_split(text: str, question: str, max_agents: int) -> tuple[list[str], bool]`
  - `async def split_question(client: Any, question: str, descriptions: dict[str, str], max_agents: int) -> Split`
  - `def agent_budget(total: Budget, n: int, split_cost: float, max_turns: int) -> Budget`
  - `def agent_query(question: str, part: str) -> str`
  - `def rrf(ranked: list[list[str]], k: int) -> list[tuple[str, float, list[int]]]` — (key, score, agent indexes that returned it), best first

- [ ] **Step 1: Write the failing tests**

```python
def test_parse_split_reads_fenced_json():
    subs, fb = parse_split('```json\n{"sub_questions": ["a?", "b?", "c?"]}\n```', "Q", 4)
    assert subs == ["a?", "b?", "c?"] and fb is False

def test_parse_split_caps_and_dedupes():
    subs, _ = parse_split('{"sub_questions": ["a","a","b","c","d","e","f"]}', "Q", 4)
    assert subs == ["a", "b", "c", "d"]

@pytest.mark.parametrize("text", ["no json here", '{"sub_questions": []}', '{"sub_questions": ["only"]}'])
def test_parse_split_falls_back_to_the_question(text):
    assert parse_split(text, "Q", 4) == (["Q"], True)

async def test_split_question_falls_back_when_the_model_errors():
    split = await split_question(RaisingClient(), "Q", {"wiki": "d"}, 4)
    assert split.sub_questions == ["Q"] and split.fallback and split.usage.cost_usd == 0

def test_agent_budget_divides_the_harness_budget():
    total = Budget(max_turns=4, max_tool_calls=24, max_seconds=120, max_cost_usd=0.50)
    b = agent_budget(total, 4, split_cost=0.02, max_turns=2)
    assert (b.max_turns, b.max_tool_calls, b.max_seconds) == (2, 6, 120)
    assert b.max_cost_usd == pytest.approx(0.12)

def test_agent_budget_gives_at_least_one_call():
    assert agent_budget(Budget(max_tool_calls=3), 6, 0.0, 2).max_tool_calls == 1

def test_agent_query_text():
    assert agent_query("Q", "P") == "Main question: Q. Your part: P."

def test_rrf_sums_reciprocal_ranks():
    out = rrf([["a", "b"], ["b", "c"]], k=60)
    assert [key for key, _, _ in out] == ["b", "a", "c"]
    assert out[0][1] == pytest.approx(1/62 + 1/61) and out[0][2] == [0, 1]

def test_rrf_ties_keep_first_seen_order():
    assert [k for k, _, _ in rrf([["x"], ["y"]], k=60)] == ["x", "y"]
```

`RaisingClient` is a tiny fake in the test file whose `chat` raises `TimeoutError`.

- [ ] **Step 2: Run, expect FAIL** — `uv run pytest demo/ignite/tests/test_swarm.py -v` → import error.

- [ ] **Step 3: Implement the interfaces above in `demo/ignite/swarm.py`**

`parse_split`: find the first `{...}` block (strip ``` fences), `json.loads`, take `sub_questions`, strip, drop empties and exact duplicates, cap at `max_agents`; fewer than 2 → `([question], True)`. `split_question`: one `client.chat(SPLIT_SYSTEM, [ChatMessage(role="user", content=...)], max_tokens=800)` (same call shape as `answer.py:answer_from`); prompt lists source descriptions and asks for 3–6 independent sub-questions as `{"sub_questions": [...]}`; any exception → fallback with zero usage. `agent_budget`: keep `max_seconds` and `max_tokens` from `total`; `max_cost_usd=None` stays `None`.

- [ ] **Step 4: Run, expect PASS** — same command.

- [ ] **Step 5: Commit** — `git add demo/ignite/swarm.py demo/ignite/tests/test_swarm.py && git commit -m "demo(swarm): split, budget division, RRF"`

---

### Task 2: `Swarm` — fan-out stream and search, plus config

**Files:**
- Modify: `demo/ignite/swarm.py`, `demo/ignite/config.yaml`, `demo/ignite/build.py`
- Test: `demo/ignite/tests/test_swarm.py`

**Interfaces:**
- Consumes: Task 1; `Harness.stream(question, *, top_k, budget, snapshot_k, include_content) -> SearchStream` (per-call `budget` is supported); events `ResultsUpdated.hits: list[HitSummary]`, `FactNoted.fact`, `UsageUpdated.usage: Usage`, `SearchFinished.result: SearchResult`, `SearchFailed`; `render_event` from `agentic_search.server.models`.
- Produces:
  - `class Swarm` with `__init__(self, harness: Harness, client: Any, cfg: SwarmConfig, descriptions: dict[str, str])`
  - `def stream(self, question: str, *, snapshot_k: int = 10, include_trace: bool = False) -> AsyncIterator[dict[str, Any]]` — yields wire-format events in order: `swarm_started`, `swarm_split`, interleaved `agent_event`s, then exactly one `swarm_finished` or `swarm_failed`.
  - `async def search(self, question: str, *, top_k: int = 20, mode: Any = None, **_: Any) -> SearchResult` — runs `stream`, returns `SearchResult(question=Query.of(question), hits=[RankedHit(hit=Hit(doc_id, source), score=rrf_score)...], stop_reason=StopReason.DELEGATE_DONE, usage=Usage(...), trace=Trace(), mode="swarm")`; raises `RuntimeError` on `swarm_failed`.
  - `async def __aenter__(self) -> Swarm` / `__aexit__` delegating to the harness.
  - `def build_demo_swarm(cfg: dict[str, Any], harness: Harness, descriptions: dict[str, str], *, client: Any = None) -> Swarm` in `build.py` (client defaults to `make_client(cfg["driver"])`).
  - `config.yaml`: `budget.max_seconds: 120`; `swarm: {max_agents: 4, agent_max_turns: 2, rrf_k: 60}` with a comment pointing at the spec.

- [ ] **Step 1: Write the failing tests** (offline: `offline_harness()` from `tests/helpers.py`, a `FakeSplitClient` returning fixed JSON with cost 0.01)

```python
async def test_stream_event_order_and_merge():
    evs = [e async for e in make_swarm(subs=["a", "b", "c"]).stream("Q")]
    types = [e["type"] for e in evs]
    assert types[0:2] == ["swarm_started", "swarm_split"] and types[-1] == "swarm_finished"
    assert {e["agent"] for e in evs if e["type"] == "agent_event"} == {0, 1, 2}
    fin = evs[-1]
    assert fin["agent_status"] == ["done", "done", "done"] and len(fin["hits"]) <= 10
    assert fin["usage"]["cost_usd"] >= 0.01   # split cost included

async def test_each_agent_gets_the_divided_budget_and_its_query(spy_harness):
    [e async for e in make_swarm(subs=["a", "b", "c", "d"], harness=spy_harness).stream("Q")]
    assert {c.budget.max_tool_calls for c in spy_harness.calls} == {6}
    assert spy_harness.calls[0].question == "Main question: Q. Your part: a."

async def test_one_failed_agent_does_not_sink_the_swarm():
    fin = (await collect(make_swarm(subs=["a", "b"], fail_agent=1).stream("Q")))[-1]
    assert fin["type"] == "swarm_finished" and fin["agent_status"] == ["done", "failed"]

async def test_time_cut_keeps_partial_hits():
    fin = (await collect(make_swarm(subs=["a", "b"], stall_agent=1, max_seconds=0.5).stream("Q")))[-1]
    assert fin["agent_status"][1] == "cut" and fin["hits"]

async def test_search_returns_a_search_result_for_the_eval_runner():
    res = await make_swarm(subs=["a", "b"]).search("Q", top_k=10, mode="swarm")
    assert res.mode == "swarm" and res.hits and res.usage.turns <= 2

def test_config_has_the_swarm_block_and_120s():
    cfg = load_config()
    assert cfg["budget"]["max_seconds"] == 120 and cfg["swarm"] == {"max_agents": 4, "agent_max_turns": 2, "rrf_k": 60}
```

`spy_harness`, `fail_agent`, `stall_agent` are test-file fakes: a `Harness` subclass or wrapper whose `stream` records `(question, budget)` and, for the chosen agent, yields a `search_failed` / sleeps after one `results_updated`. `make_swarm` / `collect` are small helpers in the test file.

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: Implement `Swarm`, `build_demo_swarm`, config change**

One `asyncio.Task` per agent pumping its `harness.stream(...)` into a shared `asyncio.Queue` as `agent_event`s; track per agent last `results_updated` keys, notes from `fact_noted`, last `usage_updated`, final status. Whole run under `asyncio.timeout(budget.max_seconds)`; on timeout cancel unfinished tasks (status `cut`) and merge what exists. Agent ranked list = keys from its `search_finished.result.hits` if finished, else last `results_updated.hits`. On generator close (`aclose`/cancel), cancel all agent tasks in `finally`. Unexpected error before split → `swarm_failed`.

- [ ] **Step 4: Run, expect PASS** — `uv run pytest demo/ignite/tests -q` (whole demo suite; `test_build` must still pass with 120 s).

- [ ] **Step 5: Commit** — `git commit -m "demo(swarm): parallel agents with fair budget; 120 s for all modes"`

---

### Task 3: `POST /ignite/swarm` SSE endpoint

**Files:**
- Modify: `demo/ignite/serve.py`
- Test: `demo/ignite/tests/test_serve.py`

**Interfaces:**
- Consumes: `Swarm.stream` (Task 2), `build_demo_swarm` (Task 2).
- Produces: `create_demo_app(..., swarms: dict[str, Swarm] | None = None)`; endpoint `POST /ignite/swarm` with body `SwarmRequest(profile: str, question: str, snapshot_k: int = 10, include_trace: bool = False)`; 404 for unknown profile; response `text/event-stream`, frames `event: <type>\ndata: <json>\n\n`. `main()` builds one swarm per `PROFILE_OF` profile.

- [ ] **Step 1: Write the failing tests**

```python
async def test_swarm_stream_returns_sse(swarm_client):
    r = await swarm_client.post("/ignite/swarm", json={"profile": "ignite", "question": "Q"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    assert "event: swarm_split" in r.text and r.text.rstrip().split("event: ")[-1].startswith("swarm_finished")

async def test_swarm_unknown_profile_is_404(swarm_client):
    assert (await swarm_client.post("/ignite/swarm", json={"profile": "nope", "question": "Q"})).status_code == 404

async def test_swarm_disconnect_cancels_agents(swarm_app_with_stalling_agents):
    ...  # open the stream, read the first frame, close; assert every agent task is cancelled within 1 s
```

- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement** — `StreamingResponse` over an async generator that frames each event; watch `request.is_disconnected()` (or reuse the `receive` pattern from `agentic_search.server.sse`) and `aclose()` the swarm stream on disconnect.
- [ ] **Step 4: Run, expect PASS** — `uv run pytest demo/ignite/tests/test_serve.py -v`.
- [ ] **Step 5: Commit** — `git commit -m "demo(swarm): SSE endpoint"`

---

### Task 4: Swarm in evals and side-by-side runs

**Files:**
- Modify: `demo/ignite/eval.py`, `demo/ignite/public/eval_public.py`, `demo/ignite/run_side_by_side.py`
- Test: `demo/ignite/tests/test_eval.py`, `demo/ignite/tests/test_public_eval.py`, `demo/ignite/tests/test_runner.py`

**Interfaces:**
- Consumes: `Swarm.search`, `Swarm` async context manager, `Swarm.stream`.
- Produces: `evaluate_groups` accepts `harness_for` returning `Harness | Swarm`; for mode `swarm` it uses `concurrency=1` (8 agents at once overloads the shared model — 2026-11-10 lesson); `eval_public.py --modes` accepts `swarm`; `run_side_by_side.py --modes retrieval,harness,swarm` (default all three) and `--save` writes `runs/<stamp>-<id>-swarm.jsonl` (one wire event per line).

- [ ] **Step 1: Write the failing tests**

```python
async def test_evaluate_groups_runs_a_swarm_mode(...):
    reports = await evaluate_groups(["harness", "swarm"], harness_for, dataset, {"all": qids}, name="t")
    assert ("swarm", "all") in reports and reports[("swarm", "all")].turns

def test_eval_public_accepts_swarm_mode():
    assert build_parser().parse_args(["--dataset", "musique", "--modes", "harness,swarm"]).modes == "harness,swarm"

async def test_side_by_side_saves_a_swarm_file(tmp_path, ...):
    ...  # run hero offline with --save into tmp RUNS_DIR; a *-swarm.jsonl exists, first line type swarm_started, last swarm_finished
```

- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement.** The results-file intro line in `eval_public.py` gets one sentence: "`swarm`: one planner call splits the question; up to 4 harness agents run at the same time on the parts with the harness budget divided between them; RRF merges their lists."
- [ ] **Step 4: Run, expect PASS** — `uv run pytest demo/ignite/tests -q`.
- [ ] **Step 5: Commit** — `git commit -m "demo(swarm): eval mode and saved runs"`

---

### Task 5: Viewer third lane

**Files:**
- Modify: `demo/ignite/viewer/index.html`
- Test: `demo/ignite/tests/viewer.test.mjs`, `demo/ignite/tests/test_viewer_static.py`

**Interfaces:**
- Consumes: wire format above; existing core functions `reduce`, `initialState`, `parseSse`, `parseJsonl`, `assignFiles`, `overtime`, `verdictHtml`.
- Produces (inside `/* <core> */`, exported to the test VM):
  - `MODES = ["retrieval", "harness", "swarm"]`
  - `initialSwarm() -> {mode: "swarm", subs: [], agents: [], hits: [], notes: [], usage: null, seconds: null, done, error, fallback}` where each agent is `initialState("harness")` plus `status`
  - `reduceSwarm(state, ev) -> state` — `agent_event` folds `ev.event` into `agents[ev.agent]` with existing `reduce`
  - `swarmRowsHtml(state) -> string` — one row per agent: index, sub-question (truncated), turn dots, hit count, status
  - `swarmRequestBody(q) -> string` — `{profile, question, snapshot_k: 10, include_trace: true}`
  - `assignFiles` also maps `*-swarm.jsonl` to `swarm`
  - `overtime` threshold 120 s; overtime text "⚠ Past 120 s. Switch to replay?"
  - `verdictHtml(r, h, s)` — adds a swarm clause when `s` is present (e.g. "Swarm: 6 of 10."); unchanged output when `s` is absent

- [ ] **Step 1: Write the failing tests** (`viewer.test.mjs`)

```js
test("reduceSwarm folds agent events into each agent", () => {
  let s = initialSwarm();
  s = reduceSwarm(s, { type: "swarm_split", sub_questions: ["a", "b"], fallback: false });
  s = reduceSwarm(s, { type: "agent_event", agent: 1, event: ev("search_started", { mode: "harness" }) });
  assert.equal(s.agents.length, 2); assert.equal(s.agents[1].phase !== undefined, true);
});
test("swarm_finished sets merged hits, usage and agent status", () => { /* agents[1].status === "cut", hits.length === 2 */ });
test("assignFiles picks up a -swarm.jsonl", () => { /* out.swarm === text */ });
test("old two-lane replay: no swarm file, verdict still renders", () => {
  assert.equal(verdictHtml(r, h, null), verdictHtml(r, h));
});
test("overtime is 120 s", () => { assert.equal(overtime(119, false, false), false); assert.equal(overtime(121, false, false), true); });
test("swarmRowsHtml escapes sub-questions", () => { /* "<b>" → "&lt;b&gt;" */ });
```

- [ ] **Step 2: Run, expect FAIL** — `node --test demo/ignite/tests/viewer.test.mjs`.
- [ ] **Step 3: Implement** core functions, then the DOM part: `<section class="col" id="col-swarm">`, border colour from an existing token, live run posts `swarmRequestBody` to `/ignite/swarm` and feeds `parseSse` frames to `reduceSwarm`; replay plays `player.runs.swarm`; answer box and check-it-yourself use `state.hits` keys and `state.notes`; gold hidden until `done`. Header line "split → N agents → merge (RRF)". If three columns do not fit at 1920×1080, stack the swarm column below the others; note which in the tuning log.
- [ ] **Step 4: Run, expect PASS** — `node --test demo/ignite/tests/viewer.test.mjs && uv run pytest demo/ignite/tests/test_viewer_static.py -q` (no external URLs).
- [ ] **Step 5: Look at it** — `uv run python -m demo.ignite.serve`, open http://127.0.0.1:8765/, load a saved swarm replay from Task 4; screenshot at 1920×1080.
- [ ] **Step 6: Commit** — `git commit -m "demo(swarm): third viewer lane"`

---

### Task 6: Live numbers, rehearsal, docs

**Files:**
- Modify: `demo/ignite/results-musique.md`, `results-wands.md`, `results-wands-match.md`, `README.md`, `tuning-log.md`
- Create: `demo/ignite/runs/<stamp>-<id>-{retrieval,harness,swarm}.jsonl` backups

- [ ] **Step 1: Auth check** — `gcloud auth application-default print-access-token >/dev/null && echo ok` → `ok`. If not, ask AJ to run `! gcloud auth application-default login`.
- [ ] **Step 2: Run the three evals one at a time** (export `IGNITE_GCP_PROJECT=ajw-sandbox-497707`):
  `uv run python -m demo.ignite.public.eval_public --dataset musique --modes bm25,retrieval,harness,swarm`, then `--dataset wands`, then `--dataset wands-match`. Each rewrites its `results-*.md` (replacing the 60 s numbers). Record each run's total cost.
- [ ] **Step 3: Rehearse live** — in the viewer, `hero`, `awning`, `trays`, `match-hero`, 2× each; all three lanes finish inside 120 s. Save backups with `uv run python -m demo.ignite.run_side_by_side --question <id> --save`.
- [ ] **Step 4: Docs** — README: swarm in the story and tables, 120 s budget, updated numbers (incl. the WANDS match harness P@10 that was 0.515). tuning-log: new round with budget change, swarm wins/losses per set, cost per set, layout choice, any concurrency change. Say plainly where the swarm lost.
- [ ] **Step 5: Full test pass** — `uv run pytest -q && node --test demo/ignite/tests/viewer.test.mjs` → all pass.
- [ ] **Step 6: Commit** — `git commit -m "docs(demo): swarm results at 120 s; rehearsal backups"`
