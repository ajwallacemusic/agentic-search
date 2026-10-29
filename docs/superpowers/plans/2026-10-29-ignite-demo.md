# Ignite Demo (Retrieval vs Harness) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A laptop demo that runs one question through `retrieval` and `harness` mode over a synthetic five-source corpus, shows each run's decisions live in a browser viewer (or replays saved runs), and produces eval numbers for the talk.

**Architecture:** A consumer of the library under `demo/ignite/`. Backends are built in Python with `FilesBackend.from_documents` from committed JSONL; the driver, judge, controller and embedder come from a small YAML read through the library's own `agentic_search.config.build`. `serve.py` wraps the library's `create_app` with in-code `Harness` objects and also serves the static viewer. The viewer is one HTML file with no build step and no network calls other than to `serve.py`.

**Tech Stack:** Python 3.11, `uv`, pytest (asyncio auto mode), ruff, FastAPI/uvicorn (`server` extra), `sentence-transformers` (`local` extra), Vertex via `vertex_client` (`openai` + `gcp` extras), plain HTML/CSS/JS, Node's built-in `node --test` for the viewer's pure logic.

**Spec:** `docs/superpowers/specs/2026-10-29-ignite-demo-design.md`

## Investigation results (spec §5.2 and §8, done while writing this plan)

- **FilesBackend metadata:** loading from a directory (`root:` in YAML) only sets `path`, `ext`, `size`, `modified`, and a `.jsonl` file becomes one document. `from_documents` keeps each `Document.metadata`, and every metadata key becomes a filterable field (dates sortable). **Path taken: build backends in Python in `build.py`; `serve.py` passes those `Harness` objects to `agentic_search.server.create_app`.** `create_app(config, {name: Profile | Harness})` already accepts in-code harnesses.
- **eval_beir.py:** `download_beir` skips the download if `<data-dir>/<dataset>/corpus.jsonl` exists, so a local dataset loads. But the script builds a single `corpus` backend (no per-source metadata), only knows `anthropic:`/`openai:` drivers, reports recall@100, and has no single/multi-hop split. **Path taken: a thin `demo/ignite/eval.py` that calls `agentic_search.eval.runner.run_eval` with the demo harness and `recall_k=10`.** `scripts/` is unchanged.
- **Hit identity:** a hit's `key` is `"<source>:<doc_id>"`. Gold ids are plain `doc_id`s, so doc ids must be unique across all sources; `check_data.py` enforces it.
- **Pytest scope:** `pyproject.toml` sets `testpaths = ["tests"]`, so `uv run pytest -q` does not collect demo tests. Demo tests run with `uv run pytest demo/ignite/tests -q`. `uv run ruff check` covers `demo/` already.

## Global Constraints

- Nothing under `src/` or `scripts/` changes. If a library change looks needed, stop and raise it as a separate PR.
- All new files live under `demo/ignite/` (plus this plan and the spec).
- Everything is synthetic: company **Lumen Retail**, invented names, dates and numbers. No real client or Egen data.
- Hero question, verbatim: `Did last month's latency spike come from the new reranker, and is it fixed?`
- Hero gold ids, in hop order: `INC-4821`, `DEP-0913-07`, `PR-1203`, `DASH-W38-search`.
- Sources, in swimlane order: `incidents`, `deploys`, `pull_requests`, `dashboards`, `runbooks`.
- Budget: `max_turns: 4`, `max_tool_calls: 24`, `max_seconds: 60`, `max_cost_usd: 0.50`.
- Saved runs: `demo/ignite/runs/<timestamp>-<question>-<mode>.jsonl`, one `ev.model_dump_json()` per line.
- Viewer: one file, no build step, body text at least 18px, no colour-only signals, no external network calls.
- `serve.py` binds `127.0.0.1` only.
- `uv run pytest -q`, `uv run pytest demo/ignite/tests -q` and `uv run ruff check` stay clean.
- Line length 100 (ruff config).

## Defaults for the spec's open questions (§11)

These are the plan's defaults; change them in `demo/ignite/config.yaml` or the viewer without code changes.

1. Driver: Vertex Gemini Flash in `config.yaml`; an Anthropic block sits commented out under it.
2. Jev controller: optional `--controller typesafe` in the runner and a second `ignite-typesafe` profile in `serve.py`, built only when `TYPESAFE_API_KEY` is set. Not part of the main demo flow.
3. Branding: neutral.
4. `model` mode column: no.

## Review Focus

1. **A run that fails mid-stream** (expired `gcloud` credentials, quota, timeout): the runner prints the `search_failed` message, still saves what it got, and shows the mode as failed in the summary; the viewer shows the error in that column and does not freeze. Tests: Task 4 `test_failed_run_is_summarized`, Task 7 `reduce handles search_failed`.
2. **A saved run cut short** (Ctrl-C during `--save`, or a line from a newer schema): the file still holds every event received so far, and replay skips bad or unknown lines instead of refusing the file. Tests: Task 4 `test_save_writes_each_event_as_it_arrives`, Task 8 `parseJsonl skips bad and unknown lines`.
3. **Gold ids that clash across sources, or gold matched by `key` instead of `doc_id`**: gold recall must count by `doc_id`, and duplicate ids must fail `check_data`. Tests: Task 1 `test_duplicate_id_across_sources_fails`, Task 4 `test_gold_recall_counts_doc_ids_in_top_10`.
4. **The live stream stalls on stage**: the viewer shows a running clock per column and a visible "Switch to replay" button at all times in live mode; the harness budget (`max_seconds: 60`) bounds the search. Tests: Task 3 `test_budget_matches_spec`, Task 7 manual check step.
5. **"Last month" without a date in context**: the driver does not know today is 2026-10-29. The corpus must make September the obvious month (the only `search-api` latency incident after the August CDN one). Test: Task 1 `check_data` rule that `INC-4821` is the only `search-api` latency incident opened in September, and the Task 5 acceptance runs.

---

### Task 1: Corpus loader, `check_data.py`, and the hero chain

**Files:**
- Create: `demo/ignite/__init__.py` (empty), `demo/ignite/corpus.py`, `demo/ignite/check_data.py`
- Create: `demo/ignite/data/incidents.jsonl`, `deploys.jsonl`, `pull_requests.jsonl`, `dashboards.jsonl`, `runbooks.jsonl`, `questions.yaml` (hero chain + named distractors only in this task)
- Test: `demo/ignite/tests/__init__.py`, `demo/ignite/tests/test_check_data.py`

**Interfaces:**
- Produces:
  - `SOURCES: tuple[str, ...] = ("incidents", "deploys", "pull_requests", "dashboards", "runbooks")` in `corpus.py`
  - `DATA_DIR: Path` (the `data/` folder next to `corpus.py`)
  - `load_records(data_dir: Path = DATA_DIR) -> dict[str, list[dict]]`: source → raw JSONL rows `{"id", "title", "text", "metadata"}`
  - `load_documents(data_dir: Path = DATA_DIR) -> dict[str, list[Document]]`: `Document(doc_id=row["id"], content=[TextPart(text=f"{title}\n\n{text}")], metadata={**row["metadata"], "title": row["title"]})`
  - `Question` (pydantic): `id: str`, `text: str`, `kind: Literal["single_hop", "multi_hop"]`, `gold: list[str]`
  - `load_questions(data_dir: Path = DATA_DIR) -> dict[str, Question]` keyed by id, file order kept
  - `check(data_dir: Path = DATA_DIR) -> list[str]` in `check_data.py`: problem messages; empty means pass. `main()` prints them and exits 1, or prints `ok: <n> docs, <m> questions` and exits 0.

Required metadata per source (missing key = problem):
`incidents`: `opened` (ISO date), `service`, `severity`, `status`. `deploys`: `date`, `service`, `version`, `pr` (int or null), `author`. `pull_requests`: `number` (int), `merged` (ISO date or null), `service`. `dashboards`: `service`, `week` (`2026-Wnn`), `week_start`, `week_end`. `runbooks`: `service`.

Hero chain rules `check` enforces (`HERO_CHAIN` constant in `check_data.py`):
- `INC-4821`: `service == "search-api"`, `opened == "2026-09-14"`; its text mentions neither `reranker`, `1187` nor `1203` (no root cause recorded).
- `DEP-0913-07`: `service == "search-api"`, `date == "2026-09-13"`, `version == "4.12"`, `pr == 1187`; text contains `reranker v2`.
- `PR-1187` exists with `number == 1187`.
- `PR-1203`: `number == 1203`, `merged == "2026-09-18"`; text contains `INC-4821` and `reranker v2`.
- `DASH-W38-search`: `service == "search-api"`, `week_start <= "2026-09-18" <= week_end`; text does not contain `reranker` or `INC-4821`.
- `INC-4821` is the only `incidents` doc with `service == "search-api"`, `opened` in September 2026 and `latency` in its text.

- [ ] **Step 1: Write the failing tests**

Tests build a tiny corpus in `tmp_path` by copying `DATA_DIR` and editing it, so they keep working as the corpus grows. Helper: `_copy_data(tmp_path) -> Path`, `_edit(path, doc_id, **changes)` that rewrites one JSONL row (`metadata.<key>` via a `metadata` dict merge, `text` replaced).

```python
def test_real_data_passes():
    assert check() == []

def test_documents_carry_metadata_and_title():
    docs = {d.doc_id: d for d in load_documents()["deploys"]}
    assert docs["DEP-0913-07"].metadata["pr"] == 1187
    assert docs["DEP-0913-07"].metadata["title"]

def test_missing_gold_id_fails(tmp_path):  # remove PR-1203 row
    assert any("PR-1203" in p for p in check(data))

def test_duplicate_id_across_sources_fails(tmp_path):  # append INC-4821 row to runbooks.jsonl
    assert any("duplicate" in p and "INC-4821" in p for p in check(data))

def test_broken_deploy_link_fails(tmp_path):  # DEP-0913-07 metadata pr -> 1190
    assert any("DEP-0913-07" in p for p in check(data))

def test_incident_leaking_root_cause_fails(tmp_path):  # INC-4821 text += " reranker"
    assert any("INC-4821" in p for p in check(data))

def test_dashboard_week_must_cover_fix_date(tmp_path):  # week_start -> 2026-09-21
    assert any("DASH-W38-search" in p for p in check(data))

def test_second_september_search_latency_incident_fails(tmp_path):
    # append an incidents row: service search-api, opened 2026-09-20, text mentions latency
    assert any("only" in p for p in check(data))

def test_missing_required_metadata_fails(tmp_path):  # drop deploys row's "service"
    assert any("service" in p for p in check(data))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest demo/ignite/tests/test_check_data.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'demo.ignite.corpus'` (or `demo`). If `demo` is not importable, add `demo/__init__.py` (empty) so `demo.ignite` is a package from the repo root.

- [ ] **Step 3: Write `corpus.py` and `check_data.py` to the interfaces above**

`check_data.py` also checks: ids unique across all sources, every question's gold ids exist, question ids unique, `kind` valid. It runs as `uv run python -m demo.ignite.check_data`.

- [ ] **Step 4: Write the hero chain and the named distractors**

Realistic tone (tickets read like tickets, PRs like PRs). In this task write only:
- `incidents`: `INC-4821`; the August CDN latency distractor (`service: search-api`, opened in August, root cause "CDN config change").
- `deploys`: `DEP-0913-07`; two other `search-api` deploys on 2026-09-13/14 that are not the reranker (one is a same-day unrelated feature-flag deploy).
- `pull_requests`: `PR-1187` (reranker v2 rollout, author named), `PR-1203` (the fix), an older reranker v1 PR (distractor).
- `dashboards`: `DASH-W37-search` (spike week), `DASH-W38-search` (back to baseline), one week where a different service (`checkout-api`) had a p95 spike.
- `runbooks`: two runbooks using "latency" and "rollback".
- `questions.yaml`: the hero question only, `id: hero`, `kind: multi_hop`, gold in hop order.

- [ ] **Step 5: Run tests and the checker**

Run: `uv run pytest demo/ignite/tests/test_check_data.py -q && uv run python -m demo.ignite.check_data`
Expected: all PASS; checker prints `ok: …` and exits 0.

- [ ] **Step 6: Commit**

```bash
git add demo/__init__.py demo/ignite
git commit -m "feat(demo): ignite corpus loader, data checker and hero chain"
```

### Task 2: Full corpus and supporting questions

**Files:**
- Modify: `demo/ignite/data/*.jsonl`, `demo/ignite/data/questions.yaml`
- Modify: `demo/ignite/check_data.py` (size and mix rules)
- Test: `demo/ignite/tests/test_check_data.py`

**Interfaces:**
- Consumes: Task 1's `check`, `load_records`, `load_questions`.
- Produces: the final corpus; question ids that later tasks use (`hero` plus the ids written here).

- [ ] **Step 1: Write the failing tests**

```python
def test_source_sizes_match_spec():
    sizes = {s: len(rows) for s, rows in load_records().items()}
    assert 25 <= sizes["incidents"] <= 35
    assert 50 <= sizes["deploys"] <= 70
    assert 20 <= sizes["pull_requests"] <= 30
    assert 10 <= sizes["dashboards"] <= 14
    assert 8 <= sizes["runbooks"] <= 12

def test_question_mix_matches_spec():
    qs = [q for q in load_questions().values() if q.id != "hero"]
    assert 8 <= len(qs) <= 12
    assert 3 <= sum(q.kind == "single_hop" for q in qs) <= 4
    assert 5 <= sum(q.kind == "multi_hop" for q in qs) <= 8
    assert all(len(q.gold) == 1 for q in qs if q.kind == "single_hop")
    assert all(2 <= len(q.gold) <= 4 for q in qs if q.kind == "multi_hop")
```

Also add these as `check` rules so `check_data` fails on them.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest demo/ignite/tests/test_check_data.py -q`
Expected: the two new tests FAIL on sizes and mix.

- [ ] **Step 3: Write the rest of the corpus and the supporting questions**

Fill each source to the sizes above with realistic filler across several services (`search-api`, `checkout-api`, `catalog-svc`, `auth-gateway`, `recs-api`) and dates from July to October 2026. Single-hop questions include "Who authored the reranker v2 PR?" (gold: `PR-1187`). Each multi-hop question must need a value learned from an earlier hop, like the hero. Keep the Task 1 hero rules true.

- [ ] **Step 4: Run tests and the checker**

Run: `uv run pytest demo/ignite/tests -q && uv run python -m demo.ignite.check_data`
Expected: PASS; `ok: <~137> docs, <9–13> questions`.

- [ ] **Step 5: Commit**

```bash
git add demo/ignite
git commit -m "feat(demo): full ignite corpus and supporting questions"
```

### Task 3: `build.py` and `config.yaml`

**Files:**
- Create: `demo/ignite/build.py`, `demo/ignite/config.yaml`
- Test: `demo/ignite/tests/test_build.py`

**Interfaces:**
- Consumes: `load_documents`, `SOURCES` (Task 1).
- Produces:
  - `CONFIG_PATH: Path` (`config.yaml` next to `build.py`)
  - `load_config(path: Path = CONFIG_PATH) -> dict[str, Any]`
  - `build_demo_harness(cfg: dict[str, Any], *, controller: Literal["heuristic", "typesafe"] = "heuristic", data_dir: Path = DATA_DIR, driver: Driver | None = None, analyzer: Decider | None = None, embedder: Embedder | None = None) -> Harness`. The keyword overrides exist for offline tests. Default mode `"harness"`.
  - `SOURCE_DESCRIPTIONS: dict[str, str]`: one line per source naming its fields, e.g. deploys: "Deploy log: one record per deploy, fields date, service, version, pr (PR number), author."

`config.yaml` shape (env vars via the library's `*_env` convention):

```yaml
driver: {type: vertex, model: gemini-3.8-flash, project_env: IGNITE_GCP_PROJECT, location: global,
         price_per_mtok: [<input>, <output>]}
# driver: {type: anthropic, model: claude-sonnet-5-5, price_per_mtok: [3.0, 15.0]}
embedder: {type: sentence_transformers, model: BAAI/bge-small-en-v1.5}
fallback_embedder: {type: hash}
budget: {max_turns: 4, max_tool_calls: 24, max_seconds: 60, max_cost_usd: 0.50}
typesafe: {type: typesafe, api_key_env: TYPESAFE_API_KEY}
max_calls_per_turn: 8
```

`build_demo_harness` builds the client with `agentic_search.config.build("client", cfg["driver"], ctx)`, wraps it in `ToolCallingDriver(client, max_calls_per_turn=...)`, uses `LLMJudge(client)` as analyzer, the built-in controller (`controller=None`) for `heuristic` and `build("decider", cfg["typesafe"], ctx)` for `typesafe`. The embedder comes from `cfg["embedder"]`; if building it raises `ImportError`, it logs a warning and uses `cfg["fallback_embedder"]`. One `FilesBackend.from_documents(source, docs, embedder=embedder, description=SOURCE_DESCRIPTIONS[source])` per source, in `SOURCES` order.

- [ ] **Step 1: Write the failing tests**

Use `agentic_search.testing.ScriptedDriver`, `KeywordJudge` and `HashEmbedder` overrides; no network.

```python
def test_one_backend_per_source_in_order():
    h = build_demo_harness(load_config(), driver=..., analyzer=..., embedder=HashEmbedder())
    assert list(h.backends) == list(SOURCES)

def test_budget_matches_spec():
    assert h.budget == Budget(max_turns=4, max_tool_calls=24, max_seconds=60, max_cost_usd=0.50)

async def test_deploys_are_filterable_by_date_and_service():
    manifests = await h.setup()
    fields = {f.name: f for f in manifests["deploys"].collections[0].fields}
    assert fields["date"].type == FieldType.DATE and fields["date"].filterable
    assert fields["service"].filterable

async def test_filter_search_finds_the_hero_deploy():
    hits = await h.backends["deploys"].execute(FilterOnly(filter=<and(eq service search-api, eq date 2026-09-13)>, limit=10))
    assert "DEP-0913-07" in [x.doc_id for x in hits]

def test_heuristic_controller_is_builtin():
    assert h.controller_decider is None
```

Build the `Filter` with the library's filter model from `agentic_search.core.types`; check its constructor there.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest demo/ignite/tests/test_build.py -q`
Expected: FAIL, `build_demo_harness` not defined.

- [ ] **Step 3: Implement `build.py` and `config.yaml`**

- [ ] **Step 4: Set the real Vertex price**

Look up the chosen Gemini Flash model's input and output price per million tokens on the Vertex AI pricing page. Put it in `price_per_mtok`. Note the model, price and lookup date in a YAML comment.

- [ ] **Step 5: Run tests**

Run: `uv run pytest demo/ignite/tests -q && uv run ruff check demo`
Expected: PASS, no lint errors.

- [ ] **Step 6: Commit**

```bash
git add demo/ignite
git commit -m "feat(demo): build the ignite harness from config"
```

### Task 4: `run_side_by_side.py`

**Files:**
- Create: `demo/ignite/run_side_by_side.py`, `demo/ignite/runs/.gitkeep`
- Test: `demo/ignite/tests/test_runner.py`

**Interfaces:**
- Consumes: `build_demo_harness`, `load_config`, `load_questions`.
- Produces:
  - `RUNS_DIR: Path`
  - `gold_recall(doc_ids: list[str], gold: list[str], k: int = 10) -> int`: count of gold ids in the first `k`
  - `RunSummary` (pydantic): `mode: str`, `turns: int`, `tool_calls: int`, `seconds: float`, `cost_usd: float`, `stop_reason: str` (`"failed: <error_type>"` on `search_failed`), `top5: list[str]` (hit keys), `gold_recall: int`, `gold_total: int`
  - `format_event(ev: SearchEvent) -> str | None`: one compact line, or `None` for events not shown. Shown: `phase_started` (`… plan`), `tool_call_started` (`→ deploys.filter_search {service: search-api, date: …}`, args cut to 80 chars), `tool_call_finished` with an error, `phase_finished` for `decide` (`◆ refine: <note>`), `search_failed`.
  - `async run_one(harness: Harness, question: Question, mode: Mode, *, save_to: Path | None, print_fn: Callable[[str], None] = print) -> RunSummary`: drives `harness.stream(question.text, mode=mode, top_k=20, snapshot_k=10)`; if `save_to` is set, opens it before the first event and writes and flushes one `ev.model_dump_json()` line per event.
  - `save_path(question_id: str, mode: str, stamp: str) -> Path`: `RUNS_DIR / f"{stamp}-{question_id}-{mode}.jsonl"`
  - `write_meta(question: Question, stamp: str) -> Path`: `RUNS_DIR / f"{stamp}-{question.id}.meta.json"` holding `{"question_id", "text", "gold"}` (used by viewer replay for gold markers)
  - `summary_table(rows: list[RunSummary]) -> str`
  - CLI: `uv run python -m demo.ignite.run_side_by_side --question hero [--controller heuristic|typesafe] [--save]`. Runs `retrieval` then `harness` on one harness, prints both traces, then the table. Seconds are wall-clock around each `run_one`.

- [ ] **Step 1: Write the failing tests**

Offline harness: `build_demo_harness` with a `ScriptedDriver` that issues one `lexical_search` per turn (turn 0 on `incidents`, turn 1 on `deploys`), `KeywordJudge(["search-api"])`, `HashEmbedder()`.

```python
def test_gold_recall_counts_doc_ids_in_top_10():
    ids = ["x"] * 9 + ["INC-4821", "PR-1203"]
    assert gold_recall(ids, ["INC-4821", "PR-1203"]) == 1

async def test_run_one_summarizes_harness_mode(tmp_path):
    s = await run_one(h, hero, "harness", save_to=None, print_fn=lines.append)
    assert s.mode == "harness" and s.tool_calls >= 2 and s.turns >= 2 and s.gold_total == 4

async def test_save_writes_each_event_as_it_arrives(tmp_path):
    out = tmp_path / "r.jsonl"
    await run_one(h, hero, "retrieval", save_to=out, print_fn=lambda _: None)
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert rows[0]["type"] == "search_started" and rows[-1]["type"] == "search_finished"
    assert [r["seq"] for r in rows] == list(range(len(rows)))

async def test_failed_run_is_summarized(tmp_path):
    # harness whose driver raises (a driver class whose plan() raises RuntimeError)
    s = await run_one(bad, hero, "harness", save_to=tmp_path / "f.jsonl", print_fn=lines.append)
    assert s.stop_reason.startswith("failed")
    assert any("failed" in line for line in lines)

def test_format_event_shows_source_and_tool():
    line = format_event(ToolCallStarted(search_id="s", seq=1, turn=0, at_ms=1.0, call_id="c",
                                        source="deploys", name="filter_search", arguments={...}))
    assert "deploys" in line and "filter_search" in line
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest demo/ignite/tests/test_runner.py -q`
Expected: FAIL, names not defined.

- [ ] **Step 3: Implement `run_side_by_side.py`**

- [ ] **Step 4: Run tests**

Run: `uv run pytest demo/ignite/tests -q && uv run ruff check demo`
Expected: PASS.

- [ ] **Step 5: Smoke-test against the real model once**

Run: `uv sync --extra server --extra local --extra openai --extra gcp && IGNITE_GCP_PROJECT=<project> uv run python -m demo.ignite.run_side_by_side --question hero`
Expected: two traces and a table with a non-zero cost. If Vertex auth fails, run `gcloud auth application-default login` and retry; if Vertex is blocked, switch `config.yaml` to the Anthropic block.

- [ ] **Step 6: Commit**

```bash
git add demo/ignite
git commit -m "feat(demo): side-by-side runner with trace, summary and saved events"
```

### Task 5: Tune the corpus to the acceptance criteria (live)

**Files:**
- Modify: `demo/ignite/data/*.jsonl` (and `check_data.py` rules if a new chain rule is needed)
- Create: `demo/ignite/tuning-log.md` (one line per round)

**Interfaces:**
- Consumes: the runner CLI (Task 4).

- [ ] **Step 1: Run the hero question 5 times**

Run (5×): `uv run python -m demo.ignite.run_side_by_side --question hero --save`
Record each round in `tuning-log.md`: retrieval gold recall, harness gold recall, harness seconds, harness cost.

- [ ] **Step 2: Compare with the acceptance criteria**

Pass when: harness gets 4/4 gold in the top 10 in ≥4 of 5 runs, each under 45 s; retrieval misses at least one gold doc in ≥3 of 5 runs.

- [ ] **Step 3: If it fails, change the corpus, not the modes**

Retrieval wins too often → make distractors closer (a `reranker` mention in a non-gold PR, more `search-api` p95 dashboards) or make the hop link less obvious (the dashboard names the week, not the PR). Harness misses → make the link from the previous hop findable (the exact PR number or date appears in the next document). Never change the retrieval mode, the budget, or the judge. Re-run `check_data`, then go back to Step 1.

- [ ] **Step 4: Check the single-hop questions**

Run each single-hop question once in both modes. Expected: same gold recall in both modes for each one. If harness does worse, note it in `tuning-log.md` for the talk; do not hide it.

- [ ] **Step 5: Commit the tuned corpus and the two cleanest runs**

Keep the two cleanest hero run pairs (and their `.meta.json`) in `runs/`; delete the rest.

```bash
git add demo/ignite
git commit -m "feat(demo): tune ignite corpus; keep two clean backup runs"
```

### Task 6: `serve.py`

**Files:**
- Create: `demo/ignite/serve.py`
- Test: `demo/ignite/tests/test_serve.py`

**Interfaces:**
- Consumes: `build_demo_harness`, `load_config`, `load_questions`.
- Produces:
  - `VIEWER_DIR: Path` (`viewer/` next to `serve.py`)
  - `create_demo_app(profiles: dict[str, Harness], questions: dict[str, Question], *, viewer_dir: Path = VIEWER_DIR) -> FastAPI`: `agentic_search.server.create_app(ServiceConfig(auth=AuthConfig(type="none")), profiles)`, plus `GET /ignite/questions` → `[{"id", "text", "kind"}]` (no gold), plus `app.mount("/", StaticFiles(directory=viewer_dir, html=True))` registered last. Same origin, so no CORS.
  - CLI: `uv run python -m demo.ignite.serve [--port 8765]`. Always `host="127.0.0.1"`. Profiles: `ignite` (heuristic, default) and `ignite-typesafe` only when `TYPESAFE_API_KEY` is set. Prints `http://127.0.0.1:<port>/` on start.

- [ ] **Step 1: Write the failing tests**

Use the offline harness from Task 4's tests and `httpx.AsyncClient(transport=httpx.ASGITransport(app))`. Write a placeholder `viewer/index.html` in `tmp_path`.

```python
async def test_questions_endpoint_hides_gold():
    rows = (await client.get("/ignite/questions")).json()
    assert rows[0]["id"] == "hero" and "gold" not in rows[0]

async def test_stream_returns_sse_for_both_modes():
    for mode in ("retrieval", "harness"):
        r = await client.post("/v1/search/stream", json={"profile": "ignite", "question": hero.text, "mode": mode})
        assert r.status_code == 200 and "event: search_finished" in r.text

async def test_viewer_is_served_at_root():
    assert (await client.get("/")).status_code == 200

def test_cli_binds_localhost_only():
    # parse serve.py's argparse: there is no --host option
    assert "--host" not in build_parser().format_help()
```

`build_parser() -> argparse.ArgumentParser` is part of the interface for this test.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest demo/ignite/tests/test_serve.py -q`
Expected: FAIL, names not defined.

- [ ] **Step 3: Implement `serve.py`**

- [ ] **Step 4: Run tests**

Run: `uv run pytest demo/ignite/tests -q && uv run ruff check demo`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add demo/ignite
git commit -m "feat(demo): localhost service for the ignite viewer"
```

### Task 7: Viewer, live mode

**Files:**
- Create: `demo/ignite/viewer/index.html`
- Test: `demo/ignite/tests/viewer.test.mjs`, `demo/ignite/tests/test_viewer_static.py`

**Interfaces:**
- Consumes: `GET /ignite/questions`, `POST /v1/search/stream` (Task 6); event shapes from `src/agentic_search/events.py` (reference: `clients/typescript/src/events.ts`, `sse.ts`).
- Produces: the pure logic lives in one `<script>` block between the comments `/* <core> */` and `/* </core> */`, with no DOM access, and defines:
  - `parseSse(buffer: string) -> {events: object[], rest: string}`: splits on blank lines; reads `event:` and `data:` lines; ignores `:` comments (keep-alive)
  - `initialState(mode) -> State` and `reduce(state, ev) -> State`. `State` holds `phase`, `calls: [{turn, source, name, args, atMs, error}]`, `decisions: [{turn, action, note}]`, `hits: HitSummary[]`, `metrics: {turns, toolCalls, seconds, costUsd, stopReason}`, `error: string | null`. Unknown event types return the state unchanged.
  - `SOURCES` (same order as `corpus.SOURCES`)

Layout (spec §7): two columns, **Retrieval mode (one pass)** left, **Harness mode (full loop)** right. Per column: phase strip (`plan → query → judge → decide`, active phase marked by text weight and a ▶ marker, not colour alone); swimlane with one row per source and a labelled dot per tool call placed by turn; decision badges with the note; results list; metrics footer; a running clock. Header: question picker (from `/ignite/questions`), **Run** (starts both streams at the same time), **Switch to replay** (always visible). 18px minimum body text, high contrast, system fonts only.

- [ ] **Step 1: Write the failing tests**

`viewer.test.mjs` reads `index.html`, extracts the core block, and runs it in `node:vm`:

```js
test("parseSse splits frames and keeps the partial tail", ...)  // two frames + half a frame
test("parseSse ignores keep-alive comments", ...)
test("reduce records a tool call in its source row", ...)       // tool_call_started source=deploys
test("reduce records decide notes", ...)                        // phase_finished decide action=refine note=...
test("reduce handles search_failed", ...)                       // state.error set, stopReason "failed"
test("reduce ignores unknown event types", ...)
```

`test_viewer_static.py`:

```python
def test_no_external_urls():
    html = VIEWER.read_text()
    assert not re.search(r"https?://(?!127\.0\.0\.1|localhost)", html)

def test_body_font_is_at_least_18px():
    assert re.search(r"body\s*{[^}]*font-size:\s*(1[89]|[2-9]\d)px", VIEWER.read_text())

def test_core_tests_pass():
    subprocess.run(["node", "--test", str(TESTS / "viewer.test.mjs")], check=True)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest demo/ignite/tests/test_viewer_static.py -q`
Expected: FAIL, `index.html` missing.

- [ ] **Step 3: Implement `index.html` (live mode)**

Stream with `fetch(..., {method: "POST", body})` and `response.body.getReader()` feeding `parseSse`. One `AbortController` per column; **Switch to replay** aborts both.

- [ ] **Step 4: Run tests**

Run: `uv run pytest demo/ignite/tests -q`
Expected: PASS.

- [ ] **Step 5: Check it by eye**

Run: `uv run python -m demo.ignite.serve`, open `http://127.0.0.1:8765/`, pick `hero`, press **Run**.
Expected: both columns fill; the harness column shows more than one turn and at least one decision badge; tool-call dots sit in different source rows; text is readable from across a room at 1920×1080. Kill the network mid-run: the column shows an error and **Switch to replay** still works.

- [ ] **Step 6: Commit**

```bash
git add demo/ignite
git commit -m "feat(demo): ignite trace viewer, live mode"
```

### Task 8: Viewer, replay mode

**Files:**
- Modify: `demo/ignite/viewer/index.html`
- Test: `demo/ignite/tests/viewer.test.mjs`

**Interfaces:**
- Consumes: Task 4 saved files (`*-retrieval.jsonl`, `*-harness.jsonl`, `*.meta.json`); Task 7 `reduce`.
- Produces (in the core block):
  - `parseJsonl(text: string) -> object[]`: skips blank, unparsable and unknown-type lines
  - `schedule(events: object[], speed: 1 | 2 | 4) -> [{delayMs, ev}]`: `delayMs` from each event's `at_ms`, divided by `speed`
  - `assignFiles(files: {name, text}[]) -> {retrieval, harness, meta}`: by file-name suffix (`-retrieval.jsonl`, `-harness.jsonl`, `.meta.json`); `meta` optional

Replay works when `index.html` is opened directly from disk (`file://`) with a multi-file picker, and when served. Controls: 1×/2×/4×, pause/resume, restart. Gold markers (a ★ and the word "gold") appear only in replay and only when a `.meta.json` was loaded.

- [ ] **Step 1: Write the failing tests**

```js
test("parseJsonl skips bad and unknown lines", ...)   // valid, "{bad", {"type":"future_thing"}, valid
test("schedule keeps relative timing and scales by speed", ...)  // at_ms 0,100,300 at 2x -> 0,50,150
test("assignFiles matches files by suffix", ...)
test("assignFiles works without meta", ...)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `node --test demo/ignite/tests/viewer.test.mjs`
Expected: new tests FAIL.

- [ ] **Step 3: Implement replay**

- [ ] **Step 4: Run tests**

Run: `uv run pytest demo/ignite/tests -q`
Expected: PASS.

- [ ] **Step 5: Check it by eye, offline**

Turn Wi-Fi off. Open `demo/ignite/viewer/index.html` from Finder. Load one saved pair plus its `.meta.json` from `demo/ignite/runs/`.
Expected: the run plays with its original timing; 2× and 4× are faster; pause works; gold docs show ★ gold.

- [ ] **Step 6: Commit**

```bash
git add demo/ignite
git commit -m "feat(demo): replay saved runs in the ignite viewer"
```

### Task 9: BEIR export and eval

**Files:**
- Create: `demo/ignite/export_beir.py`, `demo/ignite/eval.py`, `demo/ignite/beir/` (generated, committed), `demo/ignite/results.md` (generated, committed)
- Test: `demo/ignite/tests/test_eval.py`

**Interfaces:**
- Consumes: `load_records`, `load_questions`, `build_demo_harness`; library `agentic_search.eval.datasets.load_beir`, `agentic_search.eval.runner.run_eval`.
- Produces:
  - `export_beir(out_dir: Path = BEIR_DIR, data_dir: Path = DATA_DIR) -> Path`: writes `corpus.jsonl` (`{"_id", "title", "text"}`), `queries.jsonl` (`{"_id", "text"}`), `qrels/test.tsv` (header `query-id\tcorpus-id\tscore`, score 1 per gold id)
  - `AllSourcesBM25Driver(sources: list[str], limit: int = 20)`: turn 0 issues one `lexical_search` per source with the question text; later turns plan nothing. The `bm25` row uses it with `mode="retrieval"` and no analyzer.
  - `async evaluate(modes: list[str], harness_for: Callable[[str], Harness]) -> dict[tuple[str, str], EvalReport]`: key `(mode, kind)`, `kind` in `{"single_hop", "multi_hop"}`; calls `run_eval(..., query_ids=<ids of that kind>, k=10, recall_k=10, concurrency=2, mode=mode)` with the dataset from `load_beir(export_beir())`
  - `results_markdown(reports) -> str`: one table per kind; columns mode, nDCG@10, recall@10, MRR, avg turns, avg tool calls, avg cost, n. `QueryRun` has no turns field, so `evaluate` passes `run_eval` a `TurnRecorder(harness)`: a thin proxy whose `async search(question, **kw)` calls the real harness, stores `result.usage.turns` under the question text, and returns the result. `run_eval` only calls `.search`, so the proxy is enough.
  - CLI: `uv run python -m demo.ignite.eval --modes bm25,retrieval,harness` writes `beir/` and `results.md` and prints the tables. The retrieval row uses the same judged harness as the demo (same model, one pass); say so under the table.

- [ ] **Step 1: Write the failing tests**

```python
def test_export_round_trips_through_load_beir(tmp_path):
    ds = load_beir(export_beir(tmp_path))
    assert ds.qrels["hero"] == {g: 1 for g in HERO_GOLD}
    assert set(ds.corpus) == {r["id"] for rows in load_records().values() for r in rows}

async def test_bm25_driver_queries_every_source():
    plan = await AllSourcesBM25Driver(list(SOURCES)).plan(view_at_turn_0, tools=[])
    assert [c.arguments["source"] for c in plan.calls] == list(SOURCES)

async def test_evaluate_splits_by_kind(tmp_path):  # bm25 only, offline
    reports = await evaluate(["bm25"], harness_for)
    assert set(reports) == {("bm25", "single_hop"), ("bm25", "multi_hop")}
    assert "recall@10" in results_markdown(reports)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest demo/ignite/tests/test_eval.py -q`
Expected: FAIL, names not defined.

- [ ] **Step 3: Implement `export_beir.py` and `eval.py`**

- [ ] **Step 4: Run tests**

Run: `uv run pytest demo/ignite/tests -q && uv run ruff check demo`
Expected: PASS.

- [ ] **Step 5: Run the real eval**

Run: `uv run python -m demo.ignite.eval --modes bm25,retrieval,harness`
Expected: `results.md` written. Multi-hop: harness recall@10 above retrieval. Single-hop: within a few points. If not, go back to Task 5 Step 3.

- [ ] **Step 6: Commit**

```bash
git add demo/ignite
git commit -m "feat(demo): BEIR export and ignite eval results"
```

### Task 10: README, rehearsal notes, final checks

**Files:**
- Create: `demo/ignite/README.md`

**Interfaces:**
- Consumes: every CLI above.

- [ ] **Step 1: Write `README.md`**

Sections: what this is (one paragraph); setup (`uv sync --extra server --extra local --extra openai --extra gcp`, `gcloud auth application-default login`, `IGNITE_GCP_PROJECT`); the build path taken and why (from "Investigation results" above); commands (`check_data`, `run_side_by_side`, `serve`, `eval`); switching the driver to Anthropic; the optional Jev run; rehearsal checklist (spec §10: three rehearsals on the presentation laptop and network, warm-up run, reload viewer); fallback (stall past 60 s or error → **Switch to replay**, say so plainly; keep `results.md` and one saved run open in a tab); which files in `runs/` are the backup.

- [ ] **Step 2: Run every check**

Run: `uv run python -m demo.ignite.check_data && uv run pytest -q && uv run pytest demo/ignite/tests -q && uv run ruff check && git diff --stat main -- src scripts`
Expected: checker `ok`; both test runs PASS; ruff clean; the last command prints nothing.

- [ ] **Step 3: Confirm the backup runs**

Run: `ls demo/ignite/runs/`
Expected: at least two `hero` pairs (`-retrieval.jsonl` + `-harness.jsonl`) and their `.meta.json`.

- [ ] **Step 4: Commit**

```bash
git add demo/ignite
git commit -m "docs(demo): ignite README and rehearsal notes"
```

Rehearsals on the presentation laptop (spec §10) are AJ's, after this plan.
