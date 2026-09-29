# Plan 2: SQL & Search Backends — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Postgres + pgvector, MySQL, BigQuery and OpenSearch backends behind the existing `Backend` protocol. Put a read-only guard in front of native queries, and add a docker-compose contract suite that every backend passes. First, close Plan 1's two required follow-ups: secret scrubbing and cutting off delegate model spend once the budget is exhausted.

**Architecture:**
- SQL backends share `backends/sql.py` (identifier quoting, parameter collection, Filter AST → WHERE, row → Hit) and the `SqlBackend` base class in `backends/sql_backend.py`.
  - The base class implements filter, regex, aggregate, fetch, native and hybrid once.
  - Each subclass supplies its connection, table discovery, and dialect-specific lexical/vector SQL.
- OpenSearch is its own adapter, translating the Filter AST to bool-query DSL.
- `backends/native_guard.py` gates native SQL (via sqlglot), Cypher (used in Plan 3) and OpenSearch bodies.
- `core/secrets.py` masks registered secrets and URL userinfo in ToolErrors, traces and setup errors.

**Tech Stack:** Python 3.12, psycopg 3 + psycopg-pool, aiomysql, google-cloud-bigquery, opensearch-py[async], sqlglot, Docker Compose (pgvector/pgvector:pg16, mysql:8.4, opensearch 2.17.1).

**Spec:** `docs/superpowers/specs/2026-09-29-agentic-search-harness-design.md`. Plan 1 follow-ups: `docs/superpowers/plans/2026-09-29-plan-1-followups.md`.

**Verification status:** every code block below was run before this plan was written.
- Unit suite: 197 passed, 4 skipped (the skips are the files backend's native test in the contract suite plus the live BigQuery test).
- Integration contract suite against the docker services: 28 passed, 2 skipped (vector search on MySQL, and the SQL-only read-only-session test on files).
- ruff: clean.
- BigQuery SQL is verified only against a fake client (no emulator). The `live` test is the check against the real service.

## Global Constraints

- Python `>=3.11`, pydantic v2, async throughout. Blocking clients (BigQuery) run via `asyncio.to_thread`.
- Read-only everywhere:
  - Adapters issue only SELECT/search/get calls.
  - Postgres sessions set `default_transaction_read_only = on`; MySQL sessions run `SET SESSION TRANSACTION READ ONLY`.
  - `native_query` is off by default and always passes `native_guard` first.
- Every identifier interpolated into SQL goes through `quote_ident` (regex-validated). Every value is a bind parameter.
- Secrets (DSNs, passwords, `*_env` values) never appear in ToolErrors, traces or `setup_errors`, because `core/secrets.scrub` masks them.
- A vector op uses only the embedder bound to the column via `embedders: {"<table>.<column>": "<embedder id>"}`.
- Tool and validation failures become `ToolError`s. Adapters raise `BackendError`/`UnsupportedOperation`, never driver-specific exceptions.
- `uv run pytest` must pass with no services running. Docker-backed tests are marked `integration` and gated by `AGENTIC_SEARCH_INTEGRATION=1`.
- Commit after every task. Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## File Map

```
docker-compose.yml                        test services (ports 55432 / 53306 / 59200)
src/agentic_search/core/secrets.py        register_secret, scrub, scrub_data
src/agentic_search/backends/sql.py        quote_ident, Params, filter_sql, where_clause, parse_metric, row_to_hit
src/agentic_search/backends/native_guard.py  guard_sql, guard_cypher, guard_opensearch
src/agentic_search/backends/sql_backend.py   TableInfo, SqlBackend base class
src/agentic_search/backends/postgres.py   PostgresBackend
src/agentic_search/backends/mysql.py      MySQLBackend
src/agentic_search/backends/bigquery.py   BigQueryBackend
src/agentic_search/backends/opensearch.py OpenSearchBackend, filter_dsl
tests/contract/corpus.py                  shared corpus + per-service seeding
tests/contract/test_backend_contract.py   the contract every backend passes
Modified: core/state.py, core/harness.py, roles/executor.py, config.py, models/base.py,
          models/driver.py, pyproject.toml, README.md, tests/core/test_harness.py,
          tests/models/test_driver.py, tests/test_config.py
```

---

### Task 1: Secret scrubbing (Plan 1 required follow-up)

**Files:**
- Create: `src/agentic_search/core/secrets.py`
- Modify: `src/agentic_search/roles/executor.py`, `src/agentic_search/core/state.py`, `src/agentic_search/core/harness.py`, `src/agentic_search/config.py`
- Test: `tests/core/test_secrets.py`

**Interfaces:**
- Produces:
  - `register_secret(value: str | None) -> None` (ignores values shorter than 4 characters) and `clear_secrets() -> None`.
  - `scrub(text: str) -> str`, which masks registered secrets and `scheme://userinfo@` with `***`.
  - `scrub_data(value) -> value`, which does the same recursively through dicts, lists and tuples.
- Callers:
  - `Trace.add` scrubs its `data`.
  - The executor's `_short()` scrubs exception text.
  - `Harness.setup` scrubs `setup_errors`.
  - `config.resolve_env` registers every value it resolves from the environment.
  - Plan 2 adapters register their DSNs and passwords in their constructors.

- [ ] **Step 1: Write the failing tests**

`tests/core/test_secrets.py`:
```python
import pytest

from agentic_search.core.secrets import clear_secrets, register_secret, scrub, scrub_data
from agentic_search.core.state import Trace


@pytest.fixture(autouse=True)
def _clean():
    clear_secrets()
    yield
    clear_secrets()


def test_scrub_registered_values_and_url_userinfo():
    register_secret("hunter2secret")
    register_secret("abc")  # too short: ignored
    text = "auth failed for hunter2secret at postgresql://bob:pw@db:5432/x (abc)"
    assert scrub(text) == "auth failed for *** at postgresql://***@db:5432/x (abc)"


def test_scrub_data_recurses():
    register_secret("tok-123456")
    data = {"a": ["x tok-123456", ("tok-123456",)], "b": {"c": "tok-123456"}, "n": 3}
    assert scrub_data(data) == {"a": ["x ***", ("***",)], "b": {"c": "***"}, "n": 3}


def test_trace_events_are_scrubbed():
    register_secret("s3cr3t-value")
    t = Trace()
    ev = t.add("tool_error", 0, message="boom s3cr3t-value", nested={"dsn": "mysql://u:s3cr3t-value@h/db"})
    assert ev.data == {"message": "boom ***", "nested": {"dsn": "mysql://***@h/db"}}


async def test_executor_tool_error_is_scrubbed(docs_backend):
    from agentic_search.core.state import CandidatePool
    from agentic_search.core.types import Capability, CollectionInfo, Manifest, Query
    from agentic_search.embedders.base import EmbedderRegistry
    from agentic_search.roles.executor import Executor
    from agentic_search.testing import call

    register_secret("pa55word-xyz")

    class Leaky:
        name, backend_type = "leaky", "x"

        def capabilities(self):
            return {Capability.LEXICAL}

        async def discover(self, detail="full", collection=None):
            return Manifest(source="leaky", backend_type="x", capabilities=self.capabilities(),
                            collections=[CollectionInfo(name="c")])

        async def execute(self, op):
            raise ConnectionError("could not connect with password pa55word-xyz")

        async def close(self):
            pass

    b = Leaky()
    ex = Executor({"leaky": b}, {"leaky": await b.discover()}, EmbedderRegistry())
    trace = Trace()
    res = await ex.run([call("lexical_search", source="leaky", text="x")], question=Query.of("q"),
                       turn=0, pool=CandidatePool(), trace=trace)
    assert "pa55word-xyz" not in res.errors[0].message and "***" in res.errors[0].message
    assert "pa55word-xyz" not in trace.model_dump_json()


async def test_setup_errors_are_scrubbed(docs_backend):
    from agentic_search import Harness
    from agentic_search.testing import ScriptedDriver

    register_secret("topsecret-dsn-pw")

    class Broken:
        name, backend_type = "broken", "x"

        def capabilities(self):
            return set()

        async def discover(self, detail="full", collection=None):
            raise ConnectionError("login failed: topsecret-dsn-pw")

        async def execute(self, op):
            return []

        async def close(self):
            pass

    h = Harness([docs_backend, Broken()], ScriptedDriver([]))
    await h.setup()
    assert h.setup_errors["broken"] == "ConnectionError: login failed: ***"


def test_resolve_env_registers_secrets(monkeypatch):
    from agentic_search.config import resolve_env

    monkeypatch.setenv("AS_TEST_KEY", "sk-live-abcdef")
    resolve_env({"api_key_env": "AS_TEST_KEY"})
    assert scrub("key sk-live-abcdef") == "key ***"
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/core/test_secrets.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'agentic_search.core.secrets'`

- [ ] **Step 3: Implement the module**

`src/agentic_search/core/secrets.py`:
```python
"""Keep credentials out of error messages and traces.

Values are registered when config resolves `*_env` keys and when adapters receive DSNs or
passwords; `scrub()` replaces them (and any URL userinfo) before text reaches a ToolError,
the trace, or `Harness.setup_errors`."""

from __future__ import annotations

import re
from typing import Any

MASK = "***"
_MIN_SECRET_LEN = 4
_secrets: set[str] = set()
_URL_USERINFO = re.compile(r"(?P<scheme>\b[a-zA-Z][a-zA-Z0-9+.-]*://)[^/@\s]+@")


def register_secret(value: str | None) -> None:
    """Remember a secret so scrub() masks it. Very short values are ignored (too many false hits)."""
    if value and len(value) >= _MIN_SECRET_LEN:
        _secrets.add(value)


def clear_secrets() -> None:
    """For tests."""
    _secrets.clear()


def scrub(text: str) -> str:
    for secret in sorted(_secrets, key=len, reverse=True):
        text = text.replace(secret, MASK)
    return _URL_USERINFO.sub(lambda m: f"{m.group('scheme')}{MASK}@", text)


def scrub_data(value: Any) -> Any:
    """Recursively scrub every string inside dicts, lists and tuples."""
    if isinstance(value, str):
        return scrub(value)
    if isinstance(value, dict):
        return {k: scrub_data(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(scrub_data(v) for v in value)
    return value
```

- [ ] **Step 4: Wire it in (four small edits)**

In `src/agentic_search/roles/executor.py`, add the import after the `core.hooks` import:
```python
from agentic_search.core.secrets import scrub
```
and replace `_short`:
```python
def _short(exc: BaseException) -> str:
    return scrub(str(exc))[:500]
```

In `src/agentic_search/core/state.py`, add `from agentic_search.core.secrets import scrub_data` before the `core.types` import, and in `Trace.add` change `duration_ms=duration_ms, data=data)` to:
```python
                           duration_ms=duration_ms, data=scrub_data(data))
```

In `src/agentic_search/core/harness.py`, add `from agentic_search.core.secrets import scrub` after the `core.hooks` import, and in `setup()` replace the `setup_errors` line with:
```python
                    self.setup_errors[name] = scrub(f"{type(res).__name__}: {res}")
```

In `src/agentic_search/config.py`, add `from agentic_search.core.secrets import register_secret` after the `core.harness` import, and in `resolve_env` register each resolved value:
```python
                out[k[: -len("_env")]] = os.environ[v]
                register_secret(os.environ[v])
```

- [ ] **Step 5: Run to verify pass**

Run: `uv run pytest -q`
Expected: all pass (the 6 new tests included)

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat(core): scrub secrets from tool errors, traces and setup errors

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Stop delegate model spend once the budget is exhausted (Plan 1 required follow-up)

**Files:**
- Modify: `src/agentic_search/models/base.py`, `src/agentic_search/core/harness.py`, `src/agentic_search/models/driver.py`
- Test: `tests/models/test_driver.py`, `tests/core/test_harness.py`

**Interfaces:**
- Produces: the `ToolRuntime` protocol gains `budget_exhausted() -> bool`. `_DelegateRuntime` implements it. `ToolCallingDriver.run_delegate` breaks out of its turn loop into the forced `finish` call as soon as `budget_exhausted()` is true.

- [ ] **Step 1: Write the failing tests**

In `tests/models/test_driver.py`, add this method to the existing `Runtime` test class, after `report_usage`:
```python
    def budget_exhausted(self):
        return False
```
and append:
```python
class ExhaustedAfterFirstBatch(Runtime):
    def budget_exhausted(self):
        return len(self.batches) >= 1


async def test_delegate_stops_calling_model_once_budget_exhausted():
    client = FakeLLMClient([
        ChatResponse(tool_calls=[tc(1)]),
        ChatResponse(tool_calls=[tc(2, name="finish", ranked_keys=["k"])]),
        ChatResponse(tool_calls=[tc(3)]),
        ChatResponse(tool_calls=[tc(4)]),
    ])
    rt = ExhaustedAfterFirstBatch()
    res = await ToolCallingDriver(client).run_delegate(Query.of("q"), TOOLS, rt, Budget(max_turns=4))
    assert len(client.requests) == 2 and len(rt.batches) == 1
    assert client.requests[1]["tool_choice"] == "finish"
    assert res.ranked_keys == ["k"]
```

Append to `tests/core/test_harness.py`:
```python
async def test_model_mode_cost_budget_cuts_off_model_calls(docs_backend):
    from agentic_search.core.types import ModelUsage
    from agentic_search.models.driver import ToolCallingDriver
    from agentic_search.models.llm import ChatResponse
    from agentic_search.testing import FakeLLMClient

    usage = ModelUsage(cost_usd=0.3)
    client = FakeLLMClient([ChatResponse(tool_calls=[lex(t, id=t)], usage=usage)
                            for t in ("headache", "fever", "pain", "castles", "press")])
    res = await make(docs_backend, ToolCallingDriver(client)).search(
        "q", mode="model", budget=Budget(max_cost_usd=0.5, max_turns=4))
    assert len(client.requests) == 3  # two tool turns, then the forced finish — not max_turns
    assert client.requests[2]["tool_choice"] == "finish"
    assert res.stop_reason is StopReason.BUDGET_COST
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/models/test_driver.py tests/core/test_harness.py -q`
Expected:
- `test_delegate_stops_calling_model_once_budget_exhausted` FAILS with `assert None == 'finish'`: the second request is not the forced finish.
- `test_model_mode_cost_budget_cuts_off_model_calls` FAILS: 5 requests are made instead of 3.

- [ ] **Step 3: Implement**

In `src/agentic_search/models/base.py`, add to the `ToolRuntime` protocol after `report_usage`:
```python
    def budget_exhausted(self) -> bool:
        """True once the harness has refused tool calls for budget reasons; drivers stop calling
        the model and finish."""
        ...
```

In `src/agentic_search/core/harness.py`, add to `_DelegateRuntime`, before `unreported`:
```python
    def budget_exhausted(self) -> bool:
        return self.exhausted is not None or self.controller.budget_stop(self.state) is not None
```

In `src/agentic_search/models/driver.py` `run_delegate`, directly after the `messages.extend(... zip(calls, outputs, strict=True))` statement inside the loop, add:
```python
            if runtime.budget_exhausted():
                break
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest -q`
Expected: all pass, including the existing `test_model_mode_enforces_cost_budget_without_double_counting`.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "fix(model-mode): stop calling the model once the delegate budget is exhausted

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Dependencies, test services, and the contract suite

**Files:**
- Modify: `pyproject.toml`
- Create: `docker-compose.yml`, `tests/contract/__init__.py` (empty), `tests/contract/corpus.py`, `tests/contract/test_backend_contract.py`

**Interfaces:**
- Produces:
  - Extras `postgres`, `mysql`, `bigquery` and `opensearch`, with all of them in the dev group.
  - The `integration` marker, which is excluded by default.
  - In `tests/contract/corpus.py`: `EMBEDDER = HashEmbedder(dim=64, id="hash64")`, `ROWS` (d1–d5), `PG_DSN`, `MYSQL_DSN`, `OPENSEARCH_URL`, `documents()`, and the async `seed_postgres()`, `seed_mysql()` and `seed_opensearch()`.
  - The contract suite, parametrized over `files` (always runs), `postgres`, `mysql` and `opensearch` (docker-backed). It imports each adapter lazily inside `make_backend`, so it runs before Tasks 6–8 exist.

- [ ] **Step 1: Update `pyproject.toml`**

Add to `[project.optional-dependencies]`, after `local = [...]`:
```toml
postgres = ["psycopg[binary,pool]>=3.2", "sqlglot>=25"]
mysql = ["aiomysql>=0.2", "sqlglot>=25"]
bigquery = ["google-cloud-bigquery>=3.25", "sqlglot>=25"]
opensearch = ["opensearch-py[async]>=2.6"]
```
Add to the `dev` dependency group, after `"openai>=1.50",`:
```toml
  "sqlglot>=25",
  "psycopg[binary,pool]>=3.2",
  "aiomysql>=0.2",
  "google-cloud-bigquery>=3.25",
  "opensearch-py[async]>=2.6",
```
Replace the `markers` entries' closing bracket, `addopts`, and add `filterwarnings` so the pytest section ends:
```toml
markers = [
  "live: calls real network services (skipped by default)",
  "slow: downloads or runs local models",
  "integration: needs docker compose services (AGENTIC_SEARCH_INTEGRATION=1)",
]
addopts = "-m 'not live and not slow and not integration'"
filterwarnings = [
  "ignore:enable_cleanup_closed ignored:DeprecationWarning",
]
```
(The filter silences a deprecation warning emitted inside aiohttp, which opensearch-py uses, on Python 3.12.)

Run: `uv sync`

- [ ] **Step 2: Create the test services**

`docker-compose.yml`:
```yaml
# Test services for the backend contract suite: `docker compose up -d --wait`, then
# `AGENTIC_SEARCH_INTEGRATION=1 uv run pytest -m integration`.
name: agentic-search-test
services:
  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_PASSWORD: agentic
      POSTGRES_DB: agentic
    ports: ["55432:5432"]
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres -d agentic"]
      interval: 2s
      retries: 30
  mysql:
    image: mysql:8.4
    environment:
      MYSQL_ROOT_PASSWORD: agentic
      MYSQL_DATABASE: agentic
    ports: ["53306:3306"]
    healthcheck:
      test: ["CMD-SHELL", "mysqladmin ping -h 127.0.0.1 -uroot -pagentic --silent"]
      interval: 2s
      retries: 60
  opensearch:
    image: opensearchproject/opensearch:2.17.1
    environment:
      discovery.type: single-node
      DISABLE_SECURITY_PLUGIN: "true"
      DISABLE_INSTALL_DEMO_CONFIG: "true"
      OPENSEARCH_JAVA_OPTS: "-Xms512m -Xmx512m"
    ports: ["59200:9200"]
    healthcheck:
      test: ["CMD-SHELL", "curl -sf http://localhost:9200/_cluster/health || exit 1"]
      interval: 3s
      retries: 60
```

Run: `docker compose up -d --wait`
Expected: the three containers report `Healthy`. If Docker is not running, start Docker Desktop first.

- [ ] **Step 3: Write the corpus and seeding helpers**

`tests/contract/corpus.py`:
```python
"""The shared fixture corpus every backend is seeded with, plus per-service seeding helpers.
Seeding is test-only and uses separate write connections; backends themselves are read-only."""

from __future__ import annotations

import json

from agentic_search.core.types import Document, TextPart
from agentic_search.embedders.local import HashEmbedder

EMBEDDER = HashEmbedder(dim=64, id="hash64")
ROWS = [
    ("d1", "Aspirin", "Aspirin reduces fever and relieves headache pain", "drug", 2020),
    ("d2", "Ibuprofen", "Ibuprofen is an anti-inflammatory used for pain", "drug", 2021),
    ("d3", "Printing press", "The history of the printing press in Europe", "history", 1999),
    ("d4", "Acetaminophen", "Acetaminophen treats headache and fever", "drug", 2019),
    ("d5", "Castles", "Medieval castles and their architecture", "history", 2005),
]
PG_DSN = "postgresql://postgres:agentic@localhost:55432/agentic"
MYSQL_DSN = "mysql://root:agentic@127.0.0.1:53306/agentic"
OPENSEARCH_URL = "http://localhost:59200"


async def embeddings() -> list[list[float]]:
    return await EMBEDDER.embed([TextPart(text=f"{t}\n{b}") for _, t, b, _, _ in ROWS], "document")


def documents() -> list[Document]:
    return [Document(doc_id=i, content=[TextPart(text=f"{t}\n{b}")],
                     metadata={"title": t, "type": ty, "year": y}) for i, t, b, ty, y in ROWS]


async def seed_postgres() -> None:
    import psycopg

    vectors = await embeddings()
    async with await psycopg.AsyncConnection.connect(PG_DSN, autocommit=True) as conn:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        await conn.execute("DROP TABLE IF EXISTS docs")
        await conn.execute("CREATE TABLE docs (id TEXT PRIMARY KEY, title TEXT, body TEXT, "
                           "type VARCHAR(20), year INT, embedding vector(64))")
        for (i, t, b, ty, y), v in zip(ROWS, vectors):
            await conn.execute("INSERT INTO docs VALUES (%s, %s, %s, %s, %s, %s::vector)",
                               [i, t, b, ty, y, "[" + ",".join(map(str, v)) + "]"])
        await conn.execute("ANALYZE docs")


async def seed_mysql() -> None:
    import aiomysql

    conn = await aiomysql.connect(host="127.0.0.1", port=53306, user="root", password="agentic",
                                  db="agentic", autocommit=True)
    try:
        async with conn.cursor() as cur:
            await cur.execute("SET sql_notes = 0")
            await cur.execute("DROP TABLE IF EXISTS docs")
            await cur.execute("CREATE TABLE docs (id VARCHAR(20) PRIMARY KEY, title VARCHAR(200), "
                              "body TEXT, type VARCHAR(20), year INT, "
                              "FULLTEXT KEY ft_docs (title, body)) ENGINE=InnoDB")
            await cur.executemany("INSERT INTO docs VALUES (%s, %s, %s, %s, %s)", ROWS)
            await cur.execute("ANALYZE TABLE docs")
    finally:
        conn.close()


async def seed_opensearch() -> None:
    from opensearchpy import AsyncOpenSearch

    vectors = await embeddings()
    client = AsyncOpenSearch(hosts=[OPENSEARCH_URL])
    try:
        if await client.indices.exists(index="docs"):
            await client.indices.delete(index="docs")
        await client.indices.create(index="docs", body={
            "settings": {"index": {"knn": True, "number_of_shards": 1, "number_of_replicas": 0}},
            "mappings": {"properties": {
                "title": {"type": "text"}, "body": {"type": "text"},
                "type": {"type": "keyword"}, "year": {"type": "integer"},
                "embedding": {"type": "knn_vector", "dimension": 64, "method": {
                    "name": "hnsw", "engine": "lucene", "space_type": "cosinesimil"}}}}})
        lines = []
        for (i, t, b, ty, y), v in zip(ROWS, vectors):
            lines.append(json.dumps({"index": {"_index": "docs", "_id": i}}))
            lines.append(json.dumps({"title": t, "body": b, "type": ty, "year": y, "embedding": v}))
        await client.bulk(body="\n".join(lines) + "\n", refresh=True)
    finally:
        await client.close()
```

- [ ] **Step 4: Write the contract suite**

`tests/contract/test_backend_contract.py`:
```python
"""Every backend must pass this suite. `files` always runs; the docker-backed params run with
`docker compose up -d --wait` and `AGENTIC_SEARCH_INTEGRATION=1 uv run pytest -m integration`."""

import os

import pytest

from agentic_search import Harness
from agentic_search.backends.base import Backend, BackendError, UnsupportedOperation
from agentic_search.backends.files import FilesBackend
from agentic_search.core.types import (
    Aggregate,
    Capability,
    Eq,
    Fetch,
    FilterOnly,
    Lexical,
    Native,
    Range,
    Regex,
    StructuredPart,
    TextPart,
    Traverse,
    Vector,
)
from agentic_search.testing import KeywordJudge, ScriptedDriver, call

from . import corpus

INTEGRATION = os.environ.get("AGENTIC_SEARCH_INTEGRATION") == "1"
docker = pytest.mark.skipif(not INTEGRATION, reason="set AGENTIC_SEARCH_INTEGRATION=1 with docker compose up")

NATIVE = {
    "postgres": ("SELECT id FROM docs WHERE type = 'history'", "DELETE FROM docs"),
    "mysql": ("SELECT id FROM docs WHERE type = 'history'", "DELETE FROM docs"),
    "opensearch": ('{"query": {"term": {"type": "history"}}}', '{"query": {"match_all": {}}, "script": {}}'),
}
NATIVE_DIALECT = {"postgres": "sql", "mysql": "sql", "opensearch": "opensearch_dsl"}


async def make_backend(kind: str) -> Backend:
    if kind == "files":
        return FilesBackend.from_documents("files", corpus.documents(), embedder=corpus.EMBEDDER,
                                           collection="docs")
    if kind == "postgres":
        from agentic_search.backends.postgres import PostgresBackend
        await corpus.seed_postgres()
        return PostgresBackend("pg", corpus.PG_DSN, tables=["docs"],
                               embedders={"docs.embedding": "hash64"}, native_query=True)
    if kind == "mysql":
        from agentic_search.backends.mysql import MySQLBackend
        await corpus.seed_mysql()
        return MySQLBackend("my", corpus.MYSQL_DSN, tables=["docs"], native_query=True)
    if kind == "opensearch":
        from agentic_search.backends.opensearch import OpenSearchBackend
        await corpus.seed_opensearch()
        return OpenSearchBackend("os", corpus.OPENSEARCH_URL, indices=["docs"],
                                 embedders={"docs.embedding": "hash64"}, native_query=True)
    raise AssertionError(kind)


@pytest.fixture(params=[
    "files",
    pytest.param("postgres", marks=[pytest.mark.integration, docker]),
    pytest.param("mysql", marks=[pytest.mark.integration, docker]),
    pytest.param("opensearch", marks=[pytest.mark.integration, docker]),
])
async def backend(request):
    b = await make_backend(request.param)
    b.kind = request.param
    yield b
    await b.close()


def ids(hits):
    return [h.doc_id for h in hits]


def text(hit):
    return "\n".join(p.text for p in hit.content if isinstance(p, TextPart))


async def test_discover(backend):
    assert isinstance(backend, Backend)
    m = await backend.discover()
    coll = m.resolve_collection("docs")
    assert coll is not None and coll.count in (None, 5)
    assert {"type", "year"} <= {f.name for f in coll.fields}
    assert {Capability.LEXICAL, Capability.FILTER, Capability.FETCH, Capability.AGGREGATE} <= m.capabilities
    if Capability.VECTOR in m.capabilities:
        emb = coll.field("embedding")
        assert emb.embedder_id == "hash64" and emb.vector_dim == 64


async def test_lexical(backend):
    hits = await backend.execute(Lexical(source=backend.name, collection="docs", text="headache", limit=5))
    assert set(ids(hits)) == {"d1", "d4"}
    assert all(h.raw_score is not None for h in hits)


async def test_lexical_with_filter(backend):
    hits = await backend.execute(Lexical(source=backend.name, collection="docs", text="pain",
                                         filter=Eq(field="year", value=2021)))
    assert ids(hits) == ["d2"]


async def test_filters(backend):
    drugs = await backend.execute(FilterOnly(source=backend.name, collection="docs",
                                             filter=Eq(field="type", value="drug")))
    assert set(ids(drugs)) == {"d1", "d2", "d4"}
    recent = await backend.execute(FilterOnly(source=backend.name, collection="docs",
                                              filter=Range(field="year", gte=2020)))
    assert set(ids(recent)) == {"d1", "d2"}
    two = await backend.execute(FilterOnly(source=backend.name, collection="docs",
                                           filter=Range(field="year", gte=0), limit=2))
    assert len(two) == 2


async def test_vector(backend):
    m = await backend.discover()
    if Capability.VECTOR not in m.capabilities:
        pytest.skip("no vector support")
    [q] = await corpus.EMBEDDER.embed([TextPart(text="headache fever")], "query")
    hits = await backend.execute(Vector(source=backend.name, collection="docs", field="embedding",
                                        hyde_text="x", vector=q, limit=3))
    assert hits[0].doc_id in {"d1", "d4"}


async def test_regex_fetch_aggregate(backend):
    rx = await backend.execute(Regex(source=backend.name, collection="docs", pattern="print.*"))
    assert ids(rx) == ["d3"]
    [f] = await backend.execute(Fetch(source=backend.name, collection="docs", doc_ids=["d3"]))
    assert f.doc_id == "d3" and "printing" in text(f).lower()
    agg = await backend.execute(Aggregate(source=backend.name, collection="docs", group_by=["type"]))
    counts = {h.content[0].data["type"]: h.content[0].data["count"] for h in agg
              if isinstance(h.content[0], StructuredPart)}
    assert counts == {"drug": 3, "history": 2}


async def test_native_read_only(backend):
    m = await backend.discover()
    if Capability.NATIVE not in m.capabilities:
        pytest.skip("native queries not enabled")
    select, write = NATIVE[backend.kind]
    dialect = NATIVE_DIALECT[backend.kind]
    rows = await backend.execute(Native(source=backend.name, collection="docs", dialect=dialect, query=select))
    assert len(rows) == 2
    with pytest.raises(BackendError):
        await backend.execute(Native(source=backend.name, collection="docs", dialect=dialect, query=write))


async def test_sql_sessions_are_read_only(backend):
    from agentic_search.backends.sql_backend import SqlBackend

    if not isinstance(backend, SqlBackend):
        pytest.skip("not a SQL backend")
    await backend.discover()
    with pytest.raises(BackendError):
        await backend._query("DELETE FROM docs WHERE id = 'd1'", None)
    [still] = await backend.execute(Fetch(source=backend.name, collection="docs", doc_ids=["d1"]))
    assert still.doc_id == "d1"


async def test_unsupported_op(backend):
    with pytest.raises((UnsupportedOperation, BackendError)):
        await backend.execute(Traverse(source=backend.name, collection="docs", start=Eq(field="type", value="x")))


async def test_harness_end_to_end(backend):
    driver = ScriptedDriver([[call("lexical_search", source=backend.name, collection="docs", text="headache")]])
    h = Harness([backend], driver, embedders=[corpus.EMBEDDER], analyzer=KeywordJudge(["headache"]))
    res = await h.search("what treats headache?")
    assert set(res.keys()) == {f"{backend.name}:d1", f"{backend.name}:d4"}
    assert all(r.judged and r.p_relevant == 1.0 for r in res.hits)
```

- [ ] **Step 5: Run it (files only; the adapters don't exist yet)**

Run: `uv run pytest tests/contract -q`
Expected: `8 passed, 2 skipped` (files has no native queries, and the read-only-session check applies to SQL backends only). The docker-backed params are deselected.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "test: backend contract suite, shared corpus and docker-compose services

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: SQL building blocks

**Files:**
- Create: `src/agentic_search/backends/sql.py`
- Test: `tests/backends/test_sql.py`

**Interfaces:**
- Produces:
  - `Dialect = Literal["postgres","mysql","bigquery"]`, and `quote_ident(name, dialect)`, which raises `BackendError` on a non-identifier.
  - `Params(dialect)` with `.add(value) -> placeholder`, returning `%s` for postgres/mysql and `@pN` for bigquery; `.values` holds the bound values.
  - `filter_sql(f, dialect, params, columns) -> str` and `where_clause(f, dialect, params, columns, extra=None) -> str`.
  - `parse_metric(metric, columns, dialect) -> (sql_expr, alias)`, accepting `count` and `sum|avg|min|max:<col>`.
  - `jsonable(value)`, `row_to_hit(row, *, source, id_column, text_columns, score=None, fallback_id="") -> Hit`, `vector_literal(vec) -> str` and `dumps(value) -> str`.

- [ ] **Step 1: Write the failing tests**

`tests/backends/test_sql.py`:
```python
import datetime as dt
import decimal

import pytest

from agentic_search.backends.base import BackendError
from agentic_search.backends.sql import (
    Params,
    filter_sql,
    parse_metric,
    quote_ident,
    row_to_hit,
    vector_literal,
    where_clause,
)
from agentic_search.core.types import (
    And,
    Contains,
    Eq,
    Exists,
    In,
    Not,
    Or,
    Range,
    StructuredPart,
    TextPart,
)

COLS = {"type", "year", "title", "body"}


def test_quote_ident():
    assert quote_ident("year", "postgres") == '"year"'
    assert quote_ident("year", "mysql") == "`year`"
    assert quote_ident("year", "bigquery") == "`year`"
    with pytest.raises(BackendError):
        quote_ident('x"; DROP TABLE t; --', "postgres")


def test_params_styles():
    p = Params("postgres")
    assert (p.add(1), p.add("a"), p.values) == ("%s", "%s", [1, "a"])
    b = Params("bigquery")
    assert (b.add(1), b.add(2)) == ("@p0", "@p1")


def test_filter_translation_postgres():
    p = Params("postgres")
    f = And(clauses=[Eq(field="type", value="drug"),
                     Or(clauses=[Range(field="year", gte=2020, lt=2022), Not(clause=Exists(field="title"))]),
                     In(field="type", values=["a", "b"]), Contains(field="body", value="50%_off")])
    sql = filter_sql(f, "postgres", p, COLS)
    assert sql == ('("type" = %s AND (("year" >= %s AND "year" < %s) OR (NOT "title" IS NOT NULL)) '
                   'AND "type" IN (%s, %s) AND CAST("body" AS TEXT) ILIKE %s)')
    assert p.values == ["drug", 2020, 2022, "a", "b", "%50\\%\\_off%"]


def test_filter_translation_other_dialects():
    m = Params("mysql")
    assert filter_sql(Contains(field="body", value="x"), "mysql", m, COLS) == \
        "LOWER(CAST(`body` AS CHAR)) LIKE LOWER(%s)"
    b = Params("bigquery")
    assert filter_sql(Contains(field="body", value="x"), "bigquery", b, COLS) == \
        "CONTAINS_SUBSTR(CAST(`body` AS STRING), @p0)"
    assert filter_sql(In(field="type", values=[]), "mysql", Params("mysql"), COLS) == "FALSE"
    assert filter_sql(And(clauses=[]), "mysql", Params("mysql"), COLS) == "TRUE"


def test_filter_rejects_unknown_columns():
    with pytest.raises(BackendError, match="unknown column"):
        filter_sql(Eq(field="nope", value=1), "postgres", Params("postgres"), COLS)


def test_where_clause_and_metrics():
    p = Params("postgres")
    assert where_clause(None, "postgres", p, COLS) == ""
    assert where_clause(Eq(field="year", value=1), "postgres", p, COLS, extra=["x IS NOT NULL"]) == \
        ' WHERE x IS NOT NULL AND "year" = %s'
    assert parse_metric("count", COLS, "mysql") == ("COUNT(*)", "count")
    assert parse_metric("avg:year", COLS, "mysql") == ("AVG(`year`)", "avg_year")
    for bad in ("median:year", "sum:", "sum:nope"):
        with pytest.raises(BackendError):
            parse_metric(bad, COLS, "mysql")


def test_row_to_hit():
    row = {"id": 7, "title": "T", "body": "B", "price": decimal.Decimal("1.5"),
           "at": dt.date(2024, 1, 2)}
    h = row_to_hit(row, source="s", id_column="id", text_columns=["title", "body"], score=0.5)
    assert h.key == "s:7" and h.content == [TextPart(text="T\nB")] and h.raw_score == 0.5
    assert h.metadata == {"id": 7, "price": 1.5, "at": "2024-01-02"}
    s = row_to_hit({"n": 1}, source="s", id_column=None, text_columns=[], fallback_id="native:0")
    assert s.doc_id == "native:0" and isinstance(s.content[0], StructuredPart)


def test_vector_literal():
    assert vector_literal([0.5, 1.0, -2.25]) == "[0.5,1,-2.25]"
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/backends/test_sql.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'agentic_search.backends.sql'`

- [ ] **Step 3: Implement**

`src/agentic_search/backends/sql.py`:
```python
"""Shared SQL building blocks: identifier quoting, parameter collection, Filter AST → WHERE clause,
and row → Hit conversion for the Postgres, MySQL and BigQuery adapters."""

from __future__ import annotations

import datetime as dt
import decimal
import json
import re
import uuid
from typing import Any, Literal

from agentic_search.backends.base import BackendError
from agentic_search.core.types import (
    And,
    Contains,
    Content,
    Eq,
    Exists,
    Filter,
    Hit,
    In,
    Not,
    Or,
    Range,
    StructuredPart,
    TextPart,
)

Dialect = Literal["postgres", "mysql", "bigquery"]
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
AGG_FUNCS = {"sum": "SUM", "avg": "AVG", "min": "MIN", "max": "MAX"}


def quote_ident(name: str, dialect: Dialect) -> str:
    """Quote a column/table name after checking it is a plain identifier (no injection surface)."""
    if not _IDENT.match(name):
        raise BackendError(f"invalid identifier {name!r}")
    if dialect == "postgres":
        return f'"{name}"'
    return f"`{name}`"


class Params:
    """Collects bind values in placeholder order: %s for postgres/mysql, @pN for bigquery."""

    def __init__(self, dialect: Dialect):
        self.dialect = dialect
        self.values: list[Any] = []

    def add(self, value: Any) -> str:
        self.values.append(value)
        return f"@p{len(self.values) - 1}" if self.dialect == "bigquery" else "%s"


def _like_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def filter_sql(f: Filter, dialect: Dialect, params: Params, columns: set[str]) -> str:
    """Translate a filter to a parameterized boolean SQL expression over known columns."""
    if isinstance(f, And):
        return "(" + " AND ".join(filter_sql(c, dialect, params, columns) for c in f.clauses) + ")" \
            if f.clauses else "TRUE"
    if isinstance(f, Or):
        return "(" + " OR ".join(filter_sql(c, dialect, params, columns) for c in f.clauses) + ")" \
            if f.clauses else "FALSE"
    if isinstance(f, Not):
        return f"(NOT {filter_sql(f.clause, dialect, params, columns)})"
    if f.field not in columns:
        raise BackendError(f"unknown column {f.field!r}")
    col = quote_ident(f.field, dialect)
    if isinstance(f, Eq):
        return f"{col} = {params.add(f.value)}"
    if isinstance(f, In):
        if not f.values:
            return "FALSE"
        return f"{col} IN ({', '.join(params.add(v) for v in f.values)})"
    if isinstance(f, Range):
        parts = [f"{col} {sym} {params.add(bound)}" for bound, sym in
                 ((f.gte, ">="), (f.gt, ">"), (f.lte, "<="), (f.lt, "<")) if bound is not None]
        return "(" + " AND ".join(parts) + ")" if parts else "TRUE"
    if isinstance(f, Exists):
        return f"{col} IS NOT NULL"
    if isinstance(f, Contains):
        if dialect == "bigquery":
            return f"CONTAINS_SUBSTR(CAST({col} AS STRING), {params.add(f.value)})"
        pattern = params.add(f"%{_like_escape(f.value)}%")
        if dialect == "postgres":
            return f"CAST({col} AS TEXT) ILIKE {pattern}"
        return f"LOWER(CAST({col} AS CHAR)) LIKE LOWER({pattern})"
    raise BackendError(f"unsupported filter node {type(f).__name__}")


def where_clause(f: Filter | None, dialect: Dialect, params: Params, columns: set[str],
                 extra: list[str] | None = None) -> str:
    parts = list(extra or [])
    if f is not None:
        parts.append(filter_sql(f, dialect, params, columns))
    return " WHERE " + " AND ".join(parts) if parts else ""


def parse_metric(metric: str, columns: set[str], dialect: Dialect) -> tuple[str, str]:
    """'count' → COUNT(*); 'sum:price' → SUM(`price`). Returns (sql_expr, output_name)."""
    if metric == "count":
        return "COUNT(*)", "count"
    fn, _, col = metric.partition(":")
    if fn not in AGG_FUNCS or not col:
        raise BackendError(f"unsupported metric {metric!r}; use count or sum|avg|min|max:<column>")
    if col not in columns:
        raise BackendError(f"unknown column {col!r} in metric {metric!r}")
    return f"{AGG_FUNCS[fn]}({quote_ident(col, dialect)})", f"{fn}_{col}"


def jsonable(value: Any) -> Any:
    """Make DB values JSON-friendly for Hit metadata / StructuredPart."""
    if isinstance(value, (dt.datetime, dt.date, dt.time)):
        return value.isoformat()
    if isinstance(value, decimal.Decimal):
        return float(value)
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value).hex()
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    return value


def row_to_hit(row: dict[str, Any], *, source: str, id_column: str | None, text_columns: list[str],
               score: float | None = None, fallback_id: str = "") -> Hit:
    """TEXT columns become the hit's text content; everything else becomes metadata."""
    clean = {k: jsonable(v) for k, v in row.items()}
    doc_id = str(clean.get(id_column)) if id_column and clean.get(id_column) is not None else fallback_id
    texts = [str(clean[c]) for c in text_columns if clean.get(c) not in (None, "")]
    content: list[Content] = [TextPart(text="\n".join(texts))] if texts else [StructuredPart(data=clean)]
    metadata = {k: v for k, v in clean.items() if k not in text_columns}
    return Hit(doc_id=doc_id, source=source, content=content, metadata=metadata, raw_score=score)


def vector_literal(vector: list[float]) -> str:
    """pgvector text form: '[0.1,0.2,…]'."""
    return "[" + ",".join(f"{x:.8g}" for x in vector) + "]"


def dumps(value: Any) -> str:
    return json.dumps(jsonable(value), sort_keys=True, default=str)
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/backends/test_sql.py -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(backends): SQL identifier quoting, parameters and filter translation

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Native query guard

**Files:**
- Create: `src/agentic_search/backends/native_guard.py`
- Test: `tests/backends/test_native_guard.py`

**Interfaces:**
- Produces:
  - `NativeQueryRejected(BackendError)`.
  - `guard_sql(query, dialect, max_rows) -> str`: allows one SELECT or set operation; rejects INSERT/UPDATE/DELETE/MERGE/DDL/INTO/locks/TRUNCATE anywhere, including inside CTEs; caps an existing LIMIT or adds one.
  - `guard_cypher(query, max_rows) -> str`, used by Plan 3's Neo4j backend.
  - `guard_opensearch(body_json, max_rows) -> dict`.

- [ ] **Step 1: Write the failing tests**

`tests/backends/test_native_guard.py`:
```python
import pytest

from agentic_search.backends.native_guard import (
    NativeQueryRejected,
    guard_cypher,
    guard_opensearch,
    guard_sql,
)

WRITES = [
    "DELETE FROM docs",
    "UPDATE docs SET title = 'x'",
    "INSERT INTO docs (id) VALUES ('z')",
    "DROP TABLE docs",
    "CREATE TABLE t (x int)",
    "SELECT 1; DELETE FROM docs",
    "WITH d AS (DELETE FROM docs RETURNING *) SELECT * FROM d",
    "SELECT * INTO copy FROM docs",
    "SELECT * FROM docs FOR UPDATE",
    "TRUNCATE docs",
]


@pytest.mark.parametrize("query", WRITES)
@pytest.mark.parametrize("dialect", ["postgres", "mysql"])
def test_sql_rejects_writes(query, dialect):
    if dialect == "mysql" and query.startswith(("WITH d AS", "SELECT * INTO")):
        pytest.skip("postgres-only syntax")
    with pytest.raises(NativeQueryRejected):
        guard_sql(query, dialect, 100)


def test_sql_limits():
    assert guard_sql("SELECT id FROM docs", "postgres", 50) == "SELECT id FROM docs LIMIT 50"
    assert guard_sql("SELECT id FROM docs LIMIT 5", "mysql", 50) == "SELECT id FROM docs LIMIT 5"
    assert guard_sql("SELECT id FROM docs LIMIT 500", "postgres", 50) == "SELECT id FROM docs LIMIT 50"
    assert "LIMIT 10" in guard_sql("SELECT a FROM x UNION ALL SELECT a FROM y", "bigquery", 10)
    with pytest.raises(NativeQueryRejected, match="parse"):
        guard_sql("SELEC nonsense((", "postgres", 10)


def test_cypher_guard():
    assert guard_cypher("MATCH (n:Drug) RETURN n", 20) == "MATCH (n:Drug) RETURN n LIMIT 20"
    assert guard_cypher("MATCH (n) RETURN n LIMIT 5;", 20) == "MATCH (n) RETURN n LIMIT 5"
    assert guard_cypher("MATCH (n) RETURN n LIMIT 500", 20) == "MATCH (n) RETURN n LIMIT 20"
    assert guard_cypher("MATCH (n) WHERE n.name = 'set' RETURN n.offset", 5).endswith("LIMIT 5")
    assert guard_cypher("CALL db.index.fulltext.queryNodes('i', 'x') YIELD node RETURN node", 5)
    for bad in ("MATCH (n) DETACH DELETE n", "MERGE (n:X)", "MATCH (n) SET n.x = 1",
                "CALL dbms.components()", "MATCH (n) RETURN n; MATCH (m) DELETE m",
                "LOAD CSV FROM 'x' AS row RETURN row"):
        with pytest.raises(NativeQueryRejected):
            guard_cypher(bad, 5)


def test_opensearch_guard():
    body = guard_opensearch('{"query": {"term": {"type": "drug"}}, "size": 500}', 50)
    assert body == {"query": {"term": {"type": "drug"}}, "size": 50}
    assert guard_opensearch('{"query": {"match_all": {}}}', 50)["size"] == 50
    for bad in ('[1]', 'not json', '{"script": {}}', '{"query": {"script_score": {}}}',
                '{"query": {}, "index": "x"}'):
        with pytest.raises(NativeQueryRejected):
            guard_opensearch(bad, 50)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/backends/test_native_guard.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'agentic_search.backends.native_guard'`

- [ ] **Step 3: Implement**

`src/agentic_search/backends/native_guard.py`:
```python
"""Read-only gate for `native_query`: SQL (via sqlglot), Cypher (keyword scan), OpenSearch DSL.

Every guard either returns a query that is safe to run (with a row cap applied) or raises
NativeQueryRejected. Adapters still connect with read-only sessions; this is the first line."""

from __future__ import annotations

import json
import re
from typing import Any

from agentic_search.backends.base import BackendError


class NativeQueryRejected(BackendError):
    """The native query is not a single read-only statement."""


def guard_sql(query: str, dialect: str, max_rows: int) -> str:
    """Allow exactly one SELECT/set-operation with no DML/DDL/locking; cap or add LIMIT."""
    import sqlglot
    from sqlglot import exp

    try:
        statements = [s for s in sqlglot.parse(query, read=dialect) if s is not None]
    except sqlglot.errors.ParseError as exc:
        raise NativeQueryRejected(f"could not parse SQL: {str(exc)[:200]}") from exc
    if len(statements) != 1:
        raise NativeQueryRejected("native SQL must be exactly one statement")
    stmt = statements[0]
    if not isinstance(stmt, exp.Query):
        raise NativeQueryRejected(f"only SELECT queries are allowed, got {stmt.key.upper()}")
    forbidden = (exp.Insert, exp.Update, exp.Delete, exp.Merge, exp.Create, exp.Drop, exp.Alter,
                 exp.Command, exp.Into, exp.Lock, exp.TruncateTable)
    for node in stmt.walk():
        if isinstance(node, forbidden):
            raise NativeQueryRejected(f"{type(node).__name__.upper()} is not allowed in native SQL")
    limit = stmt.args.get("limit")
    current = None
    if limit is not None:
        literal = limit.expression if hasattr(limit, "expression") else None
        if isinstance(literal, exp.Literal) and literal.is_int:
            current = int(literal.this)
    if current is None or current > max_rows:
        stmt = stmt.limit(max_rows)
    return stmt.sql(dialect=dialect)


_CYPHER_WRITE = re.compile(
    r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|FOREACH|LOAD\s+CSV)\b", re.IGNORECASE)
_CYPHER_CALL = re.compile(r"\bCALL\s+([A-Za-z0-9_.]+)", re.IGNORECASE)
_CYPHER_ALLOWED_PROCS = {"db.index.fulltext.querynodes", "db.index.vector.querynodes",
                         "db.labels", "db.relationshiptypes", "db.propertykeys"}
_CYPHER_LIMIT = re.compile(r"\bLIMIT\s+(\d+)\s*;?\s*$", re.IGNORECASE)
_CYPHER_STRING = re.compile(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"")


def guard_cypher(query: str, max_rows: int) -> str:
    """Reject write clauses and non-allowlisted procedures; cap or add a trailing LIMIT."""
    code = _CYPHER_STRING.sub("''", query)  # ignore keywords inside string literals
    if ";" in code.strip().rstrip(";"):
        raise NativeQueryRejected("native Cypher must be exactly one statement")
    match = _CYPHER_WRITE.search(code)
    if match:
        raise NativeQueryRejected(f"{match.group(1).upper()} is not allowed in native Cypher")
    for proc in _CYPHER_CALL.findall(code):
        if proc.lower() not in _CYPHER_ALLOWED_PROCS:
            raise NativeQueryRejected(f"procedure {proc} is not allowed in native Cypher")
    q = query.strip().rstrip(";").rstrip()
    limit = _CYPHER_LIMIT.search(q)
    if limit is None:
        return f"{q} LIMIT {max_rows}"
    if int(limit.group(1)) > max_rows:
        return q[: limit.start()] + f"LIMIT {max_rows}"
    return q


_DSL_TOP_KEYS = {"query", "size", "from", "_source", "sort", "aggs", "aggregations",
                 "highlight", "track_total_hits"}


def _walk_keys(value: Any) -> list[str]:
    if isinstance(value, dict):
        return [k for k in value] + [k2 for v in value.values() for k2 in _walk_keys(v)]
    if isinstance(value, list):
        return [k for v in value for k in _walk_keys(v)]
    return []


def guard_opensearch(body_json: str, max_rows: int) -> dict[str, Any]:
    """Accept only a search body with known top-level keys, no scripts; cap size."""
    try:
        body = json.loads(body_json)
    except json.JSONDecodeError as exc:
        raise NativeQueryRejected(f"native OpenSearch query must be a JSON search body: {exc}") from exc
    if not isinstance(body, dict):
        raise NativeQueryRejected("native OpenSearch query must be a JSON object")
    unknown = set(body) - _DSL_TOP_KEYS
    if unknown:
        raise NativeQueryRejected(f"top-level keys not allowed: {sorted(unknown)}")
    scripted = [k for k in _walk_keys(body) if "script" in k.lower()]
    if scripted:
        raise NativeQueryRejected(f"scripts are not allowed ({scripted[0]})")
    size = body.get("size")
    body["size"] = max_rows if not isinstance(size, int) or size > max_rows else size
    return body
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/backends/test_native_guard.py -q`
Expected: all pass. Two tests are skipped because their syntax is Postgres-only.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(backends): read-only guard for native SQL, Cypher and OpenSearch queries

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: SqlBackend base + PostgresBackend

**Files:**
- Create: `src/agentic_search/backends/sql_backend.py`, `src/agentic_search/backends/postgres.py`
- Test: the contract suite from Task 3 (`-k postgres`)

**Interfaces:**
- Consumes: Tasks 1, 4 and 5; Plan 1's `Backend` protocol, `rrf_merge` and core types.
- Produces:
  - `TableInfo`: `name`, `id_column`, `fields`, `count` and `fulltext`, plus the properties `columns`, `text_columns`, `searchable_columns`, `vector_columns`, `select_columns` and the method `to_collection()`.
  - `field_flags(ftype)`, `any_terms(text)`, and `SqlBackend(name, *, tables=None, id_columns=None, embedders=None, vector_metric="cosine", native_query=False, description=None, max_rows=100, sample_values=True)`.
  - Subclass hooks: `_query`, `_discover_tables`, `_table_ref`, `_as_text`, `_regex_expr`, `_lexical_sql`, `_vector_sql` and `_supports_lexical`.
  - `PostgresBackend(name, dsn, *, schema="public", text_search_config="english", pool_size=4, statement_timeout_ms=30000, **SqlBackend kwargs)`, with `backend_type="postgres"`.
- Key behaviours:
  - Tables need a primary key or an `id_columns` entry; tables without one are listed in the manifest description as skipped.
  - TEXT columns become hit content, and every other non-vector column becomes metadata.
  - Lexical search ORs the query's terms unless the text contains a quote. It defaults to TEXT columns.
  - Vector scores are `1 - cosine distance`.

- [ ] **Step 1: Confirm the contract fails for postgres (RED)**

Run: `AGENTIC_SEARCH_INTEGRATION=1 uv run pytest tests/contract -m integration -k postgres -q`
Expected: every test ERRORs with `ModuleNotFoundError: No module named 'agentic_search.backends.postgres'`

- [ ] **Step 2: Implement the base class**

`src/agentic_search/backends/sql_backend.py`:
```python
"""Base class for SQL backends. Subclasses supply connection handling, table discovery and
dialect-specific lexical/vector SQL; this class implements everything else once."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any

from agentic_search.backends.base import (
    BackendError,
    DiscoverDetail,
    UnsupportedOperation,
    rrf_merge,
)
from agentic_search.backends.native_guard import guard_sql
from agentic_search.backends.sql import (
    Dialect,
    Params,
    dumps,
    parse_metric,
    quote_ident,
    row_to_hit,
    where_clause,
)
from agentic_search.core.types import (
    Aggregate,
    Capability,
    CollectionInfo,
    Fetch,
    FieldSpec,
    FieldType,
    FilterOnly,
    Hit,
    Hybrid,
    Lexical,
    Manifest,
    Native,
    QueryOp,
    Regex,
    StructuredPart,
    Vector,
)

METRICS = ("cosine", "l2", "ip")
_WORD = re.compile(r"\w+")
SAMPLE_DISTINCT_MAX = 20
EXACT_COUNT_BELOW = 100_000


@dataclass
class TableInfo:
    name: str
    id_column: str
    fields: list[FieldSpec]
    count: int | None = None
    fulltext: list[list[str]] = field(default_factory=list)  # MySQL FULLTEXT column sets

    @property
    def columns(self) -> set[str]:
        return {f.name for f in self.fields}

    @property
    def text_columns(self) -> list[str]:
        return [f.name for f in self.fields if f.type is FieldType.TEXT]

    @property
    def searchable_columns(self) -> list[str]:
        return [f.name for f in self.fields if f.searchable]

    @property
    def vector_columns(self) -> dict[str, FieldSpec]:
        return {f.name: f for f in self.fields if f.type is FieldType.VECTOR}

    @property
    def select_columns(self) -> list[str]:
        return [f.name for f in self.fields if f.type is not FieldType.VECTOR]

    def to_collection(self) -> CollectionInfo:
        return CollectionInfo(name=self.name, fields=self.fields, count=self.count)


def field_flags(ftype: FieldType) -> dict[str, bool]:
    return {
        "searchable": ftype in (FieldType.TEXT, FieldType.KEYWORD),
        "filterable": ftype not in (FieldType.JSON, FieldType.VECTOR),
        "sortable": ftype in (FieldType.INT, FieldType.FLOAT, FieldType.DATE),
    }


def any_terms(text: str) -> list[str]:
    """Distinct lowercase word tokens, capped — lexical search is OR-of-terms, ranked."""
    return list(dict.fromkeys(t.lower() for t in _WORD.findall(text)))[:16]


class SqlBackend:
    dialect: Dialect
    backend_type: str
    native_dialects: tuple[str, ...] = ("sql",)

    def __init__(self, name: str, *, tables: list[str] | None = None,
                 id_columns: dict[str, str] | None = None, embedders: dict[str, str] | None = None,
                 vector_metric: str = "cosine", native_query: bool = False,
                 description: str | None = None, max_rows: int = 100, sample_values: bool = True):
        if vector_metric not in METRICS:
            raise ValueError(f"vector_metric must be one of {METRICS}")
        self.name = name
        self.table_names = tables
        self.id_columns = id_columns or {}
        self.embedders = embedders or {}
        self.vector_metric = vector_metric
        self.native_query = native_query
        self.description = description
        self.max_rows = max_rows
        self.sample_values = sample_values
        self.skipped: list[str] = []
        self._tables: dict[str, TableInfo] | None = None
        self._discover_lock = asyncio.Lock()

    # ---- subclass hooks -------------------------------------------------------

    async def _query(self, sql: str, params: list[Any] | None) -> list[dict[str, Any]]:
        raise NotImplementedError

    async def _discover_tables(self) -> dict[str, TableInfo]:
        raise NotImplementedError

    def _table_ref(self, table: str) -> str:
        return quote_ident(table, self.dialect)

    def _as_text(self, expr: str) -> str:
        return f"CAST({expr} AS {'TEXT' if self.dialect == 'postgres' else 'CHAR'})"

    def _regex_expr(self, column_sql: str, placeholder: str) -> str:
        raise NotImplementedError

    def _lexical_sql(self, op: Lexical, table: TableInfo, limit: int) -> tuple[str, list[Any]]:
        raise NotImplementedError

    def _vector_sql(self, op_field: str, vector: list[float], table: TableInfo, op_filter: Any,
                    limit: int) -> tuple[str, list[Any]]:
        raise UnsupportedOperation(f"{self.backend_type} backend does not support vector search")

    def _supports_lexical(self, tables: dict[str, TableInfo]) -> bool:
        return any(t.searchable_columns for t in tables.values())

    # ---- protocol ---------------------------------------------------------------

    def capabilities(self) -> set[Capability]:
        caps = {Capability.FILTER, Capability.REGEX, Capability.AGGREGATE, Capability.FETCH}
        tables = self._tables or {}
        if self._tables is None or self._supports_lexical(tables):
            caps.add(Capability.LEXICAL)
        if any(t.vector_columns for t in tables.values()):
            caps |= {Capability.VECTOR, Capability.HYBRID}
        if self.native_query:
            caps.add(Capability.NATIVE)
        return caps

    async def _ensure_tables(self) -> dict[str, TableInfo]:
        async with self._discover_lock:
            if self._tables is None:
                self._tables = await self._discover_tables()
            return self._tables

    async def discover(self, detail: DiscoverDetail = "full",
                       collection: str | None = None) -> Manifest:
        tables = await self._ensure_tables()
        description = self.description
        if self.skipped:
            note = f"Skipped (no primary key / id column): {', '.join(self.skipped)}."
            description = f"{description} {note}" if description else note
        return Manifest(source=self.name, backend_type=self.backend_type,
                        capabilities=self.capabilities(), description=description,
                        collections=[t.to_collection() for t in tables.values()])

    async def execute(self, op: QueryOp) -> list[Hit]:
        tables = await self._ensure_tables()
        if isinstance(op, Native):
            return await self._native(op, tables)
        table = self._resolve(op.collection, tables)
        limit = min(op.limit, self.max_rows)
        if isinstance(op, Lexical):
            if op.fields and not set(op.fields) <= set(table.searchable_columns):
                raise BackendError(f"not searchable in {table.name}: {sorted(set(op.fields) - set(table.searchable_columns))}")
            sql, params = self._lexical_sql(op, table, limit)
            return self._hits(await self._query(sql, params), table)
        if isinstance(op, Vector):
            return self._hits(await self._query(*self._vector(op.field, op.vector, table, op.filter, limit)), table)
        if isinstance(op, Hybrid):
            return await self._hybrid(op, table, limit)
        if isinstance(op, FilterOnly):
            return await self._filter_only(op, table, limit)
        if isinstance(op, Regex):
            return await self._regex(op, table, limit)
        if isinstance(op, Aggregate):
            return await self._aggregate(op, table, limit)
        if isinstance(op, Fetch):
            return await self._fetch(op, table)
        raise UnsupportedOperation(f"{self.backend_type} backend does not support {op.type}")

    async def close(self) -> None:
        return None

    # ---- shared op implementations -----------------------------------------------

    def _resolve(self, name: str | None, tables: dict[str, TableInfo]) -> TableInfo:
        if name is None:
            if len(tables) == 1:
                return next(iter(tables.values()))
            raise BackendError(f"collection required; one of {sorted(tables)}")
        if name not in tables:
            raise BackendError(f"unknown collection {name!r}; one of {sorted(tables)}")
        return tables[name]

    def _select(self, table: TableInfo) -> str:
        return ", ".join(quote_ident(c, self.dialect) for c in table.select_columns)

    def _hits(self, rows: list[dict[str, Any]], table: TableInfo) -> list[Hit]:
        hits = []
        for row in rows:
            score = row.pop("_score", None)
            hits.append(row_to_hit(row, source=self.name, id_column=table.id_column,
                                   text_columns=table.text_columns,
                                   score=float(score) if score is not None else None))
        return hits

    def _vector(self, column: str, vector: list[float] | None, table: TableInfo, op_filter: Any,
                limit: int) -> tuple[str, list[Any]]:
        if vector is None:
            raise BackendError("vector op reached the backend without an embedded query vector")
        if column not in table.vector_columns:
            raise BackendError(f"{column!r} is not a vector column of {table.name}")
        return self._vector_sql(column, vector, table, op_filter, limit)

    async def _hybrid(self, op: Hybrid, table: TableInfo, limit: int) -> list[Hit]:
        depth = min(max(limit * 5, 50), 500)
        lex_rows = await self._query(*self._lexical_sql(
            Lexical(source=op.source, text=op.text, filter=op.filter, limit=depth), table, depth))
        vec_rows = await self._query(*self._vector(op.field, op.vector, table, op.filter, depth))
        by_id: dict[str, Hit] = {}
        rankings = []
        for rows, weight in ((lex_rows, op.lexical_weight), (vec_rows, 1.0 - op.lexical_weight)):
            hits = self._hits(rows, table)
            for h in hits:
                by_id.setdefault(h.doc_id, h)
            rankings.append(([h.doc_id for h in hits], weight))
        return [by_id[i].model_copy(update={"raw_score": s}) for i, s in rrf_merge(rankings)[:limit]]

    async def _filter_only(self, op: FilterOnly, table: TableInfo, limit: int) -> list[Hit]:
        p = Params(self.dialect)
        where = where_clause(op.filter, self.dialect, p, table.columns)
        sql = (f"SELECT {self._select(table)} FROM {self._table_ref(table.name)}{where} "
               f"ORDER BY {quote_ident(table.id_column, self.dialect)} LIMIT {p.add(limit)}")
        return self._hits(await self._query(sql, p.values), table)

    async def _regex(self, op: Regex, table: TableInfo, limit: int) -> list[Hit]:
        fields = op.fields or table.text_columns or table.searchable_columns
        unknown = set(fields) - table.columns
        if unknown:
            raise BackendError(f"unknown columns {sorted(unknown)}")
        p = Params(self.dialect)
        ors = " OR ".join(self._regex_expr(self._as_text(quote_ident(c, self.dialect)), p.add(op.pattern))
                          for c in fields)
        where = where_clause(op.filter, self.dialect, p, table.columns, extra=[f"({ors})"])
        sql = (f"SELECT {self._select(table)} FROM {self._table_ref(table.name)}{where} "
               f"ORDER BY {quote_ident(table.id_column, self.dialect)} LIMIT {p.add(limit)}")
        return self._hits(await self._query(sql, p.values), table)

    async def _aggregate(self, op: Aggregate, table: TableInfo, limit: int) -> list[Hit]:
        unknown = set(op.group_by) - table.columns
        if unknown:
            raise BackendError(f"unknown columns {sorted(unknown)}")
        groups = [quote_ident(g, self.dialect) for g in op.group_by]
        metrics = [parse_metric(m, table.columns, self.dialect) for m in op.metrics]
        p = Params(self.dialect)
        where = where_clause(op.filter, self.dialect, p, table.columns)
        select = ", ".join(groups + [f"{expr} AS {quote_ident(alias, self.dialect)}" for expr, alias in metrics])
        sql = (f"SELECT {select} FROM {self._table_ref(table.name)}{where} GROUP BY {', '.join(groups)} "
               f"ORDER BY {quote_ident(metrics[0][1], self.dialect)} DESC LIMIT {p.add(limit)}")
        hits = []
        for row in await self._query(sql, p.values):
            data = {k: v for k, v in row.items()}
            key = dumps({g: data.get(g) for g in op.group_by})
            first = data.get(metrics[0][1])
            hit = row_to_hit(data, source=self.name, id_column=None, text_columns=[],
                             score=float(first) if first is not None else None, fallback_id=f"agg:{key}")
            hits.append(hit.model_copy(update={"content": [StructuredPart(data=hit.metadata)], "metadata": {}}))
        return hits

    async def _fetch(self, op: Fetch, table: TableInfo) -> list[Hit]:
        p = Params(self.dialect)
        ids = ", ".join(p.add(str(i)) for i in op.doc_ids)
        id_expr = self._as_text(quote_ident(table.id_column, self.dialect))
        sql = (f"SELECT {self._select(table)} FROM {self._table_ref(table.name)} "
               f"WHERE {id_expr} IN ({ids})")
        order = {doc_id: i for i, doc_id in enumerate(op.doc_ids)}
        hits = self._hits(await self._query(sql, p.values), table)
        return sorted(hits, key=lambda h: order.get(h.doc_id, len(order)))

    async def _native(self, op: Native, tables: dict[str, TableInfo]) -> list[Hit]:
        if not self.native_query:
            raise UnsupportedOperation("native queries are disabled for this source")
        if op.dialect.lower() not in self.native_dialects:
            raise BackendError(f"dialect must be one of {self.native_dialects}, got {op.dialect!r}")
        sql = guard_sql(op.query, self.dialect, min(op.limit, self.max_rows))
        table = tables.get(op.collection) if op.collection else (
            next(iter(tables.values())) if len(tables) == 1 else None)
        hits = []
        for i, row in enumerate(await self._query(sql, None)):
            id_col = table.id_column if table and table.id_column in row else None
            hits.append(row_to_hit(row, source=self.name, id_column=id_col, text_columns=[],
                                   fallback_id=f"native:{i}"))
        return hits

    # ---- discovery helpers ----------------------------------------------------

    def _vector_spec(self, table: str, column: str, dim: int | None) -> FieldSpec:
        return FieldSpec(name=column, type=FieldType.VECTOR, vector_dim=dim,
                         vector_metric=self.vector_metric,
                         embedder_id=self.embedders.get(f"{table}.{column}"))

    async def _samples(self, table: str, column: str) -> list[Any] | None:
        if not self.sample_values:
            return None
        col = quote_ident(column, self.dialect)
        p = Params(self.dialect)
        rows = await self._query(
            f"SELECT DISTINCT {col} AS v FROM {self._table_ref(table)} WHERE {col} IS NOT NULL "
            f"LIMIT {p.add(SAMPLE_DISTINCT_MAX + 1)}", p.values)
        values = [r["v"] for r in rows]
        return values if len(values) <= SAMPLE_DISTINCT_MAX else None

    async def _count(self, table: str, estimate: int | None) -> int | None:
        if estimate is not None and estimate >= EXACT_COUNT_BELOW:
            return estimate
        rows = await self._query(f"SELECT COUNT(*) AS n FROM {self._table_ref(table)}", None)
        return int(rows[0]["n"]) if rows else None
```

- [ ] **Step 3: Implement Postgres**

`src/agentic_search/backends/postgres.py`:
```python
"""Postgres (+ pgvector) backend: tsvector full-text, pgvector ANN, filters, regex, aggregates,
fetch and guarded native SQL. Sessions are read-only. Needs the `postgres` extra."""

from __future__ import annotations

import asyncio
import re
from typing import Any
from urllib.parse import urlparse

from agentic_search.backends.base import BackendError
from agentic_search.backends.sql import Params, quote_ident, vector_literal, where_clause
from agentic_search.backends.sql_backend import SqlBackend, TableInfo, any_terms, field_flags
from agentic_search.core.secrets import register_secret
from agentic_search.core.types import FieldSpec, FieldType, Lexical

_TYPES = {
    "text": FieldType.TEXT, "character varying": FieldType.KEYWORD, "character": FieldType.KEYWORD,
    "uuid": FieldType.KEYWORD, "smallint": FieldType.INT, "integer": FieldType.INT,
    "bigint": FieldType.INT, "numeric": FieldType.FLOAT, "real": FieldType.FLOAT,
    "double precision": FieldType.FLOAT, "boolean": FieldType.BOOL, "date": FieldType.DATE,
    "timestamp without time zone": FieldType.DATE, "timestamp with time zone": FieldType.DATE,
    "json": FieldType.JSON, "jsonb": FieldType.JSON, "ARRAY": FieldType.JSON,
}
_SKIP_UDTS = {"tsvector", "tsquery"}
_OPS = {"cosine": "<=>", "l2": "<->", "ip": "<#>"}
_CONFIG = re.compile(r"^[a-z_]+$")
_DIM = re.compile(r"^vector\((\d+)\)$")


class PostgresBackend(SqlBackend):
    dialect = "postgres"
    backend_type = "postgres"
    native_dialects = ("sql", "postgres", "postgresql")

    def __init__(self, name: str, dsn: str, *, schema: str = "public",
                 text_search_config: str = "english", pool_size: int = 4,
                 statement_timeout_ms: int = 30_000, **kwargs: Any):
        super().__init__(name, **kwargs)
        if not _CONFIG.match(text_search_config):
            raise ValueError(f"invalid text_search_config {text_search_config!r}")
        register_secret(dsn)
        register_secret(urlparse(dsn).password)
        self.dsn = dsn
        self.schema = schema
        self.ts_config = text_search_config
        self.pool_size = pool_size
        self.statement_timeout_ms = int(statement_timeout_ms)
        self._pool: Any = None
        self._pool_lock = asyncio.Lock()

    async def _get_pool(self) -> Any:
        async with self._pool_lock:
            if self._pool is None:
                from psycopg_pool import AsyncConnectionPool

                timeout_ms = self.statement_timeout_ms

                async def configure(conn: Any) -> None:
                    await conn.set_autocommit(True)
                    await conn.execute("SET default_transaction_read_only = on")
                    await conn.execute(f"SET statement_timeout = {timeout_ms}")

                pool = AsyncConnectionPool(self.dsn, min_size=1, max_size=self.pool_size,
                                           open=False, configure=configure)
                await pool.open(wait=True, timeout=15)
                self._pool = pool
            return self._pool

    async def _query(self, sql: str, params: list[Any] | None) -> list[dict[str, Any]]:
        import psycopg
        from psycopg.rows import dict_row

        pool = await self._get_pool()
        try:
            async with pool.connection() as conn:
                async with conn.cursor(row_factory=dict_row) as cur:
                    await cur.execute(sql, params)
                    return list(await cur.fetchall()) if cur.description else []
        except psycopg.Error as exc:
            raise BackendError(f"{type(exc).__name__}: {exc}") from exc

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    def _table_ref(self, table: str) -> str:
        return f"{quote_ident(self.schema, 'postgres')}.{quote_ident(table, 'postgres')}"

    def _regex_expr(self, column_sql: str, placeholder: str) -> str:
        return f"{column_sql} ~ {placeholder}"

    # ---- discovery ------------------------------------------------------------

    async def _discover_tables(self) -> dict[str, TableInfo]:
        cols = await self._query(
            "SELECT table_name, column_name, data_type, udt_name FROM information_schema.columns "
            "WHERE table_schema = %s ORDER BY table_name, ordinal_position", [self.schema])
        pks = await self._query(
            "SELECT tc.table_name, kcu.column_name FROM information_schema.table_constraints tc "
            "JOIN information_schema.key_column_usage kcu ON tc.constraint_name = kcu.constraint_name "
            "AND tc.table_schema = kcu.table_schema AND tc.table_name = kcu.table_name "
            "WHERE tc.table_schema = %s AND tc.constraint_type = 'PRIMARY KEY' "
            "ORDER BY kcu.ordinal_position", [self.schema])
        dims = await self._query(
            "SELECT c.relname AS table_name, a.attname AS column_name, "
            "format_type(a.atttypid, a.atttypmod) AS fmt FROM pg_attribute a "
            "JOIN pg_class c ON a.attrelid = c.oid JOIN pg_namespace n ON c.relnamespace = n.oid "
            "WHERE n.nspname = %s AND a.attnum > 0 AND NOT a.attisdropped "
            "AND format_type(a.atttypid, a.atttypmod) LIKE 'vector%%'", [self.schema])
        estimates = await self._query(
            "SELECT c.relname AS table_name, c.reltuples::bigint AS est FROM pg_class c "
            "JOIN pg_namespace n ON c.relnamespace = n.oid WHERE n.nspname = %s", [self.schema])
        pk_of: dict[str, str] = {}
        for r in pks:
            pk_of.setdefault(r["table_name"], r["column_name"])
        dim_of = {(r["table_name"], r["column_name"]): int(m.group(1))
                  for r in dims if (m := _DIM.match(r["fmt"]))}
        est_of = {r["table_name"]: r["est"] for r in estimates}
        by_table: dict[str, list[dict[str, Any]]] = {}
        for r in cols:
            if self.table_names is None or r["table_name"] in self.table_names:
                by_table.setdefault(r["table_name"], []).append(r)
        tables: dict[str, TableInfo] = {}
        for tname, rows in by_table.items():
            id_col = self.id_columns.get(tname) or pk_of.get(tname)
            if id_col is None:
                self.skipped.append(tname)
                continue
            fields = []
            for r in rows:
                name, udt = r["column_name"], r["udt_name"]
                if udt in _SKIP_UDTS:
                    continue
                if udt == "vector":
                    fields.append(self._vector_spec(tname, name, dim_of.get((tname, name))))
                    continue
                ftype = _TYPES.get(r["data_type"], FieldType.KEYWORD)
                samples = None
                if ftype in (FieldType.KEYWORD, FieldType.BOOL) and name != id_col:
                    samples = await self._samples(tname, name)
                fields.append(FieldSpec(name=name, type=ftype, sample_values=samples, **field_flags(ftype)))
            est = est_of.get(tname)
            tables[tname] = TableInfo(name=tname, id_column=id_col, fields=fields,
                                      count=await self._count(tname, est if est and est > 0 else None))
        return tables

    # ---- lexical / vector ------------------------------------------------------

    def _lexical_sql(self, op: Lexical, table: TableInfo, limit: int) -> tuple[str, list[Any]]:
        fields = op.fields or table.text_columns or table.searchable_columns
        if not fields:
            raise BackendError(f"{table.name} has no text columns")
        terms = any_terms(op.text)
        if not terms:
            raise BackendError("lexical query has no searchable terms")
        query_text = op.text if '"' in op.text else " or ".join(terms)
        doc = (f"to_tsvector('{self.ts_config}', concat_ws(' ', "
               f"{', '.join(quote_ident(c, 'postgres') for c in fields)}))")
        p = Params("postgres")
        score = f"ts_rank({doc}, websearch_to_tsquery('{self.ts_config}', {p.add(query_text)}))"
        match = f"{doc} @@ websearch_to_tsquery('{self.ts_config}', {p.add(query_text)})"
        where = where_clause(op.filter, "postgres", p, table.columns, extra=[match])
        sql = (f"SELECT {self._select(table)}, {score} AS _score FROM {self._table_ref(table.name)}"
               f"{where} ORDER BY _score DESC, {quote_ident(table.id_column, 'postgres')} "
               f"LIMIT {p.add(limit)}")
        return sql, p.values

    def _vector_sql(self, column: str, vector: list[float], table: TableInfo, op_filter: Any,
                    limit: int) -> tuple[str, list[Any]]:
        col = quote_ident(column, "postgres")
        sym = _OPS[self.vector_metric]
        lit = vector_literal(vector)
        p = Params("postgres")
        distance = f"({col} {sym} {p.add(lit)}::vector)"
        score = f"1 - {distance}" if self.vector_metric == "cosine" else f"-{distance}"
        where = where_clause(op_filter, "postgres", p, table.columns, extra=[f"{col} IS NOT NULL"])
        sql = (f"SELECT {self._select(table)}, {score} AS _score FROM {self._table_ref(table.name)}"
               f"{where} ORDER BY {col} {sym} {p.add(lit)}::vector LIMIT {p.add(limit)}")
        return sql, p.values
```

- [ ] **Step 4: Run the contract for postgres (GREEN)**

Run: `AGENTIC_SEARCH_INTEGRATION=1 uv run pytest tests/contract -m integration -k postgres -q`
Expected: `10 passed`

Run: `uv run pytest -q`
Expected: all pass, with no services needed.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(backends): SqlBackend base and Postgres + pgvector backend

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: MySQLBackend

**Files:**
- Create: `src/agentic_search/backends/mysql.py`
- Test: `tests/backends/test_mysql_unit.py`, plus the contract suite (`-k mysql`)

**Interfaces:**
- Produces: `MySQLBackend(name, dsn, *, pool_size=4, **SqlBackend kwargs)`, with `backend_type="mysql"`. The DSN has the form `mysql://user:password@host:port/db`; percent-encoded credentials are decoded.
- Lexical search requires a FULLTEXT index whose column set equals `op.fields`, or it uses the table's first FULLTEXT index. Only FULLTEXT columns are marked searchable, and there is no vector search.

- [ ] **Step 1: Write the failing unit test**

`tests/backends/test_mysql_unit.py`:
```python
import pytest

from agentic_search.backends.mysql import MySQLBackend
from agentic_search.core.secrets import scrub


def test_dsn_parsing_and_secret_registration():
    b = MySQLBackend("my", "mysql://reader:p%40ss-word@db.internal:3307/shop")
    assert b._conn_args == {"host": "db.internal", "port": 3307, "user": "reader",
                            "password": "p@ss-word", "db": "shop"}
    assert "p@ss-word" not in scrub("error for p@ss-word")


@pytest.mark.parametrize("dsn", ["postgresql://u:p@h/db", "mysql://u:p@h"])
def test_rejects_bad_dsn(dsn):
    with pytest.raises(ValueError):
        MySQLBackend("my", dsn)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/backends/test_mysql_unit.py -q` and `AGENTIC_SEARCH_INTEGRATION=1 uv run pytest tests/contract -m integration -k mysql -q`
Expected: both fail with `ModuleNotFoundError: No module named 'agentic_search.backends.mysql'`

- [ ] **Step 3: Implement**

`src/agentic_search/backends/mysql.py`:
```python
"""MySQL backend: FULLTEXT (natural-language mode) search, filters, REGEXP, aggregates, fetch and
guarded native SQL. Sessions are READ ONLY. No vector search in v1. Needs the `mysql` extra."""

from __future__ import annotations

import asyncio
from typing import Any
from urllib.parse import unquote, urlparse

from agentic_search.backends.base import BackendError
from agentic_search.backends.sql import Params, quote_ident, where_clause
from agentic_search.backends.sql_backend import SqlBackend, TableInfo, field_flags
from agentic_search.core.secrets import register_secret
from agentic_search.core.types import FieldSpec, FieldType, Lexical

_TYPES = {
    "varchar": FieldType.KEYWORD, "char": FieldType.KEYWORD, "enum": FieldType.KEYWORD,
    "text": FieldType.TEXT, "tinytext": FieldType.TEXT, "mediumtext": FieldType.TEXT,
    "longtext": FieldType.TEXT, "tinyint": FieldType.INT, "smallint": FieldType.INT,
    "mediumint": FieldType.INT, "int": FieldType.INT, "bigint": FieldType.INT,
    "decimal": FieldType.FLOAT, "float": FieldType.FLOAT, "double": FieldType.FLOAT,
    "date": FieldType.DATE, "datetime": FieldType.DATE, "timestamp": FieldType.DATE,
    "json": FieldType.JSON,
}


class MySQLBackend(SqlBackend):
    dialect = "mysql"
    backend_type = "mysql"
    native_dialects = ("sql", "mysql")

    def __init__(self, name: str, dsn: str, *, pool_size: int = 4, **kwargs: Any):
        super().__init__(name, **kwargs)
        parsed = urlparse(dsn)
        if parsed.scheme not in ("mysql", "mysql+aiomysql") or not parsed.path.strip("/"):
            raise ValueError("dsn must look like mysql://user:password@host:port/database")
        register_secret(dsn)
        register_secret(unquote(parsed.password or ""))
        self._conn_args = {
            "host": parsed.hostname or "localhost", "port": parsed.port or 3306,
            "user": unquote(parsed.username or ""), "password": unquote(parsed.password or ""),
            "db": parsed.path.strip("/"),
        }
        self.pool_size = pool_size
        self._pool: Any = None
        self._pool_lock = asyncio.Lock()

    async def _get_pool(self) -> Any:
        async with self._pool_lock:
            if self._pool is None:
                import aiomysql

                self._pool = await aiomysql.create_pool(
                    minsize=1, maxsize=self.pool_size, autocommit=True, charset="utf8mb4",
                    connect_timeout=10, init_command="SET SESSION TRANSACTION READ ONLY",
                    **self._conn_args)
            return self._pool

    async def _query(self, sql: str, params: list[Any] | None) -> list[dict[str, Any]]:
        import aiomysql
        import pymysql

        pool = await self._get_pool()
        try:
            async with pool.acquire() as conn:
                async with conn.cursor(aiomysql.DictCursor) as cur:
                    await cur.execute(sql, params)
                    return list(await cur.fetchall())
        except pymysql.MySQLError as exc:
            raise BackendError(f"{type(exc).__name__}: {exc}") from exc

    async def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            await self._pool.wait_closed()
            self._pool = None

    def _regex_expr(self, column_sql: str, placeholder: str) -> str:
        return f"{column_sql} REGEXP {placeholder}"

    def _supports_lexical(self, tables: dict[str, TableInfo]) -> bool:
        return any(t.fulltext for t in tables.values())

    # ---- discovery ------------------------------------------------------------

    async def _discover_tables(self) -> dict[str, TableInfo]:
        cols = await self._query(
            "SELECT TABLE_NAME AS table_name, COLUMN_NAME AS column_name, DATA_TYPE AS data_type, "
            "COLUMN_TYPE AS column_type FROM information_schema.COLUMNS "
            "WHERE TABLE_SCHEMA = DATABASE() ORDER BY TABLE_NAME, ORDINAL_POSITION", None)
        pks = await self._query(
            "SELECT TABLE_NAME AS table_name, COLUMN_NAME AS column_name "
            "FROM information_schema.KEY_COLUMN_USAGE WHERE TABLE_SCHEMA = DATABASE() "
            "AND CONSTRAINT_NAME = 'PRIMARY' ORDER BY ORDINAL_POSITION", None)
        fts = await self._query(
            "SELECT TABLE_NAME AS table_name, INDEX_NAME AS index_name, COLUMN_NAME AS column_name "
            "FROM information_schema.STATISTICS WHERE TABLE_SCHEMA = DATABASE() "
            "AND INDEX_TYPE = 'FULLTEXT' ORDER BY TABLE_NAME, INDEX_NAME, SEQ_IN_INDEX", None)
        estimates = await self._query(
            "SELECT TABLE_NAME AS table_name, TABLE_ROWS AS est FROM information_schema.TABLES "
            "WHERE TABLE_SCHEMA = DATABASE()", None)
        pk_of: dict[str, str] = {}
        for r in pks:
            pk_of.setdefault(r["table_name"], r["column_name"])
        ft_of: dict[str, dict[str, list[str]]] = {}
        for r in fts:
            ft_of.setdefault(r["table_name"], {}).setdefault(r["index_name"], []).append(r["column_name"])
        est_of = {r["table_name"]: r["est"] for r in estimates}
        by_table: dict[str, list[dict[str, Any]]] = {}
        for r in cols:
            if self.table_names is None or r["table_name"] in self.table_names:
                by_table.setdefault(r["table_name"], []).append(r)
        tables: dict[str, TableInfo] = {}
        for tname, rows in by_table.items():
            id_col = self.id_columns.get(tname) or pk_of.get(tname)
            if id_col is None:
                self.skipped.append(tname)
                continue
            fulltext = list(ft_of.get(tname, {}).values())
            ft_cols = {c for idx in fulltext for c in idx}
            fields = []
            for r in rows:
                name = r["column_name"]
                if r["data_type"] == "tinyint" and r["column_type"].startswith("tinyint(1)"):
                    ftype = FieldType.BOOL
                else:
                    ftype = _TYPES.get(r["data_type"], FieldType.KEYWORD)
                flags = field_flags(ftype)
                flags["searchable"] = name in ft_cols
                samples = None
                if ftype in (FieldType.KEYWORD, FieldType.BOOL) and name != id_col:
                    samples = await self._samples(tname, name)
                fields.append(FieldSpec(name=name, type=ftype, sample_values=samples, **flags))
            est = est_of.get(tname)
            tables[tname] = TableInfo(name=tname, id_column=id_col, fields=fields, fulltext=fulltext,
                                      count=await self._count(tname, int(est) if est else None))
        return tables

    # ---- lexical ---------------------------------------------------------------

    def _lexical_sql(self, op: Lexical, table: TableInfo, limit: int) -> tuple[str, list[Any]]:
        if not table.fulltext:
            raise BackendError(f"{table.name} has no FULLTEXT index; lexical search needs one")
        if op.fields:
            match = next((idx for idx in table.fulltext if set(idx) == set(op.fields)), None)
            if match is None:
                raise BackendError(f"no FULLTEXT index on exactly {sorted(op.fields)}; "
                                   f"indexes: {table.fulltext}")
        else:
            match = table.fulltext[0]
        cols = ", ".join(quote_ident(c, "mysql") for c in match)
        p = Params("mysql")
        score = f"MATCH({cols}) AGAINST ({p.add(op.text)} IN NATURAL LANGUAGE MODE)"
        cond = f"MATCH({cols}) AGAINST ({p.add(op.text)} IN NATURAL LANGUAGE MODE)"
        where = where_clause(op.filter, "mysql", p, table.columns, extra=[cond])
        sql = (f"SELECT {self._select(table)}, {score} AS _score FROM {self._table_ref(table.name)}"
               f"{where} ORDER BY _score DESC, {quote_ident(table.id_column, 'mysql')} "
               f"LIMIT {p.add(limit)}")
        return sql, p.values
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/backends/test_mysql_unit.py -q`
Expected: 3 passed

Run: `AGENTIC_SEARCH_INTEGRATION=1 uv run pytest tests/contract -m integration -k mysql -q`
Expected: `9 passed, 1 skipped` (vector search is skipped). The output must be pristine, which is why the seeding helper sets `sql_notes = 0`.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(backends): MySQL backend with FULLTEXT search and read-only sessions

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: OpenSearchBackend

**Files:**
- Create: `src/agentic_search/backends/opensearch.py`
- Test: `tests/backends/test_opensearch_unit.py`, plus the contract suite (`-k opensearch`)

**Interfaces:**
- Produces:
  - `filter_dsl(f) -> dict`.
  - `OpenSearchBackend(name, url, *, indices=None, embedders=None, native_query=False, description=None, max_rows=100, verify_certs=True, client=None, sample_values=True)`, with `backend_type="opensearch"`.
- Behaviour:
  - Collections are indices; the default is every non-dot index.
  - `text` fields are searchable but not filterable, and a `.keyword` sub-field is exposed as its own filterable KEYWORD field.
  - `knn_vector` fields become VECTOR, with the embedder taken from `embedders["index.field"]`.
  - Hybrid search is fused client-side with RRF.
  - Aggregations use `terms` for one field and `multi_terms` for several.
  - Native queries must be an OpenSearch search body, which passes `guard_opensearch`.

- [ ] **Step 1: Write the failing unit test**

`tests/backends/test_opensearch_unit.py`:
```python
from agentic_search.backends.opensearch import filter_dsl
from agentic_search.core.types import And, Contains, Eq, Exists, In, Not, Or, Range


def test_filter_dsl():
    f = And(clauses=[Eq(field="type", value="drug"), In(field="year", values=[2020, 2021]),
                     Or(clauses=[Range(field="year", gte=2020), Not(clause=Exists(field="title"))]),
                     Contains(field="type", value="ru")])
    assert filter_dsl(f) == {"bool": {"filter": [
        {"term": {"type": "drug"}},
        {"terms": {"year": [2020, 2021]}},
        {"bool": {"should": [{"range": {"year": {"gte": 2020}}},
                             {"bool": {"must_not": [{"exists": {"field": "title"}}]}}],
                  "minimum_should_match": 1}},
        {"wildcard": {"type": {"value": "*ru*", "case_insensitive": True}}},
    ]}}
    assert filter_dsl(Range(field="year")) == {"match_all": {}}
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/backends/test_opensearch_unit.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'agentic_search.backends.opensearch'`

- [ ] **Step 3: Implement**

`src/agentic_search/backends/opensearch.py`:
```python
"""OpenSearch backend: multi_match full-text, k-NN vector search, client-side hybrid (RRF),
bool filters, regexp, terms/multi_terms aggregations, mget and guarded native search bodies.
Only search-type APIs are called. Needs the `opensearch` extra."""

from __future__ import annotations

import asyncio
from typing import Any
from urllib.parse import unquote, urlparse

from agentic_search.backends.base import (
    BackendError,
    DiscoverDetail,
    UnsupportedOperation,
    rrf_merge,
)
from agentic_search.backends.native_guard import guard_opensearch
from agentic_search.backends.sql import dumps, jsonable, parse_metric
from agentic_search.core.secrets import register_secret
from agentic_search.core.types import (
    Aggregate,
    And,
    Capability,
    CollectionInfo,
    Contains,
    Content,
    Eq,
    Exists,
    Fetch,
    FieldSpec,
    FieldType,
    Filter,
    FilterOnly,
    Hit,
    Hybrid,
    In,
    Lexical,
    Manifest,
    Native,
    Not,
    Or,
    QueryOp,
    Range,
    Regex,
    StructuredPart,
    TextPart,
    Vector,
)

_TYPES = {
    "text": FieldType.TEXT, "match_only_text": FieldType.TEXT, "keyword": FieldType.KEYWORD,
    "constant_keyword": FieldType.KEYWORD, "wildcard": FieldType.KEYWORD, "long": FieldType.INT,
    "integer": FieldType.INT, "short": FieldType.INT, "byte": FieldType.INT,
    "unsigned_long": FieldType.INT, "float": FieldType.FLOAT, "double": FieldType.FLOAT,
    "half_float": FieldType.FLOAT, "scaled_float": FieldType.FLOAT, "boolean": FieldType.BOOL,
    "date": FieldType.DATE, "date_nanos": FieldType.DATE, "knn_vector": FieldType.VECTOR,
}
SAMPLE_DISTINCT_MAX = 20


def filter_dsl(f: Filter) -> dict[str, Any]:
    """Filter AST → OpenSearch query DSL (filter context)."""
    if isinstance(f, And):
        return {"bool": {"filter": [filter_dsl(c) for c in f.clauses]}}
    if isinstance(f, Or):
        return {"bool": {"should": [filter_dsl(c) for c in f.clauses], "minimum_should_match": 1}}
    if isinstance(f, Not):
        return {"bool": {"must_not": [filter_dsl(f.clause)]}}
    if isinstance(f, Eq):
        return {"term": {f.field: f.value}}
    if isinstance(f, In):
        return {"terms": {f.field: f.values}}
    if isinstance(f, Range):
        bounds = {k: v for k, v in (("gte", f.gte), ("gt", f.gt), ("lte", f.lte), ("lt", f.lt))
                  if v is not None}
        return {"range": {f.field: bounds}} if bounds else {"match_all": {}}
    if isinstance(f, Exists):
        return {"exists": {"field": f.field}}
    if isinstance(f, Contains):
        return {"wildcard": {f.field: {"value": f"*{f.value}*", "case_insensitive": True}}}
    raise BackendError(f"unsupported filter node {type(f).__name__}")


class OpenSearchBackend:
    backend_type = "opensearch"

    def __init__(self, name: str, url: str, *, indices: list[str] | None = None,
                 embedders: dict[str, str] | None = None, native_query: bool = False,
                 description: str | None = None, max_rows: int = 100, verify_certs: bool = True,
                 client: Any = None, sample_values: bool = True):
        parsed = urlparse(url)
        register_secret(url)
        register_secret(unquote(parsed.password or ""))
        self.name = name
        self.url = url
        self.index_names = indices
        self.embedders = embedders or {}
        self.native_query = native_query
        self.description = description
        self.max_rows = max_rows
        self.verify_certs = verify_certs
        self.sample_values = sample_values
        self._client = client
        self._indices: dict[str, CollectionInfo] | None = None
        self._lock = asyncio.Lock()

    def _get_client(self) -> Any:
        if self._client is None:
            from opensearchpy import AsyncOpenSearch

            self._client = AsyncOpenSearch(hosts=[self.url], verify_certs=self.verify_certs,
                                           ssl_show_warn=False)
        return self._client

    async def _call(self, fn: str, **kwargs: Any) -> Any:
        from opensearchpy.exceptions import OpenSearchException

        client = self._get_client()
        target: Any = client
        for part in fn.split("."):
            target = getattr(target, part)
        try:
            return await target(**kwargs)
        except OpenSearchException as exc:
            raise BackendError(f"{type(exc).__name__}: {exc}") from exc

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None

    # ---- discovery ------------------------------------------------------------

    def capabilities(self) -> set[Capability]:
        caps = {Capability.LEXICAL, Capability.FILTER, Capability.REGEX, Capability.AGGREGATE,
                Capability.FETCH}
        indices = self._indices or {}
        if any(f.type is FieldType.VECTOR for c in indices.values() for f in c.fields):
            caps |= {Capability.VECTOR, Capability.HYBRID}
        if self.native_query:
            caps.add(Capability.NATIVE)
        return caps

    async def _ensure_indices(self) -> dict[str, CollectionInfo]:
        async with self._lock:
            if self._indices is None:
                self._indices = await self._discover_indices()
            return self._indices

    async def _discover_indices(self) -> dict[str, CollectionInfo]:
        names = self.index_names
        if names is None:
            rows = await self._call("cat.indices", format="json")
            names = sorted(r["index"] for r in rows if not r["index"].startswith("."))
        out: dict[str, CollectionInfo] = {}
        for index in names:
            mapping = await self._call("indices.get_mapping", index=index)
            props = next(iter(mapping.values()))["mappings"].get("properties", {})
            fields = self._fields(index, props)
            count = (await self._call("count", index=index))["count"]
            await self._add_samples(index, fields)
            out[index] = CollectionInfo(name=index, fields=fields, count=count)
        return out

    def _fields(self, index: str, props: dict[str, Any], prefix: str = "") -> list[FieldSpec]:
        fields: list[FieldSpec] = []
        for name, spec in props.items():
            full = f"{prefix}{name}"
            if "properties" in spec:
                fields.extend(self._fields(index, spec["properties"], f"{full}."))
                continue
            ftype = _TYPES.get(spec.get("type", "object"), FieldType.JSON)
            if ftype is FieldType.VECTOR:
                method = spec.get("method", {})
                fields.append(FieldSpec(name=full, type=ftype, vector_dim=spec.get("dimension"),
                                        vector_metric=method.get("space_type"),
                                        embedder_id=self.embedders.get(f"{index}.{full}")))
                continue
            fields.append(FieldSpec(
                name=full, type=ftype, searchable=ftype is FieldType.TEXT,
                filterable=ftype not in (FieldType.TEXT, FieldType.JSON),
                sortable=ftype in (FieldType.INT, FieldType.FLOAT, FieldType.DATE)))
            for sub, sub_spec in spec.get("fields", {}).items():
                if sub_spec.get("type") == "keyword":
                    fields.append(FieldSpec(name=f"{full}.{sub}", type=FieldType.KEYWORD, filterable=True))
        return fields

    async def _add_samples(self, index: str, fields: list[FieldSpec]) -> None:
        keywords = [f for f in fields if f.type in (FieldType.KEYWORD, FieldType.BOOL)]
        if not self.sample_values or not keywords:
            return
        aggs = {f"s{i}": {"terms": {"field": f.name, "size": SAMPLE_DISTINCT_MAX + 1}}
                for i, f in enumerate(keywords)}
        resp = await self._call("search", index=index, body={"size": 0, "aggs": aggs})
        for i, f in enumerate(keywords):
            buckets = resp["aggregations"][f"s{i}"]["buckets"]
            if len(buckets) <= SAMPLE_DISTINCT_MAX:
                f.sample_values = [b.get("key_as_string", b["key"]) for b in buckets]

    async def discover(self, detail: DiscoverDetail = "full",
                       collection: str | None = None) -> Manifest:
        indices = await self._ensure_indices()
        return Manifest(source=self.name, backend_type=self.backend_type,
                        capabilities=self.capabilities(), description=self.description,
                        collections=list(indices.values()))

    # ---- execution -------------------------------------------------------------

    def _resolve(self, name: str | None, indices: dict[str, CollectionInfo]) -> CollectionInfo:
        if name is None:
            if len(indices) == 1:
                return next(iter(indices.values()))
            raise BackendError(f"collection required; one of {sorted(indices)}")
        if name not in indices:
            raise BackendError(f"unknown collection {name!r}; one of {sorted(indices)}")
        return indices[name]

    def _hit(self, raw: dict[str, Any], coll: CollectionInfo, score: float | None = None) -> Hit:
        source = raw.get("_source", {}) or {}
        vectors = {f.name for f in coll.fields if f.type is FieldType.VECTOR}
        texts = [str(source[f.name]) for f in coll.fields
                 if f.type is FieldType.TEXT and source.get(f.name) not in (None, "")]
        content: list[Content] = [TextPart(text="\n".join(texts))] if texts else []
        text_names = {f.name for f in coll.fields if f.type is FieldType.TEXT}
        metadata = {k: jsonable(v) for k, v in source.items() if k not in vectors | text_names}
        if not content:
            content = [StructuredPart(data=metadata)]
        s = raw.get("_score") if score is None else score
        return Hit(doc_id=str(raw["_id"]), source=self.name, content=content, metadata=metadata,
                   raw_score=float(s) if s is not None else None)

    def _source_filter(self, coll: CollectionInfo) -> dict[str, Any]:
        vectors = [f.name for f in coll.fields if f.type is FieldType.VECTOR]
        return {"excludes": vectors} if vectors else {}

    async def _search(self, coll: CollectionInfo, body: dict[str, Any]) -> list[Hit]:
        body = {**body, "_source": self._source_filter(coll)}
        resp = await self._call("search", index=coll.name, body=body)
        return [self._hit(h, coll) for h in resp["hits"]["hits"]]

    def _with_filter(self, query: dict[str, Any], f: Filter | None) -> dict[str, Any]:
        if f is None:
            return query
        return {"bool": {"must": [query], "filter": [filter_dsl(f)]}}

    def _lexical_body(self, op: Lexical | Hybrid, coll: CollectionInfo, limit: int) -> dict[str, Any]:
        text_fields = [f.name for f in coll.fields if f.searchable]
        fields = getattr(op, "fields", None) or text_fields
        if not fields:
            raise BackendError(f"{coll.name} has no text fields")
        query = {"multi_match": {"query": op.text, "fields": fields}}
        return {"size": limit, "query": self._with_filter(query, op.filter)}

    def _vector_body(self, field: str, vector: list[float] | None, coll: CollectionInfo,
                     f: Filter | None, limit: int) -> dict[str, Any]:
        if vector is None:
            raise BackendError("vector op reached the backend without an embedded query vector")
        spec = coll.field(field)
        if spec is None or spec.type is not FieldType.VECTOR:
            raise BackendError(f"{field!r} is not a knn_vector field of {coll.name}")
        knn: dict[str, Any] = {"vector": vector, "k": limit}
        if f is not None:
            knn["filter"] = filter_dsl(f)
        return {"size": limit, "query": {"knn": {field: knn}}}

    async def execute(self, op: QueryOp) -> list[Hit]:
        indices = await self._ensure_indices()
        if isinstance(op, Native):
            return await self._native(op, indices)
        coll = self._resolve(op.collection, indices)
        limit = min(op.limit, self.max_rows)
        if isinstance(op, Lexical):
            return await self._search(coll, self._lexical_body(op, coll, limit))
        if isinstance(op, Vector):
            return await self._search(coll, self._vector_body(op.field, op.vector, coll, op.filter, limit))
        if isinstance(op, Hybrid):
            depth = min(max(limit * 5, 50), 500)
            lex = await self._search(coll, self._lexical_body(op, coll, depth))
            vec = await self._search(coll, self._vector_body(op.field, op.vector, coll, op.filter, depth))
            by_id = {h.doc_id: h for h in [*vec, *lex]}
            fused = rrf_merge([([h.doc_id for h in lex], op.lexical_weight),
                               ([h.doc_id for h in vec], 1.0 - op.lexical_weight)])
            return [by_id[i].model_copy(update={"raw_score": s}) for i, s in fused[:limit]]
        if isinstance(op, FilterOnly):
            return await self._search(coll, {"size": limit, "query": {"bool": {"filter": [filter_dsl(op.filter)]}},
                                             "sort": [{"_doc": "asc"}]})
        if isinstance(op, Regex):
            fields = op.fields or [f.name for f in coll.fields if f.searchable]
            should = [{"regexp": {f: {"value": op.pattern, "case_insensitive": True}}} for f in fields]
            query = self._with_filter({"bool": {"should": should, "minimum_should_match": 1}}, op.filter)
            return await self._search(coll, {"size": limit, "query": query})
        if isinstance(op, Aggregate):
            return await self._aggregate(op, coll, limit)
        if isinstance(op, Fetch):
            resp = await self._call("mget", index=coll.name, body={"ids": op.doc_ids},
                                    _source_excludes=self._source_filter(coll).get("excludes"))
            return [self._hit(d, coll) for d in resp["docs"] if d.get("found")]
        raise UnsupportedOperation(f"opensearch backend does not support {op.type}")

    async def _aggregate(self, op: Aggregate, coll: CollectionInfo, limit: int) -> list[Hit]:
        columns = {f.name for f in coll.fields}
        unknown = set(op.group_by) - columns
        if unknown:
            raise BackendError(f"unknown fields {sorted(unknown)}")
        sub: dict[str, Any] = {}
        names = []
        for m in op.metrics:
            _, alias = parse_metric(m, columns, "mysql")
            names.append(alias)
            if m != "count":
                fn, _, col = m.partition(":")
                sub[alias] = {fn: {"field": col}}
        if len(op.group_by) == 1:
            agg: dict[str, Any] = {"terms": {"field": op.group_by[0], "size": limit}}
        else:
            agg = {"multi_terms": {"terms": [{"field": g} for g in op.group_by], "size": limit}}
        if sub:
            agg["aggs"] = sub
        query = {"bool": {"filter": [filter_dsl(op.filter)]}} if op.filter else {"match_all": {}}
        resp = await self._call("search", index=coll.name,
                                body={"size": 0, "query": query, "aggs": {"g": agg}})
        hits = []
        for bucket in resp["aggregations"]["g"]["buckets"]:
            key = bucket["key"] if isinstance(bucket["key"], list) else [bucket["key"]]
            data: dict[str, Any] = dict(zip(op.group_by, key))
            for alias in names:
                data[alias] = bucket["doc_count"] if alias == "count" else bucket[alias]["value"]
            first = data[names[0]]
            hits.append(Hit(doc_id=f"agg:{dumps({g: data[g] for g in op.group_by})}", source=self.name,
                            content=[StructuredPart(data=data)],
                            raw_score=float(first) if first is not None else None))
        return hits

    async def _native(self, op: Native, indices: dict[str, CollectionInfo]) -> list[Hit]:
        if not self.native_query:
            raise UnsupportedOperation("native queries are disabled for this source")
        if op.dialect.lower() not in ("opensearch_dsl", "opensearch", "dsl", "json"):
            raise BackendError(f"dialect must be opensearch_dsl, got {op.dialect!r}")
        coll = self._resolve(op.collection, indices)
        body = guard_opensearch(op.query, min(op.limit, self.max_rows))
        return await self._search(coll, body)
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/backends/test_opensearch_unit.py -q`
Expected: 1 passed

Run: `AGENTIC_SEARCH_INTEGRATION=1 uv run pytest tests/contract -m integration -k opensearch -q`
Expected: `9 passed, 1 skipped` (the read-only-session test applies to SQL backends only).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(backends): OpenSearch backend (multi_match, k-NN, hybrid, aggregations)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: BigQueryBackend

**Files:**
- Create: `src/agentic_search/backends/bigquery.py`
- Test: `tests/backends/test_bigquery.py`

**Interfaces:**
- Produces: `BigQueryBackend(name, project, dataset, *, client=None, max_bytes_billed=1_000_000_000, location=None, text_columns=None, vector_dims=None, sample_values=False, **SqlBackend kwargs)`, with `backend_type="bigquery"`.
- Behaviour:
  - Every query is dry-run first; if it would scan more than `max_bytes_billed`, it raises `BackendError` before running. It then runs with `maximum_bytes_billed` set.
  - Lexical search ranks rows by how many query terms `CONTAINS_SUBSTR` finds (case-insensitive substring match; no search index needed).
  - Vector search uses `VECTOR_SEARCH`, and REPEATED FLOAT64 columns are vectors.
  - STRING columns are TEXT unless `text_columns[table]` narrows the list.
  - The id column comes from `id_columns`, else the table's primary-key constraint, else `id`.
  - Counts come from table metadata and never from a scan.
- Tests use a fake client. `test_live_bigquery` (marked `live`) runs against a real dataset when `GOOGLE_CLOUD_PROJECT` and `AGENTIC_SEARCH_BQ_DATASET` are set.

- [ ] **Step 1: Write the failing tests**

`tests/backends/test_bigquery.py`:
```python
"""BigQuery has no local emulator we rely on: these tests drive the backend with a fake client and
check the SQL, parameters and byte-cap behaviour. `test_live_bigquery` runs against a real dataset."""

import os

import pytest
from google.cloud import bigquery

from agentic_search.backends.base import BackendError
from agentic_search.backends.bigquery import BigQueryBackend
from agentic_search.core.types import (
    Aggregate,
    Capability,
    Eq,
    Fetch,
    FieldType,
    Lexical,
    Native,
    Regex,
    Vector,
)

ROWS = [{"id": "d1", "title": "Aspirin", "body": "Aspirin relieves headache", "type": "drug", "year": 2020}]


class FakeJob:
    def __init__(self, bytes_, rows):
        self.total_bytes_processed = bytes_
        self._rows = rows

    def result(self):
        return [dict(r) for r in self._rows]


class FakeTable:
    def __init__(self, schema, num_rows=5):
        self.schema = schema
        self.num_rows = num_rows
        self.table_constraints = None


class FakeClient:
    def __init__(self, bytes_=1000, rows=ROWS):
        self.bytes_ = bytes_
        self.rows = rows
        self.queries = []

    def list_tables(self, dataset):
        return [type("T", (), {"table_id": "docs"})()]

    def get_table(self, ref):
        S = bigquery.SchemaField
        return FakeTable([S("id", "STRING"), S("title", "STRING"), S("body", "STRING"),
                          S("type", "STRING"), S("year", "INTEGER"),
                          S("embedding", "FLOAT64", mode="REPEATED")])

    def query(self, sql, job_config):
        self.queries.append((sql, job_config.dry_run, {p.name: getattr(p, "value", getattr(p, "values", None))
                                                        for p in job_config.query_parameters}))
        return FakeJob(self.bytes_, [] if job_config.dry_run else self.rows)


def make(client=None, **kw):
    return BigQueryBackend("bq", "proj", "ds", client=client or FakeClient(),
                           text_columns={"docs": ["title", "body"]},
                           vector_dims={"docs.embedding": 64},
                           embedders={"docs.embedding": "hash64"}, **kw)


async def test_discover_maps_schema():
    m = await make(native_query=True).discover()
    coll = m.resolve_collection("docs")
    assert coll.count == 5
    assert coll.field("title").type is FieldType.TEXT and coll.field("type").type is FieldType.KEYWORD
    emb = coll.field("embedding")
    assert emb.type is FieldType.VECTOR and emb.vector_dim == 64 and emb.embedder_id == "hash64"
    assert {Capability.LEXICAL, Capability.VECTOR, Capability.HYBRID, Capability.NATIVE} <= m.capabilities


async def test_lexical_sql_dry_runs_then_runs():
    client = FakeClient()
    b = make(client)
    hits = await b.execute(Lexical(source="bq", text="Headache relief", filter=Eq(field="year", value=2020)))
    assert [h.doc_id for h in hits] == ["d1"] and "headache" in hits[0].content[0].text.lower()
    (dry_sql, dry, params), (sql, run_dry, _) = client.queries
    assert dry is True and not run_dry and dry_sql == sql
    assert "CONTAINS_SUBSTR((`title`, `body`), @p0)" in sql and "`year` = @p2" in sql
    assert "FROM `proj.ds.docs`" in sql and "LIMIT @p3" in sql
    assert params == {"p0": "headache", "p1": "relief", "p2": 2020, "p3": 20}


async def test_byte_cap_refuses_before_running():
    client = FakeClient(bytes_=5_000_000_000)
    with pytest.raises(BackendError, match="would scan"):
        await make(client).execute(Regex(source="bq", pattern="asp"))
    assert len(client.queries) == 1 and client.queries[0][1] is True


async def test_vector_search_sql():
    client = FakeClient()
    await make(client).execute(Vector(source="bq", field="embedding", hyde_text="x", vector=[0.5, 1.0], limit=3))
    sql, _, params = client.queries[1]
    assert "FROM VECTOR_SEARCH((SELECT * FROM `proj.ds.docs`), 'embedding'" in sql
    assert "(SELECT @p0 AS `embedding`), top_k => 3, distance_type => 'COSINE'" in sql
    assert "base.`title` AS `title`" in sql and "`embedding` AS" not in sql.split("FROM")[0]
    assert params == {"p0": [0.5, 1.0]}


async def test_regex_fetch_aggregate_native_sql():
    client = FakeClient(rows=[{"type": "drug", "count": 3}])
    b = make(client, native_query=True)
    await b.execute(Regex(source="bq", pattern="print.*", fields=["body"]))
    assert "REGEXP_CONTAINS(CAST(`body` AS STRING), @p0)" in client.queries[-1][0]
    await b.execute(Fetch(source="bq", doc_ids=["d1", "d2"]))
    assert "CAST(`id` AS STRING) IN (@p0, @p1)" in client.queries[-1][0]
    [agg] = await b.execute(Aggregate(source="bq", group_by=["type"], metrics=["count", "avg:year"]))
    assert "GROUP BY `type`" in client.queries[-1][0] and "AVG(`year`) AS `avg_year`" in client.queries[-1][0]
    assert agg.content[0].data == {"type": "drug", "count": 3}
    await b.execute(Native(source="bq", dialect="bigquery", query="SELECT id FROM `proj.ds.docs`"))
    assert client.queries[-1][0].endswith("LIMIT 20")
    with pytest.raises(BackendError):
        await b.execute(Native(source="bq", dialect="sql", query="DELETE FROM `proj.ds.docs` WHERE TRUE"))


@pytest.mark.live
async def test_live_bigquery():
    project, dataset = os.environ.get("GOOGLE_CLOUD_PROJECT"), os.environ.get("AGENTIC_SEARCH_BQ_DATASET")
    if not (project and dataset):
        pytest.skip("set GOOGLE_CLOUD_PROJECT and AGENTIC_SEARCH_BQ_DATASET (a dataset with a `docs` table)")
    b = BigQueryBackend("bq", project, dataset, tables=["docs"], max_bytes_billed=100_000_000)
    m = await b.discover()
    assert m.resolve_collection("docs") is not None
    hits = await b.execute(Lexical(source="bq", collection="docs", text="headache", limit=5))
    assert isinstance(hits, list)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/backends/test_bigquery.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'agentic_search.backends.bigquery'`

- [ ] **Step 3: Implement**

`src/agentic_search/backends/bigquery.py`:
```python
"""BigQuery backend: term-match lexical search (CONTAINS_SUBSTR), VECTOR_SEARCH, filters,
REGEXP_CONTAINS, aggregates, fetch and guarded native SQL. Every query is dry-run first and
refused above `max_bytes_billed`. Needs the `bigquery` extra. The client is synchronous, so calls
run in a worker thread."""

from __future__ import annotations

import asyncio
from typing import Any

from agentic_search.backends.base import BackendError
from agentic_search.backends.sql import Params, quote_ident, where_clause
from agentic_search.backends.sql_backend import SqlBackend, TableInfo, any_terms, field_flags
from agentic_search.core.types import FieldSpec, FieldType, Lexical

_TYPES = {
    "STRING": FieldType.TEXT, "INTEGER": FieldType.INT, "INT64": FieldType.INT,
    "FLOAT": FieldType.FLOAT, "FLOAT64": FieldType.FLOAT, "NUMERIC": FieldType.FLOAT,
    "BIGNUMERIC": FieldType.FLOAT, "BOOLEAN": FieldType.BOOL, "BOOL": FieldType.BOOL,
    "DATE": FieldType.DATE, "DATETIME": FieldType.DATE, "TIMESTAMP": FieldType.DATE,
    "JSON": FieldType.JSON, "RECORD": FieldType.JSON, "STRUCT": FieldType.JSON,
}
_DISTANCE = {"cosine": "COSINE", "l2": "EUCLIDEAN", "ip": "DOT_PRODUCT"}


class BigQueryBackend(SqlBackend):
    dialect = "bigquery"
    backend_type = "bigquery"
    native_dialects = ("sql", "bigquery")

    def __init__(self, name: str, project: str, dataset: str, *, client: Any = None,
                 max_bytes_billed: int = 1_000_000_000, location: str | None = None,
                 text_columns: dict[str, list[str]] | None = None,
                 vector_dims: dict[str, int] | None = None, sample_values: bool = False,
                 **kwargs: Any):
        super().__init__(name, sample_values=sample_values, **kwargs)
        self.project = project
        self.dataset = dataset
        self.max_bytes_billed = int(max_bytes_billed)
        self.location = location
        self.text_column_overrides = text_columns or {}
        self.vector_dims = vector_dims or {}
        self._client = client

    def _get_client(self) -> Any:
        if self._client is None:
            from google.cloud import bigquery

            self._client = bigquery.Client(project=self.project, location=self.location)
        return self._client

    def _table_ref(self, table: str) -> str:
        quote_ident(table, "bigquery")  # validate
        return f"`{self.project}.{self.dataset}.{table}`"

    def _as_text(self, expr: str) -> str:
        return f"CAST({expr} AS STRING)"

    def _regex_expr(self, column_sql: str, placeholder: str) -> str:
        return f"REGEXP_CONTAINS({column_sql}, {placeholder})"

    @staticmethod
    def _parameters(values: list[Any]) -> list[Any]:
        from google.cloud import bigquery

        out = []
        for i, v in enumerate(values):
            name = f"p{i}"
            if isinstance(v, list):
                out.append(bigquery.ArrayQueryParameter(name, "FLOAT64", [float(x) for x in v]))
            elif isinstance(v, bool):
                out.append(bigquery.ScalarQueryParameter(name, "BOOL", v))
            elif isinstance(v, int):
                out.append(bigquery.ScalarQueryParameter(name, "INT64", v))
            elif isinstance(v, float):
                out.append(bigquery.ScalarQueryParameter(name, "FLOAT64", v))
            else:
                out.append(bigquery.ScalarQueryParameter(name, "STRING", None if v is None else str(v)))
        return out

    def _run(self, sql: str, params: list[Any] | None) -> list[dict[str, Any]]:
        from google.api_core.exceptions import GoogleAPIError
        from google.cloud import bigquery

        client = self._get_client()
        query_params = self._parameters(params or [])
        try:
            dry = client.query(sql, job_config=bigquery.QueryJobConfig(
                dry_run=True, use_query_cache=False, query_parameters=query_params))
            scanned = dry.total_bytes_processed or 0
            if scanned > self.max_bytes_billed:
                raise BackendError(f"query would scan {scanned} bytes, above the "
                                   f"{self.max_bytes_billed}-byte cap")
            job = client.query(sql, job_config=bigquery.QueryJobConfig(
                query_parameters=query_params, maximum_bytes_billed=self.max_bytes_billed))
            return [dict(row.items()) for row in job.result()]
        except GoogleAPIError as exc:
            raise BackendError(f"{type(exc).__name__}: {exc}") from exc

    async def _query(self, sql: str, params: list[Any] | None) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._run, sql, params)

    async def _count(self, table: str, estimate: int | None) -> int | None:
        return estimate  # num_rows from table metadata; never scan to count

    # ---- discovery ------------------------------------------------------------

    def _list_tables(self) -> list[tuple[str, Any]]:
        client = self._get_client()
        names = self.table_names or [t.table_id for t in client.list_tables(f"{self.project}.{self.dataset}")]
        return [(n, client.get_table(f"{self.project}.{self.dataset}.{n}")) for n in names]

    async def _discover_tables(self) -> dict[str, TableInfo]:
        from google.api_core.exceptions import GoogleAPIError

        try:
            listed = await asyncio.to_thread(self._list_tables)
        except GoogleAPIError as exc:
            raise BackendError(f"{type(exc).__name__}: {exc}") from exc
        tables: dict[str, TableInfo] = {}
        for tname, meta in listed:
            names = [f.name for f in meta.schema]
            id_col = self.id_columns.get(tname) or self._primary_key(meta) or ("id" if "id" in names else None)
            if id_col is None:
                self.skipped.append(tname)
                continue
            overrides = self.text_column_overrides.get(tname)
            fields = []
            for f in meta.schema:
                ftype_name = f.field_type.upper()
                if f.mode == "REPEATED" and ftype_name in ("FLOAT", "FLOAT64"):
                    fields.append(self._vector_spec(tname, f.name, self.vector_dims.get(f"{tname}.{f.name}")))
                    continue
                if f.mode == "REPEATED":
                    ftype = FieldType.JSON
                else:
                    ftype = _TYPES.get(ftype_name, FieldType.KEYWORD)
                if ftype is FieldType.TEXT and overrides is not None and f.name not in overrides:
                    ftype = FieldType.KEYWORD
                samples = None
                if ftype in (FieldType.KEYWORD, FieldType.BOOL) and f.name != id_col:
                    samples = await self._samples(tname, f.name)
                fields.append(FieldSpec(name=f.name, type=ftype, description=f.description or None,
                                        sample_values=samples, **field_flags(ftype)))
            tables[tname] = TableInfo(name=tname, id_column=id_col, fields=fields,
                                      count=int(meta.num_rows) if meta.num_rows is not None else None)
        return tables

    @staticmethod
    def _primary_key(meta: Any) -> str | None:
        constraints = getattr(meta, "table_constraints", None)
        pk = getattr(constraints, "primary_key", None) if constraints else None
        cols = getattr(pk, "columns", None) if pk else None
        return cols[0] if cols else None

    # ---- lexical / vector ------------------------------------------------------

    def _lexical_sql(self, op: Lexical, table: TableInfo, limit: int) -> tuple[str, list[Any]]:
        fields = op.fields or table.text_columns or table.searchable_columns
        if not fields:
            raise BackendError(f"{table.name} has no text columns")
        terms = any_terms(op.text)
        if not terms:
            raise BackendError("lexical query has no searchable terms")
        p = Params("bigquery")
        quoted = [quote_ident(c, "bigquery") for c in fields]
        target = quoted[0] if len(quoted) == 1 else f"({', '.join(quoted)})"
        score = " + ".join(f"IF(CONTAINS_SUBSTR({target}, {p.add(t)}), 1, 0)" for t in terms)
        where = where_clause(op.filter, "bigquery", p, table.columns)
        id_col = quote_ident(table.id_column, "bigquery")
        sql = (f"SELECT * FROM (SELECT {self._select(table)}, ({score}) AS _score "
               f"FROM {self._table_ref(table.name)}{where}) WHERE _score > 0 "
               f"ORDER BY _score DESC, {id_col} LIMIT {p.add(limit)}")
        return sql, p.values

    def _vector_sql(self, column: str, vector: list[float], table: TableInfo, op_filter: Any,
                    limit: int) -> tuple[str, list[Any]]:
        col = quote_ident(column, "bigquery")
        p = Params("bigquery")
        where = where_clause(op_filter, "bigquery", p, table.columns)
        select = ", ".join(f"base.{quote_ident(c, 'bigquery')} AS {quote_ident(c, 'bigquery')}"
                           for c in table.select_columns)
        score = "1 - distance" if self.vector_metric == "cosine" else "-distance"
        sql = (f"SELECT {select}, {score} AS _score FROM VECTOR_SEARCH("
               f"(SELECT * FROM {self._table_ref(table.name)}{where}), '{column}', "
               f"(SELECT {p.add(vector)} AS {col}), top_k => {int(limit)}, "
               f"distance_type => '{_DISTANCE[self.vector_metric]}') ORDER BY distance")
        return sql, p.values
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/backends/test_bigquery.py -q`
Expected: 5 passed, 1 deselected (live)

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(backends): BigQuery backend with dry-run byte cap and VECTOR_SEARCH

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10: Config registration, README, full verification

**Files:**
- Modify: `src/agentic_search/config.py`, `README.md`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces:
  - Backend config types `postgres` (alias `pgvector`), `mysql`, `bigquery` and `opensearch`.
  - Backend-level `embedders: {"table.column": id}` must name embedders defined in the top-level `embedders:` list; otherwise it raises `ConfigError`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_config.py`:
```python
async def test_build_sql_and_search_backends(tmp_path, monkeypatch):
    from agentic_search.backends.bigquery import BigQueryBackend
    from agentic_search.backends.mysql import MySQLBackend
    from agentic_search.backends.opensearch import OpenSearchBackend
    from agentic_search.backends.postgres import PostgresBackend
    from agentic_search.core.secrets import scrub

    monkeypatch.setenv("PG_DSN", "postgresql://app:pg-pass-123@db:5432/app")
    h = build_harness({
        "embedders": [{"type": "hash", "id": "hash64", "dim": 64}],
        "backends": [
            {"name": "pg", "type": "pgvector", "dsn_env": "PG_DSN", "tables": ["docs"],
             "embedders": {"docs.embedding": "hash64"}, "native_query": True},
            {"name": "my", "type": "mysql", "dsn": "mysql://u:my-pass-456@h:3306/app"},
            {"name": "bq", "type": "bigquery", "project": "p", "dataset": "d", "max_bytes_billed": 10},
            {"name": "os", "type": "opensearch", "url": "https://admin:os-pass-789@search:9200",
             "indices": ["docs"], "embedders": {"docs.embedding": "hash64"}},
        ],
        "driver": {"type": "openai_compat", "model": "local", "base_url": "http://localhost:8000/v1"},
    }, base_dir=tmp_path)
    b = h.backends
    assert isinstance(b["pg"], PostgresBackend) and b["pg"].native_query and b["pg"].table_names == ["docs"]
    assert isinstance(b["my"], MySQLBackend) and isinstance(b["os"], OpenSearchBackend)
    assert isinstance(b["bq"], BigQueryBackend) and b["bq"].max_bytes_billed == 10
    leaked = "pg-pass-123 my-pass-456 os-pass-789 https://admin:os-pass-789@search:9200"
    assert all(s not in scrub(leaked) for s in ("pg-pass-123", "my-pass-456", "os-pass-789"))


def test_backend_embedder_reference_must_exist(tmp_path):
    with pytest.raises(ConfigError, match="unknown embedder"):
        build_harness({"backends": [{"name": "pg", "type": "postgres", "dsn": "postgresql://x@h/db",
                                     "embedders": {"docs.embedding": "nope"}}],
                       "driver": {"type": "openai_compat", "model": "m", "base_url": "http://x/v1"}},
                      base_dir=tmp_path)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_config.py -q`
Expected: FAIL with `ConfigError: unknown backend type 'pgvector'`

- [ ] **Step 3: Register the backend types**

In `src/agentic_search/config.py`, add after `_files` (before `_hash`):
```python
_SQL_KEYS = ("tables", "id_columns", "embedders", "vector_metric", "native_query", "description",
             "max_rows", "sample_values")


def _backend_kwargs(cfg: dict[str, Any], ctx: BuildContext, keys: tuple[str, ...]) -> dict[str, Any]:
    """Pass through known keys; check that `embedders: {table.column: id}` names known embedders."""
    unknown = sorted(set((cfg.get("embedders") or {}).values()) - set(ctx.embedders))
    if unknown:
        raise ConfigError(f"backend {cfg['name']!r} references unknown embedder(s) {unknown}")
    return {k: cfg[k] for k in keys if k in cfg}


def _postgres(cfg: dict[str, Any], ctx: BuildContext) -> Any:
    from agentic_search.backends.postgres import PostgresBackend
    return PostgresBackend(cfg["name"], cfg["dsn"],
                           **_backend_kwargs(cfg, ctx, _SQL_KEYS + ("schema", "text_search_config",
                                                                    "pool_size", "statement_timeout_ms")))


def _mysql(cfg: dict[str, Any], ctx: BuildContext) -> Any:
    from agentic_search.backends.mysql import MySQLBackend
    return MySQLBackend(cfg["name"], cfg["dsn"], **_backend_kwargs(cfg, ctx, _SQL_KEYS + ("pool_size",)))


def _bigquery(cfg: dict[str, Any], ctx: BuildContext) -> Any:
    from agentic_search.backends.bigquery import BigQueryBackend
    return BigQueryBackend(cfg["name"], cfg["project"], cfg["dataset"],
                           **_backend_kwargs(cfg, ctx, _SQL_KEYS + ("max_bytes_billed", "location",
                                                                    "text_columns", "vector_dims")))


def _opensearch(cfg: dict[str, Any], ctx: BuildContext) -> Any:
    from agentic_search.backends.opensearch import OpenSearchBackend
    return OpenSearchBackend(cfg["name"], cfg["url"], **_backend_kwargs(
        cfg, ctx, ("indices", "embedders", "native_query", "description", "max_rows",
                   "verify_certs", "sample_values")))
```
and extend the registration list after `("backend", "files", _files),`:
```python
    ("backend", "postgres", _postgres),
    ("backend", "pgvector", _postgres),
    ("backend", "mysql", _mysql),
    ("backend", "bigquery", _bigquery),
    ("backend", "opensearch", _opensearch),
```

- [ ] **Step 4: Update the README**

In `README.md`, replace the **Backends** bullet under "Roles" with:
```markdown
- **Backends**: files, Postgres + pgvector, MySQL, BigQuery and OpenSearch (see below). Graph and
  vector stores (Neo4j, Milvus) come in Plan 3.
```
Insert this section directly before `## Evaluate`:
````markdown
## Backends

| type | install extra | lexical | vector | notes |
|---|---|---|---|---|
| `files` | – | BM25 | local index | directory or in-memory documents |
| `postgres` / `pgvector` | `postgres` | tsvector + `websearch_to_tsquery` | pgvector `<=>`/`<->`/`<#>` | read-only sessions, statement timeout |
| `mysql` | `mysql` | FULLTEXT (natural language) | – | lexical needs a FULLTEXT index; READ ONLY sessions |
| `bigquery` | `bigquery` | term match (`CONTAINS_SUBSTR`) | `VECTOR_SEARCH` | every query dry-run; refused above `max_bytes_billed` |
| `opensearch` | `opensearch` | `multi_match` | k-NN (`knn_vector`) | search APIs only |

All backends support filters, regex, aggregates (`count`, `sum|avg|min|max:<column>`) and fetch.
Set `native_query: true` on a backend to let the planner run read-only native SQL / search bodies;
they pass `backends/native_guard.py` (single SELECT, no DML/DDL/locks/scripts, row cap) first.
Vector columns need `embedders: {<table>.<column>: <embedder id>}` so queries are embedded with the
same model as the stored vectors. DSNs and passwords are masked in errors and traces.

```yaml
backends:
  - {name: notes, type: pgvector, dsn_env: NOTES_DSN, tables: [notes],
     embedders: {notes.embedding: "st:BAAI/bge-small-en-v1.5"}, native_query: true}
  - {name: orders, type: mysql, dsn_env: ORDERS_DSN}
  - {name: warehouse, type: bigquery, project: my-proj, dataset: clinical, max_bytes_billed: 500000000}
  - {name: search, type: opensearch, url_env: SEARCH_URL, indices: [articles]}
```
````
Replace the `## Develop` code block with:
```bash
uv sync && uv run pytest                      # unit tests, no services
docker compose up -d --wait                   # Postgres+pgvector, MySQL, OpenSearch
AGENTIC_SEARCH_INTEGRATION=1 uv run pytest -m integration   # backend contract suite
```

- [ ] **Step 5: Full verification**

Run: `uv run ruff check --fix src tests scripts && uv run pytest -q && uv run ruff check src tests scripts`
Expected: `197 passed, 4 skipped`, and `All checks passed!`

Run: `AGENTIC_SEARCH_INTEGRATION=1 uv run pytest -m integration -q`
Expected: `28 passed, 2 skipped`

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat(config): postgres/pgvector, mysql, bigquery and opensearch backend types; docs

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

## Spec coverage (Plan 2)

| Spec § / follow-up | Where |
|---|---|
| §3.3 adapters: pgvector, MySQL, BigQuery, OpenSearch | Tasks 6–9 |
| §3.3 `discover()` via information_schema / `_mapping` / table metadata; samples; counts | Tasks 6–9 |
| §6 read-only (sessions + native guard), LIMIT injection, BigQuery dry-run byte cap | Tasks 5–9 |
| §7 config for new types, `*_env` DSNs | Task 10 |
| §8 adapter contract suite with docker-compose, shared corpus; BigQuery live-only | Tasks 3, 6–9 |
| §9 criterion 1 (backends pass contract) for Plan 2 backends | Tasks 6–8 (BigQuery via fake + live) |
| §9 criterion 5 (native rejects writes) | Task 5 tests + contract `test_native_read_only` |
| Follow-up: secret scrubbing | Task 1 |
| Follow-up: delegate spend cutoff | Task 2 |
| Plan 3 dependency: `guard_cypher` | Task 5 |

## Known limitations (documented, not bugs)

- MySQL has no vector search, since community MySQL lacks ANN. Its lexical search needs a FULLTEXT index.
- The DB-side statement timeout is set for Postgres only. For the other backends, the executor's `call_timeout` bounds the call on the client side.
- Aggregations on OpenSearch return the top `limit` buckets by document count.
- BigQuery's lexical search is substring-based (`CONTAINS_SUBSTR`). Adding a search index and switching to `SEARCH()` is a later optimization.
- Hybrid search is fused client-side with RRF for every backend.
