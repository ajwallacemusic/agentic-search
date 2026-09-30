# Lattice Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Three additions that let lattice run this library inside its retrieval service: a Gemini on Vertex model client, an allow list for native SQL, and a Postgres password read fresh at each new connection. Then a `v0.2.0` tag that lattice pins.

**Architecture:**
- `OpenAICompatClient` takes an optional `Auth` and sends its headers on every call. `models/vertex.py` builds one pointed at Vertex AI's OpenAI-compatible endpoint, with Google credentials by default.
- `guard_sql` takes an optional `SqlAllowList`. With one, sqlglot's `qualify` resolves every column against the allowed columns only, the resolved query is the one that runs, and tables and functions are checked against the list.
- `SqlBackend` takes `columns`, which trims discovery to the allowed columns and builds the allow list for native SQL.
- `PostgresBackend` takes `password`, an async callable asked for a password at each new connection.

**Tech Stack:** Python ≥ 3.11, pydantic 2, sqlglot `>=25,<31` (locked at 30.20.0), psycopg 3 with psycopg_pool, the `openai` SDK, pytest with `asyncio_mode = "auto"`.

**Spec:** The lattice design `lattice/.axis/plans/design/agentic-search.md` on lattice branch `docs/agentic-search`, section "Steps", under "Library". That design is the spec for this plan. The lattice side is planned separately in lattice.

## Global Constraints

- **Backward compatible.** Every new parameter defaults to today's behavior. Existing tests pass unchanged.
- **Non-blocking.** Nothing added here runs blocking I/O on the event loop. `GcpAdc` already refreshes in `asyncio.to_thread`.
- **Secrets.** Every token or password the library handles passes through `register_secret` before any error can print it.
- **Errors.** A refused native query raises `NativeQueryRejected`, with a message that names what was refused.
- **Commands.** Run from the repo root: `uv run pytest -q` and `uv run ruff check`. Both are clean at the end of every task.
- **Commits.** One commit per task, authored by the person running the session, with no co-author trailer.
- **Do not push.** The branch and the tag stay local until the owner pushes them.

## Review Focus

- **An alias that reuses a hidden column's name.** `SELECT dx AS ssn, ssn FROM visits` must never read `ssn`. `qualify` resolves the second `ssn` to the alias, and the resolved query is what runs. Pinned in Task 3.
- **A query that reads a table outside the list and names no column.** `SELECT COUNT(*) FROM people` passes `qualify` because it names no column, so the explicit table check must refuse it. Pinned in Task 3.
- **`SELECT *`.** `qualify` expands it to the allowed columns only, so it must run and must not return a hidden column. Pinned in Task 3.
- **A table named in another dataset or project.** `other.visits` or `` `q-2.ds.visits` `` must be refused even when `visits` is allowed. Pinned in Task 3.
- **A token that expires between connections.** A pool that opens a second connection an hour later must ask for a new password, not reuse the first. Pinned in Task 5.

---

## File Structure

| File | Change | Responsibility |
|---|---|---|
| `src/agentic_search/models/openai_compat.py` | Modify | Accept `auth` and send its headers per call |
| `src/agentic_search/models/vertex.py` | Create | Vertex endpoint URL and the client factory |
| `src/agentic_search/config.py` | Modify | `client: vertex`; `columns` and `native_functions` on SQL backends |
| `src/agentic_search/backends/native_guard.py` | Modify | `SqlAllowList`, the default function list, the allow-list check |
| `src/agentic_search/backends/sql_backend.py` | Modify | `columns`, `native_functions`, trimming, building the allow list |
| `src/agentic_search/backends/bigquery.py` | Modify | Report its project and dataset as the allow list's qualifiers |
| `src/agentic_search/backends/postgres.py` | Modify | Report its schema; `password` provider through a connection class |
| `tests/models/test_openai_compat.py` | Modify | Auth header tests |
| `tests/models/test_vertex.py` | Create | Vertex URL and factory tests |
| `tests/backends/test_native_guard.py` | Modify | Allow-list tests |
| `tests/backends/test_sql_backend.py` | Modify | Column trimming and native allow-list tests |
| `tests/backends/test_postgres_unit.py` | Modify | Password provider tests |
| `tests/test_config.py` | Modify | Config tests for the new keys |
| `README.md`, `pyproject.toml` | Modify | Document the additions; version `0.2.0` |

---

### Task 1: Per-call auth headers on the OpenAI-compatible client

**Files:**
- Modify: `src/agentic_search/models/openai_compat.py`
- Test: `tests/models/test_openai_compat.py`

**Interfaces:**
- Consumes: `Auth` from `agentic_search.embedders.auth` (`async def headers(self) -> dict[str, str]`).
- Produces: `OpenAICompatClient(..., auth: Auth | None = None)`. When `auth` is set, every `chat()` call passes `extra_headers=await auth.headers()` to `chat.completions.create`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/models/test_openai_compat.py`:

```python
class CountingAuth:
    def __init__(self):
        self.calls = 0

    async def headers(self):
        self.calls += 1
        return {"Authorization": f"Bearer tok{self.calls}"}


async def test_auth_headers_are_fetched_on_every_call():
    fake = sdk()
    auth = CountingAuth()
    c = OpenAICompatClient("m", client=fake, auth=auth)
    await c.chat("s", [ChatMessage(role="user", content="q")])
    assert fake.chat.completions.kwargs["extra_headers"] == {"Authorization": "Bearer tok1"}
    await c.chat("s", [ChatMessage(role="user", content="q")])
    assert fake.chat.completions.kwargs["extra_headers"] == {"Authorization": "Bearer tok2"}


async def test_no_auth_sends_no_extra_headers():
    fake = sdk()
    c = OpenAICompatClient("m", client=fake)
    await c.chat("s", [ChatMessage(role="user", content="q")])
    assert "extra_headers" not in fake.chat.completions.kwargs
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/models/test_openai_compat.py -q -k "auth"`
Expected: FAIL with `TypeError: OpenAICompatClient.__init__() got an unexpected keyword argument 'auth'`

- [ ] **Step 3: Implement**

In `src/agentic_search/models/openai_compat.py`, change `from typing import Any` to
`from typing import TYPE_CHECKING, Any`, and add below the other imports:

```python
if TYPE_CHECKING:
    from agentic_search.embedders.auth import Auth
```

Change the constructor signature and body:

```python
    def __init__(self, model: str, *, base_url: str | None = None, api_key: str | None = None,
                 client: Any = None, supports_images: bool = False,
                 price_per_mtok: tuple[float, float] | None = None, id: str | None = None,
                 max_tokens_param: str = "max_tokens", max_retries: int = 3,
                 auth: Auth | None = None):
        if client is None:
            from openai import AsyncOpenAI
            key = api_key or os.environ.get("OPENAI_API_KEY") or "unused"
            client = AsyncOpenAI(base_url=base_url, api_key=key, max_retries=max_retries)
        self.model = model
        self._client = client
        self.supports_images = supports_images
        self.price = price_per_mtok
        self.max_tokens_param = max_tokens_param
        self.auth = auth
        self.id = id or f"openai:{model}"
```

In `chat()`, directly before `resp = await self._client.chat.completions.create(**kwargs)`:

```python
        if self.auth is not None:
            # Headers per call, so a short-lived token is refreshed rather than frozen at startup.
            kwargs["extra_headers"] = await self.auth.headers()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/models/test_openai_compat.py -q`
Expected: PASS, including the existing tests.

- [ ] **Step 5: Commit**

```bash
git add src/agentic_search/models/openai_compat.py tests/models/test_openai_compat.py
git commit -m "feat(models): send auth headers per call on the OpenAI-compatible client"
```

---

### Task 2: Gemini on Vertex

**Files:**
- Create: `src/agentic_search/models/vertex.py`
- Modify: `src/agentic_search/config.py`
- Test: `tests/models/test_vertex.py`, `tests/test_config.py`

**Interfaces:**
- Consumes: `OpenAICompatClient(..., auth=...)` from Task 1; `Auth`, `GcpAdc`, `build_auth` from `agentic_search.embedders.auth`.
- Produces:
  - `vertex_base_url(project: str, location: str = "global") -> str`
  - `vertex_client(model: str, *, project: str, location: str = "global", auth: Auth | None = None, client: Any = None, supports_images: bool = True, price_per_mtok: tuple[float, float] | None = None, id: str | None = None, max_retries: int = 3) -> OpenAICompatClient`. It sends `google/<model>` as the model name unless `model` already has a publisher prefix, and its `id` is `vertex:<model>`.
  - Config: `{type: vertex, model, project, location?, auth?, id?, supports_images?, price_per_mtok?}` under `driver:` or an `llm_judge` client.

- [ ] **Step 1: Write the failing tests**

Create `tests/models/test_vertex.py`:

```python
from types import SimpleNamespace

import pytest

from agentic_search.embedders.auth import GcpAdc
from agentic_search.models.llm import ChatMessage
from agentic_search.models.vertex import vertex_base_url, vertex_client


class FakeCompletions:
    def __init__(self):
        self.kwargs = None

    async def create(self, **kwargs):
        self.kwargs = kwargs
        msg = SimpleNamespace(content="ok", tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")],
                               usage=SimpleNamespace(prompt_tokens=3, completion_tokens=1))


class FixedAuth:
    async def headers(self):
        return {"Authorization": "Bearer ya29.test"}


def test_global_endpoint_has_no_region_prefix():
    assert vertex_base_url("my-project-1") == (
        "https://aiplatform.googleapis.com/v1/projects/my-project-1/locations/global/endpoints/openapi")


def test_regional_endpoint_uses_the_region_host():
    assert vertex_base_url("my-project-1", "us-central1") == (
        "https://us-central1-aiplatform.googleapis.com/v1/projects/my-project-1"
        "/locations/us-central1/endpoints/openapi")


@pytest.mark.parametrize("project,location", [("Bad_Project", "global"), ("my-project-1", "us central1"),
                                              ("my-project-1", "../x")])
def test_rejects_malformed_project_or_location(project, location):
    with pytest.raises(ValueError):
        vertex_base_url(project, location)


async def test_client_sends_publisher_model_and_auth_header():
    completions = FakeCompletions()
    fake = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    c = vertex_client("gemini-3.8-flash", project="my-project-1", auth=FixedAuth(), client=fake)
    assert c.id == "vertex:gemini-3.8-flash"
    assert c.supports_images is True
    r = await c.chat("s", [ChatMessage(role="user", content="q")])
    assert r.text == "ok"
    assert completions.kwargs["model"] == "google/gemini-3.8-flash"
    assert completions.kwargs["extra_headers"] == {"Authorization": "Bearer ya29.test"}


def test_publisher_prefix_is_kept():
    fake = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    c = vertex_client("meta/llama-4", project="my-project-1", auth=FixedAuth(), client=fake)
    assert c.model == "meta/llama-4" and c.id == "vertex:meta/llama-4"


def test_default_auth_is_application_default_credentials():
    fake = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    c = vertex_client("gemini-3.8-flash", project="my-project-1", client=fake)
    assert isinstance(c.auth, GcpAdc)
```

Append to `tests/test_config.py`:

```python
def test_vertex_client_from_config(tmp_path):
    from agentic_search.config import BuildContext, build
    from agentic_search.embedders.auth import Bearer

    ctx = BuildContext(base_dir=tmp_path)
    c = build("client", {"type": "vertex", "model": "gemini-3.8-flash", "project": "my-project-1",
                         "location": "us-central1", "auth": {"type": "bearer", "token": "t0k"}}, ctx)
    assert c.id == "vertex:gemini-3.8-flash"
    assert c.model == "google/gemini-3.8-flash"
    assert isinstance(c.auth, Bearer)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/models/test_vertex.py tests/test_config.py -q -k "vertex"`
Expected: FAIL with `ModuleNotFoundError: No module named 'agentic_search.models.vertex'`

- [ ] **Step 3: Implement the module**

Create `src/agentic_search/models/vertex.py`:

```python
"""Gemini, and other Model Garden models, on Vertex AI through its OpenAI-compatible endpoint.

Needs the `openai` extra, and the `gcp` extra unless `auth` is supplied. The access token is
fetched per call through `auth`, so a caller with its own credentials (an async service with its
own transport) passes an `Auth` and the library never refreshes a token itself."""

from __future__ import annotations

import re
from typing import Any

from agentic_search.embedders.auth import Auth, GcpAdc
from agentic_search.models.openai_compat import OpenAICompatClient

_PROJECT = re.compile(r"[a-z][a-z0-9-]{4,28}[a-z0-9]")
_LOCATION = re.compile(r"[a-z]+(-[a-z0-9]+)*")


def vertex_base_url(project: str, location: str = "global") -> str:
    if not _PROJECT.fullmatch(project):
        raise ValueError(f"invalid Google Cloud project {project!r}")
    if not _LOCATION.fullmatch(location):
        raise ValueError(f"invalid Vertex location {location!r}")
    host = "aiplatform.googleapis.com" if location == "global" else f"{location}-aiplatform.googleapis.com"
    return f"https://{host}/v1/projects/{project}/locations/{location}/endpoints/openapi"


def vertex_client(model: str, *, project: str, location: str = "global", auth: Auth | None = None,
                  client: Any = None, supports_images: bool = True,
                  price_per_mtok: tuple[float, float] | None = None, id: str | None = None,
                  max_retries: int = 3) -> OpenAICompatClient:
    base_url = vertex_base_url(project, location)
    publisher_model = model if "/" in model else f"google/{model}"
    # api_key is a placeholder the SDK requires; the Authorization header from `auth` replaces it.
    return OpenAICompatClient(publisher_model, base_url=base_url, api_key="unused", client=client,
                              supports_images=supports_images, price_per_mtok=price_per_mtok,
                              id=id or f"vertex:{model}", max_retries=max_retries,
                              auth=auth if auth is not None else GcpAdc())
```

- [ ] **Step 4: Register it in config**

In `src/agentic_search/config.py`, add after `_openai_compat`:

```python
def _vertex_llm(cfg: dict[str, Any], ctx: BuildContext) -> Any:
    from agentic_search.embedders.auth import build_auth
    from agentic_search.models.vertex import vertex_client
    return vertex_client(cfg["model"], project=cfg["project"], location=cfg.get("location", "global"),
                         auth=build_auth(cfg["auth"]) if cfg.get("auth") else None, id=cfg.get("id"),
                         supports_images=cfg.get("supports_images", True), price_per_mtok=_price(cfg))
```

Add to the registration list, after `("client", "openai_compat", _openai_compat),`:

```python
    ("client", "vertex", _vertex_llm),
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/models tests/test_config.py -q`
Expected: PASS

- [ ] **Step 6: Check one live call (manual, needs Google credentials)**

Tool calling through Vertex's OpenAI-compatible endpoint is the part no unit test reaches. With `gcloud auth application-default login` done and a project where Vertex AI is enabled:

```bash
uv run --extra openai --extra gcp python - <<'EOF'
import asyncio, os
from agentic_search.models.base import ToolSpec
from agentic_search.models.llm import ChatMessage
from agentic_search.models.vertex import vertex_client

async def main():
    c = vertex_client("gemini-3.8-flash", project=os.environ["VERTEX_PROJECT"])
    tool = ToolSpec(name="lexical_search", description="Search by words.",
                    parameters={"type": "object", "properties": {"text": {"type": "string"}},
                                "required": ["text"]})
    r = await c.chat("Use the tool.", [ChatMessage(role="user", content="find chest pain")],
                     tools=[tool], tool_choice="lexical_search")
    print(r.tool_calls, r.usage)

asyncio.run(main())
EOF
```

Expected: one `ToolCall` named `lexical_search` with a `text` argument, and nonzero token counts. If Vertex refuses the named `tool_choice`, record the error text in the commit body and change `ToolCallingDriver`'s call site only if the planner needs a named tool choice. Do not change the unit tests to match.

- [ ] **Step 7: Commit**

```bash
git add src/agentic_search/models/vertex.py src/agentic_search/config.py tests/models/test_vertex.py tests/test_config.py
git commit -m "feat(models): add a Gemini on Vertex client"
```

---

### Task 3: An allow list for native SQL

**Files:**
- Modify: `src/agentic_search/backends/native_guard.py`
- Test: `tests/backends/test_native_guard.py`

**Interfaces:**
- Produces:
  - `DEFAULT_SQL_FUNCTIONS: frozenset[str]`, uppercase sqlglot function names.
  - `SqlAllowList(tables: Mapping[str, frozenset[str]], db: str | None = None, catalog: str | None = None, functions: frozenset[str] = DEFAULT_SQL_FUNCTIONS)`, frozen dataclass. `db` is a Postgres schema or a BigQuery dataset. `catalog` is a BigQuery project and needs `db`.
  - `guard_sql(query: str, dialect: str, max_rows: int, allow: SqlAllowList | None = None) -> str`. With `allow`, the returned SQL is the resolved query, with every column qualified, `*` expanded to the allowed columns, and unqualified tables filled in with `catalog` and `db`.

The resolution uses `sqlglot.optimizer.qualify.qualify` with a schema that holds only the allowed columns. A column outside the list cannot resolve, so `qualify` raises. Running the resolved query is what makes an alias that reuses a hidden column's name harmless: `qualify` rewrites that reference to the aliased allowed column.

- [ ] **Step 1: Write the failing tests**

Append to `tests/backends/test_native_guard.py`:

```python
from agentic_search.backends.native_guard import SqlAllowList

BQ = SqlAllowList(tables={"visits": frozenset({"id", "dx", "note"})}, db="ds", catalog="p-1")
PG = SqlAllowList(tables={"docs": frozenset({"id", "title"})}, db="public")


def test_allow_list_resolves_and_qualifies():
    assert guard_sql("SELECT dx FROM visits", "bigquery", 50, BQ) == (
        "SELECT visits.dx AS dx FROM `p-1`.ds.visits AS visits LIMIT 50")
    assert guard_sql("SELECT title FROM docs", "postgres", 50, PG) == (
        "SELECT docs.title AS title FROM public.docs AS docs LIMIT 50")


def test_allow_list_accepts_ctes_aggregates_and_backticks():
    q = ("WITH c AS (SELECT dx, COUNT(*) AS n FROM `p-1.ds.visits` v "
         "WHERE LOWER(v.note) LIKE '%pain%' GROUP BY dx) SELECT dx, n FROM c ORDER BY n DESC")
    out = guard_sql(q, "bigquery", 10, BQ)
    assert "COUNT(*)" in out and out.endswith("LIMIT 10")
    assert guard_sql("SELECT `dx` FROM `p-1`.`ds`.`visits`", "bigquery", 5, BQ).endswith("LIMIT 5")


@pytest.mark.parametrize("query", [
    "SELECT ssn FROM visits",
    "SELECT dx FROM visits WHERE ssn = '1'",
    "WITH ssn AS (SELECT 1 AS x) SELECT ssn FROM visits",
])
def test_allow_list_rejects_hidden_columns(query):
    with pytest.raises(NativeQueryRejected, match="allowed tables and columns"):
        guard_sql(query, "bigquery", 10, BQ)


def test_alias_reusing_a_hidden_name_never_reads_it():
    out = guard_sql("SELECT dx AS ssn, ssn FROM visits", "bigquery", 10, BQ)
    assert "visits.ssn" not in out and "visits.dx" in out


def test_star_expands_to_allowed_columns_only():
    out = guard_sql("SELECT * FROM visits", "bigquery", 10, BQ)
    assert out == ("SELECT visits.dx AS dx, visits.id AS id, visits.note AS note "
                   "FROM `p-1`.ds.visits AS visits LIMIT 10")


@pytest.mark.parametrize("query", [
    "SELECT COUNT(*) FROM people",
    "SELECT COUNT(*) FROM `p-1.other.visits`",
    "SELECT COUNT(*) FROM `q-2.ds.visits`",
])
def test_allow_list_rejects_other_tables(query):
    with pytest.raises(NativeQueryRejected, match="table"):
        guard_sql(query, "bigquery", 10, BQ)


@pytest.mark.parametrize("query,name", [
    ("SELECT EXTERNAL_QUERY('c', 'SELECT 1') FROM visits", "EXTERNAL_QUERY"),
    ("SELECT SESSION_USER() FROM visits", "SESSION_USER"),
    ("SELECT x FROM UNNEST([1, 2]) AS x", "UNNEST"),
])
def test_allow_list_rejects_functions_outside_the_list(query, name):
    with pytest.raises(NativeQueryRejected):
        guard_sql(query, "bigquery", 10, BQ)


def test_allow_list_takes_custom_functions():
    only_count = SqlAllowList(tables=BQ.tables, db="ds", catalog="p-1", functions=frozenset({"COUNT"}))
    assert "COUNT(*)" in guard_sql("SELECT COUNT(*) FROM visits", "bigquery", 10, only_count)
    with pytest.raises(NativeQueryRejected, match="LOWER"):
        guard_sql("SELECT LOWER(dx) FROM visits", "bigquery", 10, only_count)


def test_catalog_needs_db():
    with pytest.raises(ValueError):
        SqlAllowList(tables={"t": frozenset({"a"})}, catalog="p-1")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/backends/test_native_guard.py -q`
Expected: FAIL with `ImportError: cannot import name 'SqlAllowList'`

- [ ] **Step 3: Implement**

In `src/agentic_search/backends/native_guard.py`, change the imports at the top:

```python
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
```

Add after `_SQL_FUNC_DENYLIST`:

```python
DEFAULT_SQL_FUNCTIONS = frozenset({
    "COUNT", "COUNT_IF", "SUM", "AVG", "MIN", "MAX", "APPROX_DISTINCT",
    "LOWER", "UPPER", "LENGTH", "TRIM", "SUBSTRING", "CONCAT", "COALESCE",
    "STARTS_WITH", "ENDS_WITH", "CONTAINS", "REGEXP_LIKE",
    "CAST", "TRY_CAST", "IF", "CASE", "ROUND", "ABS", "SAFE_DIVIDE",
    "DATE", "EXTRACT", "DATE_TRUNC", "TIMESTAMP_TRUNC", "DATEDIFF", "CURRENT_DATE",
    "CURRENT_TIMESTAMP",
})
"""Uppercase sqlglot names (`Expression.sql_name()`, or the name of an unknown function).
BigQuery's CONTAINS_SUBSTR parses as CONTAINS and REGEXP_CONTAINS as REGEXP_LIKE."""


@dataclass(frozen=True)
class SqlAllowList:
    """What native SQL may read: tables and their columns, and functions by sqlglot name.

    `db` is the Postgres schema or BigQuery dataset every allowed table sits in; `catalog` is
    the BigQuery project. An unqualified table is read as one of these."""

    tables: Mapping[str, frozenset[str]]
    db: str | None = None
    catalog: str | None = None
    functions: frozenset[str] = DEFAULT_SQL_FUNCTIONS

    def __post_init__(self) -> None:
        if self.catalog is not None and self.db is None:
            raise ValueError("SqlAllowList: a catalog needs a db")

    def schema(self) -> dict[str, Any]:
        """The nested schema `qualify` resolves against: allowed columns only, sorted so `*`
        expands in a stable order."""
        schema: dict[str, Any] = {table: {column: "UNKNOWN" for column in sorted(columns)}
                                  for table, columns in self.tables.items()}
        if self.db is not None:
            schema = {self.db: schema}
        if self.catalog is not None:
            schema = {self.catalog: schema}
        return schema


def _apply_allow_list(stmt: Any, dialect: str, allow: SqlAllowList) -> Any:
    """Resolve every column against the allowed columns and return the resolved query.

    The resolved query is what runs, so `*` reads only allowed columns and a reference that
    `qualify` binds to a select alias never reaches a hidden column of the same name."""
    from sqlglot import exp
    from sqlglot.errors import OptimizeError
    from sqlglot.optimizer.qualify import qualify

    try:
        stmt = qualify(stmt, schema=allow.schema(), dialect=dialect, catalog=allow.catalog,
                       db=allow.db, validate_qualify_columns=True, quote_identifiers=False)
    except OptimizeError as exc:
        raise NativeQueryRejected(
            f"native SQL may use only the allowed tables and columns: {exc}") from exc
    ctes = {cte.alias_or_name for cte in stmt.find_all(exp.CTE)}
    for table in stmt.find_all(exp.Table):
        if table.name in ctes and not table.db:
            continue
        # A query naming no column, such as COUNT(*), resolves against any table; check each one.
        if (table.name not in allow.tables or (table.db or None) != allow.db
                or (table.catalog or None) != allow.catalog):
            raise NativeQueryRejected(f"table {table.sql(dialect=dialect)} is not allowed in native SQL")
    for func in stmt.find_all(exp.Func):
        name = (func.name if isinstance(func, exp.Anonymous) else func.sql_name()).upper()
        if name not in allow.functions:
            raise NativeQueryRejected(f"function {name} is not allowed in native SQL")
    return stmt
```

Change `guard_sql`'s signature and the lines before the limit handling:

```python
def guard_sql(query: str, dialect: str, max_rows: int, allow: SqlAllowList | None = None) -> str:
    """Allow exactly one SELECT/set-operation with no DML/DDL/locking; cap or add LIMIT.
    With `allow`, also resolve it against the allowed tables, columns and functions."""
```

Directly after the `for node in stmt.walk():` loop ends and before `limit = stmt.args.get("limit")`:

```python
    if allow is not None:
        stmt = _apply_allow_list(stmt, dialect, allow)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/backends/test_native_guard.py -q`
Expected: PASS. If `test_star_expands_to_allowed_columns_only` or `test_allow_list_resolves_and_qualifies` differs only in quoting or alias spelling for the locked sqlglot 30.20.0, print the actual output. Confirm it names only allowed columns, then pin that exact string. Do not loosen the rejection tests.

- [ ] **Step 5: Commit**

```bash
git add src/agentic_search/backends/native_guard.py tests/backends/test_native_guard.py
git commit -m "feat(backends): add an allow list for native SQL"
```

---

### Task 4: Allowed columns on SQL backends

**Files:**
- Modify: `src/agentic_search/backends/sql_backend.py`, `src/agentic_search/backends/bigquery.py`, `src/agentic_search/backends/postgres.py`, `src/agentic_search/config.py`
- Test: `tests/backends/test_sql_backend.py`, `tests/test_config.py`

**Interfaces:**
- Consumes: `SqlAllowList`, `DEFAULT_SQL_FUNCTIONS`, `guard_sql(..., allow)` from Task 3.
- Produces:
  - `SqlBackend(..., columns: dict[str, list[str]] | None = None, native_functions: list[str] | None = None)`.
  - With `columns` set, discovery keeps only the named tables and, in each, only the named columns plus the table's id column. Native SQL runs through `guard_sql` with an allow list built from what discovery kept.
  - Hooks `_native_db() -> str | None` and `_native_catalog() -> str | None`. The base returns `None` for both. `PostgresBackend` returns `self.schema` as db. `BigQueryBackend` returns `self.dataset` as db and `self.project` as catalog.
  - Config keys `columns` and `native_functions` on every SQL backend.

The id column always stays, because a hit needs its key. A caller that must hide the id column cannot use this backend.

- [ ] **Step 1: Write the failing tests**

Append to `tests/backends/test_sql_backend.py`:

```python
from agentic_search.backends.bigquery import BigQueryBackend
from agentic_search.backends.native_guard import NativeQueryRejected
from agentic_search.backends.sql_backend import TableInfo, field_flags
from agentic_search.core.types import FieldSpec, FieldType, Native


def _text(name: str) -> FieldSpec:
    return FieldSpec(name=name, type=FieldType.TEXT, **field_flags(FieldType.TEXT))


class TwoTables(SqlBackend):
    dialect = "postgres"
    backend_type = "fake"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__("fake", native_query=True, **kwargs)
        self.queries: list[str] = []

    async def _discover_tables(self) -> dict[str, TableInfo]:
        return {"visits": TableInfo("visits", "id", [_text("id"), _text("dx"), _text("ssn")]),
                "people": TableInfo("people", "id", [_text("id"), _text("name")])}

    def _native_db(self) -> str | None:
        return "public"

    async def _query(self, sql: str, params: list[Any] | None) -> list[dict[str, Any]]:
        self.queries.append(sql)
        return [{"id": "v1", "dx": "pain"}]


async def test_columns_trim_discovery_and_keep_the_id():
    b = TwoTables(columns={"visits": ["dx"]})
    manifest = await b.discover()
    [visits] = manifest.collections
    assert visits.name == "visits"
    assert [f.name for f in visits.fields] == ["id", "dx"]


async def test_no_columns_keeps_everything():
    b = TwoTables()
    manifest = await b.discover()
    assert {c.name for c in manifest.collections} == {"visits", "people"}


async def test_native_sql_runs_the_resolved_query():
    b = TwoTables(columns={"visits": ["dx"]})
    hits = await b.execute(Native(source="fake", collection="visits", dialect="sql",
                                  query="SELECT id, dx FROM visits"))
    assert b.queries == ["SELECT visits.id AS id, visits.dx AS dx FROM public.visits AS visits LIMIT 20"]
    assert hits[0].doc_id


@pytest.mark.parametrize("query", ["SELECT ssn FROM visits", "SELECT COUNT(*) FROM people"])
async def test_native_sql_refuses_what_discovery_hid(query):
    b = TwoTables(columns={"visits": ["dx"]})
    with pytest.raises(NativeQueryRejected):
        await b.execute(Native(source="fake", collection="visits", dialect="sql", query=query))


async def test_native_functions_narrow_the_default_list():
    b = TwoTables(columns={"visits": ["dx"]}, native_functions=["count"])
    await b.execute(Native(source="fake", dialect="sql", query="SELECT COUNT(*) FROM visits"))
    with pytest.raises(NativeQueryRejected, match="LOWER"):
        await b.execute(Native(source="fake", dialect="sql", query="SELECT LOWER(dx) FROM visits"))


def test_qualifiers_per_backend():
    bq = BigQueryBackend("bq", "p-1", "ds", client=object())
    assert (bq._native_catalog(), bq._native_db()) == ("p-1", "ds")
    pg = PostgresBackend("pg", "postgresql://u:p@h/db", schema="clinic")
    assert (pg._native_catalog(), pg._native_db()) == (None, "clinic")
```

Append to `tests/test_config.py`:

```python
def test_sql_backend_columns_from_config(tmp_path):
    from agentic_search.config import BuildContext, build

    b = build("backend", {"name": "bq", "type": "bigquery", "project": "p-1", "dataset": "ds",
                          "native_query": True, "columns": {"visits": ["dx"]},
                          "native_functions": ["COUNT"]}, BuildContext(base_dir=tmp_path))
    assert b.allowed_columns == {"visits": ["dx"]}
    assert b.native_functions == frozenset({"COUNT"})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/backends/test_sql_backend.py tests/test_config.py -q`
Expected: FAIL with `TypeError: SqlBackend.__init__() got an unexpected keyword argument 'columns'`

- [ ] **Step 3: Implement in `SqlBackend`**

In `src/agentic_search/backends/sql_backend.py`, change the imports:

```python
from dataclasses import dataclass, field, replace
```

```python
from agentic_search.backends.native_guard import DEFAULT_SQL_FUNCTIONS, SqlAllowList, guard_sql
```

Change the constructor signature and add two attributes after `self.sample_values = sample_values`:

```python
    def __init__(self, name: str, *, tables: list[str] | None = None,
                 id_columns: dict[str, str] | None = None, embedders: dict[str, str] | None = None,
                 vector_metric: str = "cosine", native_query: bool = False,
                 description: str | None = None, max_rows: int = 100, sample_values: bool = True,
                 columns: dict[str, list[str]] | None = None,
                 native_functions: list[str] | None = None):
```

```python
        self.allowed_columns = columns
        self.native_functions = (frozenset(f.upper() for f in native_functions)
                                 if native_functions else DEFAULT_SQL_FUNCTIONS)
```

Add the two hooks in the "subclass hooks" block:

```python
    def _native_db(self) -> str | None:
        return None

    def _native_catalog(self) -> str | None:
        return None
```

Replace `_ensure_tables`:

```python
    async def _ensure_tables(self) -> dict[str, TableInfo]:
        async with self._discover_lock:
            if self._tables is None:
                self._tables = self._restrict(await self._discover_tables())
            return self._tables

    def _restrict(self, tables: dict[str, TableInfo]) -> dict[str, TableInfo]:
        """Keep only the allowed tables and, in each, the allowed columns plus the id column."""
        if self.allowed_columns is None:
            return tables
        kept: dict[str, TableInfo] = {}
        for name, info in tables.items():
            allowed = self.allowed_columns.get(name)
            if allowed is None:
                continue
            keep = set(allowed) | {info.id_column}
            kept[name] = replace(info, fields=[f for f in info.fields if f.name in keep],
                                 fulltext=[cols for cols in info.fulltext if set(cols) <= keep])
        return kept

    def _allow_list(self, tables: dict[str, TableInfo]) -> SqlAllowList | None:
        if self.allowed_columns is None:
            return None
        return SqlAllowList(tables={name: frozenset(t.columns) for name, t in tables.items()},
                            db=self._native_db(), catalog=self._native_catalog(),
                            functions=self.native_functions)
```

In `_native`, change the `guard_sql` line:

```python
        sql = guard_sql(op.query, self.dialect, min(op.limit, self.max_rows), self._allow_list(tables))
```

- [ ] **Step 4: Implement the backend hooks**

In `src/agentic_search/backends/bigquery.py`, add to `BigQueryBackend` after `_table_ref`:

```python
    def _native_db(self) -> str | None:
        return self.dataset

    def _native_catalog(self) -> str | None:
        return self.project
```

In `src/agentic_search/backends/postgres.py`, add to `PostgresBackend` after `_table_ref`:

```python
    def _native_db(self) -> str | None:
        return self.schema
```

- [ ] **Step 5: Pass the keys through config**

In `src/agentic_search/config.py`, change `_SQL_KEYS`:

```python
_SQL_KEYS = ("tables", "id_columns", "embedders", "vector_metric", "native_query", "description",
             "max_rows", "sample_values", "columns", "native_functions")
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest -q`
Expected: PASS across the suite. The contract tests in `tests/contract/` that need Docker skip the same way they do on `main`.

- [ ] **Step 7: Commit**

```bash
git add src/agentic_search/backends/sql_backend.py src/agentic_search/backends/bigquery.py \
        src/agentic_search/backends/postgres.py src/agentic_search/config.py \
        tests/backends/test_sql_backend.py tests/test_config.py
git commit -m "feat(backends): limit SQL backends to allowed columns"
```

---

### Task 5: A Postgres password asked for at each new connection

**Files:**
- Modify: `src/agentic_search/backends/postgres.py`
- Test: `tests/backends/test_postgres_unit.py`

**Interfaces:**
- Produces:
  - `PasswordProvider = Callable[[], Awaitable[str]]` in `agentic_search.backends.postgres`.
  - `PostgresBackend(..., password: PasswordProvider | None = None)`. With it, the pool's connection class asks the provider at each new connection and registers the answer as a secret. The DSN then carries no password.
  - `_fresh_password_class(psycopg: Any, provider: PasswordProvider) -> type`.

A Cloud SQL IAM database token lasts one hour. A pool that froze the first token into its DSN fails every connection it opens after that.

- [ ] **Step 1: Write the failing tests**

Append to `tests/backends/test_postgres_unit.py`:

```python
from agentic_search.backends.postgres import _fresh_password_class


@pytest.mark.asyncio
async def test_password_provider_is_asked_at_each_connect(monkeypatch):
    psycopg = pytest.importorskip("psycopg")
    seen = []

    async def fake_connect(cls, conninfo="", **kwargs):
        seen.append(kwargs["password"])
        return object()

    monkeypatch.setattr(psycopg.AsyncConnection, "connect", classmethod(fake_connect))
    tokens = iter(["tok-first-1234", "tok-second-5678"])

    async def provider():
        return next(tokens)

    connection_class = _fresh_password_class(psycopg, provider)
    await connection_class.connect("postgresql://sa@127.0.0.1/db")
    await connection_class.connect("postgresql://sa@127.0.0.1/db")
    assert seen == ["tok-first-1234", "tok-second-5678"]
    assert "tok-second-5678" not in scrub("leaked tok-second-5678")


@pytest.mark.asyncio
async def test_backend_uses_the_provider_for_its_pool(monkeypatch):
    psycopg = pytest.importorskip("psycopg")
    asked = []

    async def fake_connect(cls, conninfo="", **kwargs):
        raise psycopg.OperationalError("refused")

    monkeypatch.setattr(psycopg.AsyncConnection, "connect", classmethod(fake_connect))

    async def provider():
        asked.append(True)
        return "tok-pool-9012"

    backend = PostgresBackend("pg", "postgresql://sa@127.0.0.1:1/db", password=provider,
                              connect_timeout_s=1)
    with pytest.raises(BackendError):
        await backend.discover()
    assert asked
    await backend.close()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/backends/test_postgres_unit.py -q`
Expected: FAIL with `ImportError: cannot import name '_fresh_password_class'`

- [ ] **Step 3: Implement**

In `src/agentic_search/backends/postgres.py`, add to the imports:

```python
from collections.abc import Awaitable, Callable
```

Add after `_require_psycopg`:

```python
PasswordProvider = Callable[[], Awaitable[str]]


def _fresh_password_class(psycopg: Any, provider: PasswordProvider) -> type:
    """An AsyncConnection that asks `provider` for a password at each new connection, so a
    short-lived token (a Cloud SQL IAM token) is minted fresh rather than frozen into the DSN."""

    class FreshPassword(psycopg.AsyncConnection):  # type: ignore[misc, name-defined]
        @classmethod
        async def connect(cls, conninfo: str = "", **kwargs: Any) -> Any:
            password = await provider()
            register_secret(password)
            kwargs["password"] = password
            return await super().connect(conninfo, **kwargs)

    return FreshPassword
```

Change the constructor signature and store the provider:

```python
    def __init__(self, name: str, dsn: str, *, schema: str = "public",
                 text_search_config: str = "english", pool_size: int = 4,
                 statement_timeout_ms: int = 30_000, connect_timeout_s: float = 15,
                 tsvector_columns: dict[str, str] | None = None,
                 password: PasswordProvider | None = None, **kwargs: Any):
```

```python
        self.password = password
```

In `_get_pool`, take `psycopg` from `_require_psycopg()` and pass a connection class:

```python
                psycopg, _, AsyncConnectionPool = _require_psycopg()
                connection_class = (_fresh_password_class(psycopg, self.password)
                                    if self.password is not None else psycopg.AsyncConnection)
```

```python
                pool = AsyncConnectionPool(self.dsn, min_size=1, max_size=self.pool_size,
                                           open=False, configure=configure,
                                           connection_class=connection_class)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/backends/test_postgres_unit.py -q`
Expected: PASS, including the existing tests.

- [ ] **Step 5: Commit**

```bash
git add src/agentic_search/backends/postgres.py tests/backends/test_postgres_unit.py
git commit -m "feat(backends): ask for a Postgres password at each new connection"
```

---

### Task 6: Document and tag `v0.2.0`

**Files:**
- Modify: `README.md`, `pyproject.toml`

- [ ] **Step 1: Document the additions in `README.md`**

In the "Roles" section, change the Driver line to:

```markdown
- **Driver**: plans and calls tools (`ToolCallingDriver` over `AnthropicClient`, `OpenAICompatClient`, or `vertex_client(...)` for Gemini on Vertex).
```

Add after the paragraph that starts "Set `native_query: true` on a backend":

```markdown
Set `columns: {<table>: [<column>, ...]}` on a SQL backend to limit it to those tables and
columns. Discovery shows only them, plus each table's id column. Native SQL is resolved against
them with sqlglot's `qualify`, and the resolved query is what runs, so `SELECT *` reads only the
allowed columns. A table, column or function outside the list is refused by name. Functions come
from `DEFAULT_SQL_FUNCTIONS` unless `native_functions` names others.

`PostgresBackend(..., password=<async callable>)` asks for a password at each new connection,
for a short-lived token such as a Cloud SQL IAM database token. The DSN then carries no password.
```

In the "Service" section's YAML example, add a profile after `strict:`:

```yaml
  vertex:
    backends: [{name: notes, type: files, root: ./docs, glob: "**/*.md"}]
    driver: {type: vertex, model: gemini-3.8-flash, project: my-project, location: global}
```

- [ ] **Step 2: Bump the version**

In `pyproject.toml`, change `version = "0.1.0"` to `version = "0.2.0"`. Run `uv lock` so the lock file records it.

- [ ] **Step 3: Run the whole suite and lint**

Run: `uv run pytest -q && uv run ruff check`
Expected: PASS and no lint findings.

- [ ] **Step 4: Commit and tag**

```bash
git add README.md pyproject.toml uv.lock
git commit -m "release v0.2.0"
git tag -a v0.2.0 -m "v0.2.0: Vertex client, native SQL allow list, Postgres password provider"
```

Do not push. The owner opens the pull request, merges it, and pushes the tag. Lattice pins `agentic-search @ git+https://github.com/ajwallacemusic/agentic-search@v0.2.0`.
