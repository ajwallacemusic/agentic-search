# Plan 3: Graph & Vector Backends, Remote Embedders, TypeSafe — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the spec's v1 scope:
- Neo4j and Milvus backends.
- Remote embedders: Vertex AI, OpenAI-compatible, Hugging Face TEI, and a configurable HTTP embedder for custom containers such as MedSigLIP on Azure. They sit behind pluggable auth (API key, bearer, GCP ADC, Azure identity).
- TypeSafe System One deciders.
- The follow-ups Plans 1–2 left for this plan: hooked backend document embeddings, a stored tsvector column for Postgres full-text search, and ConfigError for constructor validation.

**Architecture:**
- **Backends:** Neo4j and Milvus implement the existing `Backend` protocol. Neo4j collections are node labels, and it adds TRAVERSE and guarded Cypher. Milvus collections map one-to-one, with BM25 via its sparse function plus dense ANN.
- **Embedders:** remote embedders share `embedders/http.py` (batching, auth headers, retries, response checks). Credentials live in `embedders/auth.py`.
- **Deciders:** `models/typesafe.py` calls the TypeSafe System One HTTP API.
  - `judge` sends one `noul` per hit over a shared state.
  - `decide` sends one `choice` over the controller actions.
- **Hooks:** backends that embed their own content get their embedder wrapped by the Harness (`bind_hooks`). Non-local embedders then pass through `Hooks.before_model_call` and `SourcePolicy`.

**Tech Stack:** httpx (core), google-auth, azure-identity, neo4j (async driver), pymilvus, Docker (neo4j:5-community, Milvus 2.5 standalone).

**Spec:** `docs/superpowers/specs/2026-09-29-agentic-search-harness-design.md`. Follow-ups: `docs/superpowers/plans/2026-09-29-plan-1-followups.md`. This plan builds on the Plan 2 branch (`feat/plan-2-backends`).

**Verification status:** every code block below was run before this plan was written.
- Unit suite: `300 passed, 5 skipped`.
- Integration contract suite against live Postgres/pgvector, MySQL, OpenSearch, Neo4j and Milvus: `63 passed, 13 skipped`.
- ruff: clean.
- Remote embedders and TypeSafe are verified against `httpx.MockTransport` (exact requests and responses). `live`-marked tests call the real services when credentials are set.

## Global Constraints

- Python `>=3.11`, pydantic v2, async throughout. Blocking clients run via `asyncio.to_thread`.
- Read-only everywhere:
  - Neo4j uses READ access-mode sessions, and native Cypher passes `guard_cypher`.
  - Milvus issues only search/query/describe calls.
  - Identifiers from models are validated against discovery before use. Every value is a bound parameter (Cypher `$params`, Milvus `filter_params`).
- Every payload bound for an external model or embedder passes through `Hooks.before_model_call`.
  - That includes backend document embeddings with non-local embedders (`HookedEmbedder`).
  - `SourcePolicy` applies to embedders too.
  - In-process embedders declare `local = True`.
- A vector op uses only the embedder bound to the field (`embedders: {"<collection>.<field>": "<embedder id>"}`).
- Adapters raise only `BackendError`/`UnsupportedOperation`; embedders raise `EmbedderError`; TypeSafe raises `TypeSafeError`. A missing optional extra raises one of these errors and names the extra.
- Secrets (API keys, tokens, passwords, DSNs) are registered with `core.secrets.register_secret` and never appear in errors or traces.
- `uv run pytest` passes with no services running or credentials set. Docker-backed tests are marked `integration`, and real-API tests are marked `live`.
- Commit after every task. Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## File Map

```
src/agentic_search/embedders/base.py       + EmbedderError, CachedEmbedder.local
src/agentic_search/embedders/local.py      + local = True on in-process embedders
src/agentic_search/embedders/hooked.py     HookedEmbedder
src/agentic_search/embedders/auth.py       NoAuth, ApiKey, Bearer, GcpAdc, AzureIdentity, build_auth
src/agentic_search/embedders/http.py       HttpEmbedder base (batching, retries, validation)
src/agentic_search/embedders/openai_compat.py, tei.py, vertex.py, http_generic.py
src/agentic_search/models/typesafe.py      TypeSafeDecider
src/agentic_search/backends/neo4j.py       Neo4jBackend
src/agentic_search/backends/milvus.py      MilvusBackend
Modified: core/harness.py (bind_hooks), backends/files.py (bind_hooks), backends/postgres.py (tsvector),
          config.py, pyproject.toml, docker-compose.yml, README.md, tests/contract/*
```

---

### Task 1: Capability-aware contract suite

**Files:**
- Modify: `tests/contract/test_backend_contract.py`

**Interfaces:**
- Produces:
  - A `require(backend, capability)` helper that skips when a backend doesn't advertise the capability.
  - The combined regex/fetch/aggregate test is split into `test_regex`, `test_fetch` and `test_aggregate`.
  - A new `test_traverse` (collection `conditions`, `TREATS` in-edges to `docs`).
  - `neo4j` and `milvus` params, which import `make_neo4j`/`make_milvus` lazily from `seed_neo4j.py`/`seed_milvus.py` (Tasks 9–10).
  - A Neo4j entry in `NATIVE`/`NATIVE_DIALECT`.
  - `test_discover` now requires only FILTER and FETCH.

- [ ] **Step 1: Replace `tests/contract/test_backend_contract.py` with:**

```python
"""Every backend must pass this suite. `files` always runs; the docker-backed params run with
`docker compose up -d --wait` and `AGENTIC_SEARCH_INTEGRATION=1 uv run pytest -m integration`."""

import os

import pytest

from agentic_search import Harness
from agentic_search.backends.base import Backend, BackendError, UnsupportedOperation
from agentic_search.backends.files import FilesBackend
from agentic_search.backends.sql_backend import SqlBackend
from agentic_search.core.types import (
    Aggregate,
    Capability,
    Eq,
    Fetch,
    FilterOnly,
    Hybrid,
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
NATIVE["neo4j"] = ("MATCH (n:docs) WHERE n.type = 'history' RETURN n.id AS id",
                   "MATCH (n:docs) DETACH DELETE n")
NATIVE_DIALECT = {"postgres": "sql", "mysql": "sql", "opensearch": "opensearch_dsl", "neo4j": "cypher"}


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
    if kind == "neo4j":
        from .seed_neo4j import make_neo4j
        return await make_neo4j()
    if kind == "milvus":
        from .seed_milvus import make_milvus
        return await make_milvus()
    raise AssertionError(kind)


@pytest.fixture(params=[
    "files",
    pytest.param("postgres", marks=[pytest.mark.integration, docker]),
    pytest.param("mysql", marks=[pytest.mark.integration, docker]),
    pytest.param("opensearch", marks=[pytest.mark.integration, docker]),
    pytest.param("neo4j", marks=[pytest.mark.integration, docker]),
    pytest.param("milvus", marks=[pytest.mark.integration, docker]),
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
    assert {Capability.FILTER, Capability.FETCH} <= m.capabilities
    if Capability.VECTOR in m.capabilities:
        emb = coll.field("embedding")
        assert emb.embedder_id == "hash64" and emb.vector_dim == 64


async def require(backend, cap):
    if cap not in (await backend.discover()).capabilities:
        pytest.skip(f"no {cap.value} support")


async def test_lexical(backend):
    await require(backend, Capability.LEXICAL)
    hits = await backend.execute(Lexical(source=backend.name, collection="docs", text="headache", limit=5))
    assert set(ids(hits)) == {"d1", "d4"}
    assert all(h.raw_score is not None for h in hits)


async def test_lexical_with_filter(backend):
    await require(backend, Capability.LEXICAL)
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


async def test_hybrid(backend):
    m = await backend.discover()
    if Capability.HYBRID not in m.capabilities:
        pytest.skip("no hybrid support")
    [q] = await corpus.EMBEDDER.embed([TextPart(text="headache fever")], "query")
    hits = await backend.execute(Hybrid(source=backend.name, collection="docs", text="headache",
                                        field="embedding", vector=q, limit=3))
    assert hits and hits[0].doc_id in {"d1", "d4"}
    assert all(h.raw_score is not None for h in hits)


async def test_regex(backend):
    await require(backend, Capability.REGEX)
    rx = await backend.execute(Regex(source=backend.name, collection="docs", pattern="print.*"))
    assert ids(rx) == ["d3"]


async def test_fetch(backend):
    [f] = await backend.execute(Fetch(source=backend.name, collection="docs", doc_ids=["d3"]))
    assert f.doc_id == "d3" and "printing" in text(f).lower()


async def test_aggregate(backend):
    await require(backend, Capability.AGGREGATE)
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
    if not isinstance(backend, SqlBackend):
        pytest.skip("not a SQL backend")
    await backend.discover()
    with pytest.raises(BackendError):
        await backend._query("DELETE FROM docs WHERE id = 'd1'", None)
    [still] = await backend.execute(Fetch(source=backend.name, collection="docs", doc_ids=["d1"]))
    assert still.doc_id == "d1"


async def test_traverse(backend):
    await require(backend, Capability.TRAVERSE)
    hits = await backend.execute(Traverse(source=backend.name, collection="conditions",
                                          start=Eq(field="id", value="headache"), rel_types=["TREATS"],
                                          direction="in", depth=1, target_label="docs"))
    assert set(ids(hits)) == {"d1", "d4"}


async def test_unsupported_op(backend):
    if Capability.TRAVERSE in (await backend.discover()).capabilities:
        pytest.skip("backend supports traverse")
    with pytest.raises((UnsupportedOperation, BackendError)):
        await backend.execute(Traverse(source=backend.name, collection="docs", start=Eq(field="type", value="x")))


async def test_harness_end_to_end(backend):
    await require(backend, Capability.LEXICAL)
    driver = ScriptedDriver([[call("lexical_search", source=backend.name, collection="docs", text="headache")]])
    h = Harness([backend], driver, embedders=[corpus.EMBEDDER], analyzer=KeywordJudge(["headache"]))
    res = await h.search("what treats headache?")
    assert set(res.keys()) == {f"{backend.name}:d1", f"{backend.name}:d4"}
    assert all(r.judged and r.p_relevant == 1.0 for r in res.hits)
```

- [ ] **Step 2: Run**

Run: `uv run pytest tests/contract -q`
Expected: `11 passed, 3 skipped` for files (skipped: native, SQL read-only, traverse).

Run: `AGENTIC_SEARCH_INTEGRATION=1 uv run pytest -m integration -q -k "not neo4j and not milvus"`
Expected: 0 failures (the neo4j/milvus params are excluded until Tasks 9–10).

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "test(contract): capability-aware contract suite; traverse test; neo4j/milvus params

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Hooked backend document embeddings (Plan 1 required follow-up)

**Files:**
- Create: `src/agentic_search/embedders/hooked.py`
- Modify: `src/agentic_search/embedders/base.py`, `src/agentic_search/embedders/local.py`, `src/agentic_search/backends/files.py`, `src/agentic_search/core/harness.py`
- Test: `tests/embedders/test_hooked.py`

**Interfaces:**
- Produces:
  - `EmbedderError(Exception)` in `embedders/base.py`.
  - `local = True` on `HashEmbedder` and `SentenceTransformerEmbedder`, and a `CachedEmbedder.local` property that passes through to the inner embedder.
  - `HookedEmbedder(inner, hooks, *, source, policy)`. It checks `SourcePolicy`, then calls `hooks.before_model_call(inner.id, items)`, then `inner.embed`.
  - `FilesBackend.bind_hooks(hooks, policy)` wraps its embedder when it is not local and not already hooked.
  - `Harness.__init__` calls `bind_hooks` on every backend that has it.

- [ ] **Step 1: Write the failing tests**

`tests/embedders/test_hooked.py`:
```python
import pytest

from agentic_search import Harness
from agentic_search.backends.files import FilesBackend
from agentic_search.core.hooks import Hooks, SourcePolicy
from agentic_search.core.types import TextPart
from agentic_search.embedders.base import EmbedderError
from agentic_search.embedders.hooked import HookedEmbedder
from agentic_search.embedders.local import HashEmbedder
from agentic_search.testing import ScriptedDriver


class RemoteLike(HashEmbedder):
    """Stands in for an HTTP embedder: same vectors, but not in-process."""

    local = False


class Recording(Hooks):
    def __init__(self):
        self.calls = []

    async def before_model_call(self, model_id, payload):
        self.calls.append((model_id, payload))
        if isinstance(payload, list):
            return [TextPart(text="[redacted]") for _ in payload]
        return payload


async def test_hooked_embedder_applies_hooks_and_policy():
    hooks = Recording()
    inner = RemoteLike(dim=8, id="remote-embedder")
    e = HookedEmbedder(inner, hooks, source="notes", policy=SourcePolicy())
    assert e.id == "remote-embedder" and e.dim == 8
    [v] = await e.embed([TextPart(text="patient John Doe")], "document")
    assert v == (await inner.embed([TextPart(text="[redacted]")], "document"))[0]
    assert hooks.calls[0][0] == "remote-embedder"
    blocked = HookedEmbedder(inner, hooks, source="phi", policy=SourcePolicy({"phi": {"local-llm"}}))
    with pytest.raises(EmbedderError, match="forbids"):
        await blocked.embed([TextPart(text="x")], "document")


async def test_harness_binds_files_backend_embedder(medical_docs):
    hooks = Recording()
    backend = FilesBackend.from_documents("docs", medical_docs, embedder=RemoteLike(dim=8, id="emb"))
    h = Harness([backend], ScriptedDriver([]), hooks=hooks)
    assert isinstance(backend.embedder, HookedEmbedder)
    Harness([backend], ScriptedDriver([]), hooks=hooks)  # binding twice does not double-wrap
    assert not isinstance(backend.embedder.inner, HookedEmbedder)
    await h.setup()
    assert any(model == "emb" and isinstance(p, list) for model, p in hooks.calls)
    assert h.manifests["docs"].resolve_collection(None).field("embedding").embedder_id == "emb"


async def test_policy_blocks_document_embedding_at_setup(medical_docs):
    backend = FilesBackend.from_documents("phi", medical_docs, embedder=RemoteLike(dim=8, id="cloud-emb"))
    h = Harness([backend, FilesBackend.from_documents("ok", medical_docs)], ScriptedDriver([]),
                source_policy={"phi": {"local-llm"}})
    await h.setup()
    assert "phi" in h.setup_errors and "forbids" in h.setup_errors["phi"]


async def test_local_embedders_are_not_wrapped(medical_docs):
    backend = FilesBackend.from_documents("phi", medical_docs, embedder=HashEmbedder(dim=8, id="hash8"))
    h = Harness([backend], ScriptedDriver([]), source_policy={"phi": {"local-llm"}})
    assert not isinstance(backend.embedder, HookedEmbedder)
    await h.setup()
    assert h.setup_errors == {}
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/embedders/test_hooked.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'agentic_search.embedders.hooked'`

- [ ] **Step 3: Implement**

In `src/agentic_search/embedders/base.py`, add after `Embedding = list[float]`:
```python
class EmbedderError(Exception):
    """An embedder could not produce vectors (bad config, auth, transport or response shape)."""
```
and add to `CachedEmbedder`, after its `dim` property:
```python
    @property
    def local(self) -> bool:
        return bool(getattr(self.inner, "local", False))
```

In `src/agentic_search/embedders/local.py`, add `local = True  # runs in-process: document content never leaves the machine` as the first class attribute of `HashEmbedder`, and `local = True` as the first class attribute of `SentenceTransformerEmbedder`.

`src/agentic_search/embedders/hooked.py`:
```python
"""Route a backend's document embeddings through Hooks and SourcePolicy.

Backends that embed their own content (FilesBackend) would otherwise send it to an embedder
without passing `Hooks.before_model_call`; with a remote embedder that is a leak. The Harness wraps
such embedders with HookedEmbedder via `Backend.bind_hooks` (see core/harness.py). Embedders that
declare `local = True` (in-process models) are left unwrapped: nothing leaves the machine."""

from __future__ import annotations

from typing import Any

from agentic_search.core.types import Content
from agentic_search.embedders.base import Embedder, EmbedderError, Embedding, Purpose


class HookedEmbedder:
    def __init__(self, inner: Embedder, hooks: Any, *, source: str, policy: Any):
        self.inner = inner
        self.hooks = hooks
        self.source = source
        self.policy = policy
        self.id = inner.id
        self.modalities = inner.modalities

    @property
    def dim(self) -> int:
        return self.inner.dim

    async def embed(self, items: list[Content], purpose: Purpose) -> list[Embedding]:
        if not self.policy.allows(self.source, self.id):
            raise EmbedderError(f"source policy forbids sending {self.source!r} content to "
                                f"embedder {self.id!r}")
        items = await self.hooks.before_model_call(self.id, items)
        return await self.inner.embed(items, purpose)
```

In `src/agentic_search/backends/files.py`, add before `async def close`:
```python
    def bind_hooks(self, hooks: Any, policy: Any) -> None:
        """Route document embeddings through Hooks and SourcePolicy (called by Harness)."""
        from agentic_search.embedders.hooked import HookedEmbedder

        if (self.embedder is not None and not isinstance(self.embedder, HookedEmbedder)
                and not getattr(self.embedder, "local", False)):
            self.embedder = HookedEmbedder(self.embedder, hooks, source=self.name, policy=policy)
```

In `src/agentic_search/core/harness.py` `Harness.__init__`, directly after `self.settings = settings or HarnessSettings()`:
```python
        for backend in backends:  # backends that embed their own content get hooked embedders
            bind = getattr(backend, "bind_hooks", None)
            if callable(bind):
                bind(self.hooks, self.policy)
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest -q`
Expected: all pass. Tests that pair a restrictive `source_policy` with a local embedder must still pass; that's the reason for `local = True`.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: route backend document embeddings through hooks and source policy

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Auth providers and the HTTP embedder base

**Files:**
- Modify: `pyproject.toml`
- Create: `src/agentic_search/embedders/auth.py`, `src/agentic_search/embedders/http.py`, `tests/embedders/mock_http.py`
- Test: `tests/embedders/test_auth.py`, `tests/embedders/test_http_base.py` (the base tests use the Task 4 providers, so they run GREEN in Task 4)

**Interfaces:**
- Produces:
  - `Auth` protocol (`async headers() -> dict`) and `NoAuth`.
  - `ApiKey(key, *, header="x-api-key", prefix="")` and `Bearer(token)`.
  - `GcpAdc(*, scopes, credentials=None)`: ADC via google-auth, refreshed in a thread.
  - `AzureIdentity(scope, *, credential=None)`: `DefaultAzureCredential`, token cached until 60 s before expiry.
  - `build_auth(cfg)`.
  - `HttpEmbedder(id, dim, modalities, *, auth, batch_size=32, concurrency=4, timeout_s=30, max_retries=4, backoff_s=0.5, client=None)`, with `post_json(url, payload, extra_headers)`, `embed`, abstract `_embed_batch`, and `close`.
  - Retries cover 408/429/5xx/529 and transport errors. Everything else raises `EmbedderError` with the message scrubbed.

- [ ] **Step 1: Update `pyproject.toml`**

Add `"httpx>=0.27",` to `[project] dependencies`. (The Anthropic and OpenAI SDK versions in the lockfile no longer depend on httpx.) Add extras after `opensearch = [...]`:
```toml
gcp = ["google-auth[requests]>=2.29"]
azure = ["azure-identity>=1.17"]
```
Add `"google-auth[requests]>=2.29",` and `"azure-identity>=1.17",` at the end of the `dev` group. Run `uv sync`.

- [ ] **Step 2: Write the failing tests**

`tests/embedders/mock_http.py`:
```python
"""httpx.MockTransport helpers for remote-embedder and model tests (no network)."""

import json

import httpx


def mock(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


class Recorder:
    def __init__(self, respond):
        self.requests: list[httpx.Request] = []
        self.respond = respond

    def __call__(self, request):
        self.requests.append(request)
        return self.respond(request)

    def body(self, i=0):
        return json.loads(self.requests[i].content)
```

`tests/embedders/test_auth.py`:
```python
import time
from types import SimpleNamespace

import pytest

from agentic_search.core.secrets import scrub
from agentic_search.embedders.auth import (
    ApiKey,
    AzureIdentity,
    Bearer,
    GcpAdc,
    NoAuth,
    build_auth,
)
from agentic_search.embedders.base import EmbedderError


async def test_static_auth_headers_and_registration():
    assert await NoAuth().headers() == {}
    assert await ApiKey("key-abc-123", header="api-key").headers() == {"api-key": "key-abc-123"}
    assert await Bearer("tok-xyz-789").headers() == {"Authorization": "Bearer tok-xyz-789"}
    assert scrub("key-abc-123 tok-xyz-789") == "*** ***"
    with pytest.raises(ValueError):
        ApiKey("")


class FakeGoogleCreds:
    def __init__(self):
        self.valid = False
        self.token = None
        self.refreshes = 0

    def refresh(self, request):
        self.refreshes += 1
        self.token = f"ya29.fake-{self.refreshes}"
        self.valid = True


async def test_gcp_adc_refreshes_only_when_invalid():
    creds = FakeGoogleCreds()
    auth = GcpAdc(credentials=creds)
    assert await auth.headers() == {"Authorization": "Bearer ya29.fake-1"}
    assert await auth.headers() == {"Authorization": "Bearer ya29.fake-1"}
    assert creds.refreshes == 1
    creds.valid = False
    assert (await auth.headers())["Authorization"] == "Bearer ya29.fake-2"
    assert scrub("ya29.fake-2") == "***"


async def test_gcp_adc_failure_is_embedder_error():
    class Broken(FakeGoogleCreds):
        def refresh(self, request):
            raise RuntimeError("metadata server unreachable")

    with pytest.raises(EmbedderError, match="metadata server unreachable"):
        await GcpAdc(credentials=Broken()).headers()


class FakeAzureCred:
    def __init__(self, ttl=3600):
        self.calls = 0
        self.ttl = ttl
        self.closed = False

    async def get_token(self, scope):
        self.calls += 1
        return SimpleNamespace(token=f"eyJ.fake-{self.calls}", expires_on=time.time() + self.ttl)

    async def close(self):
        self.closed = True


async def test_azure_identity_caches_until_near_expiry():
    cred = FakeAzureCred()
    auth = AzureIdentity("api://medsiglip/.default", credential=cred)
    assert await auth.headers() == {"Authorization": "Bearer eyJ.fake-1"}
    await auth.headers()
    assert cred.calls == 1
    short = FakeAzureCred(ttl=30)  # inside the 60 s refresh margin
    auth2 = AzureIdentity("s", credential=short)
    await auth2.headers()
    await auth2.headers()
    assert short.calls == 2
    await auth.close()
    assert cred.closed and scrub("eyJ.fake-1") == "***"


async def test_azure_failure_is_embedder_error():
    class Broken(FakeAzureCred):
        async def get_token(self, scope):
            raise RuntimeError("no managed identity")

    with pytest.raises(EmbedderError, match="no managed identity"):
        await AzureIdentity("s", credential=Broken()).headers()


def test_build_auth():
    assert isinstance(build_auth(None), NoAuth)
    assert isinstance(build_auth({"type": "bearer", "token": "abcd-token"}), Bearer)
    assert isinstance(build_auth({"type": "api_key", "key": "k-1234"}), ApiKey)
    assert isinstance(build_auth({"type": "gcp_adc"}), GcpAdc)
    assert isinstance(build_auth({"type": "azure_identity", "scope": "s"}), AzureIdentity)
    with pytest.raises(ValueError):
        build_auth({"type": "kerberos"})
```

- [ ] **Step 3: Run to verify failure**

Run: `uv run pytest tests/embedders/test_auth.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'agentic_search.embedders.auth'`

- [ ] **Step 4: Implement**

`src/agentic_search/embedders/auth.py`:
```python
"""Credentials for remote embedders and models, kept separate from the endpoint adapters.

Every provider returns request headers and registers the secret it handles with
core.secrets so it is masked in errors and traces."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Protocol

from agentic_search.core.secrets import register_secret
from agentic_search.embedders.base import EmbedderError

GCP_SCOPE = "https://www.googleapis.com/auth/cloud-platform"
_REFRESH_MARGIN_S = 60


class Auth(Protocol):
    async def headers(self) -> dict[str, str]: ...


class NoAuth:
    async def headers(self) -> dict[str, str]:
        return {}


class ApiKey:
    """A static key sent in a header, e.g. `x-api-key: ...` or `api-key: ...` (Azure OpenAI)."""

    def __init__(self, key: str, *, header: str = "x-api-key", prefix: str = ""):
        if not key:
            raise ValueError("ApiKey needs a non-empty key")
        register_secret(key)
        self._value = f"{prefix}{key}"
        self.header = header

    async def headers(self) -> dict[str, str]:
        return {self.header: self._value}


class Bearer:
    def __init__(self, token: str):
        if not token:
            raise ValueError("Bearer needs a non-empty token")
        register_secret(token)
        self._token = token

    async def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._token}"}


class GcpAdc:
    """Google Application Default Credentials (service account, workload identity, gcloud login).
    Needs the `gcp` extra unless `credentials` (a google.auth Credentials object) is injected."""

    def __init__(self, *, scopes: tuple[str, ...] = (GCP_SCOPE,), credentials: Any = None):
        self.scopes = scopes
        self._credentials = credentials
        self._lock = asyncio.Lock()

    def _load_and_refresh(self) -> str:
        try:
            import google.auth
            import google.auth.transport.requests
        except ImportError as exc:
            raise EmbedderError("GcpAdc needs the `gcp` extra: pip install 'agentic-search[gcp]'") from exc
        if self._credentials is None:
            self._credentials, _ = google.auth.default(scopes=list(self.scopes))
        if not self._credentials.valid:
            self._credentials.refresh(google.auth.transport.requests.Request())
        return self._credentials.token

    async def headers(self) -> dict[str, str]:
        async with self._lock:
            try:
                token = await asyncio.to_thread(self._load_and_refresh)
            except EmbedderError:
                raise
            except Exception as exc:
                raise EmbedderError(f"GCP credentials unavailable: {type(exc).__name__}: {exc}") from exc
        register_secret(token)
        return {"Authorization": f"Bearer {token}"}


class AzureIdentity:
    """Microsoft Entra ID token (managed identity, workload identity, az login) for `scope`,
    e.g. "api://<app-id>/.default" or "https://cognitiveservices.azure.com/.default".
    Needs the `azure` extra unless `credential` (an async azure.core TokenCredential) is injected."""

    def __init__(self, scope: str, *, credential: Any = None):
        if not scope:
            raise ValueError("AzureIdentity needs a scope")
        self.scope = scope
        self._credential = credential
        self._token: str | None = None
        self._expires_on = 0.0
        self._lock = asyncio.Lock()

    async def headers(self) -> dict[str, str]:
        async with self._lock:
            if self._token is None or time.time() >= self._expires_on - _REFRESH_MARGIN_S:
                if self._credential is None:
                    try:
                        from azure.identity.aio import DefaultAzureCredential
                    except ImportError as exc:
                        raise EmbedderError(
                            "AzureIdentity needs the `azure` extra: pip install 'agentic-search[azure]'"
                        ) from exc
                    self._credential = DefaultAzureCredential()
                try:
                    access = await self._credential.get_token(self.scope)
                except Exception as exc:
                    raise EmbedderError(f"Azure credentials unavailable: {type(exc).__name__}: {exc}") from exc
                self._token, self._expires_on = access.token, float(access.expires_on)
                register_secret(self._token)
        return {"Authorization": f"Bearer {self._token}"}

    async def close(self) -> None:
        if self._credential is not None and hasattr(self._credential, "close"):
            await self._credential.close()


def build_auth(cfg: dict[str, Any] | None) -> Any:
    """Config form: {type: none|api_key|bearer|gcp_adc|azure_identity, ...}."""
    if not cfg:
        return NoAuth()
    kind = cfg.get("type", "none")
    if kind == "none":
        return NoAuth()
    if kind == "api_key":
        return ApiKey(cfg["key"], header=cfg.get("header", "x-api-key"), prefix=cfg.get("prefix", ""))
    if kind == "bearer":
        return Bearer(cfg["token"])
    if kind == "gcp_adc":
        return GcpAdc(scopes=tuple(cfg.get("scopes", (GCP_SCOPE,))))
    if kind == "azure_identity":
        return AzureIdentity(cfg["scope"])
    raise ValueError(f"unknown auth type {kind!r}; one of none, api_key, bearer, gcp_adc, azure_identity")
```

`src/agentic_search/embedders/http.py`:
```python
"""Shared plumbing for remote embedders: batching, auth headers, retries with backoff, response
validation. Subclasses implement `_embed_batch` for one provider's request/response shape."""

from __future__ import annotations

import asyncio
import random
from typing import Any

import httpx

from agentic_search.core.secrets import scrub
from agentic_search.core.types import Content, Modality, modality_of
from agentic_search.embedders.auth import NoAuth
from agentic_search.embedders.base import EmbedderError, Embedding, Purpose

RETRY_STATUSES = {408, 429, 500, 502, 503, 504, 529}


class HttpEmbedder:
    def __init__(self, id: str, dim: int, modalities: set[Modality], *, auth: Any = None,
                 batch_size: int = 32, concurrency: int = 4, timeout_s: float = 30.0,
                 max_retries: int = 4, backoff_s: float = 0.5, client: httpx.AsyncClient | None = None):
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.id = id
        self.dim = dim
        self.modalities = modalities
        self.auth = auth or NoAuth()
        self.batch_size = max(1, batch_size)
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.backoff_s = backoff_s
        self._client = client
        self._owns_client = client is None
        self._sem = asyncio.Semaphore(max(1, concurrency))

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout_s)
        return self._client

    async def post_json(self, url: str, payload: Any, extra_headers: dict[str, str] | None = None) -> Any:
        """POST JSON with auth headers; retry transient failures; raise EmbedderError otherwise."""
        client = self._get_client()
        last = ""
        for attempt in range(self.max_retries + 1):
            headers = {**(await self.auth.headers()), **(extra_headers or {})}
            try:
                resp = await client.post(url, json=payload, headers=headers)
            except httpx.HTTPError as exc:
                last = f"{type(exc).__name__}: {exc}"
            else:
                if resp.status_code < 300:
                    try:
                        return resp.json()
                    except ValueError as exc:
                        raise EmbedderError(f"{self.id}: response is not JSON") from exc
                last = f"HTTP {resp.status_code}: {resp.text[:300]}"
                if resp.status_code not in RETRY_STATUSES:
                    break
            if attempt < self.max_retries:
                await asyncio.sleep(self.backoff_s * (2 ** attempt) * (1 + random.random()))
        raise EmbedderError(scrub(f"{self.id}: {last}"))

    async def embed(self, items: list[Content], purpose: Purpose) -> list[Embedding]:
        for item in items:
            if modality_of(item) not in self.modalities:
                raise EmbedderError(f"{self.id} cannot embed {modality_of(item).value}")
        batches = [items[i:i + self.batch_size] for i in range(0, len(items), self.batch_size)]

        async def run(batch: list[Content]) -> list[Embedding]:
            async with self._sem:
                return await self._embed_batch(batch, purpose)

        results = await asyncio.gather(*(run(b) for b in batches))
        vectors = [v for batch in results for v in batch]
        if len(vectors) != len(items):
            raise EmbedderError(f"{self.id}: expected {len(items)} vectors, got {len(vectors)}")
        for v in vectors:
            if len(v) != self.dim:
                raise EmbedderError(f"{self.id}: expected dim {self.dim}, got {len(v)}")
        return [[float(x) for x in v] for v in vectors]

    async def _embed_batch(self, batch: list[Content], purpose: Purpose) -> list[Embedding]:
        raise NotImplementedError

    async def close(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None
        close = getattr(self.auth, "close", None)
        if close is not None:
            await close()
```

- [ ] **Step 5: Run to verify pass**

Run: `uv run pytest tests/embedders/test_auth.py -q`
Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat(embedders): auth providers (API key, bearer, GCP ADC, Azure identity) and HTTP embedder base

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: OpenAI-compatible and TEI embedders

**Files:**
- Create: `src/agentic_search/embedders/openai_compat.py`, `src/agentic_search/embedders/tei.py`
- Test: `tests/embedders/test_openai_tei.py`, `tests/embedders/test_http_base.py`

**Interfaces:**
- Produces:
  - `OpenAICompatEmbedder(model, dim, *, base_url, api_key=None, auth=None, dimensions=None, query_prefix, document_prefix, id=None, **HttpEmbedder kwargs)`. Its id defaults to `openai:{model}`.
  - `TEIEmbedder(url, dim, *, id=None, query_prefix, document_prefix, normalize=True, auth=None, **kwargs)`. Its id defaults to `tei:{url}`.

- [ ] **Step 1: Write the failing tests**

`tests/embedders/test_openai_tei.py`:
```python
import json

import httpx

from agentic_search.core.types import TextPart
from agentic_search.embedders.openai_compat import OpenAICompatEmbedder
from agentic_search.embedders.tei import TEIEmbedder

from .mock_http import Recorder, mock


async def test_openai_compat_request_and_order():
    def respond(req):
        n = len(json.loads(req.content)["input"])
        return httpx.Response(200, json={"data": [{"index": i, "embedding": [float(i), 1.0]}
                                                  for i in reversed(range(n))]})

    rec = Recorder(respond)
    e = OpenAICompatEmbedder("text-embedding-3-small", 2, base_url="https://llm.local/v1/",
                             api_key="sk-test-1234", dimensions=2, query_prefix="query: ",
                             batch_size=2, client=mock(rec))
    vecs = await e.embed([TextPart(text="a"), TextPart(text="b"), TextPart(text="c")], "query")
    assert vecs == [[0.0, 1.0], [1.0, 1.0], [0.0, 1.0]]
    assert str(rec.requests[0].url) == "https://llm.local/v1/embeddings"
    assert rec.requests[0].headers["authorization"] == "Bearer sk-test-1234"
    assert rec.body(0) == {"model": "text-embedding-3-small", "input": ["query: a", "query: b"],
                           "dimensions": 2}
    assert e.id == "openai:text-embedding-3-small"


async def test_tei_request():
    rec = Recorder(lambda req: httpx.Response(200, json=[[0.1, 0.2, 0.3]]))
    e = TEIEmbedder("http://tei:8080", 3, id="tei:bge", document_prefix="passage: ", client=mock(rec))
    assert await e.embed([TextPart(text="x")], "document") == [[0.1, 0.2, 0.3]]
    assert str(rec.requests[0].url) == "http://tei:8080/embed"
    assert rec.body() == {"inputs": ["passage: x"], "normalize": True, "truncate": True}
```

`tests/embedders/test_http_base.py`:
```python
import httpx
import pytest

from agentic_search.core.secrets import scrub
from agentic_search.core.types import ImagePart, TextPart
from agentic_search.embedders.auth import Bearer
from agentic_search.embedders.base import EmbedderError
from agentic_search.embedders.openai_compat import OpenAICompatEmbedder
from agentic_search.embedders.tei import TEIEmbedder

from .mock_http import mock


async def test_retries_transient_then_succeeds():
    calls = {"n": 0}

    def respond(req):
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(503, text="overloaded")
        return httpx.Response(200, json=[[1.0]])

    e = TEIEmbedder("http://tei", 1, client=mock(respond), backoff_s=0.001)
    assert await e.embed([TextPart(text="x")], "query") == [[1.0]]
    assert calls["n"] == 3


async def test_errors_are_embedder_errors_and_scrubbed():
    e = TEIEmbedder("http://tei", 1, client=mock(lambda r: httpx.Response(401, text="bad key sk-live-secret99")),
                    auth=Bearer("sk-live-secret99"), backoff_s=0.001)
    with pytest.raises(EmbedderError) as exc:
        await e.embed([TextPart(text="x")], "query")
    assert "401" in str(exc.value) and "sk-live-secret99" not in str(exc.value)
    wrong_dim = TEIEmbedder("http://tei", 3, client=mock(lambda r: httpx.Response(200, json=[[1.0]])))
    with pytest.raises(EmbedderError, match="dim"):
        await wrong_dim.embed([TextPart(text="x")], "query")
    text_only = TEIEmbedder("http://tei", 1, client=mock(lambda r: httpx.Response(200, json=[[1.0]])))
    with pytest.raises(EmbedderError, match="image"):
        await text_only.embed([ImagePart(data=b"x")], "query")
    shape = OpenAICompatEmbedder("m", 1, client=mock(lambda r: httpx.Response(200, json={"nope": 1})))
    with pytest.raises(EmbedderError, match="shape"):
        await shape.embed([TextPart(text="x")], "query")
    assert scrub("sk-live-secret99") == "***"


async def test_transport_errors_retry_then_fail():
    def respond(req):
        raise httpx.ConnectError("refused")

    e = TEIEmbedder("http://tei", 1, client=mock(respond), max_retries=2, backoff_s=0.001)
    with pytest.raises(EmbedderError, match="ConnectError"):
        await e.embed([TextPart(text="x")], "query")
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/embedders/test_openai_tei.py tests/embedders/test_http_base.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'agentic_search.embedders.openai_compat'`

- [ ] **Step 3: Implement**

`src/agentic_search/embedders/openai_compat.py`:
```python
"""OpenAI-compatible `/embeddings` endpoint: OpenAI, Azure OpenAI (via base_url + api-key auth),
vLLM, Ollama, Together, and most hosted open models. Text only."""

from __future__ import annotations

from typing import Any

from agentic_search.core.types import Content, Modality, text_of
from agentic_search.embedders.auth import Bearer
from agentic_search.embedders.base import EmbedderError, Embedding, Purpose
from agentic_search.embedders.http import HttpEmbedder


class OpenAICompatEmbedder(HttpEmbedder):
    def __init__(self, model: str, dim: int, *, base_url: str = "https://api.openai.com/v1",
                 api_key: str | None = None, auth: Any = None, dimensions: int | None = None,
                 query_prefix: str = "", document_prefix: str = "", id: str | None = None,
                 **kwargs: Any):
        if auth is None and api_key:
            auth = Bearer(api_key)
        super().__init__(id or f"openai:{model}", dim, {Modality.TEXT}, auth=auth, **kwargs)
        self.model = model
        self.url = base_url.rstrip("/") + "/embeddings"
        self.dimensions = dimensions
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix

    async def _embed_batch(self, batch: list[Content], purpose: Purpose) -> list[Embedding]:
        prefix = self.query_prefix if purpose == "query" else self.document_prefix
        payload: dict[str, Any] = {"model": self.model, "input": [prefix + text_of([c]) for c in batch]}
        if self.dimensions is not None:
            payload["dimensions"] = self.dimensions
        body = await self.post_json(self.url, payload)
        try:
            rows = sorted(body["data"], key=lambda r: r["index"])
            return [r["embedding"] for r in rows]
        except (KeyError, TypeError) as exc:
            raise EmbedderError(f"{self.id}: unexpected response shape") from exc
```

`src/agentic_search/embedders/tei.py`:
```python
"""Hugging Face Text Embeddings Inference (TEI) `/embed` endpoint — the usual way to self-host
text embedders (BGE, E5, GTE, …). Text only."""

from __future__ import annotations

from typing import Any

from agentic_search.core.types import Content, Modality, text_of
from agentic_search.embedders.base import EmbedderError, Embedding, Purpose
from agentic_search.embedders.http import HttpEmbedder


class TEIEmbedder(HttpEmbedder):
    def __init__(self, url: str, dim: int, *, id: str | None = None, query_prefix: str = "",
                 document_prefix: str = "", normalize: bool = True, auth: Any = None, **kwargs: Any):
        super().__init__(id or f"tei:{url}", dim, {Modality.TEXT}, auth=auth, **kwargs)
        self.url = url.rstrip("/") + "/embed"
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.normalize = normalize

    async def _embed_batch(self, batch: list[Content], purpose: Purpose) -> list[Embedding]:
        prefix = self.query_prefix if purpose == "query" else self.document_prefix
        body = await self.post_json(self.url, {
            "inputs": [prefix + text_of([c]) for c in batch],
            "normalize": self.normalize, "truncate": True,
        })
        if not isinstance(body, list):
            raise EmbedderError(f"{self.id}: unexpected response shape")
        return body
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/embedders -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(embedders): OpenAI-compatible and TEI embedders

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Vertex AI embedder

**Files:**
- Create: `src/agentic_search/embedders/vertex.py`
- Test: `tests/embedders/test_vertex.py`

**Interfaces:**
- Produces: `VertexEmbedder(model, dim, *, project, location="us-central1", auth=None (→ GcpAdc), id=None, batch_size=None, endpoint=None, **kwargs)`. Its id defaults to `vertex:{model}`.
- Text models get a task type per purpose. `multimodalembedding@*` takes one instance per request, accepts text or image, and has `modalities={TEXT, IMAGE}`.
- The batch size defaults to 1 for `gemini-embedding*` and multimodal models, 32 otherwise.

- [ ] **Step 1: Write the failing tests**

`tests/embedders/test_vertex.py`:
```python
import base64
import json

import httpx

from agentic_search.core.types import ImagePart, TextPart
from agentic_search.embedders.auth import Bearer
from agentic_search.embedders.vertex import VertexEmbedder

from .mock_http import Recorder, mock


async def test_vertex_text_request():
    rec = Recorder(lambda req: httpx.Response(200, json={"predictions": [
        {"embeddings": {"values": [0.5, 0.5]}}]}))
    e = VertexEmbedder("gemini-embedding-001", 2, project="proj-1", location="us-east1",
                       auth=Bearer("ya29.token-abc"), client=mock(rec))
    assert e.batch_size == 1 and e.id == "vertex:gemini-embedding-001"
    await e.embed([TextPart(text="headache")], "query")
    assert str(rec.requests[0].url) == (
        "https://us-east1-aiplatform.googleapis.com/v1/projects/proj-1/locations/us-east1"
        "/publishers/google/models/gemini-embedding-001:predict")
    assert rec.body() == {"instances": [{"content": "headache", "task_type": "RETRIEVAL_QUERY"}],
                          "parameters": {"outputDimensionality": 2, "autoTruncate": True}}


async def test_vertex_multimodal_image_and_text():
    def respond(req):
        inst = json.loads(req.content)["instances"][0]
        key = "imageEmbedding" if "image" in inst else "textEmbedding"
        return httpx.Response(200, json={"predictions": [{key: [1.0, 0.0] if key == "imageEmbedding" else [0.0, 1.0]}]})

    rec = Recorder(respond)
    e = VertexEmbedder("multimodalembedding@001", 2, project="p", auth=Bearer("tok-12345"),
                       client=mock(rec))
    vecs = await e.embed([ImagePart(data=b"\x89PNG"), TextPart(text="x-ray")], "query")
    assert vecs == [[1.0, 0.0], [0.0, 1.0]]
    assert rec.body(0) == {"instances": [{"image": {"bytesBase64Encoded": base64.b64encode(b"\x89PNG").decode()}}],
                           "parameters": {"dimension": 2}}
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/embedders/test_vertex.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'agentic_search.embedders.vertex'`

- [ ] **Step 3: Implement**

`src/agentic_search/embedders/vertex.py`:
```python
"""Vertex AI embeddings via the `:predict` endpoint.

- Text models (`gemini-embedding-001`, `text-embedding-005`, …): instances
  `{"content": text, "task_type": RETRIEVAL_QUERY|RETRIEVAL_DOCUMENT}`, response
  `predictions[i].embeddings.values`.
- Multimodal (`multimodalembedding@001`): one instance per request,
  `{"text": …}` or `{"image": {"bytesBase64Encoded": …}}`, response
  `predictions[0].textEmbedding` / `imageEmbedding`; text and images share one space.

Auth defaults to Google Application Default Credentials (the `gcp` extra)."""

from __future__ import annotations

import base64
from typing import Any

from agentic_search.core.types import Content, ImagePart, Modality, image_bytes, text_of
from agentic_search.embedders.auth import GcpAdc
from agentic_search.embedders.base import EmbedderError, Embedding, Purpose
from agentic_search.embedders.http import HttpEmbedder

TASK_TYPES = {"query": "RETRIEVAL_QUERY", "document": "RETRIEVAL_DOCUMENT"}


class VertexEmbedder(HttpEmbedder):
    def __init__(self, model: str, dim: int, *, project: str, location: str = "us-central1",
                 auth: Any = None, id: str | None = None, batch_size: int | None = None,
                 endpoint: str | None = None, **kwargs: Any):
        self.multimodal = model.startswith("multimodalembedding")
        if batch_size is None:
            batch_size = 1 if self.multimodal or model.startswith("gemini-embedding") else 32
        modalities = {Modality.TEXT, Modality.IMAGE} if self.multimodal else {Modality.TEXT}
        super().__init__(id or f"vertex:{model}", dim, modalities, auth=auth or GcpAdc(),
                         batch_size=batch_size, **kwargs)
        self.model = model
        host = endpoint or f"https://{location}-aiplatform.googleapis.com"
        self.url = (f"{host}/v1/projects/{project}/locations/{location}"
                    f"/publishers/google/models/{model}:predict")

    async def _embed_batch(self, batch: list[Content], purpose: Purpose) -> list[Embedding]:
        if self.multimodal:
            return [await self._multimodal(item) for item in batch]
        body = await self.post_json(self.url, {
            "instances": [{"content": text_of([c]), "task_type": TASK_TYPES[purpose]} for c in batch],
            "parameters": {"outputDimensionality": self.dim, "autoTruncate": True},
        })
        try:
            return [p["embeddings"]["values"] for p in body["predictions"]]
        except (KeyError, TypeError) as exc:
            raise EmbedderError(f"{self.id}: unexpected response shape") from exc

    async def _multimodal(self, item: Content) -> Embedding:
        if isinstance(item, ImagePart):
            instance: dict[str, Any] = {"image": {"bytesBase64Encoded":
                                                  base64.b64encode(image_bytes(item)).decode()}}
            key = "imageEmbedding"
        else:
            instance, key = {"text": text_of([item])}, "textEmbedding"
        body = await self.post_json(self.url, {"instances": [instance],
                                               "parameters": {"dimension": self.dim}})
        try:
            return body["predictions"][0][key]
        except (KeyError, IndexError, TypeError) as exc:
            raise EmbedderError(f"{self.id}: unexpected response shape") from exc
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/embedders/test_vertex.py -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(embedders): Vertex AI text and multimodal embedder

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Generic HTTP embedder (custom containers such as MedSigLIP)

**Files:**
- Create: `src/agentic_search/embedders/http_generic.py`
- Test: `tests/embedders/test_http_generic.py`

**Interfaces:**
- Produces:
  - `json_path(value, path) -> list`, supporting `$`, `.name`, `[n]`, `[*]` and `['key']`.
  - `render(template, values)`.
  - `GenericHttpEmbedder(url, dim, *, request={"text": body, "image": body}, response_path, id=None, auth=None, headers=None, **kwargs)`. It sends one request per item. Placeholders are `{{text}}`, `{{b64}}` and `{{mime}}`, and the first `response_path` match is the vector.

- [ ] **Step 1: Write the failing tests**

`tests/embedders/test_http_generic.py`:
```python
import base64
import json

import httpx
import pytest

from agentic_search.core.types import ImagePart, TextPart
from agentic_search.embedders.auth import Bearer
from agentic_search.embedders.http_generic import GenericHttpEmbedder, json_path, render

from .mock_http import Recorder, mock


def test_json_path_and_render():
    doc = {"predictions": [{"embedding": [1, 2]}, {"embedding": [3, 4]}], "a": {"b-c": 5}}
    assert json_path(doc, "$.predictions[*].embedding") == [[1, 2], [3, 4]]
    assert json_path(doc, "$.predictions[1].embedding") == [[3, 4]]
    assert json_path(doc, "$.a['b-c']") == [5]
    assert json_path(doc, "$.missing") == []
    with pytest.raises(ValueError):
        json_path(doc, "predictions")
    assert render({"x": ["{{text}}!", 3]}, {"{{text}}": "hi"}) == {"x": ["hi!", 3]}


async def test_generic_http_medsiglip_style():
    def respond(req):
        inst = json.loads(req.content)["instances"][0]
        return httpx.Response(200, json={"predictions": [{"embedding": [0.1, 0.2] if "image_b64" in inst else [0.3, 0.4]}]})

    rec = Recorder(respond)
    e = GenericHttpEmbedder(
        "https://medsiglip.azureml.net/score", 2, id="azure:medsiglip-448", auth=Bearer("eyJ.azure-tok"),
        request={"image": {"instances": [{"image_b64": "{{b64}}", "mime": "{{mime}}"}]},
                 "text": {"instances": [{"text": "{{text}}"}]}},
        response_path="$.predictions[*].embedding", client=mock(rec))
    vecs = await e.embed([ImagePart(data=b"img", mime="image/jpeg"), TextPart(text="pneumonia")], "document")
    assert vecs == [[0.1, 0.2], [0.3, 0.4]]
    assert rec.body(0) == {"instances": [{"image_b64": base64.b64encode(b"img").decode(), "mime": "image/jpeg"}]}
    assert rec.body(1) == {"instances": [{"text": "pneumonia"}]}
    assert rec.requests[0].headers["authorization"] == "Bearer eyJ.azure-tok"
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/embedders/test_http_generic.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'agentic_search.embedders.http_generic'`

- [ ] **Step 3: Implement**

`src/agentic_search/embedders/http_generic.py`:
```python
"""Configurable HTTP embedder for custom model containers (e.g. MedSigLIP on Azure ML / AKS).

One request per item. `request` holds a JSON body template per modality; placeholders:
`{{text}}` (the text), `{{b64}}` (base64 image bytes), `{{mime}}` (image MIME type).
`response_path` is a small JSONPath (`$`, `.name`, `[n]`, `[*]`) whose FIRST match is the vector."""

from __future__ import annotations

import base64
import re
from typing import Any

from agentic_search.core.types import Content, ImagePart, Modality, image_bytes, text_of
from agentic_search.embedders.base import EmbedderError, Embedding, Purpose
from agentic_search.embedders.http import HttpEmbedder

_TOKEN = re.compile(r"\.([A-Za-z_][A-Za-z0-9_]*)|\[(\*|\d+)\]|\[['\"]([^'\"]+)['\"]\]")
_PLACEHOLDERS = ("{{text}}", "{{b64}}", "{{mime}}")


def json_path(value: Any, path: str) -> list[Any]:
    """Evaluate a minimal JSONPath. Returns all matches (wildcards fan out)."""
    if not path.startswith("$"):
        raise ValueError(f"JSONPath must start with '$': {path!r}")
    rest, pos, steps = path[1:], 0, []
    while pos < len(rest):
        m = _TOKEN.match(rest, pos)
        if m is None:
            raise ValueError(f"unsupported JSONPath syntax at {rest[pos:]!r}")
        steps.append(m.group(1) or m.group(3) or m.group(2))
        pos = m.end()
    current = [value]
    for step in steps:
        nxt: list[Any] = []
        for node in current:
            if step == "*":
                if isinstance(node, list):
                    nxt.extend(node)
                elif isinstance(node, dict):
                    nxt.extend(node.values())
            elif step.isdigit() and isinstance(node, list):
                if int(step) < len(node):
                    nxt.append(node[int(step)])
            elif isinstance(node, dict) and step in node:
                nxt.append(node[step])
        current = nxt
    return current


def render(template: Any, values: dict[str, str]) -> Any:
    """Substitute placeholders in every string of a JSON template."""
    if isinstance(template, str):
        for placeholder, value in values.items():
            template = template.replace(placeholder, value)
        return template
    if isinstance(template, dict):
        return {k: render(v, values) for k, v in template.items()}
    if isinstance(template, list):
        return [render(v, values) for v in template]
    return template


class GenericHttpEmbedder(HttpEmbedder):
    def __init__(self, url: str, dim: int, *, request: dict[str, Any], response_path: str,
                 id: str | None = None, auth: Any = None, headers: dict[str, str] | None = None,
                 **kwargs: Any):
        modalities = set()
        if "text" in request:
            modalities.add(Modality.TEXT)
        if "image" in request:
            modalities.add(Modality.IMAGE)
        if not modalities:
            raise ValueError("request needs a 'text' and/or 'image' template")
        json_path({}, response_path)  # validate syntax early
        kwargs.setdefault("batch_size", 1)
        super().__init__(id or f"http:{url}", dim, modalities, auth=auth, **kwargs)
        self.url = url
        self.request = request
        self.response_path = response_path
        self.headers = headers or {}

    async def _embed_batch(self, batch: list[Content], purpose: Purpose) -> list[Embedding]:
        return [await self._one(item) for item in batch]

    async def _one(self, item: Content) -> Embedding:
        if isinstance(item, ImagePart):
            values = {"{{b64}}": base64.b64encode(image_bytes(item)).decode(), "{{mime}}": item.mime,
                      "{{text}}": ""}
            template = self.request["image"]
        else:
            values = {"{{text}}": text_of([item]), "{{b64}}": "", "{{mime}}": ""}
            template = self.request["text"]
        body = await self.post_json(self.url, render(template, values), self.headers)
        matches = json_path(body, self.response_path)
        if not matches or not isinstance(matches[0], list):
            raise EmbedderError(f"{self.id}: {self.response_path} matched no vector in the response")
        return matches[0]
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/embedders/test_http_generic.py -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(embedders): configurable HTTP embedder for custom model containers

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: TypeSafe System One decider

**Files:**
- Create: `src/agentic_search/models/typesafe.py`
- Test: `tests/models/test_typesafe.py`

**Interfaces:**
- Produces:
  - `TypeSafeError`.
  - `TypeSafeDecider(model="jev-latest", *, api_key=None (→ TYPESAFE_API_KEY), url=API_URL, client=None, id=None, batch_size=16, snippet_chars=2000, timeout_s=60, max_retries=4, backoff_s=0.5, price_per_mtok=None)`. Its id defaults to `typesafe:{model}`.
  - It implements both `judge` and `decide`.
- HTTP contract (docs.typesafe.ai/api): `POST https://api.typesafe.ai/v1/systemone`, `Authorization: Bearer <key>`, body `{state, model, questions: {id: {type: noul|choice|score, instructions, criteria}}}`. The response is `{answers: {id: {noul|choice, confidence, probabilities}}, usage}`. Retries cover 429/529/5xx.

- [ ] **Step 1: Write the failing tests**

`tests/models/test_typesafe.py`:
```python
"""TypeSafe decider against httpx.MockTransport; `test_live_typesafe` hits the real API when
TYPESAFE_API_KEY is set."""

import json
import os

import httpx
import pytest

from agentic_search.core.secrets import scrub
from agentic_search.core.types import Hit, Query, TextPart
from agentic_search.models.base import Action, ControllerView, Decider, TurnSummary
from agentic_search.models.typesafe import TypeSafeDecider, TypeSafeError

HITS = [Hit(doc_id=str(i), source="s", content=[TextPart(text=t)])
        for i, t in enumerate(["aspirin relieves headache", "medieval castles", "ibuprofen for pain"])]


def client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_judge_batches_nouls_over_shared_state():
    seen = []

    def respond(req):
        body = json.loads(req.content)
        seen.append((req, body))
        answers = {qid: {"type": "noul", "noul": 0.9 if "headache" in body["state"]["documents"][int(qid[1:])]["text"] else 0.1}
                   for qid in body["questions"]}
        return httpx.Response(200, json={"model": "jev-latest", "answers": answers,
                                         "usage": {"input_tokens": 100, "output_tokens": 3}})

    d = TypeSafeDecider(api_key="ts-key-12345", client=client(respond), batch_size=2,
                        price_per_mtok=(1.0, 1.0))
    assert isinstance(d, Decider) and d.id == "typesafe:jev-latest"
    res = await d.judge(Query.of("what treats headache?"), HITS)
    assert [(j.key, j.p_relevant) for j in res.judgments] == [("s:0", 0.9), ("s:1", 0.1), ("s:2", 0.1)]
    assert len(seen) == 2 and res.usage.input_tokens == 200 and res.usage.cost_usd == pytest.approx(206e-6)
    req, body = seen[0]
    assert str(req.url) == "https://api.typesafe.ai/v1/systemone"
    assert req.headers["authorization"] == "Bearer ts-key-12345"
    assert body["model"] == "jev-latest" and body["state"]["query"] == "what treats headache?"
    assert body["questions"]["d1"]["type"] == "noul"
    assert "`documents[1].text`" in body["questions"]["d1"]["instructions"]
    assert set(body["questions"]["d0"]["criteria"]) == {"true", "false"}


async def test_decide_uses_choice():
    def respond(req):
        body = json.loads(req.content)
        assert body["questions"]["action"]["type"] == "choice"
        assert set(body["questions"]["action"]["criteria"]) == {a.value for a in Action}
        return httpx.Response(200, json={"answers": {"action": {"type": "choice", "choice": "broaden",
                                                                 "confidence": 0.8}}})

    view = ControllerView(question=Query.of("q"), turn=1, digest="d", total_relevant=0,
                          history=[TurnSummary(turn=0, n_calls=2, n_errors=0, n_new=3, n_new_relevant=0)],
                          budget_remaining={"turns": 2})
    d = await TypeSafeDecider(api_key="ts-key-12345", client=client(respond)).decide(view)
    assert d.action is Action.BROADEN and d.confidence == 0.8


async def test_errors_and_retries():
    calls = {"n": 0}

    def flaky(req):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(529, text="overloaded")
        return httpx.Response(200, json={"answers": {"d0": {"noul": 0.5}}})

    d = TypeSafeDecider(api_key="ts-key-12345", client=client(flaky), backoff_s=0.001)
    assert (await d.judge(Query.of("q"), HITS[:1])).judgments[0].p_relevant == 0.5
    assert calls["n"] == 2

    bad = TypeSafeDecider(api_key="ts-key-12345", client=client(lambda r: httpx.Response(
        401, text="invalid key ts-key-12345")), backoff_s=0.001)
    with pytest.raises(TypeSafeError) as exc:
        await bad.judge(Query.of("q"), HITS[:1])
    assert "401" in str(exc.value) and "ts-key-12345" not in str(exc.value)

    no_choice = TypeSafeDecider(api_key="ts-key-12345", client=client(lambda r: httpx.Response(
        200, json={"answers": {}})))
    view = ControllerView(question=Query.of("q"), turn=0, digest="", total_relevant=None, history=[],
                          budget_remaining={})
    with pytest.raises(TypeSafeError, match="no usable choice"):
        await no_choice.decide(view)


async def test_missing_key(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(TypeSafeError, match="API key"):
        await TypeSafeDecider(client=client(lambda r: httpx.Response(200))).judge(Query.of("q"), HITS[:1])


async def test_env_key_registered(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-env-key-777")
    TypeSafeDecider()
    assert scrub("ts-env-key-777") == "***"


@pytest.mark.live
async def test_live_typesafe():
    if not os.environ.get("TYPESAFE_API_KEY"):
        pytest.skip("set TYPESAFE_API_KEY")
    d = TypeSafeDecider()
    try:
        res = await d.judge(Query.of("what relieves headaches?"), HITS)
    finally:
        await d.close()
    p = {j.key: j.p_relevant for j in res.judgments}
    assert p["s:0"] > p["s:1"]
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/models/test_typesafe.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'agentic_search.models.typesafe'`

- [ ] **Step 3: Implement**

`src/agentic_search/models/typesafe.py`:
```python
"""TypeSafe System One models (Jev, …) as deciders.

judge(): one request per batch of hits. The shared state holds the query and the documents; each
document gets its own Noul ("does it help answer the query?"), so the batch is judged in parallel
and every hit gets a calibrated probability. decide(): one Choice over the controller actions.

HTTP contract: POST https://api.typesafe.ai/v1/systemone with `Authorization: Bearer <key>`,
body {state, model, questions: {id: {type, instructions, criteria}}}; see docs.typesafe.ai/api."""

from __future__ import annotations

import asyncio
import os
import random
from typing import Any

import httpx

from agentic_search.core.secrets import register_secret, scrub
from agentic_search.core.types import Hit, ModelUsage, Query
from agentic_search.models.base import (
    Action,
    ControllerView,
    Decision,
    JudgeResult,
    Judgment,
)
from agentic_search.models.llm import usage_with_cost

API_URL = "https://api.typesafe.ai/v1/systemone"
RETRY_STATUSES = {408, 429, 500, 502, 503, 504, 529}

RELEVANCE_CRITERIA = {
    "true": "The document contains information that helps answer the query: it states facts, "
            "findings or instructions the query is asking for.",
    "false": "The document is off-topic, or only shares vocabulary or a general subject with the "
             "query without supplying what the query asks for.",
}
ACTION_CRITERIA = {
    Action.STOP.value: "Enough relevant documents have been found, or more searching is unlikely "
                       "to find more.",
    Action.REFINE.value: "Searches return too much noise; make the queries more specific.",
    Action.BROADEN.value: "Searches return too little; widen or rephrase the queries.",
    Action.SWITCH_SOURCE.value: "The current sources look exhausted; try other sources.",
    Action.CONTINUE.value: "Searches are productive; keep exploring in the same direction.",
}


class TypeSafeError(Exception):
    pass


class TypeSafeDecider:
    def __init__(self, model: str = "jev-latest", *, api_key: str | None = None,
                 url: str = API_URL, client: httpx.AsyncClient | None = None, id: str | None = None,
                 batch_size: int = 16, snippet_chars: int = 2000, timeout_s: float = 60.0,
                 max_retries: int = 4, backoff_s: float = 0.5,
                 price_per_mtok: tuple[float, float] | None = None):
        self.model = model
        self.id = id or f"typesafe:{model}"
        self._api_key = api_key or os.environ.get("TYPESAFE_API_KEY")
        if self._api_key:
            register_secret(self._api_key)
        self.url = url
        self.batch_size = max(1, batch_size)
        self.snippet_chars = snippet_chars
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.backoff_s = backoff_s
        self.price = price_per_mtok
        self._client = client
        self._owns_client = client is None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout_s)
        return self._client

    async def _ask(self, state: Any, questions: dict[str, Any]) -> tuple[dict[str, Any], ModelUsage]:
        if not self._api_key:
            raise TypeSafeError("TypeSafe needs an API key (api_key= or TYPESAFE_API_KEY)")
        headers = {"Authorization": f"Bearer {self._api_key}"}
        payload = {"state": state, "model": self.model, "questions": questions}
        last = ""
        for attempt in range(self.max_retries + 1):
            try:
                resp = await self._get_client().post(self.url, json=payload, headers=headers)
            except httpx.HTTPError as exc:
                last = f"{type(exc).__name__}: {exc}"
            else:
                if resp.status_code < 300:
                    body = resp.json()
                    usage = body.get("usage") or {}
                    return body.get("answers") or {}, usage_with_cost(
                        int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0)), self.price)
                last = f"HTTP {resp.status_code}: {resp.text[:300]}"
                if resp.status_code not in RETRY_STATUSES:
                    break
            if attempt < self.max_retries:
                await asyncio.sleep(self.backoff_s * (2 ** attempt) * (1 + random.random()))
        raise TypeSafeError(scrub(f"{self.id}: {last}"))

    async def judge(self, question: Query, hits: list[Hit]) -> JudgeResult:
        judgments: list[Judgment] = []
        usage = ModelUsage()
        for start in range(0, len(hits), self.batch_size):
            batch = hits[start:start + self.batch_size]
            state = {"query": question.as_text(),
                     "documents": [{"text": h.snippet(self.snippet_chars) or "(no content)"}
                                   for h in batch]}
            questions = {
                f"d{i}": {
                    "type": "noul",
                    "instructions": f"Does the document `documents[{i}].text` help answer the query "
                                    f"`query`?",
                    "criteria": RELEVANCE_CRITERIA,
                } for i in range(len(batch))
            }
            answers, u = await self._ask(state, questions)
            usage = usage.plus(u)
            for i, h in enumerate(batch):
                p = (answers.get(f"d{i}") or {}).get("noul")
                if isinstance(p, (int, float)):
                    judgments.append(Judgment(key=h.key, p_relevant=min(1.0, max(0.0, float(p)))))
        return JudgeResult(judgments=judgments, usage=usage)

    async def decide(self, view: ControllerView) -> Decision:
        state = {
            "question": view.question.as_text(),
            "turns": [t.model_dump() for t in view.history],
            "total_relevant": view.total_relevant,
            "budget_remaining": view.budget_remaining,
            "latest_results": view.digest,
        }
        questions = {"action": {
            "type": "choice",
            "instructions": "An agentic search loop is looking for every document relevant to "
                            "`question`. Given the per-turn statistics in `turns`, the relevant "
                            "documents found so far (`total_relevant`), `budget_remaining` and "
                            "`latest_results`, what should the search do next?",
            "criteria": ACTION_CRITERIA,
        }}
        answers, usage = await self._ask(state, questions)
        answer = answers.get("action") or {}
        try:
            action = Action(answer["choice"])
        except (KeyError, ValueError) as exc:
            raise TypeSafeError(f"{self.id}: no usable choice in response") from exc
        confidence = answer.get("confidence")
        return Decision(action=action, usage=usage, note="typesafe",
                        confidence=min(1.0, max(0.0, float(confidence))) if confidence is not None else 1.0)

    async def close(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/models/test_typesafe.py -q`
Expected: 5 passed, 1 deselected (live)

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(models): TypeSafe System One decider (noul relevance judging, choice decisions)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Postgres stored tsvector and ConfigError for invalid values (Plan 2 follow-ups)

**Files:**
- Modify: `src/agentic_search/backends/postgres.py`, `src/agentic_search/config.py`
- Test: `tests/backends/test_postgres_unit.py`, `tests/contract/test_sql_discovery.py`, `tests/test_config.py`

**Interfaces:**
- Produces:
  - `PostgresBackend(..., tsvector_columns: dict[str, str] | None = None)`. When no explicit `fields` are requested, lexical search uses a stored `tsvector` column: the configured one, or the only one on the table. That lets a GIN index serve the query. The column is not exposed as a field.
  - `config.build()` turns a `ValueError` from a factory into `ConfigError("<kind> <type> config is invalid: …")`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/backends/test_postgres_unit.py`:
```python
def test_lexical_uses_stored_tsvector_column():
    from agentic_search.backends.sql_backend import TableInfo
    from agentic_search.core.types import FieldSpec, FieldType, Lexical

    b = PostgresBackend("pg", "postgresql://u:p@h/db")
    table = TableInfo(name="docs", id_column="id", fields=[
        FieldSpec(name="id", type=FieldType.KEYWORD), FieldSpec(name="body", type=FieldType.TEXT, searchable=True)])
    b._tsv = {"docs": "search"}
    sql, _ = b._lexical_sql(Lexical(source="pg", text="headache"), table, 5)
    assert '"search" @@ websearch_to_tsquery' in sql and "to_tsvector" not in sql
    sql2, _ = b._lexical_sql(Lexical(source="pg", text="headache", fields=["body"]), table, 5)
    assert "to_tsvector('english', concat_ws(' ', \"body\"))" in sql2
```

Append to `tests/contract/test_sql_discovery.py`:
```python
async def test_postgres_uses_stored_tsvector_column():
    from agentic_search.backends.postgres import PostgresBackend
    from agentic_search.core.types import Lexical

    await _pg_exec(
        "DROP TABLE IF EXISTS tsv_docs",
        "CREATE TABLE tsv_docs (id TEXT PRIMARY KEY, title TEXT, body TEXT, search tsvector "
        "GENERATED ALWAYS AS (to_tsvector('english', coalesce(title, '') || ' ' || coalesce(body, ''))) STORED)",
        "CREATE INDEX tsv_docs_search ON tsv_docs USING GIN (search)",
        *[f"INSERT INTO tsv_docs (id, title, body) VALUES ('{i}', '{t}', '{b}')" for i, t, b, _, _ in corpus.ROWS])
    b = PostgresBackend("pg", corpus.PG_DSN, tables=["tsv_docs"])
    try:
        m = await b.discover()
        assert b._tsv == {"tsv_docs": "search"}
        assert m.resolve_collection("tsv_docs").field("search") is None  # not exposed as a field
        hits = await b.execute(Lexical(source="pg", collection="tsv_docs", text="headache"))
        assert {h.doc_id for h in hits} == {"d1", "d4"} and all(h.raw_score for h in hits)
    finally:
        await b.close()
        await _pg_exec("DROP TABLE IF EXISTS tsv_docs")
```

Append to `tests/test_config.py`:
```python
def test_constructor_value_errors_become_config_errors(tmp_path):
    with pytest.raises(ConfigError, match="invalid"):
        build_harness({"backends": [{"name": "bq", "type": "bigquery", "project": "bad project!", "dataset": "d"}],
                       "driver": {"type": "openai_compat", "model": "m", "base_url": "http://x/v1"}},
                      base_dir=tmp_path)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/backends/test_postgres_unit.py tests/test_config.py -q`
Expected:
- `test_lexical_uses_stored_tsvector_column` fails because `_tsv` is ignored and `to_tsvector` still appears in the SQL.
- `test_constructor_value_errors_become_config_errors` fails with a raw `ValueError`.

- [ ] **Step 3: Implement**

Apply this change to `src/agentic_search/backends/postgres.py`:

```diff
diff --git a/src/agentic_search/backends/postgres.py b/src/agentic_search/backends/postgres.py
index bda218b..555110c 100644
--- a/src/agentic_search/backends/postgres.py
+++ b/src/agentic_search/backends/postgres.py
@@ -50,8 +50,11 @@ class PostgresBackend(SqlBackend):
 
     def __init__(self, name: str, dsn: str, *, schema: str = "public",
                  text_search_config: str = "english", pool_size: int = 4,
-                 statement_timeout_ms: int = 30_000, connect_timeout_s: float = 15, **kwargs: Any):
+                 statement_timeout_ms: int = 30_000, connect_timeout_s: float = 15,
+                 tsvector_columns: dict[str, str] | None = None, **kwargs: Any):
         super().__init__(name, **kwargs)
+        self.tsvector_columns = dict(tsvector_columns or {})
+        self._tsv: dict[str, str] = {}
         if not _CONFIG.match(text_search_config):
             raise ValueError(f"invalid text_search_config {text_search_config!r}")
         register_secret(dsn)
@@ -150,13 +153,18 @@ class PostgresBackend(SqlBackend):
                 by_table.setdefault(r["table_name"], []).append(r)
         tables: dict[str, TableInfo] = {}
         skipped: list[str] = []
+        tsv_of: dict[str, str] = {}
         for tname, rows in by_table.items():
             id_col = self.id_columns.get(tname) or self._single_pk(tname, pk_of.get(tname), skipped)
             if id_col is None:
                 continue
             fields = []
+            tsv_cols: list[str] = []
             for r in rows:
                 name, udt = r["column_name"], r["udt_name"]
+                if udt == "tsvector":
+                    tsv_cols.append(name)
+                    continue
                 if udt in _SKIP_UDTS:
                     continue
                 if udt == "vector":
@@ -167,10 +175,16 @@ class PostgresBackend(SqlBackend):
                 if r["data_type"] in _TYPES and ftype in (FieldType.KEYWORD, FieldType.BOOL) and name != id_col:
                     samples = await self._samples(tname, name)
                 fields.append(FieldSpec(name=name, type=ftype, sample_values=samples, **field_flags(ftype)))
+            configured = self.tsvector_columns.get(tname)
+            if configured in tsv_cols:
+                tsv_of[tname] = configured
+            elif configured is None and len(tsv_cols) == 1:
+                tsv_of[tname] = tsv_cols[0]
             est = est_of.get(tname)
             tables[tname] = TableInfo(name=tname, id_column=id_col, fields=fields,
                                       count=await self._count(tname, est if est and est > 0 else None))
         self.skipped = skipped
+        self._tsv = tsv_of
         return tables
 
     # ---- lexical / vector ------------------------------------------------------
@@ -183,8 +197,12 @@ class PostgresBackend(SqlBackend):
         if not terms:
             raise BackendError("lexical query has no searchable terms")
         query_text = op.text if '"' in op.text else " or ".join(terms)
-        doc = (f"to_tsvector('{self.ts_config}', concat_ws(' ', "
-               f"{', '.join(quote_ident(c, 'postgres') for c in fields)}))")
+        stored = None if op.fields else self._tsv.get(table.name)
+        if stored is not None:  # a stored tsvector column can use its GIN index
+            doc = quote_ident(stored, "postgres")
+        else:
+            doc = (f"to_tsvector('{self.ts_config}', concat_ws(' ', "
+                   f"{', '.join(quote_ident(c, 'postgres') for c in fields)}))")
         p = Params("postgres")
         score = f"ts_rank({doc}, websearch_to_tsquery('{self.ts_config}', {p.add(query_text)}))"
         match = f"{doc} @@ websearch_to_tsquery('{self.ts_config}', {p.add(query_text)})"
```

In `src/agentic_search/config.py` `build()`, extend the `try` block's handlers:
```python
    except ValueError as exc:
        if isinstance(exc, ConfigError):
            raise
        raise ConfigError(f"{kind} {type_!r} config is invalid: {exc}") from exc
```
and add `"tsvector_columns"` to the keys `_postgres` passes through.

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest -q` and `AGENTIC_SEARCH_INTEGRATION=1 uv run pytest tests/contract/test_sql_discovery.py -m integration -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(backends): use stored tsvector columns for Postgres full-text; ConfigError for invalid values

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Neo4j backend

**Files:**
- Modify: `pyproject.toml`, `docker-compose.yml`, `tests/contract/test_backend_contract.py`
- Create: `src/agentic_search/backends/neo4j.py`, `tests/backends/test_neo4j_unit.py`, `tests/contract/seed_neo4j.py`, `tests/contract/test_neo4j_integration.py`

**Interfaces:**
- Produces: `Neo4jBackend(name, uri, *, user=None, password=None, database=None, labels=None, id_property="id", embedders=None ("Label.property" → embedder id), native_query=False, description=None, max_rows=100, sample_values=True, connect_timeout_s=10)`, with `backend_type="neo4j"`.
- Collections and fields:
  - Collections are node labels.
  - Fields come from `db.schema.nodeTypeProperties()`. A String is TEXT when a FULLTEXT index covers it, KEYWORD otherwise. A property covered by an ONLINE VECTOR index becomes VECTOR.
  - Relationship types go into `Manifest.relationship_types`.
- Capabilities:
  - LEXICAL via FULLTEXT index; VECTOR and HYBRID via VECTOR index.
  - FILTER, REGEX (`=~`, wrapped for search semantics), AGGREGATE, FETCH.
  - TRAVERSE: start label = `op.collection`, then `start` filter, validated `rel_types`, direction, depth and `target_label`; distinct end nodes, `raw_score = 1/hops`.
  - NATIVE (Cypher, through `guard_cypher`).
- Safety: every query runs in `execute_read` within READ-mode sessions. Identifiers come only from discovery and are backtick-quoted with backticks doubled; values are `$params`.
- Ids are namespaced `<label>/<id>` when the backend has more than one label. The contract suite's `bare()` helper strips that prefix for comparisons.

- [ ] **Step 1: Dependency and service**

Add `neo4j = ["neo4j>=5.20"]` to `[project.optional-dependencies]` and `"neo4j>=5.20",` to the `dev` group, then run `uv sync`. Add this service to `docker-compose.yml`:
```yaml
  neo4j:
    image: neo4j:5-community
    environment:
      NEO4J_AUTH: neo4j/agenticpass
      NEO4J_server_memory_heap_max__size: 512m
    ports: ["57687:7687", "57474:7474"]
    healthcheck:
      test: ["CMD-SHELL", "wget -qO- http://localhost:7474 >/dev/null || exit 1"]
      interval: 3s
      retries: 60
```
Run: `docker compose up -d --wait neo4j`

- [ ] **Step 2: Write the failing unit tests**

`tests/backends/test_neo4j_unit.py`:
```python
import sys

import pytest

from agentic_search.backends.base import BackendError
from agentic_search.core.secrets import scrub
from agentic_search.core.types import (
    And,
    Contains,
    Eq,
    Exists,
    In,
    Not,
    Or,
    Range,
    Traverse,
)

PROPS = {"type", "year", "title"}


def test_quote_name_doubles_backticks():
    from agentic_search.backends.neo4j import quote_name
    assert quote_name("docs") == "`docs`"
    assert quote_name("a`b") == "`a``b`"


def test_filter_cypher_translation():
    from agentic_search.backends.neo4j import filter_cypher
    params: dict = {}
    f = And(clauses=[Eq(field="type", value="drug"),
                     Or(clauses=[Range(field="year", gte=2020, lt=2022), Not(clause=Exists(field="title"))]),
                     In(field="type", values=["a", "b"]), Contains(field="title", value="Asp")])
    cypher = filter_cypher(f, params, PROPS, "n")
    assert cypher == ("(n.`type` = $p0 AND ((n.`year` >= $p1 AND n.`year` < $p2) OR (NOT n.`title` IS NOT NULL)) "
                      "AND n.`type` IN $p3 AND toLower(toString(n.`title`)) CONTAINS toLower($p4))")
    assert params == {"p0": "drug", "p1": 2020, "p2": 2022, "p3": ["a", "b"], "p4": "Asp"}
    assert filter_cypher(And(clauses=[]), {}, PROPS, "n") == "true"
    assert filter_cypher(In(field="type", values=[]), {}, PROPS, "n") == "false"


def test_filter_cypher_rejects_unknown_property():
    from agentic_search.backends.neo4j import filter_cypher
    with pytest.raises(BackendError, match="unknown property"):
        filter_cypher(Eq(field="x`) DETACH DELETE n //", value=1), {}, PROPS, "n")


def test_lucene_query_is_or_of_lowercase_terms():
    from agentic_search.backends.neo4j import lucene_query
    assert lucene_query("Headache AND pain!") == "headache OR and OR pain"
    with pytest.raises(BackendError):
        lucene_query("!!!")


async def test_traverse_validates_identifiers(monkeypatch):
    from agentic_search.backends import neo4j as mod

    b = mod.Neo4jBackend("neo", "bolt://localhost:1")
    schema = mod._Schema(labels={"docs": mod._Label(name="docs", properties={"id": None}),
                                 "conditions": mod._Label(name="conditions", properties={"id": None})},
                         relationship_types=["TREATS"])

    async def fake_schema():
        return schema

    monkeypatch.setattr(b, "_ensure_schema", fake_schema)
    for bad in (dict(rel_types=["TREATS`]->(x) DELETE x //"]), dict(target_label="nope"),
                dict(collection="nope")):
        kwargs = {"rel_types": ["TREATS"], **bad}
        with pytest.raises(BackendError):
            await b.execute(Traverse(source="neo", start=Eq(field="id", value="h"), **kwargs))


async def test_missing_extra_is_backend_error(monkeypatch):
    from agentic_search.backends.neo4j import Neo4jBackend
    monkeypatch.setitem(sys.modules, "neo4j", None)
    with pytest.raises(BackendError, match="neo4j.*extra"):
        await Neo4jBackend("neo", "bolt://localhost:57687").discover()


async def test_unreachable_is_backend_error_and_password_registered():
    from agentic_search.backends.neo4j import Neo4jBackend
    b = Neo4jBackend("neo", "bolt://127.0.0.1:1", user="neo4j", password="secretpw3", connect_timeout_s=2)
    try:
        with pytest.raises(BackendError) as excinfo:
            await b.discover()
    finally:
        await b.close()
    assert scrub("leak secretpw3 here") == "leak *** here"
    assert "secretpw3" not in scrub(str(excinfo.value))
```

Run: `uv run pytest tests/backends/test_neo4j_unit.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'agentic_search.backends.neo4j'`

- [ ] **Step 3: Implement**

`src/agentic_search/backends/neo4j.py`:
```python
"""Neo4j backend: full-text and vector index search, property filters, regex, aggregates, fetch,
graph traversal and guarded native Cypher. Collections are node labels. All work runs in READ
transactions, so the server refuses writes even if the native guard were bypassed. Needs the
`neo4j` extra."""

from __future__ import annotations

import asyncio
import importlib
from dataclasses import dataclass, field
from typing import Any

from agentic_search.backends.base import (
    BackendError,
    DiscoverDetail,
    UnsupportedOperation,
    routed_collection,
    rrf_merge,
    strip_collection,
)
from agentic_search.backends.native_guard import guard_cypher
from agentic_search.backends.sql import dumps, jsonable
from agentic_search.backends.sql_backend import any_terms
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
    Traverse,
    Vector,
)

SAMPLE_DISTINCT_MAX = 20
SAMPLE_SCAN_MAX = 10_000
_AGG_FUNCS = {"sum": "sum", "avg": "avg", "min": "min", "max": "max"}
_DATE_TYPES = {"Date", "DateTime", "LocalDateTime", "ZonedDateTime", "LocalTime", "Time"}
_ARRAY_TYPES = {"DoubleArray", "FloatArray", "LongArray", "IntegerArray"}


def _require_neo4j() -> Any:
    try:
        return importlib.import_module("neo4j")
    except ImportError as exc:
        raise BackendError(
            "Neo4jBackend needs the `neo4j` extra: pip install 'agentic-search[neo4j]'") from exc


def quote_name(name: str) -> str:
    """Backtick-quote a label, property, relationship type or index name."""
    return "`" + name.replace("`", "``") + "`"


def _param(params: dict[str, Any], value: Any) -> str:
    name = f"p{len(params)}"
    params[name] = value
    return f"${name}"


def filter_cypher(f: Filter, params: dict[str, Any], properties: set[str], var: str) -> str:
    """Filter AST → parameterized Cypher boolean expression over `var`'s known properties."""
    if isinstance(f, And):
        return "(" + " AND ".join(filter_cypher(c, params, properties, var) for c in f.clauses) + ")" \
            if f.clauses else "true"
    if isinstance(f, Or):
        return "(" + " OR ".join(filter_cypher(c, params, properties, var) for c in f.clauses) + ")" \
            if f.clauses else "false"
    if isinstance(f, Not):
        return f"(NOT {filter_cypher(f.clause, params, properties, var)})"
    if f.field not in properties:
        raise BackendError(f"unknown property {f.field!r}")
    prop = f"{var}.{quote_name(f.field)}"
    if isinstance(f, Eq):
        return f"{prop} = {_param(params, f.value)}"
    if isinstance(f, In):
        return f"{prop} IN {_param(params, list(f.values))}" if f.values else "false"
    if isinstance(f, Range):
        parts = [f"{prop} {sym} {_param(params, bound)}" for bound, sym in
                 ((f.gte, ">="), (f.gt, ">"), (f.lte, "<="), (f.lt, "<")) if bound is not None]
        return "(" + " AND ".join(parts) + ")" if parts else "true"
    if isinstance(f, Exists):
        return f"{prop} IS NOT NULL"
    if isinstance(f, Contains):
        return f"toLower(toString({prop})) CONTAINS toLower({_param(params, f.value)})"
    raise BackendError(f"unsupported filter node {type(f).__name__}")


def lucene_query(text: str) -> str:
    """OR of lowercase word terms. Lowercasing neutralizes Lucene's AND/OR/NOT operators, and
    word tokens contain no Lucene special characters."""
    terms = any_terms(text)
    if not terms:
        raise BackendError("lexical query has no searchable terms")
    return " OR ".join(terms)


def _plain(value: Any) -> Any:
    """Neo4j temporal/spatial values → JSON-friendly."""
    if hasattr(value, "iso_format"):
        return value.iso_format()
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    return jsonable(value)


@dataclass
class _Label:
    name: str
    properties: dict[str, FieldSpec | None]
    fulltext: list[tuple[str, list[str]]] = field(default_factory=list)  # (index, properties)
    vectors: dict[str, str] = field(default_factory=dict)  # property -> index name
    count: int | None = None

    def fields(self) -> list[FieldSpec]:
        return [f for f in self.properties.values() if f is not None]

    def text_properties(self) -> list[str]:
        return [f.name for f in self.fields() if f.type is FieldType.TEXT]

    def vector_properties(self) -> set[str]:
        return {f.name for f in self.fields() if f.type is FieldType.VECTOR}


@dataclass
class _Schema:
    labels: dict[str, _Label]
    relationship_types: list[str]
    all_labels: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.all_labels = self.all_labels or set(self.labels)


def _flags(ftype: FieldType) -> dict[str, bool]:
    return {"searchable": ftype is FieldType.TEXT,
            "filterable": ftype not in (FieldType.JSON, FieldType.VECTOR),
            "sortable": ftype in (FieldType.INT, FieldType.FLOAT, FieldType.DATE)}


class Neo4jBackend:
    backend_type = "neo4j"

    def __init__(self, name: str, uri: str, *, user: str | None = None, password: str | None = None,
                 database: str | None = None, labels: list[str] | None = None,
                 id_property: str = "id", embedders: dict[str, str] | None = None,
                 native_query: bool = False, description: str | None = None, max_rows: int = 100,
                 sample_values: bool = True, connect_timeout_s: float = 10):
        register_secret(uri)
        register_secret(password)
        self.name = name
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.label_names = labels
        self.id_property = id_property
        self.embedders = embedders or {}
        self.native_query = native_query
        self.description = description
        self.max_rows = max_rows
        self.sample_values = sample_values
        self.connect_timeout_s = float(connect_timeout_s)
        self._driver: Any = None
        self._schema: _Schema | None = None
        self._lock = asyncio.Lock()

    # ---- connection -------------------------------------------------------------

    def _get_driver(self) -> Any:
        if self._driver is None:
            neo4j = _require_neo4j()
            try:
                self._driver = neo4j.AsyncGraphDatabase.driver(
                    self.uri, auth=(self.user, self.password) if self.user else None,
                    connection_timeout=self.connect_timeout_s,
                    max_transaction_retry_time=self.connect_timeout_s,
                    notifications_min_severity="OFF")
            except Exception as exc:
                raise BackendError(f"{type(exc).__name__}: {exc}") from exc
        return self._driver

    async def _read(self, query: str, params: dict[str, Any] | None = None) -> list[Any]:
        """Run one query in a READ transaction; returns records."""
        async def work(tx: Any) -> list[Any]:
            result = await tx.run(query, params or {})
            return [record async for record in result]

        try:
            driver = self._get_driver()
            neo4j = _require_neo4j()
            async with driver.session(database=self.database,
                                      default_access_mode=neo4j.READ_ACCESS) as session:
                return await session.execute_read(work)
        except BackendError:
            raise
        except Exception as exc:
            raise BackendError(f"{type(exc).__name__}: {exc}") from exc

    async def close(self) -> None:
        if self._driver is not None:
            driver, self._driver = self._driver, None
            try:
                await driver.close()
            except Exception:
                pass

    # ---- discovery ------------------------------------------------------------

    def capabilities(self) -> set[Capability]:
        caps = {Capability.FILTER, Capability.REGEX, Capability.AGGREGATE, Capability.FETCH,
                Capability.TRAVERSE}
        labels = self._schema.labels.values() if self._schema else []
        if self._schema is None or any(lbl.fulltext for lbl in labels):
            caps.add(Capability.LEXICAL)
        if any(lbl.vectors for lbl in labels):
            caps |= {Capability.VECTOR, Capability.HYBRID}
        if self.native_query:
            caps.add(Capability.NATIVE)
        return caps

    async def _ensure_schema(self) -> _Schema:
        async with self._lock:
            if self._schema is None:
                self._schema = await self._discover_schema()
            return self._schema

    async def _discover_schema(self) -> _Schema:
        all_labels = {r["label"] for r in await self._read("CALL db.labels() YIELD label RETURN label")}
        wanted = [lbl for lbl in (self.label_names or sorted(all_labels)) if lbl in all_labels]
        rel_types = sorted(r["t"] for r in await self._read(
            "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType AS t"))
        indexes = await self._read(
            "SHOW INDEXES YIELD name, type, labelsOrTypes, properties, options, state "
            "RETURN name, type, labelsOrTypes, properties, options, state")
        props = await self._read(
            "CALL db.schema.nodeTypeProperties() YIELD nodeLabels, propertyName, propertyTypes "
            "RETURN nodeLabels, propertyName, propertyTypes")
        labels = {name: _Label(name=name, properties={}) for name in wanted}
        vector_meta: dict[tuple[str, str], tuple[int | None, str | None]] = {}
        for ix in indexes:
            if ix["state"] != "ONLINE" or not ix["labelsOrTypes"]:
                continue
            for lbl in ix["labelsOrTypes"]:
                if lbl not in labels:
                    continue
                if ix["type"] == "FULLTEXT":
                    labels[lbl].fulltext.append((ix["name"], list(ix["properties"])))
                elif ix["type"] == "VECTOR" and len(ix["properties"]) == 1:
                    prop = ix["properties"][0]
                    labels[lbl].vectors[prop] = ix["name"]
                    cfg = (ix["options"] or {}).get("indexConfig", {})
                    sim = cfg.get("vector.similarity_function")
                    vector_meta[(lbl, prop)] = (cfg.get("vector.dimensions"),
                                                sim.lower() if isinstance(sim, str) else None)
        for row in props:
            name = row["propertyName"]
            if name is None:
                continue
            types = row["propertyTypes"] or []
            for lbl in row["nodeLabels"] or []:
                if lbl in labels and name not in labels[lbl].properties:
                    labels[lbl].properties[name] = self._field(labels[lbl], name, types, vector_meta)
        for lbl in labels.values():
            lbl.count = await self._count(lbl.name)
            for spec in lbl.fields():
                if spec.type in (FieldType.KEYWORD, FieldType.BOOL) and spec.name != self.id_property:
                    spec.sample_values = await self._samples(lbl.name, spec.name)
        return _Schema(labels=labels, relationship_types=rel_types, all_labels=all_labels)

    def _field(self, lbl: _Label, name: str, types: list[str],
               vector_meta: dict[tuple[str, str], tuple[int | None, str | None]]) -> FieldSpec:
        ptype = types[0] if types else "String"
        in_fulltext = any(name in idx_props for _, idx_props in lbl.fulltext)
        if name in lbl.vectors:
            dim, sim = vector_meta.get((lbl.name, name), (None, None))
            return FieldSpec(name=name, type=FieldType.VECTOR, vector_dim=dim, vector_metric=sim,
                             embedder_id=self.embedders.get(f"{lbl.name}.{name}"))
        if ptype == "String":
            ftype = FieldType.TEXT if in_fulltext else FieldType.KEYWORD
        elif ptype in ("Long", "Integer"):
            ftype = FieldType.INT
        elif ptype in ("Double", "Float"):
            ftype = FieldType.FLOAT
        elif ptype == "Boolean":
            ftype = FieldType.BOOL
        elif ptype in _DATE_TYPES:
            ftype = FieldType.DATE
        else:
            ftype = FieldType.JSON
        return FieldSpec(name=name, type=ftype, **_flags(ftype))

    async def _count(self, label: str) -> int | None:
        try:
            rows = await self._read(f"MATCH (n:{quote_name(label)}) RETURN count(n) AS c")
        except BackendError:
            return None
        return int(rows[0]["c"]) if rows else None

    async def _samples(self, label: str, prop: str) -> list[Any] | None:
        if not self.sample_values:
            return None
        p = f"n.{quote_name(prop)}"
        try:
            rows = await self._read(
                f"MATCH (n:{quote_name(label)}) WHERE {p} IS NOT NULL WITH {p} AS v LIMIT $scan "
                f"RETURN DISTINCT v LIMIT $k", {"scan": SAMPLE_SCAN_MAX, "k": SAMPLE_DISTINCT_MAX + 1})
        except BackendError:
            return None
        values = [_plain(r["v"]) for r in rows]
        return values if len(values) <= SAMPLE_DISTINCT_MAX else None

    async def discover(self, detail: DiscoverDetail = "full",
                       collection: str | None = None) -> Manifest:
        schema = await self._ensure_schema()
        return Manifest(
            source=self.name, backend_type=self.backend_type, capabilities=self.capabilities(),
            description=self.description, relationship_types=schema.relationship_types,
            collections=[CollectionInfo(name=lbl.name, fields=lbl.fields(), count=lbl.count)
                         for lbl in schema.labels.values()])

    # ---- helpers ----------------------------------------------------------------

    def _resolve(self, name: str | None, schema: _Schema) -> _Label:
        if name is None:
            if len(schema.labels) == 1:
                return next(iter(schema.labels.values()))
            raise BackendError(f"collection required; one of {sorted(schema.labels)}")
        if name not in schema.labels:
            raise BackendError(f"unknown collection {name!r}; one of {sorted(schema.labels)}")
        return schema.labels[name]

    def _where(self, f: Filter | None, params: dict[str, Any], lbl: _Label, var: str = "n",
               extra: list[str] | None = None) -> str:
        parts = list(extra or [])
        if f is not None:
            parts.append(filter_cypher(f, params, set(lbl.properties), var))
        return " WHERE " + " AND ".join(parts) if parts else ""

    def _hit(self, node: Any, element_id: str, lbl: _Label, multi: bool,
             score: float | None = None) -> Hit:
        props = {k: _plain(v) for k, v in dict(node).items()}
        vectors = lbl.vector_properties()
        texts = lbl.text_properties()
        pk = props.get(self.id_property)
        doc_id = str(pk) if pk is not None else element_id
        if multi:
            doc_id = f"{lbl.name}/{doc_id}"
        text = [str(props[t]) for t in texts if props.get(t) not in (None, "")]
        metadata = {k: v for k, v in props.items() if k not in vectors and k not in texts}
        content: list[Content] = [TextPart(text="\n".join(text))] if text else [StructuredPart(data=metadata)]
        return Hit(doc_id=doc_id, source=self.name, content=content, metadata=metadata,
                   raw_score=float(score) if score is not None else None)

    def _hits(self, rows: list[Any], lbl: _Label, schema: _Schema, score_key: str = "score") -> list[Hit]:
        multi = len(schema.labels) > 1
        return [self._hit(r["n"], r["eid"], lbl, multi,
                          r[score_key] if score_key in r.keys() else None) for r in rows]

    def _order(self, lbl: _Label) -> str:
        if self.id_property in lbl.properties:
            return f"n.{quote_name(self.id_property)}"
        return "elementId(n)"

    # ---- execution ---------------------------------------------------------------

    async def execute(self, op: QueryOp) -> list[Hit]:
        schema = await self._ensure_schema()
        if isinstance(op, Native):
            return await self._native(op, schema)
        if isinstance(op, Traverse):
            return await self._traverse(op, schema)
        lbl = self._resolve(routed_collection(op, schema.labels) if isinstance(op, Fetch)
                            else op.collection, schema)
        limit = min(op.limit, self.max_rows)
        if isinstance(op, Lexical):
            return self._hits(await self._read(*self._lexical(op.text, op.fields, op.filter, lbl, limit)),
                              lbl, schema)
        if isinstance(op, Vector):
            return self._hits(await self._read(*self._vector(op.field, op.vector, op.filter, lbl, limit)),
                              lbl, schema)
        if isinstance(op, Hybrid):
            return await self._hybrid(op, lbl, schema, limit)
        if isinstance(op, FilterOnly):
            params: dict[str, Any] = {"lim": limit}
            q = (f"MATCH (n:{quote_name(lbl.name)}){self._where(op.filter, params, lbl)} "
                 f"RETURN n, elementId(n) AS eid ORDER BY {self._order(lbl)} LIMIT $lim")
            return self._hits(await self._read(q, params), lbl, schema)
        if isinstance(op, Regex):
            return await self._regex(op, lbl, schema, limit)
        if isinstance(op, Aggregate):
            return await self._aggregate(op, lbl, limit)
        if isinstance(op, Fetch):
            return await self._fetch(op, lbl, schema)
        raise UnsupportedOperation(f"neo4j backend does not support {op.type}")

    def _lexical(self, text: str, fields: list[str] | None, f: Filter | None, lbl: _Label,
                 limit: int) -> tuple[str, dict[str, Any]]:
        if not lbl.fulltext:
            raise BackendError(f"label {lbl.name} has no FULLTEXT index")
        if fields:
            index = next((name for name, props in lbl.fulltext if set(props) == set(fields)), None)
            if index is None:
                raise BackendError(f"no FULLTEXT index on exactly {sorted(fields)}; "
                                   f"indexes: {[props for _, props in lbl.fulltext]}")
        else:
            index = lbl.fulltext[0][0]
        params: dict[str, Any] = {"idx": index, "q": lucene_query(text), "lim": limit}
        where = self._where(f, params, lbl, extra=[f"n:{quote_name(lbl.name)}"])
        q = (f"CALL db.index.fulltext.queryNodes($idx, $q) YIELD node AS n, score{where} "
             f"RETURN n, elementId(n) AS eid, score ORDER BY score DESC LIMIT $lim")
        return q, params

    def _vector(self, prop: str, vector: list[float] | None, f: Filter | None, lbl: _Label,
                limit: int) -> tuple[str, dict[str, Any]]:
        if vector is None:
            raise BackendError("vector op reached the backend without an embedded query vector")
        if prop not in lbl.vectors:
            raise BackendError(f"{prop!r} is not a vector-indexed property of {lbl.name}")
        k = limit if f is None else min(max(limit * 10, 100), 1000)
        params: dict[str, Any] = {"idx": lbl.vectors[prop], "k": k, "vec": list(vector), "lim": limit}
        where = self._where(f, params, lbl)
        q = (f"CALL db.index.vector.queryNodes($idx, $k, $vec) YIELD node AS n, score{where} "
             f"RETURN n, elementId(n) AS eid, score ORDER BY score DESC LIMIT $lim")
        return q, params

    async def _hybrid(self, op: Hybrid, lbl: _Label, schema: _Schema, limit: int) -> list[Hit]:
        depth = min(max(limit * 5, 50), 500)
        lex = self._hits(await self._read(*self._lexical(op.text, None, op.filter, lbl, depth)), lbl, schema)
        vec = self._hits(await self._read(*self._vector(op.field, op.vector, op.filter, lbl, depth)),
                         lbl, schema)
        by_id = {h.doc_id: h for h in [*vec, *lex]}
        fused = rrf_merge([([h.doc_id for h in lex], op.lexical_weight),
                           ([h.doc_id for h in vec], 1.0 - op.lexical_weight)])
        return [by_id[i].model_copy(update={"raw_score": s}) for i, s in fused[:limit]]

    async def _regex(self, op: Regex, lbl: _Label, schema: _Schema, limit: int) -> list[Hit]:
        strings = [f.name for f in lbl.fields() if f.type in (FieldType.TEXT, FieldType.KEYWORD)]
        props = op.fields or lbl.text_properties() or strings
        unknown = set(props) - set(lbl.properties)
        if unknown:
            raise BackendError(f"unknown properties {sorted(unknown)}")
        params: dict[str, Any] = {"pat": f"(?s).*(?:{op.pattern}).*", "lim": limit}
        ors = " OR ".join(f"toString(n.{quote_name(p)}) =~ $pat" for p in props)
        q = (f"MATCH (n:{quote_name(lbl.name)}){self._where(op.filter, params, lbl, extra=[f'({ors})'])} "
             f"RETURN n, elementId(n) AS eid ORDER BY {self._order(lbl)} LIMIT $lim")
        return self._hits(await self._read(q, params), lbl, schema)

    async def _aggregate(self, op: Aggregate, lbl: _Label, limit: int) -> list[Hit]:
        if not op.metrics:
            raise BackendError("at least one metric is required")
        unknown = set(op.group_by) - set(lbl.properties)
        if unknown:
            raise BackendError(f"unknown properties {sorted(unknown)}")
        returns = [f"n.{quote_name(g)} AS {quote_name(g)}" for g in op.group_by]
        aliases = []
        for m in op.metrics:
            if m == "count":
                returns.append("count(*) AS `count`")
                aliases.append("count")
                continue
            fn, _, prop = m.partition(":")
            if fn not in _AGG_FUNCS or not prop:
                raise BackendError(f"unsupported metric {m!r}; use count or sum|avg|min|max:<property>")
            if prop not in lbl.properties:
                raise BackendError(f"unknown property {prop!r} in metric {m!r}")
            alias = f"{fn}_{prop}"
            returns.append(f"{_AGG_FUNCS[fn]}(n.{quote_name(prop)}) AS {quote_name(alias)}")
            aliases.append(alias)
        params: dict[str, Any] = {"lim": limit}
        q = (f"MATCH (n:{quote_name(lbl.name)}){self._where(op.filter, params, lbl)} "
             f"RETURN {', '.join(returns)} ORDER BY {quote_name(aliases[0])} DESC LIMIT $lim")
        hits = []
        for row in await self._read(q, params):
            data = {k: _plain(row[k]) for k in row.keys()}
            first = data.get(aliases[0])
            hits.append(Hit(doc_id=f"agg:{dumps({g: data.get(g) for g in op.group_by})}", source=self.name,
                            content=[StructuredPart(data=data)],
                            raw_score=float(first) if first is not None else None))
        return hits

    async def _fetch(self, op: Fetch, lbl: _Label, schema: _Schema) -> list[Hit]:
        multi = len(schema.labels) > 1
        pks = [strip_collection(i, lbl.name) if multi else i for i in op.doc_ids]
        params: dict[str, Any] = {"ids": pks}
        if self.id_property in lbl.properties:
            cond = f"toString(n.{quote_name(self.id_property)}) IN $ids"
        else:
            cond = "elementId(n) IN $ids"
        q = f"MATCH (n:{quote_name(lbl.name)}) WHERE {cond} RETURN n, elementId(n) AS eid"
        hits = self._hits(await self._read(q, params), lbl, schema)
        order = {(f"{lbl.name}/{pk}" if multi else pk): i for i, pk in enumerate(pks)}
        return sorted(hits, key=lambda h: order.get(h.doc_id, len(order)))

    async def _traverse(self, op: Traverse, schema: _Schema) -> list[Hit]:
        start_lbl = self._resolve(op.collection, schema) if op.collection is not None else None
        unknown_rels = set(op.rel_types) - set(schema.relationship_types)
        if unknown_rels:
            raise BackendError(f"unknown relationship types {sorted(unknown_rels)}; "
                               f"known: {schema.relationship_types}")
        if op.target_label is not None and op.target_label not in schema.labels:
            raise BackendError(f"unknown target label {op.target_label!r}; one of {sorted(schema.labels)}")
        start_props = set(start_lbl.properties) if start_lbl else \
            {p for lbl in schema.labels.values() for p in lbl.properties}
        params: dict[str, Any] = {"lim": min(op.limit, self.max_rows)}
        start_cond = filter_cypher(op.start, params, start_props, "s")
        filters = [start_cond]
        if op.filter is not None and op.target_label is not None:
            filters.append(filter_cypher(op.filter, params, set(schema.labels[op.target_label].properties), "n"))
        rels = "|".join(quote_name(t) for t in op.rel_types)
        rel = f"[{':' + rels if rels else ''}*1..{int(op.depth)}]"
        s = f"(s{':' + quote_name(start_lbl.name) if start_lbl else ''})"
        n = f"(n{':' + quote_name(op.target_label) if op.target_label else ''})"
        pattern = {"out": f"{s}-{rel}->{n}", "in": f"{s}<-{rel}-{n}", "both": f"{s}-{rel}-{n}"}[op.direction]
        q = (f"MATCH p = {pattern} WHERE {' AND '.join(filters)} "
             f"WITH n, min(length(p)) AS hops RETURN n, elementId(n) AS eid, labels(n) AS labels, hops "
             f"ORDER BY hops, eid LIMIT $lim")
        multi = len(schema.labels) > 1
        hits = []
        for row in await self._read(q, params):
            label = next((lbl for lbl in row["labels"] if lbl in schema.labels), None)
            if label is None:
                continue
            hits.append(self._hit(row["n"], row["eid"], schema.labels[label], multi,
                                  1.0 / max(int(row["hops"]), 1)))
        return hits

    async def _native(self, op: Native, schema: _Schema) -> list[Hit]:
        if not self.native_query:
            raise UnsupportedOperation("native queries are disabled for this source")
        if op.dialect.lower() not in ("cypher", "neo4j"):
            raise BackendError(f"dialect must be cypher, got {op.dialect!r}")
        query = guard_cypher(op.query, min(op.limit, self.max_rows))
        multi = len(schema.labels) > 1
        hits = []
        for i, row in enumerate(await self._read(query)):
            node_hit = None
            for key in row.keys():
                value = row[key]
                labels = getattr(value, "labels", None)
                if labels is not None and hasattr(value, "element_id"):
                    label = next((lbl for lbl in labels if lbl in schema.labels), None)
                    if label is not None:
                        node_hit = self._hit(value, value.element_id, schema.labels[label], multi)
                        break
            if node_hit is None:
                data = {k: _plain(row[k]) for k in row.keys()}
                node_hit = Hit(doc_id=f"native:{i}", source=self.name,
                               content=[StructuredPart(data=data)], metadata={})
            hits.append(node_hit)
        return hits
```

- [ ] **Step 4: Seeding, contract update, and integration tests**

`tests/contract/seed_neo4j.py`:
```python
"""Seed the Neo4j test service (separate write session) and build the backend under test."""

from __future__ import annotations

from . import corpus

NEO4J_URI = "bolt://localhost:57687"
NEO4J_AUTH = ("neo4j", "agenticpass")


async def seed_neo4j() -> None:
    import neo4j

    vectors = await corpus.embeddings()
    driver = neo4j.AsyncGraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH, notifications_min_severity="OFF")
    try:
        async with driver.session() as s:
            await (await s.run("MATCH (n) DETACH DELETE n")).consume()
            for row in await (await s.run("SHOW INDEXES YIELD name, type WHERE type IN ['FULLTEXT', 'VECTOR'] "
                                          "RETURN name")).data():
                await (await s.run(f"DROP INDEX `{row['name']}`")).consume()
            for (i, t, b, ty, y), v in zip(corpus.ROWS, vectors):
                await (await s.run("CREATE (:docs {id: $id, title: $t, body: $b, type: $ty, year: $y, "
                                   "embedding: $v})", id=i, t=t, b=b, ty=ty, y=y, v=v)).consume()
            await (await s.run("CREATE (:conditions {id: 'headache'})")).consume()
            await (await s.run("MATCH (d:docs), (c:conditions {id: 'headache'}) WHERE d.id IN ['d1', 'd4'] "
                               "CREATE (d)-[:TREATS]->(c)")).consume()
            await (await s.run("CREATE FULLTEXT INDEX docs_text FOR (n:docs) ON EACH [n.title, n.body]")).consume()
            await (await s.run("CREATE VECTOR INDEX docs_embedding FOR (n:docs) ON n.embedding OPTIONS "
                               "{indexConfig: {`vector.dimensions`: 64, `vector.similarity_function`: 'cosine'}}")
                   ).consume()
            await (await s.run("CALL db.awaitIndexes(300)")).consume()
    finally:
        await driver.close()


async def make_neo4j():
    from agentic_search.backends.neo4j import Neo4jBackend

    await seed_neo4j()
    return Neo4jBackend("neo", NEO4J_URI, user=NEO4J_AUTH[0], password=NEO4J_AUTH[1],
                        labels=["docs", "conditions"], embedders={"docs.embedding": "hash64"},
                        native_query=True)
```

`tests/contract/test_neo4j_integration.py`:
```python
"""Neo4j-specific integration checks beyond the shared contract."""

import os

import pytest

from agentic_search.backends.base import BackendError
from agentic_search.core.types import Fetch

pytestmark = [pytest.mark.integration,
              pytest.mark.skipif(os.environ.get("AGENTIC_SEARCH_INTEGRATION") != "1",
                                 reason="set AGENTIC_SEARCH_INTEGRATION=1 with docker compose up")]


@pytest.fixture
async def neo():
    from .seed_neo4j import make_neo4j
    b = await make_neo4j()
    yield b
    await b.close()


async def test_server_refuses_writes_in_read_sessions(neo):
    await neo.discover()
    with pytest.raises(BackendError, match="AccessMode|read access"):
        await neo._read("MATCH (n:docs {id: 'd1'}) SET n.title = 'x'")
    [still] = await neo.execute(Fetch(source="neo", collection="docs", doc_ids=["d1"]))
    assert still.metadata["id"] == "d1"


async def test_namespaced_fetch_routes_by_prefix(neo):
    [hit] = await neo.execute(Fetch(source="neo", doc_ids=["docs/d3"]))
    assert hit.doc_id == "docs/d3"
    with pytest.raises(BackendError, match="collection required"):
        await neo.execute(Fetch(source="neo", doc_ids=["d3"]))
```

Replace `tests/contract/test_backend_contract.py` with this version, which adds `bare()` for namespaced ids:
```python
"""Every backend must pass this suite. `files` always runs; the docker-backed params run with
`docker compose up -d --wait` and `AGENTIC_SEARCH_INTEGRATION=1 uv run pytest -m integration`."""

import os

import pytest

from agentic_search import Harness
from agentic_search.backends.base import Backend, BackendError, UnsupportedOperation
from agentic_search.backends.files import FilesBackend
from agentic_search.backends.sql_backend import SqlBackend
from agentic_search.core.types import (
    Aggregate,
    Capability,
    Eq,
    Fetch,
    FilterOnly,
    Hybrid,
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
NATIVE["neo4j"] = ("MATCH (n:docs) WHERE n.type = 'history' RETURN n.id AS id",
                   "MATCH (n:docs) DETACH DELETE n")
NATIVE_DIALECT = {"postgres": "sql", "mysql": "sql", "opensearch": "opensearch_dsl", "neo4j": "cypher"}


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
    if kind == "neo4j":
        from .seed_neo4j import make_neo4j
        return await make_neo4j()
    if kind == "milvus":
        from .seed_milvus import make_milvus
        return await make_milvus()
    raise AssertionError(kind)


@pytest.fixture(params=[
    "files",
    pytest.param("postgres", marks=[pytest.mark.integration, docker]),
    pytest.param("mysql", marks=[pytest.mark.integration, docker]),
    pytest.param("opensearch", marks=[pytest.mark.integration, docker]),
    pytest.param("neo4j", marks=[pytest.mark.integration, docker]),
    pytest.param("milvus", marks=[pytest.mark.integration, docker]),
])
async def backend(request):
    b = await make_backend(request.param)
    b.kind = request.param
    yield b
    await b.close()


def bare(doc_id):
    """Multi-collection sources namespace ids as `<collection>/<pk>`; compare on the pk."""
    return doc_id.split("/", 1)[1] if "/" in doc_id else doc_id


def ids(hits):
    return [bare(h.doc_id) for h in hits]


def text(hit):
    return "\n".join(p.text for p in hit.content if isinstance(p, TextPart))


async def test_discover(backend):
    assert isinstance(backend, Backend)
    m = await backend.discover()
    coll = m.resolve_collection("docs")
    assert coll is not None and coll.count in (None, 5)
    assert {"type", "year"} <= {f.name for f in coll.fields}
    assert {Capability.FILTER, Capability.FETCH} <= m.capabilities
    if Capability.VECTOR in m.capabilities:
        emb = coll.field("embedding")
        assert emb.embedder_id == "hash64" and emb.vector_dim == 64


async def require(backend, cap):
    if cap not in (await backend.discover()).capabilities:
        pytest.skip(f"no {cap.value} support")


async def test_lexical(backend):
    await require(backend, Capability.LEXICAL)
    hits = await backend.execute(Lexical(source=backend.name, collection="docs", text="headache", limit=5))
    assert set(ids(hits)) == {"d1", "d4"}
    assert all(h.raw_score is not None for h in hits)


async def test_lexical_with_filter(backend):
    await require(backend, Capability.LEXICAL)
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
    assert bare(hits[0].doc_id) in {"d1", "d4"}


async def test_hybrid(backend):
    m = await backend.discover()
    if Capability.HYBRID not in m.capabilities:
        pytest.skip("no hybrid support")
    [q] = await corpus.EMBEDDER.embed([TextPart(text="headache fever")], "query")
    hits = await backend.execute(Hybrid(source=backend.name, collection="docs", text="headache",
                                        field="embedding", vector=q, limit=3))
    assert hits and bare(hits[0].doc_id) in {"d1", "d4"}
    assert all(h.raw_score is not None for h in hits)


async def test_regex(backend):
    await require(backend, Capability.REGEX)
    rx = await backend.execute(Regex(source=backend.name, collection="docs", pattern="print.*"))
    assert ids(rx) == ["d3"]


async def test_fetch(backend):
    [f] = await backend.execute(Fetch(source=backend.name, collection="docs", doc_ids=["d3"]))
    assert bare(f.doc_id) == "d3" and "printing" in text(f).lower()


async def test_aggregate(backend):
    await require(backend, Capability.AGGREGATE)
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
    if not isinstance(backend, SqlBackend):
        pytest.skip("not a SQL backend")
    await backend.discover()
    with pytest.raises(BackendError):
        await backend._query("DELETE FROM docs WHERE id = 'd1'", None)
    [still] = await backend.execute(Fetch(source=backend.name, collection="docs", doc_ids=["d1"]))
    assert still.doc_id == "d1"


async def test_traverse(backend):
    await require(backend, Capability.TRAVERSE)
    hits = await backend.execute(Traverse(source=backend.name, collection="conditions",
                                          start=Eq(field="id", value="headache"), rel_types=["TREATS"],
                                          direction="in", depth=1, target_label="docs"))
    assert set(ids(hits)) == {"d1", "d4"}


async def test_unsupported_op(backend):
    if Capability.TRAVERSE in (await backend.discover()).capabilities:
        pytest.skip("backend supports traverse")
    with pytest.raises((UnsupportedOperation, BackendError)):
        await backend.execute(Traverse(source=backend.name, collection="docs", start=Eq(field="type", value="x")))


async def test_harness_end_to_end(backend):
    await require(backend, Capability.LEXICAL)
    driver = ScriptedDriver([[call("lexical_search", source=backend.name, collection="docs", text="headache")]])
    h = Harness([backend], driver, embedders=[corpus.EMBEDDER], analyzer=KeywordJudge(["headache"]))
    res = await h.search("what treats headache?")
    assert {bare(k.split(":", 1)[1]) for k in res.keys()} == {"d1", "d4"}
    assert all(r.judged and r.p_relevant == 1.0 for r in res.hits)
```

- [ ] **Step 5: Run to verify pass**

Run: `uv run pytest tests/backends/test_neo4j_unit.py -q`
Expected: 7 passed

Run: `AGENTIC_SEARCH_INTEGRATION=1 uv run pytest tests/contract -m integration -k neo4j -q`
Expected: 0 failures. The skips are "not a SQL backend" and "backend supports traverse".

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat(backends): Neo4j backend (fulltext, vector, traverse, guarded Cypher, READ sessions)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10: Milvus backend

**Files:**
- Modify: `pyproject.toml`, `docker-compose.yml`
- Create: `docker/milvus/embedEtcd.yaml`, `docker/milvus/user.yaml`, `src/agentic_search/backends/milvus.py`, `tests/backends/test_milvus_unit.py`, `tests/contract/seed_milvus.py`

**Interfaces:**
- Produces: `MilvusBackend(name, uri, *, token=None, db_name=None, collections=None, embedders=None ("collection.field" → embedder id), description=None, max_rows=100, sample_values=True, connect_timeout_s=10)`, with `backend_type="milvus"`.
- Capabilities:
  - LEXICAL, only when a BM25 function exists (searches its sparse output field).
  - VECTOR, HYBRID (client-side RRF), FILTER and FETCH.
  - REGEX, AGGREGATE, TRAVERSE and NATIVE raise `UnsupportedOperation`.
- Filters become Milvus expressions with `filter_params` templating. The exception is `Contains`: the server rejects templates inside `LIKE`, so it uses an escaped, double-quoted literal.
- Collections load lazily, and searches use Strong consistency.

- [ ] **Step 1: Dependency and service**

Add `milvus = ["pymilvus>=2.5,<4"]` to `[project.optional-dependencies]` and `"pymilvus>=2.5,<4",` to the `dev` group, then run `uv sync`.

`docker/milvus/embedEtcd.yaml`:
```yaml
listen-client-urls: http://0.0.0.0:2379
advertise-client-urls: http://0.0.0.0:2379
quota-backend-bytes: 4294967296
auto-compaction-mode: revision
auto-compaction-retention: '1000'
```

`docker/milvus/user.yaml`:
```yaml
# Extra config for Milvus standalone test service (intentionally empty).
```

Add this service to `docker-compose.yml`:
```yaml
  milvus:
    image: milvusdb/milvus:v2.5.10
    command: ["milvus", "run", "standalone"]
    security_opt: ["seccomp:unconfined"]
    environment:
      ETCD_USE_EMBED: "true"
      ETCD_DATA_DIR: /var/lib/milvus/etcd
      ETCD_CONFIG_PATH: /milvus/configs/embedEtcd.yaml
      COMMON_STORAGETYPE: local
    volumes:
      - ./docker/milvus/embedEtcd.yaml:/milvus/configs/embedEtcd.yaml:ro
      - ./docker/milvus/user.yaml:/milvus/configs/user.yaml:ro
    ports: ["59530:19530", "59091:9091"]
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:9091/healthz || exit 1"]
      interval: 5s
      start_period: 60s
      retries: 60
```
Run: `docker compose up -d --wait milvus` (the first start takes about a minute)

- [ ] **Step 2: Write the failing unit tests**

`tests/backends/test_milvus_unit.py`:
```python
import sys

import pytest

from agentic_search.backends.base import BackendError, UnsupportedOperation
from agentic_search.backends.milvus import MilvusBackend, filter_expr
from agentic_search.core.secrets import scrub
from agentic_search.core.types import (
    Aggregate,
    And,
    Contains,
    Eq,
    Exists,
    In,
    Not,
    Or,
    Range,
    Regex,
)

FIELDS = {"type", "year", "title", "id"}


def test_filter_expr_templates_every_value():
    params: dict = {}
    expr = filter_expr(And(clauses=[Eq(field="type", value="drug"),
                                    Or(clauses=[Range(field="year", gte=2020, lt=2022),
                                                Not(clause=In(field="id", values=["a", "b"]))]),
                                    Contains(field="title", value='50%_off"x')]), FIELDS, params)
    # LIKE only accepts a literal: wildcards are LIKE-escaped, then the literal is quote-escaped.
    assert expr == ('(type == {p0} and ((year >= {p1} and year < {p2}) or (not (id in {p3}))) '
                    'and title like "%50\\\\%\\\\_off\\"x%")')
    assert params == {"p0": "drug", "p1": 2020, "p2": 2022, "p3": ["a", "b"]}


def test_filter_expr_edge_cases():
    assert filter_expr(And(clauses=[]), FIELDS, {}) == "true"
    assert filter_expr(Or(clauses=[]), FIELDS, {}) == "false"
    assert filter_expr(In(field="type", values=[]), FIELDS, {}) == "false"
    assert filter_expr(Range(field="year"), FIELDS, {}) == "true"
    assert filter_expr(Exists(field="title"), FIELDS, {}) == "title is not null"


@pytest.mark.parametrize("bad", ["nope", "year) or (1 == 1", "type\n"])
def test_filter_expr_rejects_unknown_or_bad_fields(bad):
    with pytest.raises(BackendError):
        filter_expr(Eq(field=bad, value=1), FIELDS | {"year) or (1 == 1", "type\n"}, {})


def test_constructor_registers_secrets():
    MilvusBackend("mv", "http://root:milvus-pw-123@h:19530", token="tok-abcdef-999")
    assert scrub("x milvus-pw-123 tok-abcdef-999") == "x *** ***"


async def test_missing_extra_is_backend_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "pymilvus", None)
    with pytest.raises(BackendError, match="milvus.*extra"):
        await MilvusBackend("mv", "http://localhost:59530").discover()


async def test_unreachable_uri_is_backend_error():
    b = MilvusBackend("mv", "http://127.0.0.1:1", connect_timeout_s=2)
    try:
        with pytest.raises(BackendError):
            await b.discover()
    finally:
        await b.close()


async def test_unsupported_ops_without_connecting():
    b = MilvusBackend("mv", "http://127.0.0.1:1", connect_timeout_s=2)
    b._collections = {}  # pretend discovery ran
    for op in (Regex(source="mv", pattern="x"), Aggregate(source="mv", group_by=["type"])):
        with pytest.raises(UnsupportedOperation):
            await b.execute(op)
```

Run: `uv run pytest tests/backends/test_milvus_unit.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'agentic_search.backends.milvus'`

- [ ] **Step 3: Implement**

`src/agentic_search/backends/milvus.py`:
```python
"""Milvus backend: dense ANN search, BM25 full-text search (when a BM25 function exists), client-side
hybrid (RRF), boolean filter expressions and fetch by primary key. Values are passed as
`filter_params` templates except LIKE patterns, which Milvus only accepts as escaped literals.
Needs the `milvus` extra."""

from __future__ import annotations

import asyncio
import importlib
import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import unquote, urlparse

from agentic_search.backends.base import (
    BackendError,
    DiscoverDetail,
    UnsupportedOperation,
    routed_collection,
    rrf_merge,
    strip_collection,
)
from agentic_search.backends.sql import jsonable
from agentic_search.core.secrets import register_secret
from agentic_search.core.types import (
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
    Not,
    Or,
    QueryOp,
    Range,
    StructuredPart,
    TextPart,
    Vector,
)

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
SAMPLE_DISTINCT_MAX = 20
SAMPLE_SCAN_MAX = 1000
_VECTOR_TYPES = {"FLOAT_VECTOR", "FLOAT16_VECTOR", "BFLOAT16_VECTOR"}
_SCALAR_TYPES = {
    "INT8": FieldType.INT, "INT16": FieldType.INT, "INT32": FieldType.INT, "INT64": FieldType.INT,
    "FLOAT": FieldType.FLOAT, "DOUBLE": FieldType.FLOAT, "BOOL": FieldType.BOOL,
    "JSON": FieldType.JSON, "ARRAY": FieldType.JSON, "VARCHAR": FieldType.KEYWORD,
}


def _require_pymilvus() -> Any:
    try:
        return importlib.import_module("pymilvus")
    except ImportError as exc:
        raise BackendError(
            "MilvusBackend needs the `milvus` extra: pip install 'agentic-search[milvus]'") from exc


def _ident(name: str, fields: set[str]) -> str:
    if name not in fields or not _IDENT.fullmatch(name):
        raise BackendError(f"unknown or invalid field {name!r}")
    return name


def _like_literal(value: str) -> str:
    """A double-quoted Milvus string literal for LIKE '%value%' with wildcards escaped."""
    pattern = "%" + value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
    return '"' + pattern.replace("\\", "\\\\").replace('"', '\\"') + '"'


def filter_expr(f: Filter, fields: set[str], params: dict[str, Any]) -> str:
    """Filter AST → Milvus boolean expression; values go into `params` as {pN} templates."""
    def bind(value: Any) -> str:
        key = f"p{len(params)}"
        params[key] = value
        return "{" + key + "}"

    if isinstance(f, And):
        return "(" + " and ".join(filter_expr(c, fields, params) for c in f.clauses) + ")" \
            if f.clauses else "true"
    if isinstance(f, Or):
        return "(" + " or ".join(filter_expr(c, fields, params) for c in f.clauses) + ")" \
            if f.clauses else "false"
    if isinstance(f, Not):
        return f"(not ({filter_expr(f.clause, fields, params)}))"
    name = _ident(f.field, fields)
    if isinstance(f, Eq):
        return f"{name} == {bind(f.value)}"
    if isinstance(f, In):
        return f"{name} in {bind(list(f.values))}" if f.values else "false"
    if isinstance(f, Range):
        parts = [f"{name} {sym} {bind(b)}" for b, sym in
                 ((f.gte, ">="), (f.gt, ">"), (f.lte, "<="), (f.lt, "<")) if b is not None]
        return "(" + " and ".join(parts) + ")" if parts else "true"
    if isinstance(f, Exists):
        return f"{name} is not null"
    if isinstance(f, Contains):
        return f"{name} like {_like_literal(f.value)}"
    raise BackendError(f"unsupported filter node {type(f).__name__}")


@dataclass
class _Coll:
    info: CollectionInfo
    pk: str
    text_fields: list[str]
    output_fields: list[str]
    vector_metrics: dict[str, str] = field(default_factory=dict)
    sparse_field: str | None = None
    bm25_inputs: list[str] = field(default_factory=list)

    @property
    def fields(self) -> set[str]:
        return {f.name for f in self.info.fields}


def _type_name(t: Any) -> str:
    return getattr(t, "name", str(t)).upper()


def _truthy(v: Any) -> bool:
    return v is True or str(v).lower() == "true"


class MilvusBackend:
    backend_type = "milvus"

    def __init__(self, name: str, uri: str, *, token: str | None = None, db_name: str | None = None,
                 collections: list[str] | None = None, embedders: dict[str, str] | None = None,
                 description: str | None = None, max_rows: int = 100, sample_values: bool = True,
                 connect_timeout_s: float = 10.0):
        parsed = urlparse(uri)
        register_secret(token)
        register_secret(unquote(parsed.password or ""))
        self.name = name
        self.uri = uri
        self.token = token
        self.db_name = db_name
        self.collection_names = collections
        self.embedders = embedders or {}
        self.description = description
        self.max_rows = max_rows
        self.sample_values = sample_values
        self.connect_timeout_s = float(connect_timeout_s)
        self._client: Any = None
        self._collections: dict[str, _Coll] | None = None
        self._loaded: set[str] = set()
        self._lock = asyncio.Lock()

    # ---- client ---------------------------------------------------------------

    def _get_client(self) -> Any:
        if self._client is None:
            pymilvus = _require_pymilvus()
            self._client = pymilvus.AsyncMilvusClient(
                uri=self.uri, token=self.token or "", db_name=self.db_name or "",
                timeout=self.connect_timeout_s)
        return self._client

    async def _call(self, method: str, *args: Any, timeout: float | None = None, **kwargs: Any) -> Any:
        try:
            client = self._get_client()
            coro = getattr(client, method)(*args, **kwargs)
            return await asyncio.wait_for(coro, timeout) if timeout else await coro
        except (BackendError, UnsupportedOperation):
            raise
        except Exception as exc:  # pymilvus / grpc / timeout
            raise BackendError(f"{type(exc).__name__}: {exc}") from exc

    async def close(self) -> None:
        if self._client is not None:
            client, self._client = self._client, None
            try:
                await client.close()
            except Exception:
                pass

    # ---- discovery ---------------------------------------------------------------

    def capabilities(self) -> set[Capability]:
        caps = {Capability.FILTER, Capability.FETCH}
        colls = self._collections
        if colls is None or any(c.sparse_field for c in colls.values()):
            caps.add(Capability.LEXICAL)
        if colls is None or any(c.vector_metrics for c in colls.values()):
            caps.add(Capability.VECTOR)
        if colls is not None and any(c.sparse_field and c.vector_metrics for c in colls.values()):
            caps.add(Capability.HYBRID)
        return caps

    async def _ensure(self) -> dict[str, _Coll]:
        async with self._lock:
            if self._collections is None:
                self._collections = await self._discover_collections()
            return self._collections

    async def _discover_collections(self) -> dict[str, _Coll]:
        t = self.connect_timeout_s
        names = self.collection_names or await self._call("list_collections", timeout=t)
        out: dict[str, _Coll] = {}
        for name in names:
            desc = await self._call("describe_collection", name, timeout=t)
            out[name] = await self._collection(name, desc)
        return out

    async def _collection(self, name: str, desc: dict[str, Any]) -> _Coll:
        functions = desc.get("functions") or []
        bm25 = [fn for fn in functions if _type_name(fn.get("type")) == "BM25"]
        bm25_inputs = [i for fn in bm25 for i in fn.get("input_field_names", [])]
        sparse = bm25[0]["output_field_names"][0] if bm25 else None
        pk = next((f["name"] for f in desc["fields"] if f.get("is_primary")), None)
        if pk is None:
            raise BackendError(f"collection {name!r} has no primary key")
        metrics = await self._metrics(name)
        specs: list[FieldSpec] = []
        text_fields: list[str] = []
        output_fields: list[str] = []
        vector_metrics: dict[str, str] = {}
        for f in desc["fields"]:
            fname, tname = f["name"], _type_name(f["type"])
            params = f.get("params") or {}
            if tname == "SPARSE_FLOAT_VECTOR":
                continue
            if tname in _VECTOR_TYPES:
                metric = metrics.get(fname, "COSINE")
                vector_metrics[fname] = metric
                specs.append(FieldSpec(name=fname, type=FieldType.VECTOR, vector_dim=int(params["dim"])
                                       if params.get("dim") is not None else None,
                                       vector_metric=metric.lower(),
                                       embedder_id=self.embedders.get(f"{name}.{fname}")))
                continue
            ftype = _SCALAR_TYPES.get(tname, FieldType.JSON)
            if ftype is FieldType.KEYWORD and (fname in bm25_inputs or _truthy(params.get("enable_analyzer"))):
                ftype = FieldType.TEXT
                text_fields.append(fname)
            output_fields.append(fname)
            samples = None
            if ftype in (FieldType.KEYWORD, FieldType.BOOL) and fname != pk:
                samples = await self._samples(name, fname)
            specs.append(FieldSpec(
                name=fname, type=ftype, searchable=fname in bm25_inputs,
                filterable=ftype is not FieldType.JSON,
                sortable=ftype in (FieldType.INT, FieldType.FLOAT), sample_values=samples))
        count = await self._count(name)
        return _Coll(info=CollectionInfo(name=name, fields=specs, count=count), pk=pk,
                     text_fields=text_fields, output_fields=output_fields,
                     vector_metrics=vector_metrics, sparse_field=sparse, bm25_inputs=bm25_inputs)

    async def _metrics(self, name: str) -> dict[str, str]:
        out: dict[str, str] = {}
        try:
            for index in await self._call("list_indexes", name, timeout=self.connect_timeout_s):
                info = await self._call("describe_index", name, index, timeout=self.connect_timeout_s)
                if info.get("field_name") and info.get("metric_type"):
                    out[info["field_name"]] = str(info["metric_type"]).upper()
        except BackendError:
            return out
        return out

    async def _count(self, name: str) -> int | None:
        try:
            stats = await self._call("get_collection_stats", name, timeout=self.connect_timeout_s)
            return int(stats["row_count"])
        except (BackendError, KeyError, TypeError, ValueError):
            return None

    async def _samples(self, name: str, fname: str) -> list[Any] | None:
        if not self.sample_values:
            return None
        try:
            await self._load(name)
            rows = await self._call("query", name, filter=f"{_ident(fname, {fname})} is not null",
                                    output_fields=[fname], limit=SAMPLE_SCAN_MAX,
                                    consistency_level="Strong")
        except BackendError:
            return None
        values = list(dict.fromkeys(r.get(fname) for r in rows if r.get(fname) is not None))
        return values if len(values) <= SAMPLE_DISTINCT_MAX else None

    async def _load(self, name: str) -> None:
        if name not in self._loaded:
            await self._call("load_collection", name)
            self._loaded.add(name)

    async def discover(self, detail: DiscoverDetail = "full",
                       collection: str | None = None) -> Manifest:
        colls = await self._ensure()
        return Manifest(source=self.name, backend_type=self.backend_type,
                        capabilities=self.capabilities(), description=self.description,
                        collections=[c.info for c in colls.values()])

    # ---- execution ---------------------------------------------------------------

    def _resolve(self, name: str | None, colls: dict[str, _Coll]) -> _Coll:
        if name is None:
            if len(colls) == 1:
                return next(iter(colls.values()))
            raise BackendError(f"collection required; one of {sorted(colls)}")
        if name not in colls:
            raise BackendError(f"unknown collection {name!r}; one of {sorted(colls)}")
        return colls[name]

    def _doc_id(self, coll: _Coll, pk: Any, colls: dict[str, _Coll]) -> str:
        return f"{coll.info.name}/{pk}" if len(colls) > 1 else str(pk)

    def _hit(self, coll: _Coll, row: dict[str, Any], colls: dict[str, _Coll],
             score: float | None = None) -> Hit:
        clean = {k: jsonable(v) for k, v in row.items() if k in coll.output_fields}
        texts = [str(clean[f]) for f in coll.text_fields if clean.get(f) not in (None, "")]
        content: list[Content] = [TextPart(text="\n".join(texts))] if texts else [StructuredPart(data=clean)]
        metadata = {k: v for k, v in clean.items() if k not in coll.text_fields}
        return Hit(doc_id=self._doc_id(coll, row[coll.pk], colls), source=self.name, content=content,
                   metadata=metadata, raw_score=score)

    def _filter(self, f: Filter | None, coll: _Coll) -> tuple[str, dict[str, Any]]:
        params: dict[str, Any] = {}
        return (filter_expr(f, coll.fields, params) if f is not None else ""), params

    async def _search(self, coll: _Coll, data: Any, anns_field: str, metric: str, f: Filter | None,
                      limit: int, colls: dict[str, _Coll]) -> list[Hit]:
        expr, params = self._filter(f, coll)
        await self._load(coll.info.name)
        kwargs: dict[str, Any] = {"filter_params": params} if params else {}
        results = await self._call("search", coll.info.name, data=[data], anns_field=anns_field,
                                   filter=expr, limit=limit, output_fields=coll.output_fields,
                                   search_params={"metric_type": metric}, consistency_level="Strong",
                                   **kwargs)
        hits = []
        for r in (results[0] if results else []):
            row = dict(r.get("entity") or {})
            row.setdefault(coll.pk, r.get("id"))
            distance = float(r.get("distance", 0.0))
            hits.append(self._hit(coll, row, colls, -distance if metric == "L2" else distance))
        return hits

    async def _lexical(self, op: Lexical | Hybrid, coll: _Coll, limit: int,
                       colls: dict[str, _Coll]) -> list[Hit]:
        if coll.sparse_field is None:
            raise UnsupportedOperation(f"collection {coll.info.name!r} has no BM25 function")
        fields = getattr(op, "fields", None)
        if fields and not set(fields) <= set(coll.bm25_inputs):
            raise BackendError(f"lexical search covers only {coll.bm25_inputs} in {coll.info.name!r}")
        return await self._search(coll, op.text, coll.sparse_field, "BM25", op.filter, limit, colls)

    async def _vector(self, field_name: str, vector: list[float] | None, coll: _Coll, f: Filter | None,
                      limit: int, colls: dict[str, _Coll]) -> list[Hit]:
        if vector is None:
            raise BackendError("vector op reached the backend without an embedded query vector")
        if field_name not in coll.vector_metrics:
            raise BackendError(f"{field_name!r} is not a vector field of {coll.info.name}")
        return await self._search(coll, vector, field_name, coll.vector_metrics[field_name], f, limit, colls)

    async def execute(self, op: QueryOp) -> list[Hit]:
        colls = await self._ensure()
        if not isinstance(op, (Lexical, Vector, Hybrid, FilterOnly, Fetch)):
            raise UnsupportedOperation(f"milvus backend does not support {op.type}")
        if isinstance(op, Fetch):
            return await self._fetch(op, colls)
        coll = self._resolve(op.collection, colls)
        limit = min(op.limit, self.max_rows)
        if isinstance(op, Lexical):
            return await self._lexical(op, coll, limit, colls)
        if isinstance(op, Vector):
            return await self._vector(op.field, op.vector, coll, op.filter, limit, colls)
        if isinstance(op, Hybrid):
            depth = min(max(limit * 5, 50), 500)
            lex = await self._lexical(op, coll, depth, colls)
            vec = await self._vector(op.field, op.vector, coll, op.filter, depth, colls)
            by_id = {h.doc_id: h for h in [*vec, *lex]}
            fused = rrf_merge([([h.doc_id for h in lex], op.lexical_weight),
                               ([h.doc_id for h in vec], 1.0 - op.lexical_weight)])
            return [by_id[i].model_copy(update={"raw_score": s}) for i, s in fused[:limit]]
        expr, params = self._filter(op.filter, coll)
        await self._load(coll.info.name)
        kwargs: dict[str, Any] = {"filter_params": params} if params else {}
        rows = await self._call("query", coll.info.name, filter=expr, output_fields=coll.output_fields,
                                limit=limit, consistency_level="Strong", **kwargs)
        return [self._hit(coll, r, colls) for r in rows]

    async def _fetch(self, op: Fetch, colls: dict[str, _Coll]) -> list[Hit]:
        coll = self._resolve(routed_collection(op, colls), colls)
        ids = [strip_collection(i, coll.info.name) if len(colls) > 1 else i for i in op.doc_ids]
        await self._load(coll.info.name)
        rows = await self._call("query", coll.info.name, filter=f"{coll.pk} in {{ids}}",
                                filter_params={"ids": ids}, output_fields=coll.output_fields,
                                consistency_level="Strong")
        order = {str(i): n for n, i in enumerate(ids)}
        hits = [self._hit(coll, r, colls) for r in rows]
        return sorted(hits, key=lambda h: order.get(h.doc_id.split("/", 1)[-1] if len(colls) > 1
                                                    else h.doc_id, len(order)))
```

- [ ] **Step 4: Seeding**

`tests/contract/seed_milvus.py`:
```python
"""Seed the Milvus test collection `docs` from the shared corpus (test-only writes)."""

from __future__ import annotations

from . import corpus

MILVUS_URI = "http://localhost:59530"


async def seed_milvus() -> None:
    from pymilvus import AsyncMilvusClient, DataType, Function, FunctionType, MilvusClient

    vectors = await corpus.embeddings()
    schema = MilvusClient.create_schema(auto_id=False)
    schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
    schema.add_field("title", DataType.VARCHAR, max_length=256)
    schema.add_field("body", DataType.VARCHAR, max_length=4096, enable_analyzer=True)
    schema.add_field("type", DataType.VARCHAR, max_length=32)
    schema.add_field("year", DataType.INT64)
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=64)
    schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
    schema.add_function(Function(name="body_bm25", function_type=FunctionType.BM25,
                                 input_field_names=["body"], output_field_names=["sparse"]))
    index = MilvusClient.prepare_index_params()
    index.add_index(field_name="embedding", index_type="HNSW", metric_type="COSINE",
                    params={"M": 16, "efConstruction": 64})
    index.add_index(field_name="sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
    client = AsyncMilvusClient(uri=MILVUS_URI)
    try:
        if "docs" in await client.list_collections():
            await client.drop_collection("docs")
        await client.create_collection("docs", schema=schema, index_params=index,
                                       consistency_level="Strong")
        rows = [{"id": i, "title": t, "body": b, "type": ty, "year": y, "embedding": v}
                for (i, t, b, ty, y), v in zip(corpus.ROWS, vectors)]
        await client.insert("docs", rows)
        await client.flush("docs")
        await client.load_collection("docs")
    finally:
        await client.close()


_seeded = False


async def make_milvus():
    """Seed once per test process (contract tests never write), then return a fresh backend."""
    global _seeded
    from agentic_search.backends.milvus import MilvusBackend

    if not _seeded:
        await seed_milvus()
        _seeded = True
    return MilvusBackend("mv", MILVUS_URI, collections=["docs"], embedders={"docs.embedding": "hash64"})
```

- [ ] **Step 5: Run to verify pass**

Run: `uv run pytest tests/backends/test_milvus_unit.py -q`
Expected: 9 passed

Run: `AGENTIC_SEARCH_INTEGRATION=1 uv run pytest tests/contract -m integration -k milvus -q`
Expected: 0 failures. The skips are regex, aggregate, native, traverse and SQL read-only.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat(backends): Milvus backend (BM25 sparse + dense ANN, filter templating)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 11: Config registration, README, follow-ups, full verification

**Files:**
- Modify: `src/agentic_search/config.py`, `README.md`, `docs/superpowers/plans/2026-09-29-plan-1-followups.md`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces:
  - Embedder types `openai_compat`, `tei`, `vertex` and `http`. Each accepts an `auth: {type: none|api_key|bearer|gcp_adc|azure_identity, …}` block.
  - Decider type `typesafe`.
  - Backend types `neo4j` and `milvus`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_config.py`:
```python
async def test_build_remote_embedders_and_typesafe(tmp_path, monkeypatch):
    from agentic_search.embedders.auth import AzureIdentity, Bearer, GcpAdc
    from agentic_search.embedders.http_generic import GenericHttpEmbedder
    from agentic_search.embedders.openai_compat import OpenAICompatEmbedder
    from agentic_search.embedders.tei import TEIEmbedder
    from agentic_search.embedders.vertex import VertexEmbedder
    from agentic_search.models.typesafe import TypeSafeDecider

    monkeypatch.setenv("TS_KEY", "ts-cfg-key-555")
    monkeypatch.setenv("MEDSIGLIP_URL", "https://medsiglip.example/score")
    h = build_harness({
        "embedders": [
            {"type": "openai_compat", "model": "text-embedding-3-small", "dim": 1536,
             "auth": {"type": "bearer", "token": "sk-cfg-1234"}},
            {"type": "tei", "url": "http://tei:8080", "dim": 384, "id": "tei:bge-small"},
            {"type": "vertex", "model": "gemini-embedding-001", "dim": 768, "project": "proj-1"},
            {"type": "http", "id": "azure:medsiglip-448", "url_env": "MEDSIGLIP_URL", "dim": 1152,
             "auth": {"type": "azure_identity", "scope": "api://medsiglip/.default"},
             "request": {"image": {"instances": [{"image_b64": "{{b64}}"}]}},
             "response_path": "$.predictions[*].embedding"},
        ],
        "backends": [{"name": "notes", "type": "files", "root": ".", "glob": "none/*"}],
        "driver": {"type": "openai_compat", "model": "local", "base_url": "http://localhost:8000/v1"},
        "controller": {"type": "typesafe", "api_key_env": "TS_KEY"},
    }, base_dir=tmp_path)
    emb = {e.id: e for e in h.embedders._by_id.values()}
    assert isinstance(emb["openai:text-embedding-3-small"], OpenAICompatEmbedder)
    assert isinstance(emb["openai:text-embedding-3-small"].auth, Bearer)
    assert isinstance(emb["tei:bge-small"], TEIEmbedder)
    assert isinstance(emb["vertex:gemini-embedding-001"].auth, GcpAdc)
    assert isinstance(emb["vertex:gemini-embedding-001"], VertexEmbedder)
    med = emb["azure:medsiglip-448"]
    assert isinstance(med, GenericHttpEmbedder) and isinstance(med.auth, AzureIdentity)
    assert med.url == "https://medsiglip.example/score"
    assert isinstance(h.controller_decider, TypeSafeDecider) and h.controller_decider.id == "typesafe:jev-latest"


def test_build_graph_and_vector_backends(tmp_path, monkeypatch):
    from agentic_search.backends.milvus import MilvusBackend
    from agentic_search.backends.neo4j import Neo4jBackend
    from agentic_search.core.secrets import scrub

    monkeypatch.setenv("NEO4J_PASSWORD", "neo-pass-2468")
    h = build_harness({
        "embedders": [{"type": "hash", "id": "hash64", "dim": 64}],
        "backends": [
            {"name": "kg", "type": "neo4j", "uri": "bolt://graph:7687", "user": "neo4j",
             "password_env": "NEO4J_PASSWORD", "labels": ["Drug", "Condition"],
             "embedders": {"Drug.embedding": "hash64"}, "native_query": True},
            {"name": "vec", "type": "milvus", "uri": "http://milvus:19530", "collections": ["docs"],
             "embedders": {"docs.embedding": "hash64"}},
        ],
        "driver": {"type": "openai_compat", "model": "local", "base_url": "http://localhost:8000/v1"},
    }, base_dir=tmp_path)
    assert isinstance(h.backends["kg"], Neo4jBackend) and isinstance(h.backends["vec"], MilvusBackend)
    assert scrub("neo-pass-2468") == "***"
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_config.py -q`
Expected: FAIL with `ConfigError: unknown embedder type 'openai_compat'`

- [ ] **Step 3: Implement**

In `src/agentic_search/config.py`, add before `_cross_encoder`:
```python
def _neo4j(cfg: dict[str, Any], ctx: BuildContext) -> Any:
    from agentic_search.backends.neo4j import Neo4jBackend
    return Neo4jBackend(cfg["name"], cfg["uri"], **_backend_kwargs(
        cfg, ctx, ("user", "password", "database", "labels", "id_property", "embedders", "native_query",
                   "description", "max_rows", "sample_values", "connect_timeout_s")))


def _milvus(cfg: dict[str, Any], ctx: BuildContext) -> Any:
    from agentic_search.backends.milvus import MilvusBackend
    return MilvusBackend(cfg["name"], cfg["uri"], **_backend_kwargs(
        cfg, ctx, ("token", "db_name", "collections", "embedders", "description", "max_rows",
                   "sample_values", "connect_timeout_s")))


_HTTP_EMBEDDER_KEYS = ("id", "batch_size", "concurrency", "timeout_s", "max_retries")


def _http_kwargs(cfg: dict[str, Any], extra: tuple[str, ...] = ()) -> dict[str, Any]:
    from agentic_search.embedders.auth import build_auth
    kwargs = {k: cfg[k] for k in _HTTP_EMBEDDER_KEYS + extra if k in cfg}
    if "auth" in cfg:
        kwargs["auth"] = build_auth(cfg["auth"])
    return kwargs


def _openai_embedder(cfg: dict[str, Any], ctx: BuildContext) -> Any:
    from agentic_search.embedders.openai_compat import OpenAICompatEmbedder
    return OpenAICompatEmbedder(cfg["model"], int(cfg["dim"]), **_http_kwargs(
        cfg, ("base_url", "api_key", "dimensions", "query_prefix", "document_prefix")))


def _tei(cfg: dict[str, Any], ctx: BuildContext) -> Any:
    from agentic_search.embedders.tei import TEIEmbedder
    return TEIEmbedder(cfg["url"], int(cfg["dim"]), **_http_kwargs(
        cfg, ("query_prefix", "document_prefix", "normalize")))


def _vertex(cfg: dict[str, Any], ctx: BuildContext) -> Any:
    from agentic_search.embedders.vertex import VertexEmbedder
    return VertexEmbedder(cfg["model"], int(cfg["dim"]), project=cfg["project"],
                          **_http_kwargs(cfg, ("location", "endpoint")))


def _http_embedder(cfg: dict[str, Any], ctx: BuildContext) -> Any:
    from agentic_search.embedders.http_generic import GenericHttpEmbedder
    return GenericHttpEmbedder(cfg["url"], int(cfg["dim"]), request=cfg["request"],
                               response_path=cfg["response_path"], **_http_kwargs(cfg, ("headers",)))


def _typesafe(cfg: dict[str, Any], ctx: BuildContext) -> Any:
    from agentic_search.models.typesafe import TypeSafeDecider
    return TypeSafeDecider(cfg.get("model", "jev-latest"), api_key=cfg.get("api_key"), id=cfg.get("id"),
                           batch_size=int(cfg.get("batch_size", 16)), price_per_mtok=_price(cfg))
```
and extend the registration list:
```python
    ("backend", "neo4j", _neo4j),
    ("backend", "milvus", _milvus),
    ("embedder", "openai_compat", _openai_embedder),
    ("embedder", "tei", _tei),
    ("embedder", "vertex", _vertex),
    ("embedder", "http", _http_embedder),
    ("decider", "typesafe", _typesafe),
```

- [ ] **Step 4: README and follow-ups**

Apply this change to `README.md`:

````diff
diff --git a/README.md b/README.md
index 05e2fc7..4701644 100644
--- a/README.md
+++ b/README.md
@@ -47,10 +47,9 @@ Or from YAML: `from agentic_search.config import load_harness`. The spec §7 sho
 ## Roles
 
 - **Driver**: plans and calls tools (`ToolCallingDriver` over `AnthropicClient` or `OpenAICompatClient`).
-- **Analyzer decider**: judges relevance (`LLMJudge`, `CrossEncoderJudge`, TypeSafe System One in Plan 3).
-- **Controller decider**: continue/refine/broaden/stop (`LLMJudge`, or the built-in heuristic).
-- **Backends**: files, Postgres + pgvector, MySQL, BigQuery and OpenSearch (see below). Graph and
-  vector stores (Neo4j, Milvus) come in Plan 3.
+- **Analyzer decider**: judges relevance (`LLMJudge`, `CrossEncoderJudge`, `TypeSafeDecider`).
+- **Controller decider**: continue/refine/broaden/stop (`LLMJudge`, `TypeSafeDecider`, or the built-in heuristic).
+- **Backends**: files, Postgres + pgvector, MySQL, BigQuery, OpenSearch, Neo4j and Milvus (see below).
 - **Hooks / SourcePolicy**: every model-bound payload passes through `Hooks.before_model_call`:
   the planner view, judge requests (question + hits), the controller view, embedder queries, and in
   model mode the delegate question/context and every tool output (a raising hook withholds that
@@ -66,6 +65,8 @@ Or from YAML: `from agentic_search.config import load_harness`. The spec §7 sho
 | `mysql` | `mysql` | FULLTEXT (natural language) | – | lexical needs a FULLTEXT index; READ ONLY sessions, `max_execution_time` |
 | `bigquery` | `bigquery` | term match (`CONTAINS_SUBSTR`) | `VECTOR_SEARCH` | every query dry-run; refused above `max_bytes_billed` |
 | `opensearch` | `opensearch` | `multi_match` | k-NN (`knn_vector`) | search APIs only |
+| `neo4j` | `neo4j` | FULLTEXT index | VECTOR index | collections are labels; `traverse`; READ sessions; native Cypher |
+| `milvus` | `milvus` | BM25 function (sparse) | ANN | collections map 1:1; no regex/aggregate |
 
 All backends support filters, regex, aggregates (`count`, `sum|avg|min|max:<column>`) and fetch.
 Set `native_query: true` on a backend to let the planner run read-only native SQL / search bodies;
@@ -89,6 +90,36 @@ backends:
   - {name: search, type: opensearch, url_env: SEARCH_URL, indices: [articles]}
 ```
 
+## Embedders
+
+| type | modalities | notes |
+|---|---|---|
+| `hash`, `sentence_transformers` | text (+ image for CLIP) | in-process (`local`) |
+| `openai_compat` | text | OpenAI, Azure OpenAI, vLLM, Ollama, Together |
+| `tei` | text | Hugging Face Text Embeddings Inference |
+| `vertex` | text; image with `multimodalembedding@001` | Google Application Default Credentials by default |
+| `http` | per request template | custom containers, e.g. MedSigLIP on Azure ML |
+
+Remote embedders take `auth: {type: api_key|bearer|gcp_adc|azure_identity, ...}` (install the `gcp` or
+`azure` extra for the cloud credentials). Backends that embed their own documents send them to
+non-local embedders only through `Hooks.before_model_call`, and `allowed_models` source policy applies to
+embedders too.
+
+```yaml
+embedders:
+  - {type: vertex, model: gemini-embedding-001, dim: 768, project: my-proj}
+  - id: "azure:medsiglip-448"
+    type: http
+    url_env: MEDSIGLIP_URL
+    dim: 1152
+    auth: {type: azure_identity, scope: "api://medsiglip/.default"}
+    request:
+      image: {instances: [{image_b64: "{{b64}}"}]}
+      text: {instances: [{text: "{{text}}"}]}
+    response_path: "$.predictions[*].embedding"
+controller: {type: typesafe, api_key_env: TYPESAFE_API_KEY}   # Jev judges/decides
+```
+
 ## Evaluate
 
 ```bash
@@ -103,6 +134,6 @@ the keys the model ranked, so its recall@100 is structurally lower than modes th
 
 ```bash
 uv sync && uv run pytest                      # unit tests, no services
-docker compose up -d --wait                   # Postgres+pgvector, MySQL, OpenSearch
+docker compose up -d --wait                   # Postgres+pgvector, MySQL, OpenSearch, Neo4j, Milvus
 AGENTIC_SEARCH_INTEGRATION=1 uv run pytest -m integration   # backend contract suite
 ```
````

Append to `docs/superpowers/plans/2026-09-29-plan-1-followups.md`:
```markdown
## Status after Plan 3
- Done: hooked backend embeddings (Task 2); Postgres stored tsvector and ConfigError for invalid values (Task 8).
- Open: live runs of BigQuery, Vertex, OpenAI/TEI embeddings, TypeSafe and Azure/GCP auth with real credentials; Neo4j Enterprise/Aura/TLS; auth-enabled Milvus.
```

- [ ] **Step 5: Full verification**

Run: `uv run ruff check --fix src tests scripts && uv run pytest -q && uv run ruff check src tests scripts`
Expected: `300 passed, 5 skipped`, and `All checks passed!`

Run: `docker compose up -d --wait` and then `AGENTIC_SEARCH_INTEGRATION=1 uv run pytest -m integration -q`
Expected: `63 passed, 13 skipped`

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat(config): remote embedders, TypeSafe, Neo4j and Milvus types; docs

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

## Spec coverage (Plan 3)

| Spec § / follow-up | Where |
|---|---|
| §3.3 Neo4j (traverse, fulltext, vector, native Cypher) and Milvus adapters | Tasks 9–10 |
| §3.4 remote embedders (Vertex, OpenAI-compatible, TEI, generic HTTP) and auth providers | Tasks 3–6 |
| §3.4 embedding-space binding for new backends | Tasks 9–10 (`embedders:` maps) |
| §3.5 TypeSafe System One deciders | Task 7 |
| §6 hooks on every model-bound payload, source policy on embedders | Task 2 |
| §7 config for the new types | Task 11 |
| §8 contract suite across all backends; mocked HTTP for remote adapters; live tests | Tasks 1, 3–10 |
| §9 criterion 1 (all backends pass the contract) | Tasks 1, 9, 10 |
| Follow-up: hooked backend embeddings | Task 2 |
| Follow-ups: Postgres stored tsvector; ConfigError for constructor ValueError | Task 8 |

## Known limitations

- Vertex, OpenAI, TEI, TypeSafe and Azure/GCP auth are verified against mocked HTTP only. Run the `live` tests with real credentials before relying on them.
- BigQuery is still unverified against a real dataset (carried from Plan 2).
- Neo4j is verified on 5-community only. Enterprise, non-default databases, TLS and Aura are not tested, and neither is traversal over dense graphs (depth is capped at 5).
- Milvus is verified on 2.5.10 with pymilvus 3.x. Not verified: `LIKE` escaping against data containing `%` or `_`, L2/IP scoring, multi-collection sources, and auth-enabled Milvus.
- The generic HTTP embedder sends one request per item, with a concurrency limit. Batch-capable containers should use a provider-specific adapter.
