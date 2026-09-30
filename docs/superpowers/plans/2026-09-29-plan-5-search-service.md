# Search Service (HTTP + SSE) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serve agentic searches over HTTP. `POST /v1/search` returns JSON and `POST /v1/search/stream` streams server-sent events. Operators define named profiles with per-profile limits, and the service exports the event contract (JSON Schema plus golden SSE fixtures) for clients.

**Architecture:**
- A new `agentic_search.server` package: config, request models and the lean projection, SSE framing, the FastAPI app, a demo service, schema export, and a CLI.
- It reuses `Harness.stream()` (Plan 4). A client disconnect is detected from the ASGI `receive` channel and closes the `SearchStream`, which cancels the search.
- A prerequisite fix makes `ImagePart.data` JSON-serialisable as standard base64.

**Tech Stack:** Python ≥3.11, FastAPI ≥0.115, uvicorn ≥0.30, Starlette, httpx, Pydantic v2, pytest (`asyncio_mode = "auto"`), uv, ruff.

**Spec:** `docs/superpowers/specs/2026-09-29-search-service-design.md` (it builds on `docs/superpowers/specs/2026-09-29-streaming-events-design.md`)

## Global Constraints

- **Auth.** `service.auth` must be explicit (`{type: none}` or `{type: api_key, keys_env: …}`). API keys are compared in constant time and registered for scrubbing. A bad or missing key gets `401` with `WWW-Authenticate: Bearer`. `/healthz` needs no auth.
- **No client-supplied paths.** Clients never send credentials, datastore definitions, or image paths or URIs. Images are inline base64 only; `ImageInput` forbids extra fields.
- **Lean results.** Without `include_trace` there is no `trace`, and without `include_content` there is no `hits[].hit.content`. `search_finished.result` carries the same lean result. All other events are their spec-1 JSON unchanged.
- **Status codes.** `400` request the service cannot run, `401` auth, `404` unknown profile, `422` invalid body, `429` over `max_concurrent_searches`, `503` profile unavailable. Failures during a stream are a `search_failed` event.
- **Streams.** SSE frames are `id: <seq>`, `event: <type>`, `data: <single-line JSON>`, and a blank line. After `keepalive_s` of silence the stream sends `: keep-alive`. A client disconnect cancels the search promptly, without waiting for the next write.
- **Scrubbing.** Error detail sent to clients is scrubbed. Unexpected exceptions give a generic `500` with no detail.
- **Dependencies.** New runtime dependencies only in the optional `server` extra: `fastapi>=0.115` and `uvicorn>=0.30`.
- **Commits.** Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- **Checks.** `uv run pytest -q` and `uv run ruff check` must be clean at the end of every task.

## File Structure

| File | Responsibility |
|---|---|
| `src/agentic_search/core/types.py` | `ImagePart` JSON: `data` as standard base64 |
| `src/agentic_search/server/config.py` | `AuthConfig`, `ServiceConfig`, `ProfileLimits`, `Profile`, `load_service`, `check_profiles` |
| `src/agentic_search/server/models.py` | request models, `build_query`, `resolve_budget`, `lean_result`, `render_event`, `RequestError` |
| `src/agentic_search/server/sse.py` | `format_sse`, `KEEPALIVE`, `sse_body` |
| `src/agentic_search/server/app.py` | `create_app` |
| `src/agentic_search/server/demo.py` | `demo_harness`, `demo_app` |
| `src/agentic_search/server/schema.py` | `event_schema`, `fixture_streams`, `render_files`, `export` |
| `src/agentic_search/server/cli.py` | `agentic-search serve` / `export-schema` |
| `src/agentic_search/server/__init__.py` | public exports |
| `schema/search-events.v1.{schema,fixtures}.json` | committed client contract |
| `tests/server/*` | tests |

---

### Task 1: Image bytes serialise as standard base64

**Files:**
- Modify: `src/agentic_search/core/types.py` (`ImagePart` and the pydantic import)
- Test: `tests/core/test_types.py` (append)

**Interfaces:**
- Produces: `ImagePart.model_dump_json()` writes `data` as standard base64 (`+`, `/`, `=`). JSON validation accepts standard or URL-safe base64. `model_dump()` (Python mode) still returns `bytes`.

- [ ] **Step 1: Write the failing test**

Append to `tests/core/test_types.py`:

```python


def test_image_bytes_serialise_as_standard_base64():
    import base64

    from agentic_search.core.types import ImagePart, Query, TextPart

    raw = bytes([0xFB, 0xFF, 0xBF, 0x00, 0x3E])  # not valid UTF-8; base64 uses '+' and '/'
    q = Query(content=[TextPart(text="x"), ImagePart(data=raw)])
    dumped = q.model_dump_json()
    assert '"data":"+/+/AD4="' in dumped
    assert Query.model_validate_json(dumped) == q
    urlsafe = base64.urlsafe_b64encode(raw).decode()
    assert ImagePart.model_validate_json(f'{{"data":"{urlsafe}"}}').data == raw
    assert q.model_dump()["content"][1]["data"] == raw  # python mode keeps bytes
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest -q tests/core/test_types.py`
Expected: `1 failed, 13 passed`. The new test fails with `PydanticSerializationError: ... invalid utf-8 sequence`.

- [ ] **Step 3: Implement**

In `src/agentic_search/core/types.py`:
- Add `import base64` above `import json`.
- Change the pydantic import to `from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_serializer, model_validator`.
- Replace the `ImagePart` class, including its existing `_exactly_one_source` validator, with:

```python
class ImagePart(BaseModel):
    """An image, inline (`data`) or by reference (`uri`). In JSON, `data` is standard base64."""

    model_config = ConfigDict(val_json_bytes="base64")

    kind: Literal["image"] = "image"
    uri: str | None = None
    data: bytes | None = None
    mime: str = "image/png"

    @field_serializer("data", when_used="json")
    def _data_as_base64(self, data: bytes | None) -> str | None:
        return None if data is None else base64.b64encode(data).decode("ascii")

    @model_validator(mode="after")
    def _exactly_one_source(self) -> ImagePart:
        if (self.uri is None) == (self.data is None):
            raise ValueError("ImagePart needs exactly one of uri or data")
        return self
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest -q && uv run ruff check`
Expected: `393 passed, 5 skipped, 84 deselected` and `All checks passed!`.

- [ ] **Step 5: Commit**

```bash
git add src/agentic_search/core/types.py tests/core/test_types.py
git commit -m "fix(types): serialise ImagePart.data as standard base64 in JSON

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Service configuration and request models

**Files:**
- Modify: `pyproject.toml` (the `server` extra and dev dependencies), then run `uv sync`, which updates `uv.lock`.
- Create: `src/agentic_search/server/__init__.py`, `src/agentic_search/server/config.py`, `src/agentic_search/server/models.py`, `tests/server/__init__.py` (empty), `tests/server/test_service_config.py`

**Interfaces:**
- Consumes: `agentic_search.config.build_harness`, `ConfigError`; `core.secrets.register_secret`; `core.types.Budget`, `ImagePart`, `Query`, `TextPart`; `core.result.SearchResult`; `events.SearchEvent`, `SearchFinished`.
- Produces:
  - `config.py`:
    - `AuthConfig(type: "none" | "api_key", keys_env)`, with `.keys() -> list[str]` (raises `ConfigError` if the variable is empty or missing).
    - `ServiceConfig(auth, cors_origins, max_concurrent_searches=16, default_profile, keepalive_s=15.0, max_image_bytes=10_000_000, max_images=4)`.
    - `ProfileLimits(max_budget: dict, allow_include_content=True)`.
    - The `Profile(harness, limits)` dataclass.
    - `load_service(path) -> (ServiceConfig, dict[str, Profile])`.
    - `check_profiles(config, profiles)`.
  - `models.py`:
    - The models `ImageInput`, `BudgetOverride`, `SearchRequest`, `StreamRequest`, and the exception `RequestError(ValueError)`.
    - `build_query(req, *, max_images, max_image_bytes) -> Query`.
    - `resolve_budget(base, override, ceilings) -> Budget`.
    - `lean_result(result, *, include_content, include_trace) -> dict`.
    - `render_event(event, *, include_content, include_trace) -> str`.
  - `agentic_search.server` exports `AuthConfig`, `Profile`, `ProfileLimits`, `ServiceConfig`, `load_service`. Task 4 adds `create_app`.

- [ ] **Step 1: Add the dependencies**

In `pyproject.toml`:
- Add this line after `azure = ["azure-identity>=1.17"]` in `[project.optional-dependencies]`:
  ```toml
  server = ["fastapi>=0.115", "uvicorn>=0.30"]
  ```
- Add these two lines at the end of the `dev` list in `[dependency-groups]`:
  ```toml
    "fastapi>=0.115",
    "uvicorn>=0.30",
  ```

Run: `uv sync`

- [ ] **Step 2: Write the failing test**

Create `tests/server/__init__.py` (empty) and `tests/server/test_service_config.py`:

```python
import textwrap

import pytest

from agentic_search.config import ConfigError
from agentic_search.core.types import Budget
from agentic_search.server import ProfileLimits, load_service
from agentic_search.server.models import BudgetOverride, resolve_budget

PROFILE = """
    backends: [{name: notes, type: files, root: docs, glob: "*.txt"}]
    driver: {type: openai_compat, model: local, base_url: "http://localhost:9/v1"}
"""


def write(tmp_path, text):
    (tmp_path / "docs").mkdir(exist_ok=True)
    (tmp_path / "docs" / "a.txt").write_text("aspirin treats headache")
    path = tmp_path / "service.yaml"
    path.write_text(textwrap.dedent(text))
    return path


def test_load_service(tmp_path):
    path = write(tmp_path, f"""
        service:
          auth: {{type: none}}
          default_profile: general
          max_concurrent_searches: 3
        profiles:
          general:{textwrap.indent(textwrap.dedent(PROFILE), "            ")}
          strict:{textwrap.indent(textwrap.dedent(PROFILE), "            ")}
            limits: {{max_budget: {{max_turns: 2}}, allow_include_content: false}}
    """)
    config, profiles = load_service(path)
    assert config.default_profile == "general" and config.max_concurrent_searches == 3
    assert set(profiles) == {"general", "strict"}
    assert profiles["strict"].limits.max_budget == {"max_turns": 2}
    assert not profiles["strict"].limits.allow_include_content
    assert profiles["general"].harness.backends["notes"]


@pytest.mark.parametrize("text, message", [
    ("service: {auth: {type: none}}\n", "profiles"),
    ("profiles: {a: {}}\n", "service"),
    ("service: {auth: {type: none}, bogus: 1}\nprofiles: {a: {}}\n", "service"),
])
def test_load_service_errors(tmp_path, text, message):
    with pytest.raises(ConfigError, match=message):
        load_service(write(tmp_path, text))


def test_profile_limits_validation():
    with pytest.raises(ValueError):
        ProfileLimits(max_budget={"max_wishes": 3})
    with pytest.raises(ValueError):
        ProfileLimits(max_budget={"max_turns": 0})


def test_resolve_budget():
    base = Budget(max_turns=4, max_cost_usd=None)
    got = resolve_budget(base, BudgetOverride(max_turns=9, max_tool_calls=5),
                         {"max_turns": 3, "max_cost_usd": 0.5})
    assert (got.max_turns, got.max_tool_calls, got.max_cost_usd) == (3, 5, 0.5)
    assert resolve_budget(base, None, {}) == base
```

- [ ] **Step 3: Run it to verify it fails**

Run: `uv run pytest -q tests/server/test_service_config.py`
Expected: a collection error, `ModuleNotFoundError: No module named 'agentic_search.server'`.

- [ ] **Step 4: Create `src/agentic_search/server/config.py`**

```python
"""Service configuration: auth, CORS, capacity, and named harness profiles with limits."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from agentic_search.config import ConfigError, build_harness
from agentic_search.core.harness import Harness
from agentic_search.core.secrets import register_secret
from agentic_search.core.types import Budget


class AuthConfig(BaseModel):
    """`api_key`: clients send `Authorization: Bearer <key>` or `X-API-Key: <key>`; valid keys are
    read from the comma-separated environment variable `keys_env`. `none` must be explicit."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["none", "api_key"]
    keys_env: str | None = None

    @model_validator(mode="after")
    def _keys_env_for_api_key(self) -> AuthConfig:
        if self.type == "api_key" and not self.keys_env:
            raise ValueError("auth type api_key needs keys_env")
        return self

    def keys(self) -> list[str]:
        if self.type == "none":
            return []
        raw = os.environ.get(self.keys_env or "", "")
        keys = [k.strip() for k in raw.split(",") if k.strip()]
        if not keys:
            raise ConfigError(f"environment variable {self.keys_env!r} holds no API keys")
        for k in keys:
            register_secret(k)
        return keys


class ServiceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    auth: AuthConfig
    cors_origins: list[str] = Field(default_factory=list)
    max_concurrent_searches: int = Field(default=16, ge=1)
    default_profile: str | None = None
    keepalive_s: float = Field(default=15.0, gt=0)
    max_image_bytes: int = Field(default=10_000_000, ge=1)
    max_images: int = Field(default=4, ge=0)


class ProfileLimits(BaseModel):
    """`max_budget`: per-field ceilings on the budget a request may ask for (unset = no ceiling)."""

    model_config = ConfigDict(extra="forbid")

    max_budget: dict[str, float | int] = Field(default_factory=dict)
    allow_include_content: bool = True

    @model_validator(mode="after")
    def _known_budget_fields(self) -> ProfileLimits:
        unknown = set(self.max_budget) - set(Budget.model_fields)
        if unknown:
            raise ValueError(f"unknown max_budget fields: {sorted(unknown)}")
        Budget(**self.max_budget)  # ceilings must themselves be a valid budget
        if any(v <= 0 for v in self.max_budget.values()):
            raise ValueError("max_budget ceilings must be positive")
        return self


@dataclass
class Profile:
    harness: Harness
    limits: ProfileLimits = field(default_factory=ProfileLimits)


def load_service(path: str | Path) -> tuple[ServiceConfig, dict[str, Profile]]:
    """Read a service YAML: a `service:` block and `profiles:` of harness configs, each with an
    optional `limits:` block. Every profile's harness is built here (not yet set up)."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw.get("profiles"), dict) or not raw["profiles"]:
        raise ConfigError("service config needs a non-empty `profiles:` mapping")
    try:
        config = ServiceConfig(**(raw.get("service") or {}))
    except ValueError as exc:
        raise ConfigError(f"invalid `service:` block: {exc}") from exc
    profiles: dict[str, Profile] = {}
    for name, cfg in raw["profiles"].items():
        cfg = dict(cfg or {})
        try:
            limits = ProfileLimits(**(cfg.pop("limits", None) or {}))
        except ValueError as exc:
            raise ConfigError(f"profile {name!r} has invalid limits: {exc}") from exc
        profiles[str(name)] = Profile(harness=build_harness(cfg, base_dir=path.parent), limits=limits)
    return config, profiles


def check_profiles(config: ServiceConfig, profiles: dict[str, Any]) -> None:
    if not profiles:
        raise ConfigError("the service needs at least one profile")
    if config.default_profile is not None and config.default_profile not in profiles:
        raise ConfigError(f"default_profile {config.default_profile!r} is not a profile")
```

- [ ] **Step 5: Create `src/agentic_search/server/models.py`**

```python
"""HTTP request models and the lean response projection."""

from __future__ import annotations

import base64
import binascii
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from agentic_search.core.result import SearchResult
from agentic_search.core.types import Budget, ImagePart, Query, TextPart
from agentic_search.events import SearchEvent, SearchFinished


class ImageInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data: str = Field(description="Standard or URL-safe base64 image bytes.")
    mime: str = Field(default="image/png", pattern=r"^image/[A-Za-z0-9.+-]+$")


class BudgetOverride(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_turns: int | None = Field(default=None, ge=1)
    max_tool_calls: int | None = Field(default=None, ge=1)
    max_tokens: int | None = Field(default=None, ge=1)
    max_cost_usd: float | None = Field(default=None, gt=0)
    max_seconds: float | None = Field(default=None, gt=0)


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile: str | None = None
    question: str = Field(min_length=1, max_length=10_000)
    images: list[ImageInput] = Field(default_factory=list)
    sources: list[str] | None = None
    mode: Literal["retrieval", "harness", "model"] | None = None
    top_k: int = Field(default=20, ge=1, le=1000)
    budget: BudgetOverride | None = None
    include_content: bool = False
    include_trace: bool = False


class StreamRequest(SearchRequest):
    snapshot_k: int = Field(default=10, ge=0, le=100)


class RequestError(ValueError):
    """A well-formed request the service still cannot run (maps to HTTP 400)."""


def build_query(req: SearchRequest, *, max_images: int, max_image_bytes: int) -> Query:
    """Question text plus inline images. Images are always inline bytes: the service never
    accepts a `uri`, so a client cannot make the server read its own files."""
    if len(req.images) > max_images:
        raise RequestError(f"at most {max_images} images per request")
    parts: list[Any] = [TextPart(text=req.question)]
    for i, img in enumerate(req.images):
        try:
            data = base64.b64decode(img.data.replace("-", "+").replace("_", "/"), validate=True)
        except (binascii.Error, ValueError) as exc:
            raise RequestError(f"images[{i}].data is not valid base64") from exc
        if len(data) > max_image_bytes:
            raise RequestError(f"images[{i}] is larger than {max_image_bytes} bytes")
        parts.append(ImagePart(data=data, mime=img.mime))
    return Query(content=parts)


def resolve_budget(base: Budget, override: BudgetOverride | None,
                   ceilings: dict[str, float | int]) -> Budget:
    """The profile's budget, updated with the request's fields, then capped by the profile's
    ceilings. An unlimited (None) field under a ceiling becomes the ceiling."""
    fields = base.model_dump()
    if override is not None:
        fields.update(override.model_dump(exclude_none=True))
    for name, ceiling in ceilings.items():
        value = fields.get(name)
        if value is None or value > ceiling:
            fields[name] = ceiling
    return Budget(**fields)


def lean_result(result: SearchResult, *, include_content: bool, include_trace: bool) -> dict[str, Any]:
    """The service's JSON view of a result: no trace unless asked, no hit content unless asked."""
    data = result.model_dump(mode="json", exclude=None if include_trace else {"trace"})
    if not include_content:
        for ranked in data["hits"]:
            ranked["hit"].pop("content", None)
    return data


def render_event(event: SearchEvent, *, include_content: bool, include_trace: bool) -> str:
    """One event as a single-line JSON string; `search_finished` carries the lean result."""
    if isinstance(event, SearchFinished):
        data = event.model_dump(mode="json", exclude={"result"})
        data["result"] = lean_result(event.result, include_content=include_content,
                                     include_trace=include_trace)
        return json.dumps(data, separators=(",", ":"))
    return event.model_dump_json()
```

- [ ] **Step 6: Create `src/agentic_search/server/__init__.py`**

```python
"""HTTP service for agentic search (install the `server` extra: fastapi, uvicorn)."""

from agentic_search.server.config import (
           AuthConfig,
           Profile,
           ProfileLimits,
           ServiceConfig,
           load_service,
)

__all__ = ["AuthConfig", "Profile", "ProfileLimits", "ServiceConfig", "load_service"]
```

- [ ] **Step 7: Run the tests**

Run: `uv run pytest -q tests/server/test_service_config.py`
Expected: `6 passed`.

Run: `uv run pytest -q && uv run ruff check`
Expected: `399 passed, 5 skipped, 84 deselected` and `All checks passed!`.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml uv.lock src/agentic_search/server tests/server
git commit -m "feat(server): service config, profiles with limits, request models and lean projection

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: SSE framing, keep-alives and disconnect handling

**Files:**
- Create: `src/agentic_search/server/sse.py`, `tests/server/conftest.py`, `tests/server/test_sse.py`

**Interfaces:**
- Consumes: `core.harness.SearchStream` (`__aiter__`, `aclose`), `events.SearchEvent`.
- Produces:
  - `format_sse(event_type, data, event_id=None) -> bytes`.
  - `KEEPALIVE = b": keep-alive\n\n"`.
  - `sse_body(stream, render, *, keepalive_s, receive=None) -> AsyncIterator[bytes]`. It frames every event, sends a keep-alive after `keepalive_s` of silence without cancelling the pending event, stops when `receive()` yields `http.disconnect`, and always closes the stream in `finally`, which cancels the search.
  - The test fixtures in `tests/server/conftest.py`: `lex`, `make_harness`, `service`, `client`, `parse_sse`, and the `app_factory` fixture, which imports `create_app` lazily so this file works before Task 4.

- [ ] **Step 1: Write the test fixtures and the failing test**

Create `tests/server/conftest.py`:

```python
import json

import httpx
import pytest

from agentic_search import Harness
from agentic_search.server.config import AuthConfig, Profile, ProfileLimits, ServiceConfig
from agentic_search.testing import KeywordJudge, ScriptedDriver, call


def lex(text, **kw):
    return call("lexical_search", source="docs", text=text, **kw)


def make_harness(docs_backend, turns=None, **kw):
    driver = ScriptedDriver(turns if turns is not None else [[lex("headache")]])
    kw.setdefault("analyzer", KeywordJudge(["headache"]))
    return Harness([docs_backend], driver, embedders=[docs_backend.embedder], **kw)


def service(**kw):
    kw.setdefault("auth", AuthConfig(type="none"))
    return ServiceConfig(**kw)


def client(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


def parse_sse(text):
    """Split an SSE body into (fields, comments): fields are dicts with id/event/data(JSON)."""
    frames, comments = [], []
    for block in text.split("\n\n"):
        if not block.strip():
            continue
        fields, data = {}, []
        for line in block.split("\n"):
            if line.startswith(":"):
                comments.append(line)
            elif line.startswith("data: "):
                data.append(line[6:])
            elif ": " in line:
                k, v = line.split(": ", 1)
                fields[k] = v
        if data:
            fields["data"] = json.loads("\n".join(data))
            frames.append(fields)
    return frames, comments


@pytest.fixture
def app_factory(docs_backend):
    def build(turns=None, limits=None, config=None, **harness_kw):
        from agentic_search.server.app import create_app

        h = make_harness(docs_backend, turns, **harness_kw)
        return create_app(config or service(),
                          {"demo": Profile(harness=h, limits=limits or ProfileLimits())})
    return build
```

Create `tests/server/test_sse.py`:

```python
import asyncio

import pytest

from agentic_search.server.sse import KEEPALIVE, format_sse, sse_body

from .conftest import lex, make_harness


def test_format_sse_frames_and_splits_lines():
    assert format_sse("phase_started", '{"a":1}', 4) == b'id: 4\nevent: phase_started\ndata: {"a":1}\n\n'
    assert format_sse("x", "l1\nl2") == b"event: x\ndata: l1\ndata: l2\n\n"


def slow_backend(docs_backend, entered, cancelled, delay=60):
    async def slow(op):
        entered.set()
        try:
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return []

    docs_backend.execute = slow


async def test_keepalive_during_silence(docs_backend):
    entered, cancelled = asyncio.Event(), asyncio.Event()
    slow_backend(docs_backend, entered, cancelled, delay=0.3)
    h = make_harness(docs_backend)
    chunks = [c async for c in sse_body(h.stream("q"), lambda e: e.model_dump_json(),
                                        keepalive_s=0.05)]
    assert chunks.count(KEEPALIVE) >= 3
    assert chunks[-1].startswith(b"id: ") and b"event: search_finished" in chunks[-1]


async def test_client_disconnect_cancels_search(docs_backend):
    entered, cancelled = asyncio.Event(), asyncio.Event()
    slow_backend(docs_backend, entered, cancelled)
    h = make_harness(docs_backend)
    gone = asyncio.Event()

    async def receive():
        await gone.wait()
        return {"type": "http.disconnect"}

    async def consume():
        chunks = []
        async for chunk in sse_body(h.stream("q"), lambda e: e.model_dump_json(),
                                    keepalive_s=10, receive=receive):
            chunks.append(chunk)
            if b"tool_call_started" in chunk:
                await entered.wait()
                gone.set()
        return chunks

    chunks = await asyncio.wait_for(consume(), 5)
    assert cancelled.is_set()
    assert not any(b"search_finished" in c for c in chunks)


async def test_closing_body_cancels_search(docs_backend):
    entered, cancelled = asyncio.Event(), asyncio.Event()
    slow_backend(docs_backend, entered, cancelled)
    h = make_harness(docs_backend)
    body = sse_body(h.stream("q"), lambda e: e.model_dump_json(), keepalive_s=10)

    async def consume():
        async for chunk in body:
            if b"tool_call_started" in chunk:
                await entered.wait()
                break
        await body.aclose()

    await asyncio.wait_for(consume(), 5)
    assert cancelled.is_set()


async def test_body_ends_after_terminal_event(docs_backend):
    h = make_harness(docs_backend, turns=[[lex("headache")]])
    chunks = [c async for c in sse_body(h.stream("q", mode="retrieval"),
                                        lambda e: e.model_dump_json(), keepalive_s=10)]
    assert b"event: search_finished" in chunks[-1]


@pytest.mark.parametrize("keepalive", [0.01])
async def test_keepalive_does_not_drop_events(docs_backend, keepalive):
    h = make_harness(docs_backend, turns=[[lex("headache"), lex("fever")]])
    chunks = [c async for c in sse_body(h.stream("q", mode="retrieval"),
                                        lambda e: e.model_dump_json(), keepalive_s=keepalive)]
    ids = [int(c.split(b"\n")[0][4:]) for c in chunks if c != KEEPALIVE]
    assert ids == list(range(len(ids)))
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest -q tests/server/test_sse.py`
Expected: a collection error, `ModuleNotFoundError: No module named 'agentic_search.server.sse'`.

- [ ] **Step 3: Create `src/agentic_search/server/sse.py`**

```python
"""Server-sent events framing, keep-alives and client-disconnect handling for one search."""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Awaitable, Callable

from agentic_search.core.harness import SearchStream
from agentic_search.events import SearchEvent

KEEPALIVE = b": keep-alive\n\n"


def format_sse(event_type: str, data: str, event_id: int | None = None) -> bytes:
    lines = [] if event_id is None else [f"id: {event_id}"]
    lines.append(f"event: {event_type}")
    lines.extend(f"data: {line}" for line in data.split("\n"))
    return ("\n".join(lines) + "\n\n").encode()


async def _wait_for_disconnect(receive: Callable[[], Awaitable[dict[str, Any]]]) -> None:
    while True:
        message = await receive()
        if message.get("type") == "http.disconnect":
            return


async def sse_body(stream: SearchStream, render: Callable[[SearchEvent], str], *,
                   keepalive_s: float,
                   receive: Callable[[], Awaitable[dict[str, Any]]] | None = None,
                   ) -> AsyncIterator[bytes]:
    """Frame every event of `stream` as SSE, sending a keep-alive comment after `keepalive_s`
    of silence. If the client disconnects (seen via `receive`), the search is cancelled."""
    iterator = stream.__aiter__()
    pending: asyncio.Future[SearchEvent] | None = None
    watcher = asyncio.ensure_future(_wait_for_disconnect(receive)) if receive else None
    try:
        while True:
            if pending is None:
                pending = asyncio.ensure_future(iterator.__anext__())
            waiting = {pending} if watcher is None else {pending, watcher}
            done, _ = await asyncio.wait(waiting, timeout=keepalive_s,
                                         return_when=asyncio.FIRST_COMPLETED)
            if watcher is not None and watcher in done:
                return  # client went away: the finally block cancels the search
            if pending not in done:
                yield KEEPALIVE
                continue
            finished, pending = pending, None
            try:
                event = finished.result()
            except StopAsyncIteration:
                return
            yield format_sse(event.type, render(event), event.seq)
    finally:
        if watcher is not None:
            watcher.cancel()
        if pending is not None:
            pending.cancel()
            # wait() never raises the future's own error; our own cancellation still propagates
            await asyncio.wait({pending})
            if not pending.cancelled():
                pending.exception()  # mark retrieved; the stream is being abandoned anyway
        await stream.aclose()
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest -q tests/server/test_sse.py`
Expected: `6 passed`, in about a second. A disconnect or close test that takes about 5 seconds and then fails means the search was not cancelled.

Run: `uv run pytest -q && uv run ruff check`
Expected: `405 passed, 5 skipped, 84 deselected` and `All checks passed!`.

- [ ] **Step 5: Commit**

```bash
git add src/agentic_search/server/sse.py tests/server/conftest.py tests/server/test_sse.py
git commit -m "feat(server): SSE framing with keep-alives; disconnect or close cancels the search

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: The FastAPI application

**Files:**
- Create: `src/agentic_search/server/app.py`, `tests/server/test_app.py`, `tests/server/test_live.py`
- Modify: `src/agentic_search/server/__init__.py` (add `create_app`)

**Interfaces:**
- Consumes: everything from Tasks 2 and 3; `Harness.setup/search/stream/close`, `HarnessError`; `core.secrets.scrub`.
- Produces:
  - `create_app(config: ServiceConfig, profiles: dict[str, Profile | Harness]) -> FastAPI` with routes `GET /healthz`, `GET /v1/profiles`, `POST /v1/search` and `POST /v1/search/stream`.
  - Profiles are set up in the lifespan and lazily per request. They are closed at shutdown.
  - A non-blocking capacity gate. For streams, the slot is released by a `StreamingResponse` subclass however the response ends.

- [ ] **Step 1: Write the failing tests**

Create `tests/server/test_app.py`:

```python
import asyncio
import base64

import pytest

from agentic_search.config import ConfigError
from agentic_search.server import ProfileLimits, create_app
from agentic_search.server.config import AuthConfig

from .conftest import client, lex, make_harness, parse_sse, service


async def test_search_returns_lean_result(app_factory):
    async with client(app_factory()) as c:
        r = await c.post("/v1/search", json={"question": "what treats headache?"})
    assert r.status_code == 200
    body = r.json()
    assert {h["hit"]["doc_id"] for h in body["hits"]} == {"d1", "d4"}
    assert "trace" not in body and all("content" not in h["hit"] for h in body["hits"])
    assert body["stop_reason"] == "no_plan" and body["mode"] == "harness"


async def test_include_content_and_trace(app_factory):
    async with client(app_factory()) as c:
        r = await c.post("/v1/search", json={"question": "q", "include_content": True,
                                             "include_trace": True})
    body = r.json()
    assert body["trace"]["events"] and body["hits"][0]["hit"]["content"][0]["text"]


async def test_profile_can_forbid_content(app_factory):
    app = app_factory(limits=ProfileLimits(allow_include_content=False))
    async with client(app) as c:
        r = await c.post("/v1/search", json={"question": "q", "include_content": True})
    assert r.status_code == 400 and "include_content" in r.json()["detail"]


async def test_profile_resolution(docs_backend):
    two = {"a": make_harness(docs_backend), "b": make_harness(docs_backend)}
    async with client(create_app(service(), two)) as c:
        assert (await c.post("/v1/search", json={"question": "q"})).status_code == 400
        assert (await c.post("/v1/search", json={"question": "q", "profile": "zzz"})).status_code == 404
        assert (await c.post("/v1/search", json={"question": "q", "profile": "b"})).status_code == 200
    with_default = {"a": make_harness(docs_backend), "b": make_harness(docs_backend)}
    async with client(create_app(service(default_profile="b"), with_default)) as c:
        assert (await c.post("/v1/search", json={"question": "q"})).status_code == 200
    with pytest.raises(ConfigError):
        create_app(service(default_profile="nope"), {"a": make_harness(docs_backend)})


async def test_api_key_auth(app_factory, monkeypatch):
    monkeypatch.setenv("SEARCH_KEYS", "k-one, k-two")
    app = app_factory(config=service(auth=AuthConfig(type="api_key", keys_env="SEARCH_KEYS")))
    async with client(app) as c:
        assert (await c.post("/v1/search", json={"question": "q"})).status_code == 401
        bad = await c.post("/v1/search", json={"question": "q"},
                           headers={"Authorization": "Bearer nope"})
        assert bad.status_code == 401 and bad.headers["www-authenticate"] == "Bearer"
        ok1 = await c.post("/v1/search", json={"question": "q"},
                           headers={"Authorization": "Bearer k-two"})
        ok2 = await c.get("/v1/profiles", headers={"X-API-Key": "k-one"})
        assert ok1.status_code == 200 and ok2.status_code == 200
        assert (await c.get("/healthz")).status_code == 200  # no auth on liveness
    monkeypatch.delenv("SEARCH_KEYS")
    with pytest.raises(ConfigError):
        app_factory(config=service(auth=AuthConfig(type="api_key", keys_env="SEARCH_KEYS")))


def test_auth_type_must_be_explicit():
    with pytest.raises(ValueError):
        service(auth=None)
    with pytest.raises(ValueError):
        AuthConfig(type="api_key")


async def test_profiles_and_health(app_factory, docs_backend):
    async with client(app_factory()) as c:
        body = (await c.get("/v1/profiles")).json()
        health = (await c.get("/healthz")).json()
    [p] = body["profiles"]
    assert p["name"] == "demo" and p["available"] and p["mode"] == "harness"
    [src] = p["sources"]
    assert src["name"] == "docs" and "lexical" in src["capabilities"]
    assert health["status"] == "ok"

    async def broken(*a, **kw):
        raise RuntimeError("index offline")

    docs_backend.discover = broken
    app = create_app(service(), {"down": make_harness(docs_backend)})
    async with client(app) as c:
        [p] = (await c.get("/v1/profiles")).json()["profiles"]
        search = await c.post("/v1/search", json={"question": "q"})
        health = (await c.get("/healthz")).json()
    assert not p["available"] and "index offline" in p["error"]
    assert search.status_code == 503 and health["status"] == "degraded"


async def test_budget_is_clamped_by_profile_limits(app_factory):
    turns = [[lex("headache")], [lex("fever")], [lex("pain")]]
    from agentic_search.models.base import Action
    from agentic_search.testing import ScriptedController

    app = app_factory(turns=turns, limits=ProfileLimits(max_budget={"max_turns": 1}),
                      controller=ScriptedController([Action.CONTINUE, Action.CONTINUE]))
    async with client(app) as c:
        r = await c.post("/v1/search", json={"question": "q", "budget": {"max_turns": 9}})
    assert r.json()["usage"]["turns"] == 1 and r.json()["stop_reason"] == "budget_turns"


async def test_images_are_decoded_inline_only(app_factory, docs_backend):
    app = app_factory()
    png = b"\x89PNG\r\n\x1a\n\x00\xff"
    async with client(app) as c:
        ok = await c.post("/v1/search", json={"question": "q", "images": [
            {"data": base64.b64encode(png).decode(), "mime": "image/png"}]})
        bad = await c.post("/v1/search", json={"question": "q", "images": [{"data": "@@@"}]})
        uri = await c.post("/v1/search", json={"question": "q", "images": [
            {"uri": "file:///etc/passwd"}]})
        many = await c.post("/v1/search", json={"question": "q", "images": [
            {"data": "AA=="}] * 5})
    assert ok.status_code == 200
    [img] = ok.json()["question"]["content"][1:]
    assert base64.b64decode(img["data"]) == png
    assert bad.status_code == 400 and "base64" in bad.json()["detail"]
    assert uri.status_code == 422
    assert many.status_code == 400


async def test_harness_error_maps_to_400(app_factory):
    async with client(app_factory()) as c:
        r = await c.post("/v1/search", json={"question": "q", "sources": ["nope"]})
    assert r.status_code == 400 and "nope" in r.json()["detail"]


async def test_capacity_limit_returns_429(docs_backend):
    release = asyncio.Event()

    async def slow(op):
        await release.wait()
        return []

    docs_backend.execute = slow
    app = create_app(service(max_concurrent_searches=1), {"demo": make_harness(docs_backend)})
    async with client(app) as c:
        first = asyncio.create_task(c.post("/v1/search", json={"question": "q"}))
        await asyncio.sleep(0.05)
        second = await c.post("/v1/search", json={"question": "q"})
        stream = await c.post("/v1/search/stream", json={"question": "q"})
        release.set()
        assert (await first).status_code == 200
        again = await c.post("/v1/search", json={"question": "q"})
    assert second.status_code == 429 and stream.status_code == 429
    assert again.status_code == 200


async def test_stream_is_sse_and_ends_with_lean_finish(app_factory):
    async with client(app_factory()) as c:
        r = await c.post("/v1/search/stream", json={"question": "q", "snapshot_k": 3})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    assert r.headers["cache-control"] == "no-cache" and r.headers["x-search-profile"] == "demo"
    frames, _ = parse_sse(r.text)
    assert [f["event"] for f in frames][0] == "search_started"
    assert [int(f["id"]) for f in frames] == list(range(len(frames)))
    assert all(f["event"] == f["data"]["type"] for f in frames)
    last = frames[-1]
    assert last["event"] == "search_finished"
    assert "trace" not in last["data"]["result"]
    assert all("content" not in h["hit"] for h in last["data"]["result"]["hits"])
    snaps = [f for f in frames if f["event"] == "results_updated"]
    assert snaps and len(snaps[-1]["data"]["hits"]) <= 3


async def test_stream_failures(app_factory):
    async with client(app_factory()) as c:
        failed = await c.post("/v1/search/stream", json={"question": "q", "sources": ["nope"]})
        invalid = await c.post("/v1/search/stream", json={"question": "q", "mode": "psychic"})
        negative = await c.post("/v1/search/stream", json={"question": "q", "snapshot_k": -1})
    frames, _ = parse_sse(failed.text)
    assert failed.status_code == 200 and [f["event"] for f in frames] == ["search_failed"]
    assert invalid.status_code == 422 and negative.status_code == 422


async def test_stream_releases_slot(docs_backend):
    app = create_app(service(max_concurrent_searches=1), {"demo": make_harness(docs_backend)})
    async with client(app) as c:
        for _ in range(3):
            assert (await c.post("/v1/search/stream", json={"question": "q"})).status_code == 200
```

Create `tests/server/test_live.py`:

```python
"""The service under a real uvicorn server: HTTP streaming and disconnect-cancels-search."""

import asyncio

import httpx
import uvicorn

from agentic_search.server import create_app

from .conftest import make_harness, parse_sse, service


async def serve(app):
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning",
                                           lifespan="on"))
    task = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.01)
    port = server.servers[0].sockets[0].getsockname()[1]
    return server, task, f"http://127.0.0.1:{port}"


async def test_live_stream_and_search(docs_backend):
    server, task, url = await serve(create_app(service(), {"demo": make_harness(docs_backend)}))
    try:
        async with httpx.AsyncClient(base_url=url) as c:
            async with c.stream("POST", "/v1/search/stream", json={"question": "q"}) as r:
                text = "".join([chunk async for chunk in r.aiter_text()])
            frames, _ = parse_sse(text)
            assert frames[0]["event"] == "search_started"
            assert frames[-1]["event"] == "search_finished"
            assert (await c.get("/healthz")).json()["status"] == "ok"
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, 10)


async def test_live_disconnect_cancels_backend_call(docs_backend):
    entered, cancelled = asyncio.Event(), asyncio.Event()

    async def slow(op):
        entered.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return []

    docs_backend.execute = slow
    app = create_app(service(keepalive_s=30), {"demo": make_harness(docs_backend)})
    server, task, url = await serve(app)
    try:
        async with httpx.AsyncClient(base_url=url) as c:
            async with c.stream("POST", "/v1/search/stream", json={"question": "q"}) as r:
                async for line in r.aiter_lines():
                    if line == "event: tool_call_started":
                        break
            await asyncio.wait_for(entered.wait(), 5)
        await asyncio.wait_for(cancelled.wait(), 5)
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, 10)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest -q tests/server/test_app.py tests/server/test_live.py`
Expected: 2 collection errors (`ImportError: cannot import name 'create_app' from 'agentic_search.server'`).

- [ ] **Step 3: Create `src/agentic_search/server/app.py`**

```python
"""The FastAPI application: profiles, one-shot search, and SSE-streamed search."""

from __future__ import annotations

import asyncio
import hmac
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Callable

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse

from agentic_search import __version__
from agentic_search.core.harness import Harness, HarnessError
from agentic_search.core.secrets import scrub
from agentic_search.server.config import Profile, ServiceConfig, check_profiles
from agentic_search.server.models import (
    RequestError,
    SearchRequest,
    StreamRequest,
    build_query,
    lean_result,
    render_event,
    resolve_budget,
)
from agentic_search.server.sse import sse_body


class _Gate:
    """Non-blocking concurrency limit: a search either gets a slot now or the caller gets 429."""

    def __init__(self, limit: int):
        self.limit, self.active = limit, 0

    def try_acquire(self) -> bool:
        if self.active >= self.limit:
            return False
        self.active += 1
        return True

    def release(self) -> None:
        self.active -= 1


class _GatedStreamingResponse(StreamingResponse):
    """Releases the search slot however the response ends, even if the body never starts."""

    def __init__(self, *args: Any, release: Callable[[], None], **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._release = release

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            self._release()


def create_app(config: ServiceConfig, profiles: dict[str, Profile | Harness]) -> FastAPI:
    """Build the service. Profiles are set up at startup (and lazily on first use, so the app
    also works without lifespan events) and closed at shutdown."""
    check_profiles(config, profiles)
    profs = {n: p if isinstance(p, Profile) else Profile(harness=p) for n, p in profiles.items()}
    keys = config.auth.keys()
    gate = _Gate(config.max_concurrent_searches)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        await asyncio.gather(*(_setup(p) for p in profs.values()), return_exceptions=True)
        yield
        await asyncio.gather(*(p.harness.close() for p in profs.values()), return_exceptions=True)

    app = FastAPI(title="agentic-search", version=__version__, lifespan=lifespan)
    if config.cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=config.cors_origins,
                           allow_methods=["GET", "POST"],
                           allow_headers=["Authorization", "Content-Type", "X-API-Key"])

    async def require_key(authorization: str | None = Header(default=None),
                          x_api_key: str | None = Header(default=None)) -> None:
        if config.auth.type == "none":
            return
        token = x_api_key
        if authorization and authorization.lower().startswith("bearer "):
            token = authorization[7:].strip()
        if not token or not any(hmac.compare_digest(token, k) for k in keys):
            raise HTTPException(401, "missing or invalid API key",
                                headers={"WWW-Authenticate": "Bearer"})

    def resolve(name: str | None) -> tuple[str, Profile]:
        if name is None:
            if config.default_profile is not None:
                name = config.default_profile
            elif len(profs) == 1:
                name = next(iter(profs))
            else:
                raise HTTPException(400, f"profile is required; one of {sorted(profs)}")
        if name not in profs:
            raise HTTPException(404, f"unknown profile {name!r}")
        return name, profs[name]

    async def _setup(profile: Profile) -> None:
        await profile.harness.setup()

    async def ready(name: str, profile: Profile) -> None:
        try:
            await _setup(profile)
        except HarnessError as exc:
            raise HTTPException(503, f"profile {name!r} is unavailable: {scrub(str(exc))}") from exc

    def options(req: SearchRequest, profile: Profile) -> dict[str, Any]:
        if req.include_content and not profile.limits.allow_include_content:
            raise HTTPException(400, "this profile does not allow include_content")
        try:
            query = build_query(req, max_images=config.max_images,
                                max_image_bytes=config.max_image_bytes)
        except RequestError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"sources": req.sources, "top_k": req.top_k, "mode": req.mode,
                "budget": resolve_budget(profile.harness.budget, req.budget,
                                         profile.limits.max_budget),
                "question": query}

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        ok = all(p.harness.manifests for p in profs.values())
        return {"status": "ok" if ok else "degraded", "version": __version__}

    @app.get("/v1/profiles", dependencies=[Depends(require_key)])
    async def list_profiles() -> dict[str, Any]:
        out = []
        for name, p in profs.items():
            error = None
            try:
                await _setup(p)
            except HarnessError as exc:
                error = scrub(str(exc))
            h = p.harness
            out.append({
                "name": name,
                "default": name == config.default_profile,
                "available": error is None,
                "error": error,
                "mode": h.mode,
                "budget": h.budget.model_dump(mode="json"),
                "limits": p.limits.model_dump(mode="json"),
                "sources": [{"name": m.source, "backend_type": m.backend_type,
                             "capabilities": sorted(c.value for c in m.capabilities),
                             "collections": [c.name for c in m.collections],
                             "description": m.description}
                            for m in sorted(h.manifests.values(), key=lambda m: m.source)],
                "setup_errors": dict(h.setup_errors),
            })
        return {"profiles": out}

    @app.post("/v1/search", dependencies=[Depends(require_key)])
    async def search(req: SearchRequest) -> dict[str, Any]:
        name, profile = resolve(req.profile)
        opts = options(req, profile)
        await ready(name, profile)
        if not gate.try_acquire():
            raise HTTPException(429, "too many concurrent searches")
        try:
            result = await profile.harness.search(opts.pop("question"), **opts)
        except HarnessError as exc:
            raise HTTPException(400, scrub(str(exc))) from exc
        finally:
            gate.release()
        return lean_result(result, include_content=req.include_content,
                           include_trace=req.include_trace)

    @app.post("/v1/search/stream", dependencies=[Depends(require_key)])
    async def search_stream(req: StreamRequest, request: Request) -> StreamingResponse:
        name, profile = resolve(req.profile)
        opts = options(req, profile)
        await ready(name, profile)
        try:
            stream = profile.harness.stream(opts.pop("question"), snapshot_k=req.snapshot_k,
                                            include_content=req.include_content, **opts)
        except HarnessError as exc:
            raise HTTPException(400, scrub(str(exc))) from exc
        if not gate.try_acquire():
            raise HTTPException(429, "too many concurrent searches")

        def render(event: Any) -> str:
            return render_event(event, include_content=req.include_content,
                                include_trace=req.include_trace)

        body = sse_body(stream, render, keepalive_s=config.keepalive_s, receive=request.receive)
        return _GatedStreamingResponse(body, release=gate.release, media_type="text/event-stream",
                                       headers={"Cache-Control": "no-cache",
                                                "X-Accel-Buffering": "no",
                                                "X-Search-Profile": name})

    return app
```

- [ ] **Step 4: Replace `src/agentic_search/server/__init__.py`**

```python
"""HTTP service for agentic search (install the `server` extra: fastapi, uvicorn)."""

from agentic_search.server.app import create_app
from agentic_search.server.config import (
           AuthConfig,
           Profile,
           ProfileLimits,
           ServiceConfig,
           load_service,
)

__all__ = ["AuthConfig", "Profile", "ProfileLimits", "ServiceConfig", "create_app", "load_service"]
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest -q tests/server`
Expected: `28 passed`.

Run: `uv run pytest -q && uv run ruff check`
Expected: `421 passed, 5 skipped, 84 deselected` and `All checks passed!`.

- [ ] **Step 6: Commit**

```bash
git add src/agentic_search/server/app.py src/agentic_search/server/__init__.py tests/server/test_app.py tests/server/test_live.py
git commit -m "feat(server): FastAPI app with profiles, auth, capacity gate, search and SSE stream

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Demo service, contract export, CLI and docs

**Files:**
- Create: `src/agentic_search/server/demo.py`, `src/agentic_search/server/schema.py`, `src/agentic_search/server/cli.py`, `tests/server/test_schema.py`, and `schema/search-events.v1.schema.json` plus `schema/search-events.v1.fixtures.json` (generated, not hand-written)
- Modify: `pyproject.toml` (console script), then run `uv sync`; `README.md` (Service section)

**Interfaces:**
- Consumes: `create_app`, `AuthConfig`, `Profile`, `ServiceConfig`; `events.SCHEMA_VERSION`, `SearchEvent`; `testing.EchoDriver`, `KeywordJudge`; `FilesBackend`, `HashEmbedder`.
- Produces:
  - `demo.py`:
    - `DEMO_DOCS`.
    - `demo_harness(*, slow_s=None, on_cancel=None) -> Harness`.
    - `demo_app(**profiles: Harness)`: the `demo` profile plus any extras, with auth off.
  - `schema.py`:
    - `SCHEMA_FILE` and `FIXTURES_FILE`.
    - `event_schema() -> dict`.
    - `async fixture_streams() -> dict[str, list[{id, event, data}]]`.
    - `async render_files() -> dict[str, str]`.
    - `async export(out_dir) -> list[Path]`.
  - `cli.py`: `main(argv) -> int` with subcommands `serve --config --host --port --log-level` and `export-schema --out`.
  - The console script `agentic-search`.
  - The committed contract under `schema/`, which Plan 6 (the TypeScript client) reads.

- [ ] **Step 1: Write the failing test**

Create `tests/server/test_schema.py`:

```python
from pathlib import Path

from agentic_search.server.cli import main
from agentic_search.server.schema import FIXTURES_FILE, SCHEMA_FILE, event_schema, render_files

ROOT = Path(__file__).resolve().parents[2]


async def test_committed_contract_matches_models():
    """If this fails, run `uv run agentic-search export-schema --out schema` and commit."""
    rendered = await render_files()
    for name in (SCHEMA_FILE, FIXTURES_FILE):
        assert (ROOT / "schema" / name).read_text() == rendered[name], name


def test_schema_covers_every_event_type():
    schema = event_schema()
    mapping = schema["discriminator"]["mapping"]
    assert set(mapping) == {"search_started", "phase_started", "phase_finished",
                            "tool_call_started", "tool_call_finished", "results_updated",
                            "usage_updated", "search_finished", "search_failed"}


def test_cli_export_schema(tmp_path, capsys):
    assert main(["export-schema", "--out", str(tmp_path / "out")]) == 0
    printed = capsys.readouterr().out.split()
    assert len(printed) == 2 and all((tmp_path / "out").glob("*.json"))
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest -q tests/server/test_schema.py`
Expected: a collection error, `ModuleNotFoundError: No module named 'agentic_search.server.cli'`.

- [ ] **Step 3: Create `src/agentic_search/server/demo.py`**

```python
"""A tiny in-memory demo service: four medical/history documents, a BM25 echo driver and a
keyword judge. Used for the golden contract fixtures and for client end-to-end tests; needs no
network, models or credentials."""

from __future__ import annotations

import asyncio
from typing import Any, Callable

from agentic_search.backends.files import FilesBackend
from agentic_search.core.harness import Harness
from agentic_search.core.types import Document, TextPart
from agentic_search.embedders.local import HashEmbedder
from agentic_search.server.app import create_app
from agentic_search.server.config import AuthConfig, Profile, ServiceConfig
from agentic_search.testing import EchoDriver, KeywordJudge

DEMO_DOCS = [
    ("d1", "Aspirin reduces fever and relieves headache pain", {"title": "Aspirin"}),
    ("d2", "Ibuprofen is an anti-inflammatory used for pain", {"title": "Ibuprofen"}),
    ("d3", "The history of the printing press in Europe", {"title": "Printing press"}),
    ("d4", "Acetaminophen treats headache and fever", {"title": "Acetaminophen"}),
]


def demo_harness(*, slow_s: float | None = None,
                 on_cancel: Callable[[], None] | None = None) -> Harness:
    """`slow_s`: every backend call sleeps this long first; `on_cancel` runs if one is
    cancelled (lets tests observe that closing a stream cancels the search)."""
    docs = [Document(doc_id=i, content=[TextPart(text=t)], metadata=m) for i, t, m in DEMO_DOCS]
    backend = FilesBackend.from_documents("docs", docs, embedder=HashEmbedder())
    if slow_s is not None:
        execute = backend.execute

        async def slow_execute(op: Any) -> Any:
            try:
                await asyncio.sleep(slow_s)
            except asyncio.CancelledError:
                if on_cancel is not None:
                    on_cancel()
                raise
            return await execute(op)

        backend.execute = slow_execute  # type: ignore[method-assign]
    return Harness([backend], EchoDriver("docs", limit=10), embedders=[backend.embedder],
                   analyzer=KeywordJudge(["headache"]))


def demo_app(**profiles: Harness) -> Any:
    """The demo service with auth off. Extra keyword arguments add named profiles."""
    all_profiles = {"demo": Profile(demo_harness()),
                    **{name: Profile(h) for name, h in profiles.items()}}
    return create_app(ServiceConfig(auth=AuthConfig(type="none")), all_profiles)
```

- [ ] **Step 4: Create `src/agentic_search/server/schema.py`**

```python
"""The wire contract for clients: the SearchEvent JSON Schema and golden SSE fixtures.

`agentic-search export-schema --out schema/` writes both; a test keeps the committed copies in
step with the Python models, and the TypeScript client tests against them."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
from pydantic import TypeAdapter

from agentic_search.events import SCHEMA_VERSION, SearchEvent
from agentic_search.server.demo import demo_app

SCHEMA_FILE = f"search-events.v{SCHEMA_VERSION}.schema.json"
FIXTURES_FILE = f"search-events.v{SCHEMA_VERSION}.fixtures.json"

# Each fixture: a name and the stream request that produces it.
_CASES: list[tuple[str, dict[str, Any]]] = [
    ("retrieval", {"question": "what treats headache", "mode": "retrieval", "snapshot_k": 3}),
    ("harness", {"question": "what treats headache", "snapshot_k": 2}),
    ("with_content", {"question": "fever", "mode": "retrieval", "include_content": True,
                      "include_trace": True}),
    ("failed", {"question": "anything", "sources": ["nope"]}),
]


def event_schema() -> dict[str, Any]:
    schema = TypeAdapter(SearchEvent).json_schema(mode="serialization")
    schema["title"] = f"SearchEvent v{SCHEMA_VERSION}"
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    return schema


def _normalise(value: Any) -> Any:
    """Replace run-dependent values (ids, clocks) so fixtures are byte-stable."""
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if k == "search_id":
                out[k] = "fixture"
            elif k in ("at_ms", "duration_ms") and isinstance(v, (int, float)):
                out[k] = 0.0
            else:
                out[k] = _normalise(v)
        return out
    if isinstance(value, list):
        return [_normalise(v) for v in value]
    return value


async def fixture_streams() -> dict[str, list[dict[str, Any]]]:
    """Each case's SSE frames as served, as `{"id", "event", "data"}` dicts."""
    out: dict[str, list[dict[str, Any]]] = {}
    transport = httpx.ASGITransport(app=demo_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://fixture") as client:
        for name, body in _CASES:
            r = await client.post("/v1/search/stream", json=body)
            r.raise_for_status()
            frames = []
            for block in r.text.split("\n\n"):
                fields = dict(line.split(": ", 1) for line in block.split("\n") if ": " in line
                              and not line.startswith(":"))
                if "data" in fields:
                    frames.append({"id": int(fields["id"]), "event": fields["event"],
                                   "data": _normalise(json.loads(fields["data"]))})
            out[name] = frames
    return out


def _dump(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


async def render_files() -> dict[str, str]:
    return {SCHEMA_FILE: _dump(event_schema()), FIXTURES_FILE: _dump(await fixture_streams())}


async def export(out_dir: str | Path) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for name, text in (await render_files()).items():
        (out / name).write_text(text)
        written.append(out / name)
    return written
```

- [ ] **Step 5: Create `src/agentic_search/server/cli.py`**

```python
"""`agentic-search` command line: run the service, or export the client contract."""

from __future__ import annotations

import argparse
import asyncio
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agentic-search")
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve", help="run the HTTP/SSE search service")
    serve.add_argument("--config", required=True, help="service YAML (service: + profiles:)")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8080)
    serve.add_argument("--log-level", default="info")
    export = sub.add_parser("export-schema", help="write the event JSON Schema and fixtures")
    export.add_argument("--out", default="schema")
    args = parser.parse_args(argv)
    try:
        import uvicorn

        from agentic_search.server import create_app, load_service
        from agentic_search.server.schema import export as export_schema
    except ImportError as exc:
        print(f"agentic-search: the server extra is not installed ({exc}); "
              "install with `pip install 'agentic-search[server]'`", file=sys.stderr)
        return 2
    if args.command == "serve":
        config, profiles = load_service(args.config)
        uvicorn.run(create_app(config, profiles), host=args.host, port=args.port,
                    log_level=args.log_level)
        return 0
    for path in asyncio.run(export_schema(args.out)):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: Register the console script and generate the contract**

In `pyproject.toml`, insert this block directly above `[project.optional-dependencies]`:
```toml
[project.scripts]
agentic-search = "agentic_search.server.cli:main"

```
Then run:
```bash
uv sync
uv run agentic-search export-schema --out schema
```
Expected output: the two paths `schema/search-events.v1.schema.json` and `schema/search-events.v1.fixtures.json`.
Run the export a second time. `git diff --stat schema` must show no change, because the output is byte-stable.

- [ ] **Step 7: Add the Service section to `README.md`**

Insert directly above the `## Modes` heading:

````markdown
## Service

`pip install 'agentic-search[server]'` adds an HTTP service that streams searches as
server-sent events. Operators define named **profiles** (each a harness config) at startup;
clients pick a profile and override per request, but never send credentials or datastores.

```yaml
# service.yaml
service:
  auth: {type: api_key, keys_env: SEARCH_API_KEYS}   # or {type: none}, which must be explicit
  cors_origins: ["https://app.example.com"]
  max_concurrent_searches: 16
  default_profile: general
profiles:
  general:
    backends: [{name: notes, type: files, root: ./docs, glob: "**/*.md"}]
    driver: {type: anthropic, model: claude-sonnet-5-5}
  strict:
    backends: [{name: notes, type: files, root: ./docs, glob: "**/*.md"}]
    driver: {type: anthropic, model: claude-sonnet-5-5}
    limits: {max_budget: {max_turns: 2, max_cost_usd: 0.25}, allow_include_content: false}
```

```bash
SEARCH_API_KEYS=key1,key2 agentic-search serve --config service.yaml --port 8080
```

| Endpoint | |
|---|---|
| `GET /healthz` | Liveness, no auth: `{"status": "ok" \| "degraded"}` |
| `GET /v1/profiles` | Profiles with their sources, capabilities, budget and limits |
| `POST /v1/search` | One-shot search; returns the result as JSON |
| `POST /v1/search/stream` | The same search as `text/event-stream` |

Request body: `{profile?, question, images?: [{data: <base64>, mime}], sources?, mode?, top_k?,
budget?, include_content?, include_trace?}`, plus `snapshot_k?` for the stream. Send the key as
`Authorization: Bearer <key>` or `X-API-Key`.

- Each SSE frame is `id: <seq>`, `event: <type>`, `data: <event JSON>`; a `: keep-alive`
  comment is sent after `keepalive_s` (default 15 s) of silence.
- The final `search_finished` carries a lean result: no trace unless `include_trace`, no hit
  content unless `include_content` (and the profile allows it).
- A request's budget is capped by the profile's `limits.max_budget`.
- Closing the connection cancels the search, including in-flight backend and model calls.
- `401` bad key, `404` unknown profile, `422` invalid body, `429` too many concurrent searches,
  `503` profile unavailable. Failures during a streamed search arrive as a `search_failed` event.
- Images are always inline base64; the service never reads a client-supplied path or URI.
- `agentic-search export-schema --out schema/` writes the event JSON Schema and golden SSE
  fixtures that clients test against (committed under `schema/`).

````

- [ ] **Step 8: Run the tests**

Run: `uv run pytest -q tests/server`
Expected: `31 passed`.

Run: `uv run pytest -q && uv run ruff check`
Expected: `424 passed, 5 skipped, 84 deselected` and `All checks passed!`.

Smoke-test the CLI:
```bash
uv run agentic-search --help
```
Expected: usage that lists `serve` and `export-schema`.

- [ ] **Step 9: Commit**

```bash
git add src/agentic_search/server/demo.py src/agentic_search/server/schema.py src/agentic_search/server/cli.py tests/server/test_schema.py schema pyproject.toml uv.lock README.md
git commit -m "feat(server): demo service, event schema and golden SSE fixtures, CLI, README

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

## Spec coverage

| Spec section | Task |
|---|---|
| §2 image bytes in JSON | 1 |
| §3 packaging (extra, script, modules) | 2 (extra), 5 (script) |
| §4 configuration, `load_service`, limits | 2 |
| §5 API, auth, profile resolution, images, budget, content ban, readiness, errors, capacity | 2 (models), 4 (routes) |
| §6 lean result | 2 (`lean_result`/`render_event`), 4 (routes) |
| §7 SSE framing, keep-alive, disconnect | 3, plus 4 (live disconnect test) |
| §8 contract export | 5 |
| §9 demo service | 5 |
| §10 testing | 1–5 |

