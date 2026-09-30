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
