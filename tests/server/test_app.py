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


async def test_non_ascii_api_key_is_401_not_500(app_factory, monkeypatch):
    monkeypatch.setenv("SEARCH_KEYS", "k-one")
    app = app_factory(config=service(auth=AuthConfig(type="api_key", keys_env="SEARCH_KEYS")))
    async with client(app) as c:
        x = await c.post("/v1/search", json={"question": "q"},
                         headers={"X-API-Key": "k\xe9y".encode("latin-1")})
        bearer = await c.post("/v1/search", json={"question": "q"},
                              headers={"Authorization": "Bearer k\xe9y".encode("latin-1")})
    assert x.status_code == 401 and bearer.status_code == 401


async def test_auth_header_precedence(app_factory, monkeypatch):
    monkeypatch.setenv("SEARCH_KEYS", "k-one")
    app = app_factory(config=service(auth=AuthConfig(type="api_key", keys_env="SEARCH_KEYS")))
    async with client(app) as c:
        bad_bearer = await c.get("/v1/profiles", headers={
            "Authorization": "Bearer nope", "X-API-Key": "k-one"})
        basic = await c.get("/v1/profiles", headers={
            "Authorization": "Basic abc", "X-API-Key": "k-one"})
    assert bad_bearer.status_code == 401 and basic.status_code == 200


async def test_stream_disconnect_with_blocked_send_cancels_search_and_frees_slot(
        docs_backend, monkeypatch):
    from agentic_search.core.harness import HarnessSettings

    entered, cancelled = asyncio.Event(), asyncio.Event()
    mode = {"slow": True}

    async def execute(op):
        if not mode["slow"]:
            return []
        entered.set()
        try:
            await asyncio.sleep(600)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return []

    docs_backend.execute = execute
    import agentic_search.server.app as app_module

    real_sse_body, bodies = app_module.sse_body, []  # strong refs: GC must not rescue the test

    def keep(*args, **kwargs):
        bodies.append(real_sse_body(*args, **kwargs))
        return bodies[-1]

    monkeypatch.setattr(app_module, "sse_body", keep)
    h = make_harness(docs_backend, settings=HarnessSettings(call_timeout=600))
    app = create_app(service(max_concurrent_searches=1, keepalive_s=30), {"demo": h})
    scope = {"type": "http", "method": "POST", "path": "/v1/search/stream",
             "headers": [(b"content-type", b"application/json")], "query_string": b"",
             "http_version": "1.1", "scheme": "http", "server": ("test", 80),
             "client": ("t", 1), "root_path": "",
             "asgi": {"version": "3.0", "spec_version": "2.4"}}
    disconnect, blocked, sent_body, requested = asyncio.Event(), asyncio.Event(), [], []

    async def receive():
        if not requested:
            requested.append(1)
            return {"type": "http.request", "body": b'{"question": "q"}', "more_body": False}
        await disconnect.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        if message["type"] == "http.response.body":
            if sent_body:
                blocked.set()
                await disconnect.wait()  # write-paused transport, then the peer is gone
                raise OSError("client disconnected")
            sent_body.append(message)

    task = asyncio.create_task(app(scope, receive, send))
    await asyncio.wait_for(entered.wait(), 5)
    await asyncio.wait_for(blocked.wait(), 5)
    assert not cancelled.is_set()
    disconnect.set()
    await asyncio.wait_for(cancelled.wait(), 5)
    mode["slow"] = False
    async with client(app) as c:
        for _ in range(100):  # the slot comes back once the response has wound down
            again = await c.post("/v1/search", json={"question": "q"})
            if again.status_code != 429:
                break
            await asyncio.sleep(0.05)
    assert again.status_code == 200
    task.cancel()
