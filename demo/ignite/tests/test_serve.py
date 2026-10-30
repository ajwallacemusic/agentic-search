import httpx
import pytest

from demo.ignite.serve import build_parser, create_demo_app
from demo.ignite.tests.helpers import demo_data, offline_harness

HERO = demo_data().questions["hero"]


@pytest.fixture
def client(tmp_path):
    (tmp_path / "index.html").write_text("<!doctype html><title>viewer</title>")
    app = create_demo_app({"ignite": offline_harness()}, demo_data().questions,
                          viewer_dir=tmp_path)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


async def test_questions_endpoint_hides_gold(client):
    rows = (await client.get("/ignite/questions")).json()
    assert rows[0]["id"] == "hero" and rows[0]["text"] == HERO.text
    assert all("gold" not in r for r in rows)


async def test_questions_endpoint_gives_set_and_profile(client):
    rows = (await client.get("/ignite/questions")).json()
    assert rows[0]["set"] == "musique" and rows[0]["profile"] == "ignite"


def test_each_corpus_has_its_own_profile():
    from demo.ignite.serve import PROFILE_OF
    assert PROFILE_OF == {"musique": "ignite", "wands": "ignite-wands"}


async def test_stream_returns_sse_for_both_modes(client):
    for mode in ("retrieval", "harness"):
        r = await client.post("/v1/search/stream",
                              json={"profile": "ignite", "question": HERO.text, "mode": mode})
        assert r.status_code == 200 and "event: search_finished" in r.text


async def test_viewer_is_served_at_root(client):
    r = await client.get("/")
    assert r.status_code == 200 and "viewer" in r.text


def test_cli_binds_localhost_only():
    assert "--host" not in build_parser().format_help()
