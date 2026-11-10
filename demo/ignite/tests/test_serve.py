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


def _answer_app(tmp_path, client):
    from demo.ignite.check import load_proofs
    from demo.ignite.serve import doc_texts
    (tmp_path / "index.html").write_text("<!doctype html><title>viewer</title>")
    data = demo_data()
    app = create_demo_app({"ignite": offline_harness()}, data.questions, viewer_dir=tmp_path,
                          docs=doc_texts(data.documents), answer_client=client,
                          proofs=load_proofs(data.questions))
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


async def test_answer_endpoint_reads_the_given_docs_and_reveals_gold(tmp_path):
    from demo.ignite.tests.test_answer import FakeClient
    assert HERO.answers, "the fixture's hero question should have an answer"
    fake = FakeClient(f"[1] a fact\nAnswer: {HERO.answers[0]}")
    gold_key = f"wiki:{HERO.gold[0]}"
    async with _answer_app(tmp_path, fake) as c:
        r = await c.post("/ignite/answer", json={"id": "hero", "keys": [gold_key, "wiki:nope"],
                                                 "notes": ["X was born in Y."]})
    body = r.json()
    assert r.status_code == 200 and body["answer"] == HERO.answers[0]
    assert body["steps"] == ["[1] a fact"]
    assert body["gold"] == HERO.gold
    assert body["expected"] == HERO.answers[0] and body["correct"] is True
    assert "[1]" in fake.seen[1] and "[2]" not in fake.seen[1]  # unknown keys are skipped
    assert "- X was born in Y." in fake.seen[1]


async def test_answer_endpoint_explains_each_mark(tmp_path):
    from demo.ignite.tests.test_answer import FakeClient
    gold_key = f"wiki:{HERO.gold[0]}"
    async with _answer_app(tmp_path, FakeClient("Answer: x")) as c:
        r = await c.post("/ignite/answer", json={"id": "hero", "keys": [gold_key, "wiki:nope"]})
    body = r.json()
    assert body["rule"].startswith("★ right = one of the")
    assert body["checks"][gold_key]["verdict"] == "right" and body["checks"][gold_key]["text"]
    assert body["checks"]["wiki:nope"]["verdict"] == "wrong"
    assert "not a document" in body["checks"]["wiki:nope"]["why"]
    assert body["coverage"] is None


async def test_answer_endpoint_unknown_question_is_404(tmp_path):
    from demo.ignite.tests.test_answer import FakeClient
    async with _answer_app(tmp_path, FakeClient()) as c:
        r = await c.post("/ignite/answer", json={"id": "nope", "keys": []})
    assert r.status_code == 404


async def test_answer_endpoint_model_error_is_reported_not_raised(tmp_path):
    class Broken:
        id = "broken"
        async def chat(self, *a, **k):
            raise TimeoutError("slow")
    async with _answer_app(tmp_path, Broken()) as c:
        r = await c.post("/ignite/answer", json={"id": "hero", "keys": []})
    assert r.status_code == 200 and r.json()["answer"] == "No documents to answer from."
    async with _answer_app(tmp_path, Broken()) as c:
        r = await c.post("/ignite/answer", json={"id": "hero", "keys": [f"wiki:{HERO.gold[0]}"]})
    assert r.status_code == 200 and r.json()["error"].startswith("TimeoutError")
    assert r.json()["gold"] == HERO.gold


def test_cli_binds_localhost_only():
    assert "--host" not in build_parser().format_help()
