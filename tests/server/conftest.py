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
