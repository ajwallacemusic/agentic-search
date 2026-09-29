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
