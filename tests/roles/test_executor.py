import asyncio

import numpy as np
import pytest

from agentic_search.core.hooks import Hooks, SourcePolicy
from agentic_search.core.state import CandidatePool, Trace
from agentic_search.core.types import Capability, CollectionInfo, Manifest, Modality, Query
from agentic_search.embedders.base import EmbedderRegistry
from agentic_search.embedders.local import HashEmbedder
from agentic_search.roles.executor import Executor
from agentic_search.testing import call

Q = Query.of("headache")


class StubBackend:
    backend_type = "stub"

    def __init__(self, name, behavior):
        self.name, self.behavior = name, behavior

    def capabilities(self):
        return {Capability.LEXICAL}

    async def discover(self, detail="full", collection=None):
        return Manifest(source=self.name, backend_type="stub", capabilities=self.capabilities(),
                        collections=[CollectionInfo(name="c")])

    async def execute(self, op):
        if self.behavior == "raise":
            raise RuntimeError("db down")
        await asyncio.sleep(1)
        return []

    async def close(self):
        pass


class RecordingHooks(Hooks):
    def __init__(self):
        self.model_ids = []

    async def before_model_call(self, model_id, payload):
        self.model_ids.append(model_id)
        return payload


@pytest.fixture
async def ex(docs_backend):
    m = await docs_backend.discover()
    return Executor({"docs": docs_backend}, {"docs": m}, EmbedderRegistry([HashEmbedder()]),
                    hooks=RecordingHooks())


async def run(executor, *calls, model_id=None):
    pool, trace = CandidatePool(), Trace()
    res = await executor.run(list(calls), question=Q, turn=0, pool=pool, trace=trace,
                             model_id=model_id)
    return res, pool, trace


async def test_lexical_pools_hits(ex):
    c = call("lexical_search", source="docs", text="headache")
    res, pool, trace = await run(ex, c)
    assert set(res.new_keys) == {"docs:d1", "docs:d4"} and len(pool) == 2
    assert res.hits_per_call[c.id][0] in {"docs:d1", "docs:d4"}
    assert res.outputs[c.id].startswith("2 hits (2 new)")
    assert [e.type for e in trace.events] == ["tool_call", "tool_result"]


async def test_second_call_only_reports_new(ex):
    a = call("lexical_search", source="docs", text="headache")
    b = call("lexical_search", source="docs", text="fever pain")
    res, _, _ = await run(ex, a, b)
    assert sorted(res.new_keys) == ["docs:d1", "docs:d2", "docs:d4"]


async def test_validation_errors_do_not_raise(ex):
    bad_source = call("lexical_search", source="nope", text="x")
    bad_field = call("filter_search", source="docs", filter={"op": "eq", "field": "colour", "value": 1})
    bad_args = call("lexical_search", source="docs")
    res, pool, trace = await run(ex, bad_source, bad_field, bad_args)
    assert len(pool) == 0 and len(res.errors) == 3
    assert "unknown source" in res.errors[0].message
    assert "colour" in res.errors[1].message and "known fields" in res.errors[1].message
    assert res.errors[2].kind == "validation"
    assert res.outputs[bad_source.id].startswith("ERROR [validation]")
    assert len(trace.of_type("tool_error")) == 3


async def test_vector_embeds_via_bound_embedder_and_hooks(ex):
    c = call("vector_search", source="docs", field="embedding", hyde_text="headache and fever")
    res, _, _ = await run(ex, c)
    assert res.hits_per_call[c.id][0] in {"docs:d1", "docs:d4"}
    assert ex.hooks.model_ids == ["hash"]


async def test_vector_refuses_unregistered_embedder(docs_backend):
    m = await docs_backend.discover()
    ex = Executor({"docs": docs_backend}, {"docs": m}, EmbedderRegistry())
    res, _, _ = await run(ex, call("vector_search", source="docs", field="embedding", hyde_text="x"))
    assert res.errors[0].kind == "embedder" and "refusing" in res.errors[0].message


async def test_limit_is_clamped(docs_backend):
    m = await docs_backend.discover()
    ex = Executor({"docs": docs_backend}, {"docs": m}, EmbedderRegistry(), max_limit=1)
    c = call("lexical_search", source="docs", text="headache", limit=50)
    res, _, _ = await run(ex, c)
    assert len(res.hits_per_call[c.id]) == 1


async def test_backend_error_and_timeout():
    boom, slow = StubBackend("boom", "raise"), StubBackend("slow", "sleep")
    manifests = {b.name: await b.discover() for b in (boom, slow)}
    ex = Executor({"boom": boom, "slow": slow}, manifests, EmbedderRegistry(), call_timeout=0.05)
    res, _, _ = await run(ex, call("lexical_search", source="boom", text="x"),
                          call("lexical_search", source="slow", text="x"))
    assert [e.kind for e in res.errors] == ["backend", "timeout"]
    assert "db down" in res.errors[0].message


async def test_discover_renders_manifest(ex):
    c1 = call("discover", source="docs")
    c2 = call("discover", source="docs", collection="files")
    res, _, _ = await run(ex, c1, c2)
    assert "`docs`" in res.outputs[c1.id]
    assert '"embedder_id": "hash"' in res.outputs[c2.id]


async def test_policy_withholds_content_in_outputs(docs_backend):
    m = await docs_backend.discover()
    ex = Executor({"docs": docs_backend}, {"docs": m}, EmbedderRegistry(),
                  policy=SourcePolicy({"docs": {"local"}}))
    c = call("lexical_search", source="docs", text="headache")
    res, _, _ = await run(ex, c, model_id="cloud")
    assert "withheld" in res.outputs[c.id] and "Aspirin" not in res.outputs[c.id]


class SlowEmbedder:
    """Embedder that takes 1 second to embed, for testing timeouts."""

    def __init__(self, dim: int):
        self.id = "hash"
        self.modalities = {Modality.TEXT}
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    async def embed(self, items, purpose):
        await asyncio.sleep(1)
        return [np.zeros(self._dim, dtype=np.float32) for _ in items]


async def test_embedder_timeout(docs_backend):
    m = await docs_backend.discover()
    coll = m.resolve_collection(None)
    emb_field = coll.field("embedding")
    slow = SlowEmbedder(emb_field.vector_dim)
    ex = Executor({"docs": docs_backend}, {"docs": m}, EmbedderRegistry([slow]),
                  call_timeout=0.05)
    res, _, _ = await run(ex, call("vector_search", source="docs", field="embedding", hyde_text="x"))
    assert res.errors[0].kind == "timeout" and "embedding" in res.errors[0].message
