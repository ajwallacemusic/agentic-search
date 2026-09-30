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
