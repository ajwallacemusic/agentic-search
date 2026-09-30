import pytest

from agentic_search import Harness
from agentic_search.backends.files import FilesBackend
from agentic_search.core.hooks import Hooks, SourcePolicy
from agentic_search.core.types import Modality, TextPart
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


async def test_second_harness_rebinds_embedder(medical_docs):
    """Two Harnesses sharing one FilesBackend: the second Harness's hooks/policy apply."""
    backend = FilesBackend.from_documents("docs", medical_docs, embedder=RemoteLike(dim=8, id="emb"))
    hooks1 = Recording()
    Harness([backend], ScriptedDriver([]), hooks=hooks1)
    assert isinstance(backend.embedder, HookedEmbedder)

    # Second Harness with stricter policy should rebind
    hooks2 = Recording()
    h2 = Harness([backend, FilesBackend.from_documents("ok", medical_docs)], ScriptedDriver([]),
                 hooks=hooks2, source_policy={"docs": {"local-llm"}})
    await h2.setup()
    # The second Harness's stricter policy should block discovery of 'docs'
    assert "docs" in h2.setup_errors and "forbids" in h2.setup_errors["docs"]
    # But 'ok' should succeed
    assert "ok" in h2.manifests


async def test_cached_embedder_local_property():
    """CachedEmbedder.local passes through to inner embedder."""
    from agentic_search.embedders.base import CachedEmbedder

    # With local embedder
    local_cached = CachedEmbedder(HashEmbedder(dim=8, id="hash"))
    assert local_cached.local is True

    # With remote embedder
    remote_cached = CachedEmbedder(RemoteLike(dim=8, id="remote"))
    assert remote_cached.local is False


async def test_embedder_with_no_local_attribute_is_wrapped(medical_docs):
    """Embedders without local attribute are treated as remote and wrapped."""
    # Create an embedder without local attribute
    class UnmarkedEmbedder:
        def __init__(self):
            self.id = "unmarked"
            self.dim = 8
            self.modalities = {Modality.TEXT}

        async def embed(self, items, purpose):
            # Simulate embedding
            return [[0.1] * 8 for _ in items]

    backend = FilesBackend.from_documents("docs", medical_docs, embedder=UnmarkedEmbedder())
    Harness([backend], ScriptedDriver([]))
    # Should be wrapped since no local attribute means remote
    assert isinstance(backend.embedder, HookedEmbedder)


async def test_cached_hooked_embedder_hook_called_once(medical_docs):
    """CachedEmbedder(HookedEmbedder(...)) bound to two Harnesses: hook called once per embed, not double-wrapped."""
    from agentic_search.embedders.base import CachedEmbedder

    hooks1 = Recording()
    inner = RemoteLike(dim=8, id="remote")
    hooked = HookedEmbedder(inner, hooks1, source="docs", policy=SourcePolicy())
    cached = CachedEmbedder(hooked)

    backend = FilesBackend.from_documents("docs", medical_docs, embedder=cached)
    Harness([backend], ScriptedDriver([]), hooks=hooks1)

    # Verify embedder is still CachedEmbedder(HookedEmbedder(...)), not wrapped again
    assert isinstance(backend.embedder, CachedEmbedder)
    assert isinstance(backend.embedder.inner, HookedEmbedder)

    # Bind a second Harness with different hooks - should NOT wrap again
    hooks2 = Recording()
    h2 = Harness([backend], ScriptedDriver([]), hooks=hooks2)

    # Still not wrapped
    assert isinstance(backend.embedder, CachedEmbedder)
    assert isinstance(backend.embedder.inner, HookedEmbedder)

    await h2.setup()
    # Verify the original hook (hooks1) is called exactly once per document embedding,
    # not multiple times (proving no double-wrapping)
    assert len(hooks1.calls) == 1
    assert hooks1.calls[0][0] == "remote"
