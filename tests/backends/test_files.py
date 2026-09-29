import pytest

from agentic_search.backends.base import Backend, BackendError, UnsupportedOperation, rrf_merge
from agentic_search.backends.files import FilesBackend, infer_field_type
from agentic_search.core.types import (
    Aggregate,
    Capability,
    Eq,
    Fetch,
    FieldType,
    FilterOnly,
    Hybrid,
    Lexical,
    Regex,
    StructuredPart,
    TextPart,
    Traverse,
    Vector,
)
from agentic_search.embedders.local import HashEmbedder


@pytest.fixture
def folder(tmp_path):
    (tmp_path / "a.md").write_text("The cat sat on the mat")
    (tmp_path / "b.txt").write_text("Dogs chase cats in the park")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.md").write_text("Quarterly revenue grew 10 percent")
    (tmp_path / "img.png").write_bytes(b"\x89PNG fake")
    (tmp_path / "skip.bin").write_bytes(b"\x00\x01")
    return tmp_path


def test_rrf_merge():
    fused = rrf_merge([(["a", "b"], 1.0), (["b", "c"], 1.0)], k=0)
    assert fused[0][0] == "b"


def test_infer_field_type():
    assert infer_field_type([1, 2]) is FieldType.INT
    assert infer_field_type([1, 2.5]) is FieldType.FLOAT
    assert infer_field_type([True]) is FieldType.BOOL
    assert infer_field_type(["2024-01-01T00:00:00+00:00"]) is FieldType.DATE
    assert infer_field_type(["x"]) is FieldType.KEYWORD
    assert infer_field_type([{"a": 1}]) is FieldType.JSON


async def test_discover_folder(folder):
    b = FilesBackend("fs", folder)
    assert isinstance(b, Backend)
    m = await b.discover()
    coll = m.resolve_collection(None)
    assert coll.name == "files" and coll.count == 4
    names = {f.name for f in coll.fields}
    assert {"text", "path", "ext", "size", "modified"} <= names
    assert coll.field("size").type is FieldType.INT
    assert coll.field("modified").type is FieldType.DATE
    assert set(coll.field("ext").sample_values) == {".md", ".txt", ".png"}
    assert Capability.VECTOR not in m.capabilities


async def test_lexical_exact_token(folder):
    b = FilesBackend("fs", folder)
    hits = await b.execute(Lexical(source="fs", text="cat"))
    assert [h.doc_id for h in hits] == ["a.md"]
    assert hits[0].raw_score > 0 and hits[0].metadata["ext"] == ".md"


async def test_filter_regex_fetch_aggregate(folder):
    b = FilesBackend("fs", folder)
    md = await b.execute(FilterOnly(source="fs", filter=Eq(field="ext", value=".md")))
    assert {h.doc_id for h in md} == {"a.md", "sub/c.md"}
    rx = await b.execute(Regex(source="fs", pattern=r"\d+ percent"))
    assert [h.doc_id for h in rx] == ["sub/c.md"]
    f = await b.execute(Fetch(source="fs", doc_ids=["b.txt", "missing"]))
    assert [h.doc_id for h in f] == ["b.txt"]
    agg = await b.execute(Aggregate(source="fs", group_by=["ext"]))
    top = agg[0].content[0]
    assert isinstance(top, StructuredPart) and top.data == {"ext": ".md", "count": 2}


async def test_bad_regex_and_unsupported(folder):
    b = FilesBackend("fs", folder)
    with pytest.raises(BackendError):
        await b.execute(Regex(source="fs", pattern="("))
    with pytest.raises(UnsupportedOperation):
        await b.execute(Traverse(source="fs", start=Eq(field="x", value=1)))
    with pytest.raises(UnsupportedOperation):
        await b.execute(Aggregate(source="fs", group_by=["ext"], metrics=["sum"]))


async def test_documents_vector_and_hybrid(docs_backend):
    m = await docs_backend.discover()
    coll = m.resolve_collection(None)
    assert coll.field("embedding").embedder_id == "hash"
    assert coll.field("year").type is FieldType.INT
    assert {Capability.VECTOR, Capability.HYBRID} <= m.capabilities
    [qv] = await HashEmbedder().embed([TextPart(text="headache fever")], "query")
    vec = await docs_backend.execute(Vector(source="docs", field="embedding", hyde_text="x", vector=qv))
    assert vec[0].doc_id in {"d1", "d4"}
    hyb = await docs_backend.execute(
        Hybrid(source="docs", field="embedding", text="headache", vector=qv, limit=2))
    assert {h.doc_id for h in hyb} == {"d1", "d4"}


async def test_lexical_with_filter(docs_backend):
    hits = await docs_backend.execute(
        Lexical(source="docs", text="pain", filter=Eq(field="year", value=2021)))
    assert [h.doc_id for h in hits] == ["d2"]


async def test_vector_without_resolved_vector_fails(docs_backend):
    with pytest.raises(BackendError):
        await docs_backend.execute(Vector(source="docs", field="embedding", hyde_text="x"))


async def test_unknown_collection(docs_backend):
    with pytest.raises(BackendError):
        await docs_backend.execute(Lexical(source="docs", collection="nope", text="x"))


def test_requires_exactly_one_source(medical_docs, tmp_path):
    with pytest.raises(ValueError):
        FilesBackend("x")
    with pytest.raises(ValueError):
        FilesBackend("x", tmp_path, documents=medical_docs)
