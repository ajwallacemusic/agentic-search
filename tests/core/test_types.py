import pytest
from pydantic import ValidationError

from agentic_search.core.types import (
    FILTER_ADAPTER, QUERY_OP_ADAPTER, And, Budget, Capability, CollectionInfo, Eq, FieldSpec,
    FieldType, FilterOnly, Hit, ImagePart, Lexical, Manifest, ModelUsage, Not, Query, TextPart,
    ToolError, Vector, image_bytes, required_capabilities, text_of, StructuredPart,
)


def test_query_of_and_text():
    q = Query.of("hello")
    assert q.as_text() == "hello"
    assert q.images() == []


def test_text_of_includes_structured():
    assert text_of([TextPart(text="a"), StructuredPart(data={"b": 1})]) == 'a\n{"b": 1}'


def test_image_part_requires_exactly_one_source():
    with pytest.raises(ValidationError):
        ImagePart()
    with pytest.raises(ValidationError):
        ImagePart(uri="file:///x.png", data=b"x")


def test_image_bytes_from_data_and_file(tmp_path):
    assert image_bytes(ImagePart(data=b"abc")) == b"abc"
    p = tmp_path / "x.png"
    p.write_bytes(b"png")
    assert image_bytes(ImagePart(uri=p.as_uri())) == b"png"
    with pytest.raises(ValueError):
        image_bytes(ImagePart(uri="https://example.com/x.png"))


def test_filter_roundtrip_nested():
    f = FILTER_ADAPTER.validate_python({
        "op": "and",
        "clauses": [
            {"op": "eq", "field": "a", "value": 1},
            {"op": "not", "clause": {"op": "exists", "field": "b"}},
        ],
    })
    assert isinstance(f, And)
    assert isinstance(f.clauses[1], Not)


def test_query_op_discriminates_and_defaults():
    op = QUERY_OP_ADAPTER.validate_python({"type": "lexical", "source": "s", "text": "x"})
    assert isinstance(op, Lexical)
    assert op.limit == 20


def test_vector_needs_exactly_one_query():
    with pytest.raises(ValidationError):
        Vector(source="s", field="e")
    v = Vector(source="s", field="e", hyde_text="hyp")
    assert v.query_content() == TextPart(text="hyp")


def test_filter_only_requires_filter():
    with pytest.raises(ValidationError):
        FilterOnly(source="s")


def test_required_capabilities():
    op = Vector(source="s", field="e", content=ImagePart(uri="file:///a.png"),
                filter=Eq(field="x", value=1))
    assert required_capabilities(op) == {
        Capability.VECTOR, Capability.FILTER, Capability.IMAGE_QUERY,
    }


def test_hit_key_and_snippet():
    h = Hit(doc_id="d", source="s", content=[TextPart(text="a  b\n" + "c" * 50)])
    assert h.key == "s:d"
    snip = h.snippet(10)
    assert snip.startswith("a b c") and len(snip) == 10
    assert Hit(doc_id="i", source="s", content=[ImagePart(data=b"x")]).snippet() == "[image]"


def test_manifest_resolve_and_summary():
    one = CollectionInfo(name="notes", fields=[
        FieldSpec(name="body", type=FieldType.TEXT, searchable=True),
        FieldSpec(name="emb", type=FieldType.VECTOR, vector_dim=8, embedder_id="hash"),
    ], count=3)
    m = Manifest(source="src", backend_type="files",
                 capabilities={Capability.LEXICAL}, collections=[one])
    assert m.resolve_collection(None) is one
    assert m.resolve_collection("nope") is None
    two = m.model_copy(update={"collections": [one, CollectionInfo(name="other")]})
    assert two.resolve_collection(None) is None
    s = m.summary()
    assert "`src`" in s and "`notes`" in s and "embedder=hash" in s and "3 items" in s


def test_model_usage_plus_and_budget_defaults():
    u = ModelUsage(input_tokens=1, output_tokens=2, cost_usd=0.5).plus(
        ModelUsage(input_tokens=3, output_tokens=4, cost_usd=0.25))
    assert (u.input_tokens, u.output_tokens, u.cost_usd) == (4, 6, 0.75)
    assert Budget().max_turns == 4


def test_tool_error_render():
    e = ToolError(call_id="c1", kind="validation", message="bad field")
    assert e.render() == "ERROR [validation] bad field"
