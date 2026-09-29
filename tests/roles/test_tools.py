import pytest

from agentic_search.core.types import (
    Capability,
    ImagePart,
    Lexical,
    Manifest,
    Query,
    TextPart,
    Vector,
)
from agentic_search.roles.tools import DiscoverRequest, build_tool_specs, parse_call
from agentic_search.testing import call


def manifest(name, caps):
    return Manifest(source=name, backend_type="x", capabilities=set(caps))


def test_specs_follow_capabilities():
    specs = {s.name: s for s in build_tool_specs({
        "a": manifest("a", [Capability.LEXICAL, Capability.FILTER]),
        "b": manifest("b", [Capability.VECTOR]),
    })}
    assert set(specs) == {"discover", "lexical_search", "filter_search", "vector_search"}
    assert specs["lexical_search"].parameters["properties"]["source"]["enum"] == ["a"]
    assert specs["vector_search"].parameters["properties"]["source"]["enum"] == ["b"]
    assert specs["discover"].parameters["properties"]["source"]["enum"] == ["a", "b"]
    assert specs["lexical_search"].parameters["required"] == ["source", "text"]


def test_parse_lexical_and_discover():
    q = Query.of("q")
    op = parse_call(call("lexical_search", source="a", text="x", type="vector"), q)
    assert isinstance(op, Lexical) and op.text == "x"
    d = parse_call(call("discover", source="a"), q)
    assert d == DiscoverRequest(source="a")


def test_parse_vector_strips_model_supplied_vector():
    op = parse_call(call("vector_search", source="a", field="e", hyde_text="h", vector=[1.0]),
                    Query.of("q"))
    assert isinstance(op, Vector) and op.vector is None


def test_parse_vector_uses_question_image():
    img = ImagePart(data=b"png")
    q = Query(content=[TextPart(text="find similar"), img])
    op = parse_call(call("vector_search", source="a", field="e", use_question_image=True), q)
    assert op.content == img and op.hyde_text is None
    with pytest.raises(ValueError, match="no image"):
        parse_call(call("vector_search", source="a", field="e", use_question_image=True),
                   Query.of("q"))


def test_parse_errors_are_value_errors():
    q = Query.of("q")
    with pytest.raises(ValueError, match="unknown tool"):
        parse_call(call("drop_table", source="a"), q)
    with pytest.raises(ValueError):
        parse_call(call("lexical_search", source="a"), q)  # missing text
    with pytest.raises(ValueError, match="not valid JSON"):
        parse_call(call("lexical_search", _invalid_json="{oops"), q)
