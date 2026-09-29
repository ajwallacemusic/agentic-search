from agentic_search.core.annotations import apply_annotations
from agentic_search.core.types import CollectionInfo, FieldSpec, FieldType, Manifest


def base():
    return Manifest(source="s", backend_type="x", capabilities=set(), collections=[
        CollectionInfo(name="notes", fields=[FieldSpec(name="dx_code", type=FieldType.KEYWORD)])])


def test_apply_annotations():
    m = base()
    out = apply_annotations(m, {
        "description": "Clinical notes",
        "collections": {"notes": {
            "description": "one row per note",
            "fields": {
                "dx_code": {"description": "ICD-10-CM code", "filterable": True},
                "emb": {"type": "vector", "embedder_id": "azure:medsiglip-448", "vector_dim": 1152},
            }},
            "ghost": {"description": "ignored"}},
    })
    coll = out.resolve_collection("notes")
    assert out.description == "Clinical notes" and coll.description == "one row per note"
    assert coll.field("dx_code").description == "ICD-10-CM code" and coll.field("dx_code").filterable
    emb = coll.field("emb")
    assert emb.type is FieldType.VECTOR and emb.embedder_id == "azure:medsiglip-448"
    assert m.description is None and m.resolve_collection("notes").field("emb") is None


def test_none_is_identity():
    m = base()
    assert apply_annotations(m, None) is m
