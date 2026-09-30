import sys

import pytest

from agentic_search.backends.base import BackendError, UnsupportedOperation
from agentic_search.backends.milvus import MilvusBackend, filter_expr
from agentic_search.core.secrets import scrub
from agentic_search.core.types import (
    Aggregate,
    And,
    Contains,
    Eq,
    Exists,
    In,
    Not,
    Or,
    Range,
    Regex,
)

FIELDS = {"type", "year", "title", "id"}


def test_filter_expr_templates_every_value():
    params: dict = {}
    expr = filter_expr(And(clauses=[Eq(field="type", value="drug"),
                                    Or(clauses=[Range(field="year", gte=2020, lt=2022),
                                                Not(clause=In(field="id", values=["a", "b"]))]),
                                    Contains(field="title", value='50%_off"x')]), FIELDS, params)
    # LIKE only accepts a literal: wildcards are LIKE-escaped, then the literal is quote-escaped.
    assert expr == ('(type == {p0} and ((year >= {p1} and year < {p2}) or (not (id in {p3}))) '
                    'and title like "%50\\\\%\\\\_off\\"x%")')
    assert params == {"p0": "drug", "p1": 2020, "p2": 2022, "p3": ["a", "b"]}


def test_filter_expr_edge_cases():
    assert filter_expr(And(clauses=[]), FIELDS, {}) == "true"
    assert filter_expr(Or(clauses=[]), FIELDS, {}) == "false"
    assert filter_expr(In(field="type", values=[]), FIELDS, {}) == "false"
    assert filter_expr(Range(field="year"), FIELDS, {}) == "true"
    assert filter_expr(Exists(field="title"), FIELDS, {}) == "title is not null"


@pytest.mark.parametrize("bad", ["nope", "year) or (1 == 1", "type\n"])
def test_filter_expr_rejects_unknown_or_bad_fields(bad):
    with pytest.raises(BackendError):
        filter_expr(Eq(field=bad, value=1), FIELDS | {"year) or (1 == 1", "type\n"}, {})


def test_constructor_registers_secrets():
    MilvusBackend("mv", "http://root:milvus-pw-123@h:19530", token="tok-abcdef-999")
    assert scrub("x milvus-pw-123 tok-abcdef-999") == "x *** ***"


async def test_missing_extra_is_backend_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "pymilvus", None)
    with pytest.raises(BackendError, match="milvus.*extra"):
        await MilvusBackend("mv", "http://localhost:59530").discover()


async def test_unreachable_uri_is_backend_error():
    b = MilvusBackend("mv", "http://127.0.0.1:1", connect_timeout_s=2)
    try:
        with pytest.raises(BackendError):
            await b.discover()
    finally:
        await b.close()


async def test_unsupported_ops_without_connecting():
    b = MilvusBackend("mv", "http://127.0.0.1:1", connect_timeout_s=2)
    for op in (Regex(source="mv", pattern="x"), Aggregate(source="mv", group_by=["type"])):
        with pytest.raises(UnsupportedOperation):
            await b.execute(op)


class _FakeMilvus:
    """Stands in for AsyncMilvusClient; records every call as (method, args, kwargs)."""

    def __init__(self, *, pk_type="VARCHAR", loaded=True, metric="COSINE", rows=None):
        from pymilvus import DataType
        self.calls: list[tuple[str, tuple, dict]] = []
        self.loaded = loaded
        self.metric = metric
        self.rows = rows or []
        self.desc = {"fields": [
            {"name": "id", "type": getattr(DataType, pk_type), "params": {}, "is_primary": True},
            {"name": "type", "type": DataType.VARCHAR, "params": {"max_length": 32}},
            {"name": "embedding", "type": DataType.FLOAT_VECTOR, "params": {"dim": 4}},
        ], "functions": []}

    def __getattr__(self, method):
        async def call(*args, **kwargs):
            self.calls.append((method, args, kwargs))
            return self._answer(method, args, kwargs)
        return call

    def _answer(self, method, args, kwargs):
        from pymilvus.client.types import LoadState
        if method == "describe_collection":
            return self.desc
        if method == "list_indexes":
            return ["embedding_idx"] if self.metric else []
        if method == "describe_index":
            return {"field_name": "embedding", "metric_type": self.metric}
        if method == "get_collection_stats":
            return {"row_count": len(self.rows)}
        if method == "get_load_state":
            return {"state": LoadState.Loaded if self.loaded else LoadState.NotLoad}
        if method == "load_collection":
            self.loaded = True
            return None
        if method == "query":
            return self.rows
        if method == "search":
            return [[]]
        return None

    def methods(self) -> list[str]:
        return [c[0] for c in self.calls]


def _fake_backend(fake: _FakeMilvus, **kwargs) -> MilvusBackend:
    b = MilvusBackend("mv", "http://milvus.invalid:19530", collections=["docs"], **kwargs)
    b._client = fake
    return b


async def test_fetch_coerces_ids_for_int64_primary_key():
    from agentic_search.core.types import Fetch
    fake = _FakeMilvus(pk_type="INT64", rows=[{"id": 5, "type": "drug"}, {"id": 3, "type": "x"}])
    b = _fake_backend(fake)
    hits = await b.execute(Fetch(source="mv", doc_ids=["3", "5"]))
    query = [c for c in fake.calls if c[0] == "query" and c[2].get("filter_params", {}).get("ids")]
    assert query[-1][2]["filter_params"] == {"ids": [3, 5]}
    assert [h.doc_id for h in hits] == ["3", "5"]


async def test_fetch_rejects_non_integer_id_for_int64_primary_key():
    from agentic_search.core.types import Fetch
    b = _fake_backend(_FakeMilvus(pk_type="INT64"))
    with pytest.raises(BackendError, match="INT64"):
        await b.execute(Fetch(source="mv", doc_ids=["abc"]))


async def test_fetch_with_no_ids_returns_empty_without_querying():
    from agentic_search.core.types import Fetch
    fake = _FakeMilvus()
    b = _fake_backend(fake)
    await b.discover()
    fake.calls.clear()
    assert await b.execute(Fetch.model_construct(source="mv", collection=None, doc_ids=[], limit=10)) == []
    assert "query" not in fake.methods()


async def test_does_not_load_collections_by_default():
    from agentic_search.core.types import FilterOnly
    fake = _FakeMilvus(loaded=False, rows=[{"id": "a", "type": "drug"}])
    b = _fake_backend(fake)
    manifest = await b.discover()
    assert "load_collection" not in fake.methods()
    # discovery skips sampling an unloaded collection instead of loading it
    assert "query" not in fake.methods()
    assert next(f for f in manifest.collections[0].fields if f.name == "type").sample_values is None
    with pytest.raises(BackendError, match="collection docs is not loaded"):
        await b.execute(FilterOnly(source="mv", filter=Eq(field="type", value="drug")))
    assert "load_collection" not in fake.methods()


async def test_loaded_collection_is_sampled_and_queried_without_loading():
    from agentic_search.core.types import FilterOnly
    fake = _FakeMilvus(loaded=True, rows=[{"id": "a", "type": "drug"}])
    b = _fake_backend(fake)
    manifest = await b.discover()
    assert next(f for f in manifest.collections[0].fields if f.name == "type").sample_values == ["drug"]
    hits = await b.execute(FilterOnly(source="mv", filter=Eq(field="type", value="drug")))
    assert [h.doc_id for h in hits] == ["a"]
    assert "load_collection" not in fake.methods()


async def test_load_collections_opt_in_loads_lazily():
    from agentic_search.core.types import FilterOnly
    fake = _FakeMilvus(loaded=False, rows=[{"id": "a", "type": "drug"}])
    b = _fake_backend(fake, load_collections=True)
    await b.execute(FilterOnly(source="mv", filter=Eq(field="type", value="drug")))
    assert fake.methods().count("load_collection") == 1


async def test_unknown_metric_omits_metric_type():
    from agentic_search.core.types import Vector
    fake = _FakeMilvus(metric=None)
    b = _fake_backend(fake)
    await b.execute(Vector(source="mv", field="embedding", hyde_text="x", vector=[0.1, 0.2, 0.3, 0.4]))
    search = next(c for c in fake.calls if c[0] == "search")
    assert "metric_type" not in search[2].get("search_params", {})


async def test_known_metric_is_sent():
    from agentic_search.core.types import Vector
    fake = _FakeMilvus(metric="IP")
    b = _fake_backend(fake)
    await b.execute(Vector(source="mv", field="embedding", hyde_text="x", vector=[0.1, 0.2, 0.3, 0.4]))
    search = next(c for c in fake.calls if c[0] == "search")
    assert search[2]["search_params"] == {"metric_type": "IP"}
