"""OpenSearch backend: multi_match full-text, k-NN vector search, client-side hybrid (RRF),
bool filters, regexp, terms/multi_terms aggregations, mget and guarded native search bodies.
Only search-type APIs are called. Needs the `opensearch` extra."""

from __future__ import annotations

import asyncio
from typing import Any
from urllib.parse import unquote, urlparse

from agentic_search.backends.base import (
    BackendError,
    DiscoverDetail,
    UnsupportedOperation,
    rrf_merge,
)
from agentic_search.backends.native_guard import guard_opensearch
from agentic_search.backends.sql import dumps, jsonable, parse_metric
from agentic_search.core.secrets import register_secret
from agentic_search.core.types import (
    Aggregate,
    And,
    Capability,
    CollectionInfo,
    Contains,
    Content,
    Eq,
    Exists,
    Fetch,
    FieldSpec,
    FieldType,
    Filter,
    FilterOnly,
    Hit,
    Hybrid,
    In,
    Lexical,
    Manifest,
    Native,
    Not,
    Or,
    QueryOp,
    Range,
    Regex,
    StructuredPart,
    TextPart,
    Vector,
)

_TYPES = {
    "text": FieldType.TEXT, "match_only_text": FieldType.TEXT, "keyword": FieldType.KEYWORD,
    "constant_keyword": FieldType.KEYWORD, "wildcard": FieldType.KEYWORD, "long": FieldType.INT,
    "integer": FieldType.INT, "short": FieldType.INT, "byte": FieldType.INT,
    "unsigned_long": FieldType.INT, "float": FieldType.FLOAT, "double": FieldType.FLOAT,
    "half_float": FieldType.FLOAT, "scaled_float": FieldType.FLOAT, "boolean": FieldType.BOOL,
    "date": FieldType.DATE, "date_nanos": FieldType.DATE, "knn_vector": FieldType.VECTOR,
}
SAMPLE_DISTINCT_MAX = 20


def _require_opensearch() -> Any:
    """Import opensearchpy.AsyncOpenSearch.

    Raises BackendError if import fails (e.g., missing opensearch extra).
    Returns AsyncOpenSearch class.
    """
    try:
        from opensearchpy import AsyncOpenSearch

        return AsyncOpenSearch
    except ImportError as exc:
        raise BackendError("OpenSearchBackend needs the `opensearch` extra: pip install 'agentic-search[opensearch]'") from exc


def filter_dsl(f: Filter) -> dict[str, Any]:
    """Filter AST → OpenSearch query DSL (filter context)."""
    if isinstance(f, And):
        return {"bool": {"filter": [filter_dsl(c) for c in f.clauses]}}
    if isinstance(f, Or):
        return {"bool": {"should": [filter_dsl(c) for c in f.clauses], "minimum_should_match": 1}}
    if isinstance(f, Not):
        return {"bool": {"must_not": [filter_dsl(f.clause)]}}
    if isinstance(f, Eq):
        return {"term": {f.field: f.value}}
    if isinstance(f, In):
        return {"terms": {f.field: f.values}}
    if isinstance(f, Range):
        bounds = {k: v for k, v in (("gte", f.gte), ("gt", f.gt), ("lte", f.lte), ("lt", f.lt))
                  if v is not None}
        return {"range": {f.field: bounds}} if bounds else {"match_all": {}}
    if isinstance(f, Exists):
        return {"exists": {"field": f.field}}
    if isinstance(f, Contains):
        return {"wildcard": {f.field: {"value": f"*{f.value}*", "case_insensitive": True}}}
    raise BackendError(f"unsupported filter node {type(f).__name__}")


class OpenSearchBackend:
    backend_type = "opensearch"

    def __init__(self, name: str, url: str, *, indices: list[str] | None = None,
                 embedders: dict[str, str] | None = None, native_query: bool = False,
                 description: str | None = None, max_rows: int = 100, verify_certs: bool = True,
                 client: Any = None, sample_values: bool = True):
        parsed = urlparse(url)
        register_secret(url)
        register_secret(unquote(parsed.password or ""))
        self.name = name
        self.url = url
        self.index_names = indices
        self.embedders = embedders or {}
        self.native_query = native_query
        self.description = description
        self.max_rows = max_rows
        self.verify_certs = verify_certs
        self.sample_values = sample_values
        self._client = client
        self._indices: dict[str, CollectionInfo] | None = None
        self._lock = asyncio.Lock()

    def _get_client(self) -> Any:
        if self._client is None:
            AsyncOpenSearch = _require_opensearch()
            try:
                self._client = AsyncOpenSearch(hosts=[self.url], verify_certs=self.verify_certs,
                                               ssl_show_warn=False)
            except (ValueError, Exception) as exc:
                raise BackendError(f"{type(exc).__name__}: {exc}") from exc
        return self._client

    async def _call(self, fn: str, **kwargs: Any) -> Any:
        try:
            client = self._get_client()
            target: Any = client
            for part in fn.split("."):
                target = getattr(target, part)
            return await target(**kwargs)
        except BackendError:
            raise
        except (OSError, asyncio.TimeoutError, ValueError) as exc:
            raise BackendError(f"{type(exc).__name__}: {exc}") from exc
        except Exception as exc:
            # Catch opensearchpy exceptions without requiring the module (for injected clients)
            if type(exc).__module__.startswith("opensearchpy"):
                raise BackendError(f"{type(exc).__name__}: {exc}") from exc
            raise

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None

    # ---- discovery ------------------------------------------------------------

    def capabilities(self) -> set[Capability]:
        caps = {Capability.LEXICAL, Capability.FILTER, Capability.REGEX, Capability.AGGREGATE,
                Capability.FETCH}
        indices = self._indices or {}
        if any(f.type is FieldType.VECTOR for c in indices.values() for f in c.fields):
            caps |= {Capability.VECTOR, Capability.HYBRID}
        if self.native_query:
            caps.add(Capability.NATIVE)
        return caps

    async def _ensure_indices(self) -> dict[str, CollectionInfo]:
        async with self._lock:
            if self._indices is None:
                self._indices = await self._discover_indices()
            return self._indices

    async def _discover_indices(self) -> dict[str, CollectionInfo]:
        names = self.index_names
        if names is None:
            rows = await self._call("cat.indices", format="json")
            names = sorted(r["index"] for r in rows if not r["index"].startswith("."))
        out: dict[str, CollectionInfo] = {}
        for index in names:
            mapping = await self._call("indices.get_mapping", index=index)
            props = next(iter(mapping.values()))["mappings"].get("properties", {})
            fields = self._fields(index, props)
            count = (await self._call("count", index=index))["count"]
            await self._add_samples(index, fields)
            out[index] = CollectionInfo(name=index, fields=fields, count=count)
        return out

    def _fields(self, index: str, props: dict[str, Any], prefix: str = "") -> list[FieldSpec]:
        fields: list[FieldSpec] = []
        for name, spec in props.items():
            full = f"{prefix}{name}"
            if "properties" in spec:
                fields.extend(self._fields(index, spec["properties"], f"{full}."))
                continue
            ftype = _TYPES.get(spec.get("type", "object"), FieldType.JSON)
            if ftype is FieldType.VECTOR:
                method = spec.get("method", {})
                fields.append(FieldSpec(name=full, type=ftype, vector_dim=spec.get("dimension"),
                                        vector_metric=method.get("space_type"),
                                        embedder_id=self.embedders.get(f"{index}.{full}")))
                continue
            fields.append(FieldSpec(
                name=full, type=ftype, searchable=ftype is FieldType.TEXT,
                filterable=ftype not in (FieldType.TEXT, FieldType.JSON),
                sortable=ftype in (FieldType.INT, FieldType.FLOAT, FieldType.DATE)))
            for sub, sub_spec in spec.get("fields", {}).items():
                if sub_spec.get("type") == "keyword":
                    fields.append(FieldSpec(name=f"{full}.{sub}", type=FieldType.KEYWORD, filterable=True))
        return fields

    async def _add_samples(self, index: str, fields: list[FieldSpec]) -> None:
        keywords = [f for f in fields if f.type in (FieldType.KEYWORD, FieldType.BOOL)]
        if not self.sample_values or not keywords:
            return
        aggs = {f"s{i}": {"terms": {"field": f.name, "size": SAMPLE_DISTINCT_MAX + 1}}
                for i, f in enumerate(keywords)}
        resp = await self._call("search", index=index, body={"size": 0, "aggs": aggs})
        for i, f in enumerate(keywords):
            buckets = resp["aggregations"][f"s{i}"]["buckets"]
            if len(buckets) <= SAMPLE_DISTINCT_MAX:
                f.sample_values = [b.get("key_as_string", b["key"]) for b in buckets]

    async def discover(self, detail: DiscoverDetail = "full",
                       collection: str | None = None) -> Manifest:
        indices = await self._ensure_indices()
        return Manifest(source=self.name, backend_type=self.backend_type,
                        capabilities=self.capabilities(), description=self.description,
                        collections=list(indices.values()))

    # ---- execution -------------------------------------------------------------

    def _resolve(self, name: str | None, indices: dict[str, CollectionInfo]) -> CollectionInfo:
        if name is None:
            if len(indices) == 1:
                return next(iter(indices.values()))
            raise BackendError(f"collection required; one of {sorted(indices)}")
        if name not in indices:
            raise BackendError(f"unknown collection {name!r}; one of {sorted(indices)}")
        return indices[name]

    def _hit(self, raw: dict[str, Any], coll: CollectionInfo, score: float | None = None) -> Hit:
        source = raw.get("_source", {}) or {}
        vectors = {f.name for f in coll.fields if f.type is FieldType.VECTOR}
        texts = [str(source[f.name]) for f in coll.fields
                 if f.type is FieldType.TEXT and source.get(f.name) not in (None, "")]
        content: list[Content] = [TextPart(text="\n".join(texts))] if texts else []
        text_names = {f.name for f in coll.fields if f.type is FieldType.TEXT}
        metadata = {k: jsonable(v) for k, v in source.items() if k not in vectors | text_names}
        if not content:
            content = [StructuredPart(data=metadata)]
        s = raw.get("_score") if score is None else score
        return Hit(doc_id=str(raw["_id"]), source=self.name, content=content, metadata=metadata,
                   raw_score=float(s) if s is not None else None)

    def _source_filter(self, coll: CollectionInfo) -> dict[str, Any]:
        vectors = [f.name for f in coll.fields if f.type is FieldType.VECTOR]
        return {"excludes": vectors} if vectors else {}

    async def _search(self, coll: CollectionInfo, body: dict[str, Any]) -> list[Hit]:
        body = {**body, "_source": self._source_filter(coll)}
        try:
            resp = await self._call("search", index=coll.name, body=body)
            return [self._hit(h, coll) for h in resp["hits"]["hits"]]
        except (KeyError, IndexError, TypeError) as exc:
            raise BackendError(f"unexpected OpenSearch response: {exc}") from exc

    def _with_filter(self, query: dict[str, Any], f: Filter | None) -> dict[str, Any]:
        if f is None:
            return query
        return {"bool": {"must": [query], "filter": [filter_dsl(f)]}}

    def _lexical_body(self, op: Lexical | Hybrid, coll: CollectionInfo, limit: int) -> dict[str, Any]:
        text_fields = [f.name for f in coll.fields if f.searchable]
        fields = getattr(op, "fields", None) or text_fields
        if not fields:
            raise BackendError(f"{coll.name} has no text fields")
        query = {"multi_match": {"query": op.text, "fields": fields}}
        return {"size": limit, "query": self._with_filter(query, op.filter)}

    def _vector_body(self, field: str, vector: list[float] | None, coll: CollectionInfo,
                     f: Filter | None, limit: int) -> dict[str, Any]:
        if vector is None:
            raise BackendError("vector op reached the backend without an embedded query vector")
        spec = coll.field(field)
        if spec is None or spec.type is not FieldType.VECTOR:
            raise BackendError(f"{field!r} is not a knn_vector field of {coll.name}")
        knn: dict[str, Any] = {"vector": vector, "k": limit}
        if f is not None:
            knn["filter"] = filter_dsl(f)
        return {"size": limit, "query": {"knn": {field: knn}}}

    async def execute(self, op: QueryOp) -> list[Hit]:
        indices = await self._ensure_indices()
        if isinstance(op, Native):
            return await self._native(op, indices)
        coll = self._resolve(op.collection, indices)
        limit = min(op.limit, self.max_rows)
        if isinstance(op, Lexical):
            return await self._search(coll, self._lexical_body(op, coll, limit))
        if isinstance(op, Vector):
            return await self._search(coll, self._vector_body(op.field, op.vector, coll, op.filter, limit))
        if isinstance(op, Hybrid):
            depth = min(max(limit * 5, 50), 500)
            lex = await self._search(coll, self._lexical_body(op, coll, depth))
            vec = await self._search(coll, self._vector_body(op.field, op.vector, coll, op.filter, depth))
            by_id = {h.doc_id: h for h in [*vec, *lex]}
            fused = rrf_merge([([h.doc_id for h in lex], op.lexical_weight),
                               ([h.doc_id for h in vec], 1.0 - op.lexical_weight)])
            return [by_id[i].model_copy(update={"raw_score": s}) for i, s in fused[:limit]]
        if isinstance(op, FilterOnly):
            return await self._search(coll, {"size": limit, "query": {"bool": {"filter": [filter_dsl(op.filter)]}},
                                             "sort": [{"_doc": "asc"}]})
        if isinstance(op, Regex):
            fields = op.fields or [f.name for f in coll.fields if f.searchable]
            should = [{"regexp": {f: {"value": op.pattern, "case_insensitive": True}}} for f in fields]
            query = self._with_filter({"bool": {"should": should, "minimum_should_match": 1}}, op.filter)
            return await self._search(coll, {"size": limit, "query": query})
        if isinstance(op, Aggregate):
            return await self._aggregate(op, coll, limit)
        if isinstance(op, Fetch):
            resp = await self._call("mget", index=coll.name, body={"ids": op.doc_ids},
                                    _source_excludes=self._source_filter(coll).get("excludes"))
            return [self._hit(d, coll) for d in resp["docs"] if d.get("found")]
        raise UnsupportedOperation(f"opensearch backend does not support {op.type}")

    async def _aggregate(self, op: Aggregate, coll: CollectionInfo, limit: int) -> list[Hit]:
        columns = {f.name for f in coll.fields}
        unknown = set(op.group_by) - columns
        if unknown:
            raise BackendError(f"unknown fields {sorted(unknown)}")
        if not op.metrics:
            raise BackendError("at least one metric is required")
        sub: dict[str, Any] = {}
        names = []
        for m in op.metrics:
            _, alias = parse_metric(m, columns, "mysql")
            names.append(alias)
            if m != "count":
                fn, _, col = m.partition(":")
                sub[alias] = {fn: {"field": col}}
        if len(op.group_by) == 1:
            agg: dict[str, Any] = {"terms": {"field": op.group_by[0], "size": limit}}
        else:
            agg = {"multi_terms": {"terms": [{"field": g} for g in op.group_by], "size": limit}}
        if sub:
            agg["aggs"] = sub
        query = {"bool": {"filter": [filter_dsl(op.filter)]}} if op.filter else {"match_all": {}}
        try:
            resp = await self._call("search", index=coll.name,
                                    body={"size": 0, "query": query, "aggs": {"g": agg}})
            hits = []
            for bucket in resp["aggregations"]["g"]["buckets"]:
                key = bucket["key"] if isinstance(bucket["key"], list) else [bucket["key"]]
                data: dict[str, Any] = dict(zip(op.group_by, key))
                for alias in names:
                    data[alias] = bucket["doc_count"] if alias == "count" else bucket[alias]["value"]
                first = data[names[0]]
                hits.append(Hit(doc_id=f"agg:{dumps({g: data[g] for g in op.group_by})}", source=self.name,
                                content=[StructuredPart(data=data)],
                                raw_score=float(first) if first is not None else None))
            return hits
        except (KeyError, IndexError, TypeError) as exc:
            raise BackendError(f"unexpected OpenSearch response: {exc}") from exc

    async def _native(self, op: Native, indices: dict[str, CollectionInfo]) -> list[Hit]:
        if not self.native_query:
            raise UnsupportedOperation("native queries are disabled for this source")
        if op.dialect.lower() not in ("opensearch_dsl", "opensearch", "dsl", "json"):
            raise BackendError(f"dialect must be opensearch_dsl, got {op.dialect!r}")
        coll = self._resolve(op.collection, indices)
        body = guard_opensearch(op.query, min(op.limit, self.max_rows))
        return await self._search(coll, body)
