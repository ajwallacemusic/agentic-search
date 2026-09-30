"""Milvus backend: dense ANN search, BM25 full-text search (when a BM25 function exists), client-side
hybrid (RRF), boolean filter expressions and fetch by primary key. Values are passed as
`filter_params` templates except LIKE patterns, which Milvus only accepts as escaped literals.
Needs the `milvus` extra."""

from __future__ import annotations

import asyncio
import importlib
import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import unquote, urlparse

from agentic_search.backends.base import (
    BackendError,
    DiscoverDetail,
    UnsupportedOperation,
    routed_collection,
    rrf_merge,
    strip_collection,
)
from agentic_search.backends.sql import jsonable
from agentic_search.core.secrets import register_secret
from agentic_search.core.types import (
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
    Not,
    Or,
    QueryOp,
    Range,
    StructuredPart,
    TextPart,
    Vector,
)

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
SAMPLE_DISTINCT_MAX = 20
SAMPLE_SCAN_MAX = 1000
_VECTOR_TYPES = {"FLOAT_VECTOR", "FLOAT16_VECTOR", "BFLOAT16_VECTOR"}
_SCALAR_TYPES = {
    "INT8": FieldType.INT, "INT16": FieldType.INT, "INT32": FieldType.INT, "INT64": FieldType.INT,
    "FLOAT": FieldType.FLOAT, "DOUBLE": FieldType.FLOAT, "BOOL": FieldType.BOOL,
    "JSON": FieldType.JSON, "ARRAY": FieldType.JSON, "VARCHAR": FieldType.KEYWORD,
}


def _require_pymilvus() -> Any:
    try:
        return importlib.import_module("pymilvus")
    except ImportError as exc:
        raise BackendError(
            "MilvusBackend needs the `milvus` extra: pip install 'agentic-search[milvus]'") from exc


def _ident(name: str, fields: set[str]) -> str:
    if name not in fields or not _IDENT.fullmatch(name):
        raise BackendError(f"unknown or invalid field {name!r}")
    return name


def _like_literal(value: str) -> str:
    """A double-quoted Milvus string literal for LIKE '%value%' with wildcards escaped."""
    pattern = "%" + value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
    return '"' + pattern.replace("\\", "\\\\").replace('"', '\\"') + '"'


def filter_expr(f: Filter, fields: set[str], params: dict[str, Any]) -> str:
    """Filter AST → Milvus boolean expression; values go into `params` as {pN} templates."""
    def bind(value: Any) -> str:
        key = f"p{len(params)}"
        params[key] = value
        return "{" + key + "}"

    if isinstance(f, And):
        return "(" + " and ".join(filter_expr(c, fields, params) for c in f.clauses) + ")" \
            if f.clauses else "true"
    if isinstance(f, Or):
        return "(" + " or ".join(filter_expr(c, fields, params) for c in f.clauses) + ")" \
            if f.clauses else "false"
    if isinstance(f, Not):
        return f"(not ({filter_expr(f.clause, fields, params)}))"
    name = _ident(f.field, fields)
    if isinstance(f, Eq):
        return f"{name} == {bind(f.value)}"
    if isinstance(f, In):
        return f"{name} in {bind(list(f.values))}" if f.values else "false"
    if isinstance(f, Range):
        parts = [f"{name} {sym} {bind(b)}" for b, sym in
                 ((f.gte, ">="), (f.gt, ">"), (f.lte, "<="), (f.lt, "<")) if b is not None]
        return "(" + " and ".join(parts) + ")" if parts else "true"
    if isinstance(f, Exists):
        return f"{name} is not null"
    if isinstance(f, Contains):
        return f"{name} like {_like_literal(f.value)}"
    raise BackendError(f"unsupported filter node {type(f).__name__}")


@dataclass
class _Coll:
    info: CollectionInfo
    pk: str
    text_fields: list[str]
    output_fields: list[str]
    vector_metrics: dict[str, str] = field(default_factory=dict)
    sparse_field: str | None = None
    bm25_inputs: list[str] = field(default_factory=list)

    @property
    def fields(self) -> set[str]:
        return {f.name for f in self.info.fields}


def _type_name(t: Any) -> str:
    return getattr(t, "name", str(t)).upper()


def _truthy(v: Any) -> bool:
    return v is True or str(v).lower() == "true"


class MilvusBackend:
    backend_type = "milvus"

    def __init__(self, name: str, uri: str, *, token: str | None = None, db_name: str | None = None,
                 collections: list[str] | None = None, embedders: dict[str, str] | None = None,
                 description: str | None = None, max_rows: int = 100, sample_values: bool = True,
                 connect_timeout_s: float = 10.0):
        parsed = urlparse(uri)
        register_secret(token)
        register_secret(unquote(parsed.password or ""))
        self.name = name
        self.uri = uri
        self.token = token
        self.db_name = db_name
        self.collection_names = collections
        self.embedders = embedders or {}
        self.description = description
        self.max_rows = max_rows
        self.sample_values = sample_values
        self.connect_timeout_s = float(connect_timeout_s)
        self._client: Any = None
        self._collections: dict[str, _Coll] | None = None
        self._loaded: set[str] = set()
        self._lock = asyncio.Lock()

    # ---- client ---------------------------------------------------------------

    def _get_client(self) -> Any:
        if self._client is None:
            pymilvus = _require_pymilvus()
            self._client = pymilvus.AsyncMilvusClient(
                uri=self.uri, token=self.token or "", db_name=self.db_name or "",
                timeout=self.connect_timeout_s)
        return self._client

    async def _call(self, method: str, *args: Any, timeout: float | None = None, **kwargs: Any) -> Any:
        try:
            client = self._get_client()
            coro = getattr(client, method)(*args, **kwargs)
            return await asyncio.wait_for(coro, timeout) if timeout else await coro
        except (BackendError, UnsupportedOperation):
            raise
        except Exception as exc:  # pymilvus / grpc / timeout
            raise BackendError(f"{type(exc).__name__}: {exc}") from exc

    async def close(self) -> None:
        if self._client is not None:
            client, self._client = self._client, None
            try:
                await client.close()
            except Exception:
                pass

    # ---- discovery ---------------------------------------------------------------

    def capabilities(self) -> set[Capability]:
        caps = {Capability.FILTER, Capability.FETCH}
        colls = self._collections
        if colls is None or any(c.sparse_field for c in colls.values()):
            caps.add(Capability.LEXICAL)
        if colls is None or any(c.vector_metrics for c in colls.values()):
            caps.add(Capability.VECTOR)
        if colls is not None and any(c.sparse_field and c.vector_metrics for c in colls.values()):
            caps.add(Capability.HYBRID)
        return caps

    async def _ensure(self) -> dict[str, _Coll]:
        async with self._lock:
            if self._collections is None:
                self._collections = await self._discover_collections()
            return self._collections

    async def _discover_collections(self) -> dict[str, _Coll]:
        t = self.connect_timeout_s
        names = self.collection_names or await self._call("list_collections", timeout=t)
        out: dict[str, _Coll] = {}
        for name in names:
            desc = await self._call("describe_collection", name, timeout=t)
            out[name] = await self._collection(name, desc)
        return out

    async def _collection(self, name: str, desc: dict[str, Any]) -> _Coll:
        functions = desc.get("functions") or []
        bm25 = [fn for fn in functions if _type_name(fn.get("type")) == "BM25"]
        bm25_inputs = [i for fn in bm25 for i in fn.get("input_field_names", [])]
        sparse = bm25[0]["output_field_names"][0] if bm25 else None
        pk = next((f["name"] for f in desc["fields"] if f.get("is_primary")), None)
        if pk is None:
            raise BackendError(f"collection {name!r} has no primary key")
        metrics = await self._metrics(name)
        specs: list[FieldSpec] = []
        text_fields: list[str] = []
        output_fields: list[str] = []
        vector_metrics: dict[str, str] = {}
        for f in desc["fields"]:
            fname, tname = f["name"], _type_name(f["type"])
            params = f.get("params") or {}
            if tname == "SPARSE_FLOAT_VECTOR":
                continue
            if tname in _VECTOR_TYPES:
                metric = metrics.get(fname, "COSINE")
                vector_metrics[fname] = metric
                specs.append(FieldSpec(name=fname, type=FieldType.VECTOR, vector_dim=int(params["dim"])
                                       if params.get("dim") is not None else None,
                                       vector_metric=metric.lower(),
                                       embedder_id=self.embedders.get(f"{name}.{fname}")))
                continue
            ftype = _SCALAR_TYPES.get(tname, FieldType.JSON)
            if ftype is FieldType.KEYWORD and (fname in bm25_inputs or _truthy(params.get("enable_analyzer"))):
                ftype = FieldType.TEXT
                text_fields.append(fname)
            output_fields.append(fname)
            samples = None
            if ftype in (FieldType.KEYWORD, FieldType.BOOL) and fname != pk:
                samples = await self._samples(name, fname)
            specs.append(FieldSpec(
                name=fname, type=ftype, searchable=fname in bm25_inputs,
                filterable=ftype is not FieldType.JSON,
                sortable=ftype in (FieldType.INT, FieldType.FLOAT), sample_values=samples))
        count = await self._count(name)
        return _Coll(info=CollectionInfo(name=name, fields=specs, count=count), pk=pk,
                     text_fields=text_fields, output_fields=output_fields,
                     vector_metrics=vector_metrics, sparse_field=sparse, bm25_inputs=bm25_inputs)

    async def _metrics(self, name: str) -> dict[str, str]:
        out: dict[str, str] = {}
        try:
            for index in await self._call("list_indexes", name, timeout=self.connect_timeout_s):
                info = await self._call("describe_index", name, index, timeout=self.connect_timeout_s)
                if info.get("field_name") and info.get("metric_type"):
                    out[info["field_name"]] = str(info["metric_type"]).upper()
        except BackendError:
            return out
        return out

    async def _count(self, name: str) -> int | None:
        try:
            stats = await self._call("get_collection_stats", name, timeout=self.connect_timeout_s)
            return int(stats["row_count"])
        except (BackendError, KeyError, TypeError, ValueError):
            return None

    async def _samples(self, name: str, fname: str) -> list[Any] | None:
        if not self.sample_values:
            return None
        try:
            await self._load(name)
            rows = await self._call("query", name, filter=f"{_ident(fname, {fname})} is not null",
                                    output_fields=[fname], limit=SAMPLE_SCAN_MAX,
                                    consistency_level="Strong")
        except BackendError:
            return None
        values = list(dict.fromkeys(r.get(fname) for r in rows if r.get(fname) is not None))
        return values if len(values) <= SAMPLE_DISTINCT_MAX else None

    async def _load(self, name: str) -> None:
        if name not in self._loaded:
            await self._call("load_collection", name)
            self._loaded.add(name)

    async def discover(self, detail: DiscoverDetail = "full",
                       collection: str | None = None) -> Manifest:
        colls = await self._ensure()
        return Manifest(source=self.name, backend_type=self.backend_type,
                        capabilities=self.capabilities(), description=self.description,
                        collections=[c.info for c in colls.values()])

    # ---- execution ---------------------------------------------------------------

    def _resolve(self, name: str | None, colls: dict[str, _Coll]) -> _Coll:
        if name is None:
            if len(colls) == 1:
                return next(iter(colls.values()))
            raise BackendError(f"collection required; one of {sorted(colls)}")
        if name not in colls:
            raise BackendError(f"unknown collection {name!r}; one of {sorted(colls)}")
        return colls[name]

    def _doc_id(self, coll: _Coll, pk: Any, colls: dict[str, _Coll]) -> str:
        return f"{coll.info.name}/{pk}" if len(colls) > 1 else str(pk)

    def _hit(self, coll: _Coll, row: dict[str, Any], colls: dict[str, _Coll],
             score: float | None = None) -> Hit:
        clean = {k: jsonable(v) for k, v in row.items() if k in coll.output_fields}
        texts = [str(clean[f]) for f in coll.text_fields if clean.get(f) not in (None, "")]
        content: list[Content] = [TextPart(text="\n".join(texts))] if texts else [StructuredPart(data=clean)]
        metadata = {k: v for k, v in clean.items() if k not in coll.text_fields}
        return Hit(doc_id=self._doc_id(coll, row[coll.pk], colls), source=self.name, content=content,
                   metadata=metadata, raw_score=score)

    def _filter(self, f: Filter | None, coll: _Coll) -> tuple[str, dict[str, Any]]:
        params: dict[str, Any] = {}
        return (filter_expr(f, coll.fields, params) if f is not None else ""), params

    async def _search(self, coll: _Coll, data: Any, anns_field: str, metric: str, f: Filter | None,
                      limit: int, colls: dict[str, _Coll]) -> list[Hit]:
        expr, params = self._filter(f, coll)
        await self._load(coll.info.name)
        kwargs: dict[str, Any] = {"filter_params": params} if params else {}
        results = await self._call("search", coll.info.name, data=[data], anns_field=anns_field,
                                   filter=expr, limit=limit, output_fields=coll.output_fields,
                                   search_params={"metric_type": metric}, consistency_level="Strong",
                                   **kwargs)
        hits = []
        for r in (results[0] if results else []):
            row = dict(r.get("entity") or {})
            row.setdefault(coll.pk, r.get("id"))
            distance = float(r.get("distance", 0.0))
            hits.append(self._hit(coll, row, colls, -distance if metric == "L2" else distance))
        return hits

    async def _lexical(self, op: Lexical | Hybrid, coll: _Coll, limit: int,
                       colls: dict[str, _Coll]) -> list[Hit]:
        if coll.sparse_field is None:
            raise UnsupportedOperation(f"collection {coll.info.name!r} has no BM25 function")
        fields = getattr(op, "fields", None)
        if fields and not set(fields) <= set(coll.bm25_inputs):
            raise BackendError(f"lexical search covers only {coll.bm25_inputs} in {coll.info.name!r}")
        return await self._search(coll, op.text, coll.sparse_field, "BM25", op.filter, limit, colls)

    async def _vector(self, field_name: str, vector: list[float] | None, coll: _Coll, f: Filter | None,
                      limit: int, colls: dict[str, _Coll]) -> list[Hit]:
        if vector is None:
            raise BackendError("vector op reached the backend without an embedded query vector")
        if field_name not in coll.vector_metrics:
            raise BackendError(f"{field_name!r} is not a vector field of {coll.info.name}")
        return await self._search(coll, vector, field_name, coll.vector_metrics[field_name], f, limit, colls)

    async def execute(self, op: QueryOp) -> list[Hit]:
        colls = await self._ensure()
        if not isinstance(op, (Lexical, Vector, Hybrid, FilterOnly, Fetch)):
            raise UnsupportedOperation(f"milvus backend does not support {op.type}")
        if isinstance(op, Fetch):
            return await self._fetch(op, colls)
        coll = self._resolve(op.collection, colls)
        limit = min(op.limit, self.max_rows)
        if isinstance(op, Lexical):
            return await self._lexical(op, coll, limit, colls)
        if isinstance(op, Vector):
            return await self._vector(op.field, op.vector, coll, op.filter, limit, colls)
        if isinstance(op, Hybrid):
            depth = min(max(limit * 5, 50), 500)
            lex = await self._lexical(op, coll, depth, colls)
            vec = await self._vector(op.field, op.vector, coll, op.filter, depth, colls)
            by_id = {h.doc_id: h for h in [*vec, *lex]}
            fused = rrf_merge([([h.doc_id for h in lex], op.lexical_weight),
                               ([h.doc_id for h in vec], 1.0 - op.lexical_weight)])
            return [by_id[i].model_copy(update={"raw_score": s}) for i, s in fused[:limit]]
        expr, params = self._filter(op.filter, coll)
        await self._load(coll.info.name)
        kwargs: dict[str, Any] = {"filter_params": params} if params else {}
        rows = await self._call("query", coll.info.name, filter=expr, output_fields=coll.output_fields,
                                limit=limit, consistency_level="Strong", **kwargs)
        return [self._hit(coll, r, colls) for r in rows]

    async def _fetch(self, op: Fetch, colls: dict[str, _Coll]) -> list[Hit]:
        coll = self._resolve(routed_collection(op, colls), colls)
        ids = [strip_collection(i, coll.info.name) if len(colls) > 1 else i for i in op.doc_ids]
        await self._load(coll.info.name)
        rows = await self._call("query", coll.info.name, filter=f"{coll.pk} in {{ids}}",
                                filter_params={"ids": ids}, output_fields=coll.output_fields,
                                consistency_level="Strong")
        order = {str(i): n for n, i in enumerate(ids)}
        hits = [self._hit(coll, r, colls) for r in rows]
        return sorted(hits, key=lambda h: order.get(h.doc_id.split("/", 1)[-1] if len(colls) > 1
                                                    else h.doc_id, len(order)))
