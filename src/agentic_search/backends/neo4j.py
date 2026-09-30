"""Neo4j backend: full-text and vector index search, property filters, regex, aggregates, fetch,
graph traversal and guarded native Cypher. Collections are node labels. All work runs in READ
transactions, so the server refuses writes even if the native guard were bypassed. Needs the
`neo4j` extra."""

from __future__ import annotations

import asyncio
import importlib
from dataclasses import dataclass, field
from typing import Any

from agentic_search.backends.base import (
    BackendError,
    DiscoverDetail,
    UnsupportedOperation,
    routed_collection,
    rrf_merge,
    strip_collection,
)
from agentic_search.backends.native_guard import guard_cypher
from agentic_search.backends.sql import dumps, jsonable
from agentic_search.backends.sql_backend import any_terms
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
    Traverse,
    Vector,
)

SAMPLE_DISTINCT_MAX = 20
SAMPLE_SCAN_MAX = 10_000
_AGG_FUNCS = {"sum": "sum", "avg": "avg", "min": "min", "max": "max"}
_DATE_TYPES = {"Date", "DateTime", "LocalDateTime", "ZonedDateTime", "LocalTime", "Time"}
_ARRAY_TYPES = {"DoubleArray", "FloatArray", "LongArray", "IntegerArray"}


def _require_neo4j() -> Any:
    try:
        return importlib.import_module("neo4j")
    except ImportError as exc:
        raise BackendError(
            "Neo4jBackend needs the `neo4j` extra: pip install 'agentic-search[neo4j]'") from exc


def quote_name(name: str) -> str:
    """Backtick-quote a label, property, relationship type or index name."""
    return "`" + name.replace("`", "``") + "`"


def _param(params: dict[str, Any], value: Any) -> str:
    name = f"p{len(params)}"
    params[name] = value
    return f"${name}"


def filter_cypher(f: Filter, params: dict[str, Any], properties: set[str], var: str) -> str:
    """Filter AST → parameterized Cypher boolean expression over `var`'s known properties."""
    if isinstance(f, And):
        return "(" + " AND ".join(filter_cypher(c, params, properties, var) for c in f.clauses) + ")" \
            if f.clauses else "true"
    if isinstance(f, Or):
        return "(" + " OR ".join(filter_cypher(c, params, properties, var) for c in f.clauses) + ")" \
            if f.clauses else "false"
    if isinstance(f, Not):
        return f"(NOT {filter_cypher(f.clause, params, properties, var)})"
    if f.field not in properties:
        raise BackendError(f"unknown property {f.field!r}")
    prop = f"{var}.{quote_name(f.field)}"
    if isinstance(f, Eq):
        return f"{prop} = {_param(params, f.value)}"
    if isinstance(f, In):
        return f"{prop} IN {_param(params, list(f.values))}" if f.values else "false"
    if isinstance(f, Range):
        parts = [f"{prop} {sym} {_param(params, bound)}" for bound, sym in
                 ((f.gte, ">="), (f.gt, ">"), (f.lte, "<="), (f.lt, "<")) if bound is not None]
        return "(" + " AND ".join(parts) + ")" if parts else "true"
    if isinstance(f, Exists):
        return f"{prop} IS NOT NULL"
    if isinstance(f, Contains):
        return f"toLower(toString({prop})) CONTAINS toLower({_param(params, f.value)})"
    raise BackendError(f"unsupported filter node {type(f).__name__}")


def lucene_query(text: str) -> str:
    """OR of lowercase word terms. Lowercasing neutralizes Lucene's AND/OR/NOT operators, and
    word tokens contain no Lucene special characters."""
    terms = any_terms(text)
    if not terms:
        raise BackendError("lexical query has no searchable terms")
    return " OR ".join(terms)


def _plain(value: Any) -> Any:
    """Neo4j temporal/spatial values → JSON-friendly."""
    if hasattr(value, "iso_format"):
        return value.iso_format()
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    return jsonable(value)


@dataclass
class _Label:
    name: str
    properties: dict[str, FieldSpec | None]
    fulltext: list[tuple[str, list[str]]] = field(default_factory=list)  # (index, properties)
    vectors: dict[str, str] = field(default_factory=dict)  # property -> index name
    count: int | None = None

    def fields(self) -> list[FieldSpec]:
        return [f for f in self.properties.values() if f is not None]

    def text_properties(self) -> list[str]:
        return [f.name for f in self.fields() if f.type is FieldType.TEXT]

    def vector_properties(self) -> set[str]:
        return {f.name for f in self.fields() if f.type is FieldType.VECTOR}


@dataclass
class _Schema:
    labels: dict[str, _Label]
    relationship_types: list[str]
    all_labels: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.all_labels = self.all_labels or set(self.labels)


def _flags(ftype: FieldType) -> dict[str, bool]:
    return {"searchable": ftype is FieldType.TEXT,
            "filterable": ftype not in (FieldType.JSON, FieldType.VECTOR),
            "sortable": ftype in (FieldType.INT, FieldType.FLOAT, FieldType.DATE)}


class Neo4jBackend:
    backend_type = "neo4j"

    def __init__(self, name: str, uri: str, *, user: str | None = None, password: str | None = None,
                 database: str | None = None, labels: list[str] | None = None,
                 id_property: str = "id", embedders: dict[str, str] | None = None,
                 native_query: bool = False, description: str | None = None, max_rows: int = 100,
                 sample_values: bool = True, connect_timeout_s: float = 10):
        register_secret(uri)
        register_secret(password)
        self.name = name
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.label_names = labels
        self.id_property = id_property
        self.embedders = embedders or {}
        self.native_query = native_query
        self.description = description
        self.max_rows = max_rows
        self.sample_values = sample_values
        self.connect_timeout_s = float(connect_timeout_s)
        self._driver: Any = None
        self._schema: _Schema | None = None
        self._lock = asyncio.Lock()

    # ---- connection -------------------------------------------------------------

    def _get_driver(self) -> Any:
        if self._driver is None:
            neo4j = _require_neo4j()
            try:
                self._driver = neo4j.AsyncGraphDatabase.driver(
                    self.uri, auth=(self.user, self.password) if self.user else None,
                    connection_timeout=self.connect_timeout_s,
                    max_transaction_retry_time=self.connect_timeout_s,
                    notifications_min_severity="OFF")
            except Exception as exc:
                raise BackendError(f"{type(exc).__name__}: {exc}") from exc
        return self._driver

    async def _read(self, query: str, params: dict[str, Any] | None = None) -> list[Any]:
        """Run one query in a READ transaction; returns records."""
        async def work(tx: Any) -> list[Any]:
            result = await tx.run(query, params or {})
            return [record async for record in result]

        try:
            driver = self._get_driver()
            neo4j = _require_neo4j()
            async with driver.session(database=self.database,
                                      default_access_mode=neo4j.READ_ACCESS) as session:
                return await session.execute_read(work)
        except BackendError:
            raise
        except Exception as exc:
            raise BackendError(f"{type(exc).__name__}: {exc}") from exc

    async def close(self) -> None:
        if self._driver is not None:
            driver, self._driver = self._driver, None
            try:
                await driver.close()
            except Exception:
                pass

    # ---- discovery ------------------------------------------------------------

    def capabilities(self) -> set[Capability]:
        caps = {Capability.FILTER, Capability.REGEX, Capability.AGGREGATE, Capability.FETCH,
                Capability.TRAVERSE}
        labels = self._schema.labels.values() if self._schema else []
        if self._schema is None or any(lbl.fulltext for lbl in labels):
            caps.add(Capability.LEXICAL)
        if any(lbl.vectors for lbl in labels):
            caps |= {Capability.VECTOR, Capability.HYBRID}
        if self.native_query:
            caps.add(Capability.NATIVE)
        return caps

    async def _ensure_schema(self) -> _Schema:
        async with self._lock:
            if self._schema is None:
                self._schema = await self._discover_schema()
            return self._schema

    async def _discover_schema(self) -> _Schema:
        all_labels = {r["label"] for r in await self._read("CALL db.labels() YIELD label RETURN label")}
        wanted = [lbl for lbl in (self.label_names or sorted(all_labels)) if lbl in all_labels]
        rel_types = sorted(r["t"] for r in await self._read(
            "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType AS t"))
        indexes = await self._read(
            "SHOW INDEXES YIELD name, type, labelsOrTypes, properties, options, state "
            "RETURN name, type, labelsOrTypes, properties, options, state")
        props = await self._read(
            "CALL db.schema.nodeTypeProperties() YIELD nodeLabels, propertyName, propertyTypes "
            "RETURN nodeLabels, propertyName, propertyTypes")
        labels = {name: _Label(name=name, properties={}) for name in wanted}
        vector_meta: dict[tuple[str, str], tuple[int | None, str | None]] = {}
        for ix in indexes:
            if ix["state"] != "ONLINE" or not ix["labelsOrTypes"]:
                continue
            for lbl in ix["labelsOrTypes"]:
                if lbl not in labels:
                    continue
                if ix["type"] == "FULLTEXT":
                    labels[lbl].fulltext.append((ix["name"], list(ix["properties"])))
                elif ix["type"] == "VECTOR" and len(ix["properties"]) == 1:
                    prop = ix["properties"][0]
                    labels[lbl].vectors[prop] = ix["name"]
                    cfg = (ix["options"] or {}).get("indexConfig", {})
                    sim = cfg.get("vector.similarity_function")
                    vector_meta[(lbl, prop)] = (cfg.get("vector.dimensions"),
                                                sim.lower() if isinstance(sim, str) else None)
        for row in props:
            name = row["propertyName"]
            if name is None:
                continue
            types = row["propertyTypes"] or []
            for lbl in row["nodeLabels"] or []:
                if lbl in labels and name not in labels[lbl].properties:
                    labels[lbl].properties[name] = self._field(labels[lbl], name, types, vector_meta)
        for lbl in labels.values():
            lbl.count = await self._count(lbl.name)
            for spec in lbl.fields():
                if spec.type in (FieldType.KEYWORD, FieldType.BOOL) and spec.name != self.id_property:
                    spec.sample_values = await self._samples(lbl.name, spec.name)
        return _Schema(labels=labels, relationship_types=rel_types, all_labels=all_labels)

    def _field(self, lbl: _Label, name: str, types: list[str],
               vector_meta: dict[tuple[str, str], tuple[int | None, str | None]]) -> FieldSpec:
        ptype = types[0] if types else "String"
        in_fulltext = any(name in idx_props for _, idx_props in lbl.fulltext)
        if name in lbl.vectors:
            dim, sim = vector_meta.get((lbl.name, name), (None, None))
            return FieldSpec(name=name, type=FieldType.VECTOR, vector_dim=dim, vector_metric=sim,
                             embedder_id=self.embedders.get(f"{lbl.name}.{name}"))
        if ptype == "String":
            ftype = FieldType.TEXT if in_fulltext else FieldType.KEYWORD
        elif ptype in ("Long", "Integer"):
            ftype = FieldType.INT
        elif ptype in ("Double", "Float"):
            ftype = FieldType.FLOAT
        elif ptype == "Boolean":
            ftype = FieldType.BOOL
        elif ptype in _DATE_TYPES:
            ftype = FieldType.DATE
        else:
            ftype = FieldType.JSON
        return FieldSpec(name=name, type=ftype, **_flags(ftype))

    async def _count(self, label: str) -> int | None:
        try:
            rows = await self._read(f"MATCH (n:{quote_name(label)}) RETURN count(n) AS c")
        except BackendError:
            return None
        return int(rows[0]["c"]) if rows else None

    async def _samples(self, label: str, prop: str) -> list[Any] | None:
        if not self.sample_values:
            return None
        p = f"n.{quote_name(prop)}"
        try:
            rows = await self._read(
                f"MATCH (n:{quote_name(label)}) WHERE {p} IS NOT NULL WITH {p} AS v LIMIT $scan "
                f"RETURN DISTINCT v LIMIT $k", {"scan": SAMPLE_SCAN_MAX, "k": SAMPLE_DISTINCT_MAX + 1})
        except BackendError:
            return None
        values = [_plain(r["v"]) for r in rows]
        return values if len(values) <= SAMPLE_DISTINCT_MAX else None

    async def discover(self, detail: DiscoverDetail = "full",
                       collection: str | None = None) -> Manifest:
        schema = await self._ensure_schema()
        return Manifest(
            source=self.name, backend_type=self.backend_type, capabilities=self.capabilities(),
            description=self.description, relationship_types=schema.relationship_types,
            collections=[CollectionInfo(name=lbl.name, fields=lbl.fields(), count=lbl.count)
                         for lbl in schema.labels.values()])

    # ---- helpers ----------------------------------------------------------------

    def _resolve(self, name: str | None, schema: _Schema) -> _Label:
        if name is None:
            if len(schema.labels) == 1:
                return next(iter(schema.labels.values()))
            raise BackendError(f"collection required; one of {sorted(schema.labels)}")
        if name not in schema.labels:
            raise BackendError(f"unknown collection {name!r}; one of {sorted(schema.labels)}")
        return schema.labels[name]

    def _where(self, f: Filter | None, params: dict[str, Any], lbl: _Label, var: str = "n",
               extra: list[str] | None = None) -> str:
        parts = list(extra or [])
        if f is not None:
            parts.append(filter_cypher(f, params, set(lbl.properties), var))
        return " WHERE " + " AND ".join(parts) if parts else ""

    def _hit(self, node: Any, element_id: str, lbl: _Label, multi: bool,
             score: float | None = None) -> Hit:
        props = {k: _plain(v) for k, v in dict(node).items()}
        vectors = lbl.vector_properties()
        texts = lbl.text_properties()
        pk = props.get(self.id_property)
        doc_id = str(pk) if pk is not None else element_id
        if multi:
            doc_id = f"{lbl.name}/{doc_id}"
        text = [str(props[t]) for t in texts if props.get(t) not in (None, "")]
        metadata = {k: v for k, v in props.items() if k not in vectors and k not in texts}
        content: list[Content] = [TextPart(text="\n".join(text))] if text else [StructuredPart(data=metadata)]
        return Hit(doc_id=doc_id, source=self.name, content=content, metadata=metadata,
                   raw_score=float(score) if score is not None else None)

    def _hits(self, rows: list[Any], lbl: _Label, schema: _Schema, score_key: str = "score") -> list[Hit]:
        multi = len(schema.labels) > 1
        return [self._hit(r["n"], r["eid"], lbl, multi,
                          r[score_key] if score_key in r.keys() else None) for r in rows]

    def _order(self, lbl: _Label) -> str:
        if self.id_property in lbl.properties:
            return f"n.{quote_name(self.id_property)}"
        return "elementId(n)"

    # ---- execution ---------------------------------------------------------------

    async def execute(self, op: QueryOp) -> list[Hit]:
        schema = await self._ensure_schema()
        if isinstance(op, Native):
            return await self._native(op, schema)
        if isinstance(op, Traverse):
            return await self._traverse(op, schema)
        lbl = self._resolve(routed_collection(op, schema.labels) if isinstance(op, Fetch)
                            else op.collection, schema)
        limit = min(op.limit, self.max_rows)
        if isinstance(op, Lexical):
            return self._hits(await self._read(*self._lexical(op.text, op.fields, op.filter, lbl, limit)),
                              lbl, schema)
        if isinstance(op, Vector):
            return self._hits(await self._read(*self._vector(op.field, op.vector, op.filter, lbl, limit)),
                              lbl, schema)
        if isinstance(op, Hybrid):
            return await self._hybrid(op, lbl, schema, limit)
        if isinstance(op, FilterOnly):
            params: dict[str, Any] = {"lim": limit}
            q = (f"MATCH (n:{quote_name(lbl.name)}){self._where(op.filter, params, lbl)} "
                 f"RETURN n, elementId(n) AS eid ORDER BY {self._order(lbl)} LIMIT $lim")
            return self._hits(await self._read(q, params), lbl, schema)
        if isinstance(op, Regex):
            return await self._regex(op, lbl, schema, limit)
        if isinstance(op, Aggregate):
            return await self._aggregate(op, lbl, limit)
        if isinstance(op, Fetch):
            return await self._fetch(op, lbl, schema)
        raise UnsupportedOperation(f"neo4j backend does not support {op.type}")

    def _lexical(self, text: str, fields: list[str] | None, f: Filter | None, lbl: _Label,
                 limit: int) -> tuple[str, dict[str, Any]]:
        if not lbl.fulltext:
            raise BackendError(f"label {lbl.name} has no FULLTEXT index")
        if fields:
            index = next((name for name, props in lbl.fulltext if set(props) == set(fields)), None)
            if index is None:
                raise BackendError(f"no FULLTEXT index on exactly {sorted(fields)}; "
                                   f"indexes: {[props for _, props in lbl.fulltext]}")
        else:
            index = lbl.fulltext[0][0]
        params: dict[str, Any] = {"idx": index, "q": lucene_query(text), "lim": limit}
        where = self._where(f, params, lbl, extra=[f"n:{quote_name(lbl.name)}"])
        q = (f"CALL db.index.fulltext.queryNodes($idx, $q) YIELD node AS n, score{where} "
             f"RETURN n, elementId(n) AS eid, score ORDER BY score DESC LIMIT $lim")
        return q, params

    def _vector(self, prop: str, vector: list[float] | None, f: Filter | None, lbl: _Label,
                limit: int) -> tuple[str, dict[str, Any]]:
        if vector is None:
            raise BackendError("vector op reached the backend without an embedded query vector")
        if prop not in lbl.vectors:
            raise BackendError(f"{prop!r} is not a vector-indexed property of {lbl.name}")
        k = limit if f is None else min(max(limit * 10, 100), 1000)
        params: dict[str, Any] = {"idx": lbl.vectors[prop], "k": k, "vec": list(vector), "lim": limit}
        where = self._where(f, params, lbl)
        q = (f"CALL db.index.vector.queryNodes($idx, $k, $vec) YIELD node AS n, score{where} "
             f"RETURN n, elementId(n) AS eid, score ORDER BY score DESC LIMIT $lim")
        return q, params

    async def _hybrid(self, op: Hybrid, lbl: _Label, schema: _Schema, limit: int) -> list[Hit]:
        depth = min(max(limit * 5, 50), 500)
        lex = self._hits(await self._read(*self._lexical(op.text, None, op.filter, lbl, depth)), lbl, schema)
        vec = self._hits(await self._read(*self._vector(op.field, op.vector, op.filter, lbl, depth)),
                         lbl, schema)
        by_id = {h.doc_id: h for h in [*vec, *lex]}
        fused = rrf_merge([([h.doc_id for h in lex], op.lexical_weight),
                           ([h.doc_id for h in vec], 1.0 - op.lexical_weight)])
        return [by_id[i].model_copy(update={"raw_score": s}) for i, s in fused[:limit]]

    async def _regex(self, op: Regex, lbl: _Label, schema: _Schema, limit: int) -> list[Hit]:
        strings = [f.name for f in lbl.fields() if f.type in (FieldType.TEXT, FieldType.KEYWORD)]
        props = op.fields or lbl.text_properties() or strings
        unknown = set(props) - set(lbl.properties)
        if unknown:
            raise BackendError(f"unknown properties {sorted(unknown)}")
        params: dict[str, Any] = {"pat": f"(?s).*(?:{op.pattern}).*", "lim": limit}
        ors = " OR ".join(f"toString(n.{quote_name(p)}) =~ $pat" for p in props)
        q = (f"MATCH (n:{quote_name(lbl.name)}){self._where(op.filter, params, lbl, extra=[f'({ors})'])} "
             f"RETURN n, elementId(n) AS eid ORDER BY {self._order(lbl)} LIMIT $lim")
        return self._hits(await self._read(q, params), lbl, schema)

    async def _aggregate(self, op: Aggregate, lbl: _Label, limit: int) -> list[Hit]:
        if not op.metrics:
            raise BackendError("at least one metric is required")
        unknown = set(op.group_by) - set(lbl.properties)
        if unknown:
            raise BackendError(f"unknown properties {sorted(unknown)}")
        returns = [f"n.{quote_name(g)} AS {quote_name(g)}" for g in op.group_by]
        aliases = []
        for m in op.metrics:
            if m == "count":
                returns.append("count(*) AS `count`")
                aliases.append("count")
                continue
            fn, _, prop = m.partition(":")
            if fn not in _AGG_FUNCS or not prop:
                raise BackendError(f"unsupported metric {m!r}; use count or sum|avg|min|max:<property>")
            if prop not in lbl.properties:
                raise BackendError(f"unknown property {prop!r} in metric {m!r}")
            alias = f"{fn}_{prop}"
            returns.append(f"{_AGG_FUNCS[fn]}(n.{quote_name(prop)}) AS {quote_name(alias)}")
            aliases.append(alias)
        params: dict[str, Any] = {"lim": limit}
        q = (f"MATCH (n:{quote_name(lbl.name)}){self._where(op.filter, params, lbl)} "
             f"RETURN {', '.join(returns)} ORDER BY {quote_name(aliases[0])} DESC LIMIT $lim")
        hits = []
        for row in await self._read(q, params):
            data = {k: _plain(row[k]) for k in row.keys()}
            first = data.get(aliases[0])
            hits.append(Hit(doc_id=f"agg:{dumps({g: data.get(g) for g in op.group_by})}", source=self.name,
                            content=[StructuredPart(data=data)],
                            raw_score=float(first) if first is not None else None))
        return hits

    async def _fetch(self, op: Fetch, lbl: _Label, schema: _Schema) -> list[Hit]:
        multi = len(schema.labels) > 1
        pks = [strip_collection(i, lbl.name) if multi else i for i in op.doc_ids]
        params: dict[str, Any] = {"ids": pks}
        if self.id_property in lbl.properties:
            cond = f"toString(n.{quote_name(self.id_property)}) IN $ids"
        else:
            cond = "elementId(n) IN $ids"
        q = f"MATCH (n:{quote_name(lbl.name)}) WHERE {cond} RETURN n, elementId(n) AS eid"
        hits = self._hits(await self._read(q, params), lbl, schema)
        order = {(f"{lbl.name}/{pk}" if multi else pk): i for i, pk in enumerate(pks)}
        return sorted(hits, key=lambda h: order.get(h.doc_id, len(order)))

    async def _traverse(self, op: Traverse, schema: _Schema) -> list[Hit]:
        start_lbl = self._resolve(op.collection, schema) if op.collection is not None else None
        unknown_rels = set(op.rel_types) - set(schema.relationship_types)
        if unknown_rels:
            raise BackendError(f"unknown relationship types {sorted(unknown_rels)}; "
                               f"known: {schema.relationship_types}")
        if op.target_label is not None and op.target_label not in schema.labels:
            raise BackendError(f"unknown target label {op.target_label!r}; one of {sorted(schema.labels)}")
        start_props = set(start_lbl.properties) if start_lbl else \
            {p for lbl in schema.labels.values() for p in lbl.properties}
        params: dict[str, Any] = {"lim": min(op.limit, self.max_rows)}
        start_cond = filter_cypher(op.start, params, start_props, "s")
        filters = [start_cond]
        if op.filter is not None and op.target_label is not None:
            filters.append(filter_cypher(op.filter, params, set(schema.labels[op.target_label].properties), "n"))
        rels = "|".join(quote_name(t) for t in op.rel_types)
        rel = f"[{':' + rels if rels else ''}*1..{int(op.depth)}]"
        s = f"(s{':' + quote_name(start_lbl.name) if start_lbl else ''})"
        n = f"(n{':' + quote_name(op.target_label) if op.target_label else ''})"
        pattern = {"out": f"{s}-{rel}->{n}", "in": f"{s}<-{rel}-{n}", "both": f"{s}-{rel}-{n}"}[op.direction]
        q = (f"MATCH p = {pattern} WHERE {' AND '.join(filters)} "
             f"WITH n, min(length(p)) AS hops RETURN n, elementId(n) AS eid, labels(n) AS labels, hops "
             f"ORDER BY hops, eid LIMIT $lim")
        multi = len(schema.labels) > 1
        hits = []
        for row in await self._read(q, params):
            label = next((lbl for lbl in row["labels"] if lbl in schema.labels), None)
            if label is None:
                continue
            hits.append(self._hit(row["n"], row["eid"], schema.labels[label], multi,
                                  1.0 / max(int(row["hops"]), 1)))
        return hits

    async def _native(self, op: Native, schema: _Schema) -> list[Hit]:
        if not self.native_query:
            raise UnsupportedOperation("native queries are disabled for this source")
        if op.dialect.lower() not in ("cypher", "neo4j"):
            raise BackendError(f"dialect must be cypher, got {op.dialect!r}")
        query = guard_cypher(op.query, min(op.limit, self.max_rows))
        multi = len(schema.labels) > 1
        hits = []
        for i, row in enumerate(await self._read(query)):
            node_hit = None
            for key in row.keys():
                value = row[key]
                labels = getattr(value, "labels", None)
                if labels is not None and hasattr(value, "element_id"):
                    label = next((lbl for lbl in labels if lbl in schema.labels), None)
                    if label is not None:
                        node_hit = self._hit(value, value.element_id, schema.labels[label], multi)
                        break
            if node_hit is None:
                data = {k: _plain(row[k]) for k in row.keys()}
                node_hit = Hit(doc_id=f"native:{i}", source=self.name,
                               content=[StructuredPart(data=data)], metadata={})
            hits.append(node_hit)
        return hits
