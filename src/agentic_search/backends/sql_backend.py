"""Base class for SQL backends. Subclasses supply connection handling, table discovery and
dialect-specific lexical/vector SQL; this class implements everything else once."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any

from agentic_search.backends.base import (
    BackendError,
    DiscoverDetail,
    UnsupportedOperation,
    rrf_merge,
)
from agentic_search.backends.native_guard import guard_sql
from agentic_search.backends.sql import (
    Dialect,
    Params,
    dumps,
    parse_metric,
    quote_ident,
    row_to_hit,
    where_clause,
)
from agentic_search.core.types import (
    Aggregate,
    Capability,
    CollectionInfo,
    Fetch,
    FieldSpec,
    FieldType,
    FilterOnly,
    Hit,
    Hybrid,
    Lexical,
    Manifest,
    Native,
    QueryOp,
    Regex,
    StructuredPart,
    Vector,
)

METRICS = ("cosine", "l2", "ip")
_WORD = re.compile(r"\w+")
SAMPLE_DISTINCT_MAX = 20
SAMPLE_SCAN_MAX = 10_000
EXACT_COUNT_BELOW = 100_000


@dataclass
class TableInfo:
    name: str
    id_column: str
    fields: list[FieldSpec]
    count: int | None = None
    fulltext: list[list[str]] = field(default_factory=list)  # MySQL FULLTEXT column sets

    @property
    def columns(self) -> set[str]:
        return {f.name for f in self.fields}

    @property
    def text_columns(self) -> list[str]:
        return [f.name for f in self.fields if f.type is FieldType.TEXT]

    @property
    def searchable_columns(self) -> list[str]:
        return [f.name for f in self.fields if f.searchable]

    @property
    def vector_columns(self) -> dict[str, FieldSpec]:
        return {f.name: f for f in self.fields if f.type is FieldType.VECTOR}

    @property
    def select_columns(self) -> list[str]:
        return [f.name for f in self.fields if f.type is not FieldType.VECTOR]

    def to_collection(self) -> CollectionInfo:
        return CollectionInfo(name=self.name, fields=self.fields, count=self.count)


def field_flags(ftype: FieldType) -> dict[str, bool]:
    return {
        "searchable": ftype in (FieldType.TEXT, FieldType.KEYWORD),
        "filterable": ftype not in (FieldType.JSON, FieldType.VECTOR),
        "sortable": ftype in (FieldType.INT, FieldType.FLOAT, FieldType.DATE),
    }


def any_terms(text: str) -> list[str]:
    """Distinct lowercase word tokens, capped — lexical search is OR-of-terms, ranked."""
    return list(dict.fromkeys(t.lower() for t in _WORD.findall(text)))[:16]


class SqlBackend:
    dialect: Dialect
    backend_type: str
    native_dialects: tuple[str, ...] = ("sql",)

    def __init__(self, name: str, *, tables: list[str] | None = None,
                 id_columns: dict[str, str] | None = None, embedders: dict[str, str] | None = None,
                 vector_metric: str = "cosine", native_query: bool = False,
                 description: str | None = None, max_rows: int = 100, sample_values: bool = True):
        if vector_metric not in METRICS:
            raise ValueError(f"vector_metric must be one of {METRICS}")
        self.name = name
        self.table_names = tables
        self.id_columns = id_columns or {}
        self.embedders = embedders or {}
        self.vector_metric = vector_metric
        self.native_query = native_query
        self.description = description
        self.max_rows = max_rows
        self.sample_values = sample_values
        self.skipped: list[str] = []
        self._tables: dict[str, TableInfo] | None = None
        self._discover_lock = asyncio.Lock()

    # ---- subclass hooks -------------------------------------------------------

    async def _query(self, sql: str, params: list[Any] | None) -> list[dict[str, Any]]:
        raise NotImplementedError

    async def _discover_tables(self) -> dict[str, TableInfo]:
        raise NotImplementedError

    def _table_ref(self, table: str) -> str:
        return quote_ident(table, self.dialect)

    def _as_text(self, expr: str) -> str:
        return f"CAST({expr} AS {'TEXT' if self.dialect == 'postgres' else 'CHAR'})"

    def _regex_expr(self, column_sql: str, placeholder: str) -> str:
        raise NotImplementedError

    def _lexical_sql(self, op: Lexical, table: TableInfo, limit: int) -> tuple[str, list[Any]]:
        raise NotImplementedError

    def _vector_sql(self, op_field: str, vector: list[float], table: TableInfo, op_filter: Any,
                    limit: int) -> tuple[str, list[Any]]:
        raise UnsupportedOperation(f"{self.backend_type} backend does not support vector search")

    def _supports_lexical(self, tables: dict[str, TableInfo]) -> bool:
        return any(t.searchable_columns for t in tables.values())

    # ---- protocol ---------------------------------------------------------------

    def capabilities(self) -> set[Capability]:
        caps = {Capability.FILTER, Capability.REGEX, Capability.AGGREGATE, Capability.FETCH}
        tables = self._tables or {}
        if self._tables is None or self._supports_lexical(tables):
            caps.add(Capability.LEXICAL)
        if any(t.vector_columns for t in tables.values()):
            caps |= {Capability.VECTOR, Capability.HYBRID}
        if self.native_query:
            caps.add(Capability.NATIVE)
        return caps

    async def _ensure_tables(self) -> dict[str, TableInfo]:
        async with self._discover_lock:
            if self._tables is None:
                self._tables = await self._discover_tables()
            return self._tables

    async def discover(self, detail: DiscoverDetail = "full",
                       collection: str | None = None) -> Manifest:
        tables = await self._ensure_tables()
        description = self.description
        if self.skipped:
            note = f"Skipped tables: {', '.join(self.skipped)}."
            description = f"{description} {note}" if description else note
        return Manifest(source=self.name, backend_type=self.backend_type,
                        capabilities=self.capabilities(), description=description,
                        collections=[t.to_collection() for t in tables.values()])

    async def execute(self, op: QueryOp) -> list[Hit]:
        tables = await self._ensure_tables()
        if isinstance(op, Native):
            return await self._native(op, tables)
        table = self._resolve(op.collection, tables)
        limit = min(op.limit, self.max_rows)
        if isinstance(op, Lexical):
            if op.fields and not set(op.fields) <= set(table.searchable_columns):
                raise BackendError(f"not searchable in {table.name}: {sorted(set(op.fields) - set(table.searchable_columns))}")
            sql, params = self._lexical_sql(op, table, limit)
            return self._hits(await self._query(sql, params), table)
        if isinstance(op, Vector):
            return self._hits(await self._query(*self._vector(op.field, op.vector, table, op.filter, limit)), table)
        if isinstance(op, Hybrid):
            return await self._hybrid(op, table, limit)
        if isinstance(op, FilterOnly):
            return await self._filter_only(op, table, limit)
        if isinstance(op, Regex):
            return await self._regex(op, table, limit)
        if isinstance(op, Aggregate):
            return await self._aggregate(op, table, limit)
        if isinstance(op, Fetch):
            return await self._fetch(op, table)
        raise UnsupportedOperation(f"{self.backend_type} backend does not support {op.type}")

    async def close(self) -> None:
        return None

    # ---- shared op implementations -----------------------------------------------

    def _resolve(self, name: str | None, tables: dict[str, TableInfo]) -> TableInfo:
        if name is None:
            if len(tables) == 1:
                return next(iter(tables.values()))
            raise BackendError(f"collection required; one of {sorted(tables)}")
        if name not in tables:
            raise BackendError(f"unknown collection {name!r}; one of {sorted(tables)}")
        return tables[name]

    def _select(self, table: TableInfo) -> str:
        return ", ".join(quote_ident(c, self.dialect) for c in table.select_columns)

    def _hits(self, rows: list[dict[str, Any]], table: TableInfo) -> list[Hit]:
        hits = []
        for row in rows:
            score = row.pop("_score", None)
            hits.append(row_to_hit(row, source=self.name, id_column=table.id_column,
                                   text_columns=table.text_columns,
                                   score=float(score) if score is not None else None))
        return hits

    def _vector(self, column: str, vector: list[float] | None, table: TableInfo, op_filter: Any,
                limit: int) -> tuple[str, list[Any]]:
        if vector is None:
            raise BackendError("vector op reached the backend without an embedded query vector")
        if column not in table.vector_columns:
            raise BackendError(f"{column!r} is not a vector column of {table.name}")
        return self._vector_sql(column, vector, table, op_filter, limit)

    async def _hybrid(self, op: Hybrid, table: TableInfo, limit: int) -> list[Hit]:
        depth = min(max(limit * 5, 50), 500)
        lex_rows = await self._query(*self._lexical_sql(
            Lexical(source=op.source, text=op.text, filter=op.filter, limit=depth), table, depth))
        vec_rows = await self._query(*self._vector(op.field, op.vector, table, op.filter, depth))
        by_id: dict[str, Hit] = {}
        rankings = []
        for rows, weight in ((lex_rows, op.lexical_weight), (vec_rows, 1.0 - op.lexical_weight)):
            hits = self._hits(rows, table)
            for h in hits:
                by_id.setdefault(h.doc_id, h)
            rankings.append(([h.doc_id for h in hits], weight))
        return [by_id[i].model_copy(update={"raw_score": s}) for i, s in rrf_merge(rankings)[:limit]]

    async def _filter_only(self, op: FilterOnly, table: TableInfo, limit: int) -> list[Hit]:
        p = Params(self.dialect)
        where = where_clause(op.filter, self.dialect, p, table.columns)
        sql = (f"SELECT {self._select(table)} FROM {self._table_ref(table.name)}{where} "
               f"ORDER BY {quote_ident(table.id_column, self.dialect)} LIMIT {p.add(limit)}")
        return self._hits(await self._query(sql, p.values), table)

    async def _regex(self, op: Regex, table: TableInfo, limit: int) -> list[Hit]:
        fields = op.fields or table.text_columns or table.searchable_columns
        unknown = set(fields) - table.columns
        if unknown:
            raise BackendError(f"unknown columns {sorted(unknown)}")
        p = Params(self.dialect)
        ors = " OR ".join(self._regex_expr(self._as_text(quote_ident(c, self.dialect)), p.add(op.pattern))
                          for c in fields)
        where = where_clause(op.filter, self.dialect, p, table.columns, extra=[f"({ors})"])
        sql = (f"SELECT {self._select(table)} FROM {self._table_ref(table.name)}{where} "
               f"ORDER BY {quote_ident(table.id_column, self.dialect)} LIMIT {p.add(limit)}")
        return self._hits(await self._query(sql, p.values), table)

    async def _aggregate(self, op: Aggregate, table: TableInfo, limit: int) -> list[Hit]:
        unknown = set(op.group_by) - table.columns
        if unknown:
            raise BackendError(f"unknown columns {sorted(unknown)}")
        groups = [quote_ident(g, self.dialect) for g in op.group_by]
        metrics = [parse_metric(m, table.columns, self.dialect) for m in op.metrics]
        p = Params(self.dialect)
        where = where_clause(op.filter, self.dialect, p, table.columns)
        select = ", ".join(groups + [f"{expr} AS {quote_ident(alias, self.dialect)}" for expr, alias in metrics])
        sql = (f"SELECT {select} FROM {self._table_ref(table.name)}{where} GROUP BY {', '.join(groups)} "
               f"ORDER BY {quote_ident(metrics[0][1], self.dialect)} DESC LIMIT {p.add(limit)}")
        hits = []
        for row in await self._query(sql, p.values):
            data = {k: v for k, v in row.items()}
            key = dumps({g: data.get(g) for g in op.group_by})
            first = data.get(metrics[0][1])
            hit = row_to_hit(data, source=self.name, id_column=None, text_columns=[],
                             score=float(first) if first is not None else None, fallback_id=f"agg:{key}")
            hits.append(hit.model_copy(update={"content": [StructuredPart(data=hit.metadata)], "metadata": {}}))
        return hits

    async def _fetch(self, op: Fetch, table: TableInfo) -> list[Hit]:
        p = Params(self.dialect)
        ids = ", ".join(p.add(str(i)) for i in op.doc_ids)
        id_expr = self._as_text(quote_ident(table.id_column, self.dialect))
        sql = (f"SELECT {self._select(table)} FROM {self._table_ref(table.name)} "
               f"WHERE {id_expr} IN ({ids})")
        order = {doc_id: i for i, doc_id in enumerate(op.doc_ids)}
        hits = self._hits(await self._query(sql, p.values), table)
        return sorted(hits, key=lambda h: order.get(h.doc_id, len(order)))

    async def _native(self, op: Native, tables: dict[str, TableInfo]) -> list[Hit]:
        if not self.native_query:
            raise UnsupportedOperation("native queries are disabled for this source")
        if op.dialect.lower() not in self.native_dialects:
            raise BackendError(f"dialect must be one of {self.native_dialects}, got {op.dialect!r}")
        sql = guard_sql(op.query, self.dialect, min(op.limit, self.max_rows))
        table = tables.get(op.collection) if op.collection else (
            next(iter(tables.values())) if len(tables) == 1 else None)
        hits = []
        for i, row in enumerate(await self._query(sql, None)):
            id_col = table.id_column if table and table.id_column in row else None
            hits.append(row_to_hit(row, source=self.name, id_column=id_col, text_columns=[],
                                   fallback_id=f"native:{i}"))
        return hits

    # ---- discovery helpers ----------------------------------------------------

    def _vector_spec(self, table: str, column: str, dim: int | None) -> FieldSpec:
        return FieldSpec(name=column, type=FieldType.VECTOR, vector_dim=dim,
                         vector_metric=self.vector_metric,
                         embedder_id=self.embedders.get(f"{table}.{column}"))

    async def _samples(self, table: str, column: str) -> list[Any] | None:
        if not self.sample_values:
            return None
        col = quote_ident(column, self.dialect)
        p = Params(self.dialect)
        sql = (f"SELECT DISTINCT v FROM (SELECT {col} AS v FROM {self._table_ref(table)} "
               f"WHERE {col} IS NOT NULL LIMIT {p.add(SAMPLE_SCAN_MAX)}) s "
               f"LIMIT {p.add(SAMPLE_DISTINCT_MAX + 1)}")
        try:
            rows = await self._query(sql, p.values)
        except BackendError:
            return None  # best effort: unsortable types, timeouts etc. must not sink discovery
        values = [r["v"] for r in rows]
        return values if len(values) <= SAMPLE_DISTINCT_MAX else None

    async def _count(self, table: str, estimate: int | None) -> int | None:
        if estimate is not None and estimate >= EXACT_COUNT_BELOW:
            return estimate
        rows = await self._query(f"SELECT COUNT(*) AS n FROM {self._table_ref(table)}", None)
        return int(rows[0]["n"]) if rows else None
