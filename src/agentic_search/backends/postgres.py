"""Postgres (+ pgvector) backend: tsvector full-text, pgvector ANN, filters, regex, aggregates,
fetch and guarded native SQL. Sessions are read-only. Needs the `postgres` extra."""

from __future__ import annotations

import asyncio
import re
from typing import Any
from urllib.parse import unquote, urlparse

from agentic_search.backends.base import BackendError
from agentic_search.backends.sql import Params, quote_ident, vector_literal, where_clause
from agentic_search.backends.sql_backend import SqlBackend, TableInfo, any_terms, field_flags
from agentic_search.core.secrets import register_secret
from agentic_search.core.types import FieldSpec, FieldType, Lexical

_TYPES = {
    "text": FieldType.TEXT, "character varying": FieldType.KEYWORD, "character": FieldType.KEYWORD,
    "uuid": FieldType.KEYWORD, "smallint": FieldType.INT, "integer": FieldType.INT,
    "bigint": FieldType.INT, "numeric": FieldType.FLOAT, "real": FieldType.FLOAT,
    "double precision": FieldType.FLOAT, "boolean": FieldType.BOOL, "date": FieldType.DATE,
    "timestamp without time zone": FieldType.DATE, "timestamp with time zone": FieldType.DATE,
    "json": FieldType.JSON, "jsonb": FieldType.JSON, "ARRAY": FieldType.JSON,
}
_SKIP_UDTS = {"tsvector", "tsquery"}
_OPS = {"cosine": "<=>", "l2": "<->", "ip": "<#>"}
_CONFIG = re.compile(r"^[a-z_]+$")
_DIM = re.compile(r"^vector\((\d+)\)$")


def _require_psycopg() -> tuple[Any, Any, Any]:
    """Import psycopg, psycopg.rows.dict_row, and AsyncConnectionPool.

    Raises BackendError if any import fails (e.g., missing postgres extra).
    Returns (psycopg module, dict_row, AsyncConnectionPool).
    """
    try:
        import psycopg
        from psycopg.rows import dict_row
        from psycopg_pool import AsyncConnectionPool
        return psycopg, dict_row, AsyncConnectionPool
    except ImportError as exc:
        raise BackendError("PostgresBackend needs the `postgres` extra: pip install 'agentic-search[postgres]'") from exc


class PostgresBackend(SqlBackend):
    dialect = "postgres"
    backend_type = "postgres"
    native_dialects = ("sql", "postgres", "postgresql")

    def __init__(self, name: str, dsn: str, *, schema: str = "public",
                 text_search_config: str = "english", pool_size: int = 4,
                 statement_timeout_ms: int = 30_000, connect_timeout_s: float = 15,
                 tsvector_columns: dict[str, str] | None = None, **kwargs: Any):
        super().__init__(name, **kwargs)
        self.tsvector_columns = dict(tsvector_columns or {})
        self._tsv: dict[str, str] = {}
        if not _CONFIG.match(text_search_config):
            raise ValueError(f"invalid text_search_config {text_search_config!r}")
        register_secret(dsn)
        register_secret(urlparse(dsn).password)
        register_secret(unquote(urlparse(dsn).password or ""))
        self.dsn = dsn
        self.schema = schema
        self.ts_config = text_search_config
        self.pool_size = pool_size
        self.statement_timeout_ms = int(statement_timeout_ms)
        self.connect_timeout_s = connect_timeout_s
        self._pool: Any = None
        self._pool_lock = asyncio.Lock()

    async def _get_pool(self) -> Any:
        async with self._pool_lock:
            if self._pool is None:
                _, _, AsyncConnectionPool = _require_psycopg()

                timeout_ms = self.statement_timeout_ms

                async def configure(conn: Any) -> None:
                    await conn.set_autocommit(True)
                    await conn.execute("SET default_transaction_read_only = on")
                    await conn.execute(f"SET statement_timeout = {timeout_ms}")

                pool = AsyncConnectionPool(self.dsn, min_size=1, max_size=self.pool_size,
                                           open=False, configure=configure)
                try:
                    await pool.open(wait=True, timeout=self.connect_timeout_s)
                except BaseException as exc:  # incl. CancelledError: don't leak a half-open pool
                    try:
                        await pool.close()
                    except Exception:
                        pass
                    self._pool = None
                    if not isinstance(exc, Exception):
                        raise
                    raise BackendError(f"{type(exc).__name__}: {exc}") from exc
                self._pool = pool
            return self._pool

    async def _query(self, sql: str, params: list[Any] | None) -> list[dict[str, Any]]:
        psycopg, dict_row, _ = _require_psycopg()

        try:
            pool = await self._get_pool()
            async with pool.connection() as conn:
                async with conn.cursor(row_factory=dict_row) as cur:
                    await cur.execute(sql, params)
                    return list(await cur.fetchall()) if cur.description else []
        except psycopg.Error as exc:
            raise BackendError(f"{type(exc).__name__}: {exc}") from exc

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    def _table_ref(self, table: str) -> str:
        return f"{quote_ident(self.schema, 'postgres')}.{quote_ident(table, 'postgres')}"

    def _regex_expr(self, column_sql: str, placeholder: str) -> str:
        return f"{column_sql} ~ {placeholder}"

    # ---- discovery ------------------------------------------------------------

    async def _discover_tables(self) -> dict[str, TableInfo]:
        cols = await self._query(
            "SELECT table_name, column_name, data_type, udt_name FROM information_schema.columns "
            "WHERE table_schema = %s ORDER BY table_name, ordinal_position", [self.schema])
        pks = await self._query(
            "SELECT tc.table_name, kcu.column_name FROM information_schema.table_constraints tc "
            "JOIN information_schema.key_column_usage kcu ON tc.constraint_name = kcu.constraint_name "
            "AND tc.table_schema = kcu.table_schema AND tc.table_name = kcu.table_name "
            "WHERE tc.table_schema = %s AND tc.constraint_type = 'PRIMARY KEY' "
            "ORDER BY kcu.ordinal_position", [self.schema])
        dims = await self._query(
            "SELECT c.relname AS table_name, a.attname AS column_name, "
            "format_type(a.atttypid, a.atttypmod) AS fmt FROM pg_attribute a "
            "JOIN pg_class c ON a.attrelid = c.oid JOIN pg_namespace n ON c.relnamespace = n.oid "
            "WHERE n.nspname = %s AND a.attnum > 0 AND NOT a.attisdropped "
            "AND format_type(a.atttypid, a.atttypmod) LIKE 'vector%%'", [self.schema])
        estimates = await self._query(
            "SELECT c.relname AS table_name, c.reltuples::bigint AS est FROM pg_class c "
            "JOIN pg_namespace n ON c.relnamespace = n.oid WHERE n.nspname = %s", [self.schema])
        pk_of: dict[str, list[str]] = {}
        for r in pks:
            pk_of.setdefault(r["table_name"], []).append(r["column_name"])
        dim_of = {(r["table_name"], r["column_name"]): int(m.group(1))
                  for r in dims if (m := _DIM.match(r["fmt"]))}
        est_of = {r["table_name"]: r["est"] for r in estimates}
        by_table: dict[str, list[dict[str, Any]]] = {}
        for r in cols:
            if self.table_names is None or r["table_name"] in self.table_names:
                by_table.setdefault(r["table_name"], []).append(r)
        tables: dict[str, TableInfo] = {}
        skipped: list[str] = []
        tsv_of: dict[str, str] = {}
        for tname, rows in by_table.items():
            id_col = self.id_columns.get(tname) or self._single_pk(tname, pk_of.get(tname), skipped)
            if id_col is None:
                continue
            fields = []
            tsv_cols: list[str] = []
            for r in rows:
                name, udt = r["column_name"], r["udt_name"]
                if udt == "tsvector":
                    tsv_cols.append(name)
                    continue
                if udt in _SKIP_UDTS:
                    continue
                if udt == "vector":
                    fields.append(self._vector_spec(tname, name, dim_of.get((tname, name))))
                    continue
                ftype = _TYPES.get(r["data_type"], FieldType.KEYWORD)
                samples = None
                if r["data_type"] in _TYPES and ftype in (FieldType.KEYWORD, FieldType.BOOL) and name != id_col:
                    samples = await self._samples(tname, name)
                fields.append(FieldSpec(name=name, type=ftype, sample_values=samples, **field_flags(ftype)))
            configured = self.tsvector_columns.get(tname)
            if configured in tsv_cols:
                tsv_of[tname] = configured
            elif configured is None and len(tsv_cols) == 1:
                tsv_of[tname] = tsv_cols[0]
            est = est_of.get(tname)
            tables[tname] = TableInfo(name=tname, id_column=id_col, fields=fields,
                                      count=await self._count(tname, est if est and est > 0 else None))
        self.skipped = skipped
        self._tsv = tsv_of
        return tables

    # ---- lexical / vector ------------------------------------------------------

    def _lexical_sql(self, op: Lexical, table: TableInfo, limit: int) -> tuple[str, list[Any]]:
        fields = op.fields or table.text_columns or table.searchable_columns
        if not fields:
            raise BackendError(f"{table.name} has no text columns")
        terms = any_terms(op.text)
        if not terms:
            raise BackendError("lexical query has no searchable terms")
        query_text = op.text if '"' in op.text else " or ".join(terms)
        stored = None if op.fields else self._tsv.get(table.name)
        if stored is not None:  # a stored tsvector column can use its GIN index
            doc = quote_ident(stored, "postgres")
        else:
            doc = (f"to_tsvector('{self.ts_config}', concat_ws(' ', "
                   f"{', '.join(quote_ident(c, 'postgres') for c in fields)}))")
        p = Params("postgres")
        score = f"ts_rank({doc}, websearch_to_tsquery('{self.ts_config}', {p.add(query_text)}))"
        match = f"{doc} @@ websearch_to_tsquery('{self.ts_config}', {p.add(query_text)})"
        where = where_clause(op.filter, "postgres", p, table.columns, extra=[match])
        sql = (f"SELECT {self._select(table)}, {score} AS _score FROM {self._table_ref(table.name)}"
               f"{where} ORDER BY _score DESC, {quote_ident(table.id_column, 'postgres')} "
               f"LIMIT {p.add(limit)}")
        return sql, p.values

    def _vector_sql(self, column: str, vector: list[float], table: TableInfo, op_filter: Any,
                    limit: int) -> tuple[str, list[Any]]:
        col = quote_ident(column, "postgres")
        sym = _OPS[self.vector_metric]
        lit = vector_literal(vector)
        p = Params("postgres")
        distance = f"({col} {sym} {p.add(lit)}::vector)"
        score = f"1 - {distance}" if self.vector_metric == "cosine" else f"-{distance}"
        where = where_clause(op_filter, "postgres", p, table.columns, extra=[f"{col} IS NOT NULL"])
        sql = (f"SELECT {self._select(table)}, {score} AS _score FROM {self._table_ref(table.name)}"
               f"{where} ORDER BY {col} {sym} {p.add(lit)}::vector LIMIT {p.add(limit)}")
        return sql, p.values
