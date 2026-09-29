"""MySQL backend: FULLTEXT (natural-language mode) search, filters, REGEXP, aggregates, fetch and
guarded native SQL. Sessions are READ ONLY. No vector search in v1. Needs the `mysql` extra."""

from __future__ import annotations

import asyncio
from typing import Any
from urllib.parse import unquote, urlparse

from agentic_search.backends.base import BackendError
from agentic_search.backends.sql import Params, quote_ident, where_clause
from agentic_search.backends.sql_backend import SqlBackend, TableInfo, field_flags
from agentic_search.core.secrets import register_secret
from agentic_search.core.types import FieldSpec, FieldType, Lexical

_TYPES = {
    "varchar": FieldType.KEYWORD, "char": FieldType.KEYWORD, "enum": FieldType.KEYWORD,
    "text": FieldType.TEXT, "tinytext": FieldType.TEXT, "mediumtext": FieldType.TEXT,
    "longtext": FieldType.TEXT, "tinyint": FieldType.INT, "smallint": FieldType.INT,
    "mediumint": FieldType.INT, "int": FieldType.INT, "bigint": FieldType.INT,
    "decimal": FieldType.FLOAT, "float": FieldType.FLOAT, "double": FieldType.FLOAT,
    "date": FieldType.DATE, "datetime": FieldType.DATE, "timestamp": FieldType.DATE,
    "json": FieldType.JSON,
}


def _require_aiomysql() -> tuple[Any, Any]:
    """Import aiomysql and pymysql.

    Raises BackendError if any import fails (e.g., missing mysql extra).
    Returns (aiomysql module, pymysql module).
    """
    try:
        import aiomysql
        import pymysql
        return aiomysql, pymysql
    except ImportError as exc:
        raise BackendError("MySQLBackend needs the `mysql` extra: pip install 'agentic-search[mysql]'") from exc


class MySQLBackend(SqlBackend):
    dialect = "mysql"
    backend_type = "mysql"
    native_dialects = ("sql", "mysql")

    def __init__(self, name: str, dsn: str, *, pool_size: int = 4, connect_timeout_s: float = 10, **kwargs: Any):
        super().__init__(name, **kwargs)
        parsed = urlparse(dsn)
        if parsed.scheme not in ("mysql", "mysql+aiomysql") or not parsed.path.strip("/"):
            raise ValueError("dsn must look like mysql://user:password@host:port/database")
        register_secret(dsn)
        register_secret(unquote(parsed.password or ""))
        self._conn_args = {
            "host": parsed.hostname or "localhost", "port": parsed.port or 3306,
            "user": unquote(parsed.username or ""), "password": unquote(parsed.password or ""),
            "db": parsed.path.strip("/"),
        }
        self.pool_size = pool_size
        self.connect_timeout_s = connect_timeout_s
        self._pool: Any = None
        self._pool_lock = asyncio.Lock()

    async def _get_pool(self) -> Any:
        async with self._pool_lock:
            if self._pool is None:
                aiomysql, _ = _require_aiomysql()

                try:
                    self._pool = await aiomysql.create_pool(
                        minsize=1, maxsize=self.pool_size, autocommit=True, charset="utf8mb4",
                        connect_timeout=self.connect_timeout_s, init_command="SET SESSION TRANSACTION READ ONLY",
                        **self._conn_args)
                except Exception as exc:
                    self._pool = None
                    raise BackendError(f"{type(exc).__name__}: {exc}") from exc
            return self._pool

    async def _query(self, sql: str, params: list[Any] | None) -> list[dict[str, Any]]:
        aiomysql, pymysql = _require_aiomysql()

        try:
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                async with conn.cursor(aiomysql.DictCursor) as cur:
                    await cur.execute(sql, params)
                    return list(await cur.fetchall())
        except pymysql.MySQLError as exc:
            raise BackendError(f"{type(exc).__name__}: {exc}") from exc

    async def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            await self._pool.wait_closed()
            self._pool = None

    def _regex_expr(self, column_sql: str, placeholder: str) -> str:
        return f"{column_sql} REGEXP {placeholder}"

    def _supports_lexical(self, tables: dict[str, TableInfo]) -> bool:
        return any(t.fulltext for t in tables.values())

    # ---- discovery ------------------------------------------------------------

    async def _discover_tables(self) -> dict[str, TableInfo]:
        cols = await self._query(
            "SELECT TABLE_NAME AS table_name, COLUMN_NAME AS column_name, DATA_TYPE AS data_type, "
            "COLUMN_TYPE AS column_type FROM information_schema.COLUMNS "
            "WHERE TABLE_SCHEMA = DATABASE() ORDER BY TABLE_NAME, ORDINAL_POSITION", None)
        pks = await self._query(
            "SELECT TABLE_NAME AS table_name, COLUMN_NAME AS column_name "
            "FROM information_schema.KEY_COLUMN_USAGE WHERE TABLE_SCHEMA = DATABASE() "
            "AND CONSTRAINT_NAME = 'PRIMARY' ORDER BY ORDINAL_POSITION", None)
        fts = await self._query(
            "SELECT TABLE_NAME AS table_name, INDEX_NAME AS index_name, COLUMN_NAME AS column_name "
            "FROM information_schema.STATISTICS WHERE TABLE_SCHEMA = DATABASE() "
            "AND INDEX_TYPE = 'FULLTEXT' ORDER BY TABLE_NAME, INDEX_NAME, SEQ_IN_INDEX", None)
        estimates = await self._query(
            "SELECT TABLE_NAME AS table_name, TABLE_ROWS AS est FROM information_schema.TABLES "
            "WHERE TABLE_SCHEMA = DATABASE()", None)
        pk_of: dict[str, str] = {}
        for r in pks:
            pk_of.setdefault(r["table_name"], r["column_name"])
        ft_of: dict[str, dict[str, list[str]]] = {}
        for r in fts:
            ft_of.setdefault(r["table_name"], {}).setdefault(r["index_name"], []).append(r["column_name"])
        est_of = {r["table_name"]: r["est"] for r in estimates}
        by_table: dict[str, list[dict[str, Any]]] = {}
        for r in cols:
            if self.table_names is None or r["table_name"] in self.table_names:
                by_table.setdefault(r["table_name"], []).append(r)
        tables: dict[str, TableInfo] = {}
        for tname, rows in by_table.items():
            id_col = self.id_columns.get(tname) or pk_of.get(tname)
            if id_col is None:
                self.skipped.append(tname)
                continue
            fulltext = list(ft_of.get(tname, {}).values())
            ft_cols = {c for idx in fulltext for c in idx}
            fields = []
            for r in rows:
                name = r["column_name"]
                if r["data_type"] == "tinyint" and r["column_type"].startswith("tinyint(1)"):
                    ftype = FieldType.BOOL
                else:
                    ftype = _TYPES.get(r["data_type"], FieldType.KEYWORD)
                flags = field_flags(ftype)
                flags["searchable"] = name in ft_cols
                samples = None
                if ftype in (FieldType.KEYWORD, FieldType.BOOL) and name != id_col:
                    samples = await self._samples(tname, name)
                fields.append(FieldSpec(name=name, type=ftype, sample_values=samples, **flags))
            est = est_of.get(tname)
            tables[tname] = TableInfo(name=tname, id_column=id_col, fields=fields, fulltext=fulltext,
                                      count=await self._count(tname, int(est) if est else None))
        return tables

    # ---- lexical ---------------------------------------------------------------

    def _lexical_sql(self, op: Lexical, table: TableInfo, limit: int) -> tuple[str, list[Any]]:
        if not table.fulltext:
            raise BackendError(f"{table.name} has no FULLTEXT index; lexical search needs one")
        if op.fields:
            match = next((idx for idx in table.fulltext if set(idx) == set(op.fields)), None)
            if match is None:
                raise BackendError(f"no FULLTEXT index on exactly {sorted(op.fields)}; "
                                   f"indexes: {table.fulltext}")
        else:
            match = table.fulltext[0]
        cols = ", ".join(quote_ident(c, "mysql") for c in match)
        p = Params("mysql")
        score = f"MATCH({cols}) AGAINST ({p.add(op.text)} IN NATURAL LANGUAGE MODE)"
        cond = f"MATCH({cols}) AGAINST ({p.add(op.text)} IN NATURAL LANGUAGE MODE)"
        where = where_clause(op.filter, "mysql", p, table.columns, extra=[cond])
        sql = (f"SELECT {self._select(table)}, {score} AS _score FROM {self._table_ref(table.name)}"
               f"{where} ORDER BY _score DESC, {quote_ident(table.id_column, 'mysql')} "
               f"LIMIT {p.add(limit)}")
        return sql, p.values
