"""BigQuery backend: term-match lexical search (CONTAINS_SUBSTR), VECTOR_SEARCH, filters,
REGEXP_CONTAINS, aggregates, fetch and guarded native SQL. Every query is dry-run first and
refused above `max_bytes_billed`. Needs the `bigquery` extra. The client is synchronous, so calls
run in a worker thread."""

from __future__ import annotations

import asyncio
import importlib
import re
from typing import Any

from agentic_search.backends.base import BackendError
from agentic_search.backends.sql import Params, quote_ident, where_clause
from agentic_search.backends.sql_backend import SqlBackend, TableInfo, any_terms, field_flags
from agentic_search.core.types import FieldSpec, FieldType, Lexical

_TYPES = {
    "STRING": FieldType.TEXT, "INTEGER": FieldType.INT, "INT64": FieldType.INT,
    "FLOAT": FieldType.FLOAT, "FLOAT64": FieldType.FLOAT, "NUMERIC": FieldType.FLOAT,
    "BIGNUMERIC": FieldType.FLOAT, "BOOLEAN": FieldType.BOOL, "BOOL": FieldType.BOOL,
    "DATE": FieldType.DATE, "DATETIME": FieldType.DATE, "TIMESTAMP": FieldType.DATE,
    "JSON": FieldType.JSON, "RECORD": FieldType.JSON, "STRUCT": FieldType.JSON,
}
_DISTANCE = {"cosine": "COSINE", "l2": "EUCLIDEAN", "ip": "DOT_PRODUCT"}
_PROJECT = re.compile(r"[A-Za-z0-9:._-]+")  # incl. domain-scoped "example.com:proj"
_DATASET = re.compile(r"[A-Za-z0-9_]+")


def _require_bigquery() -> tuple[Any, Any]:
    """Import google.cloud.bigquery and google.api_core.exceptions.

    Raises BackendError if any import fails (e.g., missing bigquery extra).
    Returns (bigquery module, google.api_core.exceptions module).
    """
    try:
        bigquery = importlib.import_module("google.cloud.bigquery")
        exceptions = importlib.import_module("google.api_core.exceptions")
        return bigquery, exceptions
    except ImportError as exc:
        msg = "BigQueryBackend needs the `bigquery` extra: pip install 'agentic-search[bigquery]'"
        raise BackendError(msg) from exc


class BigQueryBackend(SqlBackend):
    dialect = "bigquery"
    backend_type = "bigquery"
    native_dialects = ("sql", "bigquery")

    def __init__(self, name: str, project: str, dataset: str, *, client: Any = None,
                 max_bytes_billed: int = 1_000_000_000, location: str | None = None,
                 text_columns: dict[str, list[str]] | None = None,
                 vector_dims: dict[str, int] | None = None, sample_values: bool = False,
                 job_timeout_ms: int = 60_000, **kwargs: Any):
        super().__init__(name, sample_values=sample_values, **kwargs)
        if not _PROJECT.fullmatch(project):
            raise ValueError(f"invalid BigQuery project {project!r}")
        if not _DATASET.fullmatch(dataset):
            raise ValueError(f"invalid BigQuery dataset {dataset!r}")
        self.project = project
        self.dataset = dataset
        self.max_bytes_billed = int(max_bytes_billed)
        self.job_timeout_ms = int(job_timeout_ms)
        self.location = location
        self.text_column_overrides = text_columns or {}
        self.vector_dims = vector_dims or {}
        self._client = client

    def _get_client(self) -> Any:
        if self._client is None:
            bigquery, _ = _require_bigquery()
            self._client = bigquery.Client(project=self.project, location=self.location)
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            client, self._client = self._client, None
            await asyncio.to_thread(client.close)

    def _table_ref(self, table: str) -> str:
        quote_ident(table, "bigquery")  # validate
        return f"`{self.project}.{self.dataset}.{table}`"

    def _native_db(self) -> str | None:
        return self.dataset

    def _native_catalog(self) -> str | None:
        return self.project

    def _as_text(self, expr: str) -> str:
        return f"CAST({expr} AS STRING)"

    def _regex_expr(self, column_sql: str, placeholder: str) -> str:
        return f"REGEXP_CONTAINS({column_sql}, {placeholder})"

    @staticmethod
    def _parameters(values: list[Any]) -> list[Any]:
        bigquery, _ = _require_bigquery()

        out = []
        for i, v in enumerate(values):
            name = f"p{i}"
            if isinstance(v, list):
                out.append(bigquery.ArrayQueryParameter(name, "FLOAT64", [float(x) for x in v]))
            elif isinstance(v, bool):
                out.append(bigquery.ScalarQueryParameter(name, "BOOL", v))
            elif isinstance(v, int):
                out.append(bigquery.ScalarQueryParameter(name, "INT64", v))
            elif isinstance(v, float):
                out.append(bigquery.ScalarQueryParameter(name, "FLOAT64", v))
            else:
                out.append(bigquery.ScalarQueryParameter(name, "STRING", None if v is None else str(v)))
        return out

    def _run(self, sql: str, params: list[Any] | None) -> list[dict[str, Any]]:
        bigquery, exceptions = _require_bigquery()

        try:
            client = self._get_client()
            query_params = self._parameters(params or [])
            dry = client.query(sql, job_config=bigquery.QueryJobConfig(
                dry_run=True, use_query_cache=False, query_parameters=query_params))
            scanned = dry.total_bytes_processed or 0
            if scanned > self.max_bytes_billed:
                raise BackendError(f"query would scan {scanned} bytes, above the "
                                   f"{self.max_bytes_billed}-byte cap")
            job = client.query(sql, job_config=bigquery.QueryJobConfig(
                query_parameters=query_params, maximum_bytes_billed=self.max_bytes_billed,
                job_timeout_ms=self.job_timeout_ms))
            return [dict(row.items()) for row in job.result()]
        except BackendError:
            raise
        except Exception as exc:
            raise BackendError(f"{type(exc).__name__}: {exc}") from exc

    async def _query(self, sql: str, params: list[Any] | None) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._run, sql, params)

    async def _count(self, table: str, estimate: int | None) -> int | None:
        return estimate  # num_rows from table metadata; never scan to count

    # ---- discovery ------------------------------------------------------------

    def _list_tables(self) -> list[tuple[str, Any]]:
        client = self._get_client()
        names = self.table_names or [t.table_id for t in client.list_tables(f"{self.project}.{self.dataset}")]
        return [(n, client.get_table(f"{self.project}.{self.dataset}.{n}")) for n in names]

    async def _discover_tables(self) -> dict[str, TableInfo]:
        _require_bigquery()

        try:
            listed = await asyncio.to_thread(self._list_tables)
        except BackendError:
            raise
        except Exception as exc:
            raise BackendError(f"{type(exc).__name__}: {exc}") from exc
        tables: dict[str, TableInfo] = {}
        skipped: list[str] = []
        for tname, meta in listed:
            names = [f.name for f in meta.schema]
            id_col = self.id_columns.get(tname) or self._primary_key(meta) or ("id" if "id" in names else None)
            if id_col is None:
                skipped.append(f"{tname} (no primary key / id column)")
                continue
            overrides = self.text_column_overrides.get(tname)
            fields = []
            for f in meta.schema:
                ftype_name = f.field_type.upper()
                if f.mode == "REPEATED" and ftype_name in ("FLOAT", "FLOAT64"):
                    fields.append(self._vector_spec(tname, f.name, self.vector_dims.get(f"{tname}.{f.name}")))
                    continue
                if f.mode == "REPEATED":
                    ftype = FieldType.JSON
                else:
                    ftype = _TYPES.get(ftype_name, FieldType.KEYWORD)
                if ftype is FieldType.TEXT and overrides is not None and f.name not in overrides:
                    ftype = FieldType.KEYWORD
                samples = None
                if ftype in (FieldType.KEYWORD, FieldType.BOOL) and f.name != id_col:
                    samples = await self._samples(tname, f.name)
                fields.append(FieldSpec(name=f.name, type=ftype, description=f.description or None,
                                        sample_values=samples, **field_flags(ftype)))
            tables[tname] = TableInfo(name=tname, id_column=id_col, fields=fields,
                                      count=int(meta.num_rows) if meta.num_rows is not None else None)
        self.skipped = skipped
        return tables

    @staticmethod
    def _primary_key(meta: Any) -> str | None:
        constraints = getattr(meta, "table_constraints", None)
        pk = getattr(constraints, "primary_key", None) if constraints else None
        cols = getattr(pk, "columns", None) if pk else None
        return cols[0] if cols else None

    # ---- lexical / vector ------------------------------------------------------

    def _lexical_sql(self, op: Lexical, table: TableInfo, limit: int) -> tuple[str, list[Any]]:
        fields = op.fields or table.text_columns or table.searchable_columns
        if not fields:
            raise BackendError(f"{table.name} has no text columns")
        terms = any_terms(op.text)
        if not terms:
            raise BackendError("lexical query has no searchable terms")
        p = Params("bigquery")
        quoted = [quote_ident(c, "bigquery") for c in fields]
        target = quoted[0] if len(quoted) == 1 else f"({', '.join(quoted)})"
        score = " + ".join(f"IF(CONTAINS_SUBSTR({target}, {p.add(t)}), 1, 0)" for t in terms)
        where = where_clause(op.filter, "bigquery", p, table.columns)
        id_col = quote_ident(table.id_column, "bigquery")
        sql = (f"SELECT * FROM (SELECT {self._select(table)}, ({score}) AS _score "
               f"FROM {self._table_ref(table.name)}{where}) WHERE _score > 0 "
               f"ORDER BY _score DESC, {id_col} LIMIT {p.add(limit)}")
        return sql, p.values

    def _vector_sql(self, column: str, vector: list[float], table: TableInfo, op_filter: Any,
                    limit: int) -> tuple[str, list[Any]]:
        col = quote_ident(column, "bigquery")
        p = Params("bigquery")
        where = where_clause(op_filter, "bigquery", p, table.columns)
        select = ", ".join(f"base.{quote_ident(c, 'bigquery')} AS {quote_ident(c, 'bigquery')}"
                           for c in table.select_columns)
        score = "1 - distance" if self.vector_metric == "cosine" else "-distance"
        sql = (f"SELECT {select}, {score} AS _score FROM VECTOR_SEARCH("
               f"(SELECT * FROM {self._table_ref(table.name)}{where}), '{column}', "
               f"(SELECT {p.add(vector)} AS {col}), top_k => {int(limit)}, "
               f"distance_type => '{_DISTANCE[self.vector_metric]}') ORDER BY distance")
        return sql, p.values
