"""Build a Harness from a YAML/dict config. Plans 2 and 3 register more types via register()."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml

from agentic_search.core.harness import Harness, HarnessSettings
from agentic_search.core.secrets import register_secret
from agentic_search.core.types import Budget
from agentic_search.embedders.base import Embedder


class ConfigError(ValueError):
    pass


@dataclass
class BuildContext:
    base_dir: Path
    embedders: dict[str, Embedder] = field(default_factory=dict)


Factory = Callable[[dict[str, Any], BuildContext], Any]
_REGISTRIES: dict[str, dict[str, Factory]] = {"backend": {}, "embedder": {}, "client": {}, "decider": {}}


def register(kind: str, type_: str, factory: Factory) -> None:
    if kind not in _REGISTRIES:
        raise ValueError(f"unknown registry kind {kind!r}; one of {sorted(_REGISTRIES)}")
    _REGISTRIES[kind][type_] = factory


def resolve_env(cfg: Any) -> Any:
    if isinstance(cfg, dict):
        out: dict[str, Any] = {}
        for k, v in cfg.items():
            if isinstance(k, str) and k.endswith("_env") and isinstance(v, str):
                if v not in os.environ:
                    raise ConfigError(f"environment variable {v!r} (for {k!r}) is not set")
                out[k[: -len("_env")]] = os.environ[v]
                register_secret(os.environ[v])
            else:
                out[k] = resolve_env(v)
        return out
    if isinstance(cfg, list):
        return [resolve_env(v) for v in cfg]
    return cfg


def build(kind: str, cfg: dict[str, Any], ctx: BuildContext) -> Any:
    cfg = resolve_env(cfg)
    type_ = cfg.get("type")
    factory = _REGISTRIES[kind].get(type_)  # type: ignore[arg-type]
    if factory is None:
        raise ConfigError(f"unknown {kind} type {type_!r}; known: {sorted(_REGISTRIES[kind])}")
    try:
        return factory(cfg, ctx)
    except KeyError as exc:
        raise ConfigError(f"{kind} {type_!r} config is missing required key {exc}") from exc
    except ValueError as exc:
        if isinstance(exc, ConfigError):
            raise
        raise ConfigError(f"{kind} {type_!r} config is invalid: {exc}") from exc


def _price(cfg: dict[str, Any]) -> tuple[float, float] | None:
    p = cfg.get("price_per_mtok")
    return (float(p[0]), float(p[1])) if p else None


# ---- built-in factories (imports are lazy so optional extras stay optional) ----

def _files(cfg: dict[str, Any], ctx: BuildContext) -> Any:
    from agentic_search.backends.files import FilesBackend
    embedder = None
    if "embedder" in cfg:
        embedder = ctx.embedders.get(cfg["embedder"])
        if embedder is None:
            raise ConfigError(f"backend {cfg['name']!r} references unknown embedder {cfg['embedder']!r}")
    root = Path(cfg["root"])
    return FilesBackend(cfg["name"], root if root.is_absolute() else ctx.base_dir / root,
                        embedder=embedder, glob=cfg.get("glob", "**/*"),
                        collection=cfg.get("collection", "files"),
                        description=cfg.get("description"))


_SQL_KEYS = ("tables", "id_columns", "embedders", "vector_metric", "native_query", "description",
             "max_rows", "sample_values")


def _backend_kwargs(cfg: dict[str, Any], ctx: BuildContext, keys: tuple[str, ...]) -> dict[str, Any]:
    """Pass through known keys; check that `embedders: {table.column: id}` names known embedders."""
    unknown = sorted(set((cfg.get("embedders") or {}).values()) - set(ctx.embedders))
    if unknown:
        raise ConfigError(f"backend {cfg['name']!r} references unknown embedder(s) {unknown}")
    return {k: cfg[k] for k in keys if k in cfg}


def _postgres(cfg: dict[str, Any], ctx: BuildContext) -> Any:
    from agentic_search.backends.postgres import PostgresBackend
    return PostgresBackend(cfg["name"], cfg["dsn"],
                           **_backend_kwargs(cfg, ctx, _SQL_KEYS + ("schema", "text_search_config",
                                                                    "pool_size", "statement_timeout_ms",
                                                                    "tsvector_columns")))


def _mysql(cfg: dict[str, Any], ctx: BuildContext) -> Any:
    from agentic_search.backends.mysql import MySQLBackend
    return MySQLBackend(cfg["name"], cfg["dsn"], **_backend_kwargs(cfg, ctx, _SQL_KEYS + ("pool_size", "statement_timeout_ms")))


def _bigquery(cfg: dict[str, Any], ctx: BuildContext) -> Any:
    from agentic_search.backends.bigquery import BigQueryBackend
    return BigQueryBackend(cfg["name"], cfg["project"], cfg["dataset"],
                           **_backend_kwargs(cfg, ctx, _SQL_KEYS + ("max_bytes_billed", "location",
                                                                    "text_columns", "vector_dims",
                                                                    "job_timeout_ms")))


def _opensearch(cfg: dict[str, Any], ctx: BuildContext) -> Any:
    from agentic_search.backends.opensearch import OpenSearchBackend
    return OpenSearchBackend(cfg["name"], cfg["url"], **_backend_kwargs(
        cfg, ctx, ("indices", "embedders", "native_query", "description", "max_rows",
                   "verify_certs", "sample_values")))


def _hash(cfg: dict[str, Any], ctx: BuildContext) -> Any:
    from agentic_search.embedders.local import HashEmbedder
    return HashEmbedder(dim=int(cfg.get("dim", 256)), id=cfg.get("id", "hash"))


def _sentence_transformers(cfg: dict[str, Any], ctx: BuildContext) -> Any:
    from agentic_search.embedders.local import SentenceTransformerEmbedder
    return SentenceTransformerEmbedder(cfg["model"], id=cfg.get("id"), image=cfg.get("image", False),
                                       query_prefix=cfg.get("query_prefix", ""),
                                       document_prefix=cfg.get("document_prefix", ""))


def _anthropic(cfg: dict[str, Any], ctx: BuildContext) -> Any:
    from agentic_search.models.anthropic import AnthropicClient
    return AnthropicClient(cfg["model"], api_key=cfg.get("api_key"), id=cfg.get("id"),
                           supports_images=cfg.get("supports_images", True), price_per_mtok=_price(cfg))


def _openai_compat(cfg: dict[str, Any], ctx: BuildContext) -> Any:
    from agentic_search.models.openai_compat import OpenAICompatClient
    return OpenAICompatClient(cfg["model"], base_url=cfg.get("base_url"), api_key=cfg.get("api_key"),
                              id=cfg.get("id"), supports_images=cfg.get("supports_images", False),
                              price_per_mtok=_price(cfg),
                              max_tokens_param=cfg.get("max_tokens_param", "max_tokens"))


def _neo4j(cfg: dict[str, Any], ctx: BuildContext) -> Any:
    from agentic_search.backends.neo4j import Neo4jBackend
    return Neo4jBackend(cfg["name"], cfg["uri"], **_backend_kwargs(
        cfg, ctx, ("user", "password", "database", "labels", "id_property", "embedders", "native_query",
                   "description", "max_rows", "sample_values", "connect_timeout_s")))


def _milvus(cfg: dict[str, Any], ctx: BuildContext) -> Any:
    from agentic_search.backends.milvus import MilvusBackend
    return MilvusBackend(cfg["name"], cfg["uri"], **_backend_kwargs(
        cfg, ctx, ("token", "db_name", "collections", "embedders", "description", "max_rows",
                   "sample_values", "connect_timeout_s")))


_HTTP_EMBEDDER_KEYS = ("id", "batch_size", "concurrency", "timeout_s", "max_retries")


def _http_kwargs(cfg: dict[str, Any], extra: tuple[str, ...] = ()) -> dict[str, Any]:
    from agentic_search.embedders.auth import build_auth
    kwargs = {k: cfg[k] for k in _HTTP_EMBEDDER_KEYS + extra if k in cfg}
    if "auth" in cfg:
        kwargs["auth"] = build_auth(cfg["auth"])
    return kwargs


def _openai_embedder(cfg: dict[str, Any], ctx: BuildContext) -> Any:
    from agentic_search.embedders.openai_compat import OpenAICompatEmbedder
    return OpenAICompatEmbedder(cfg["model"], int(cfg["dim"]), **_http_kwargs(
        cfg, ("base_url", "api_key", "dimensions", "query_prefix", "document_prefix")))


def _tei(cfg: dict[str, Any], ctx: BuildContext) -> Any:
    from agentic_search.embedders.tei import TEIEmbedder
    return TEIEmbedder(cfg["url"], int(cfg["dim"]), **_http_kwargs(
        cfg, ("query_prefix", "document_prefix", "normalize")))


def _vertex(cfg: dict[str, Any], ctx: BuildContext) -> Any:
    from agentic_search.embedders.vertex import VertexEmbedder
    return VertexEmbedder(cfg["model"], int(cfg["dim"]), project=cfg["project"],
                          **_http_kwargs(cfg, ("location", "endpoint")))


def _http_embedder(cfg: dict[str, Any], ctx: BuildContext) -> Any:
    from agentic_search.embedders.http_generic import GenericHttpEmbedder
    return GenericHttpEmbedder(cfg["url"], int(cfg["dim"]), request=cfg["request"],
                               response_path=cfg["response_path"], **_http_kwargs(cfg, ("headers",)))


def _typesafe(cfg: dict[str, Any], ctx: BuildContext) -> Any:
    from agentic_search.models.typesafe import TypeSafeDecider
    return TypeSafeDecider(cfg.get("model", "jev-latest"), api_key=cfg.get("api_key"), id=cfg.get("id"),
                           batch_size=int(cfg.get("batch_size", 16)), price_per_mtok=_price(cfg))


def _cross_encoder(cfg: dict[str, Any], ctx: BuildContext) -> Any:
    from agentic_search.models.cross_encoder import CrossEncoderJudge
    return CrossEncoderJudge(cfg.get("model", "BAAI/bge-reranker-v2-m3"),
                             apply_sigmoid=cfg.get("apply_sigmoid", False), id=cfg.get("id"))


def _llm_judge(cfg: dict[str, Any], ctx: BuildContext) -> Any:
    from agentic_search.models.llm_judge import LLMJudge
    if "client" not in cfg:
        raise ConfigError("llm_judge needs a nested `client:` config")
    return LLMJudge(build("client", cfg["client"], ctx), id=cfg.get("id"))


for _kind, _type, _factory in [
    ("backend", "files", _files),
    ("backend", "postgres", _postgres),
    ("backend", "pgvector", _postgres),
    ("backend", "mysql", _mysql),
    ("backend", "bigquery", _bigquery),
    ("backend", "opensearch", _opensearch),
    ("backend", "neo4j", _neo4j),
    ("backend", "milvus", _milvus),
    ("embedder", "hash", _hash),
    ("embedder", "sentence_transformers", _sentence_transformers),
    ("embedder", "openai_compat", _openai_embedder),
    ("embedder", "tei", _tei),
    ("embedder", "vertex", _vertex),
    ("embedder", "http", _http_embedder),
    ("client", "anthropic", _anthropic),
    ("client", "openai_compat", _openai_compat),
    ("decider", "cross_encoder", _cross_encoder),
    ("decider", "llm_judge", _llm_judge),
    ("decider", "typesafe", _typesafe),
]:
    register(_kind, _type, _factory)


def build_harness(cfg: dict[str, Any], *, base_dir: Path | None = None) -> Harness:
    from agentic_search.models.driver import ToolCallingDriver

    ctx = BuildContext(base_dir=Path(base_dir) if base_dir else Path.cwd())
    for ecfg in cfg.get("embedders", []):
        embedder = build("embedder", ecfg, ctx)
        ctx.embedders[embedder.id] = embedder
    backends, policy = [], {}
    for bcfg in cfg.get("backends", []):
        backends.append(build("backend", bcfg, ctx))
        if "allowed_models" in bcfg:
            policy[bcfg["name"]] = set(bcfg["allowed_models"])
    if not cfg.get("driver"):
        raise ConfigError("config needs a `driver:` section")
    driver_cfg = dict(cfg["driver"])
    max_calls = int(driver_cfg.pop("max_calls_per_turn", 8))
    driver = ToolCallingDriver(build("client", driver_cfg, ctx), max_calls_per_turn=max_calls)
    analyzer = build("decider", cfg["analyzer"], ctx) if cfg.get("analyzer") else None
    controller = build("decider", cfg["controller"], ctx) if cfg.get("controller") else None
    annotations = cfg.get("annotations")
    if isinstance(annotations, str):
        annotations = yaml.safe_load((ctx.base_dir / annotations).read_text()) or {}
    return Harness(backends, driver, embedders=list(ctx.embedders.values()), analyzer=analyzer,
                   controller=controller, mode=cfg.get("mode", "harness"),
                   budget=Budget(**cfg.get("budget", {})), source_policy=policy or None,
                   annotations=annotations, settings=HarnessSettings(**cfg.get("settings", {})))


def load_harness(path: str | Path) -> Harness:
    path = Path(path)
    cfg = yaml.safe_load(path.read_text()) or {}
    return build_harness(cfg, base_dir=path.parent)
