"""Build a Harness from a YAML/dict config. Plans 2 and 3 register more types via register()."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml

from agentic_search.core.harness import Harness, HarnessSettings
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
    return factory(cfg, ctx)


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
    ("embedder", "hash", _hash),
    ("embedder", "sentence_transformers", _sentence_transformers),
    ("client", "anthropic", _anthropic),
    ("client", "openai_compat", _openai_compat),
    ("decider", "cross_encoder", _cross_encoder),
    ("decider", "llm_judge", _llm_judge),
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
