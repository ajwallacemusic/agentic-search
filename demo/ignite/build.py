"""Build the Ignite demo harness: one files backend per source, model roles from config.yaml.

The backends are built in Python because a YAML `files` backend loaded from a directory keeps
only file metadata (path, size, ...), and the deploy log needs filterable `date` and `service`."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Literal

import yaml

from agentic_search.backends.files import FilesBackend
from agentic_search.config import BuildContext, build, resolve_env
from agentic_search.core.harness import Harness, HarnessSettings
from agentic_search.core.types import Budget, Document
from agentic_search.embedders.base import Embedder
from agentic_search.models.driver import ToolCallingDriver
from agentic_search.models.llm_judge import LLMJudge
from agentic_search.models.vertex import vertex_base_url, vertex_client
from demo.ignite.corpus import DATA_DIR, SOURCES, load_documents

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent / "config.yaml"

# The corpus spans July-October 2026; saying so stops the planner guessing the year for
# "last month" when it builds date filters.
SOURCE_DESCRIPTIONS = {
    "incidents": ("Incident tickets: one per incident. Fields: opened (date), service, severity, "
                  "status. Text is the on-call write-up. Covers July-October 2026."),
    "deploys": ("Deploy log: one record per deploy. Fields: date, service, version, pr (PR "
                "number), author. Covers July-October 2026."),
    "pull_requests": ("Pull requests: title and description. Fields: number, merged (date), "
                      "service, author. Covers June-October 2026."),
    "dashboards": ("Weekly SLO reports per service: latency, availability, error budget. Fields: "
                   "service, week, week_start, week_end. Covers July-October 2026."),
    "runbooks": "Operational runbooks and how-to guides, current as of October 2026. Field: service.",
}


class SerialEmbedder:
    """Runs one `embed` call at a time. `Harness.setup()` indexes every source at once, and a
    shared in-process sentence-transformers model segfaults when two threads load or run it."""

    def __init__(self, inner: Embedder):
        self.inner = inner
        self._lock = asyncio.Lock()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    async def embed(self, items: list[Any], purpose: Any) -> list[Any]:
        async with self._lock:
            return await self.inner.embed(items, purpose)


class RequestDefaults:
    """Wraps an AsyncOpenAI client so every `chat.completions.create` call also sends
    `defaults` (e.g. reasoning_effort); the library's client has no setting for them."""

    def __init__(self, inner: Any, **defaults: Any):
        self.inner = inner
        self.defaults = defaults
        self.chat = self
        self.completions = self

    async def create(self, **kwargs: Any) -> Any:
        return await self.inner.chat.completions.create(**kwargs, **self.defaults)


def make_client(cfg: dict[str, Any]) -> Any:
    """The library's client factory, except a Vertex driver also gets `reasoning_effort`, a
    per-request timeout and a retry count: Gemini's default thinking makes judging slow, and a
    stalled request should be retried rather than eat the demo's time budget."""
    cfg = resolve_env(cfg)
    if cfg["type"] != "vertex":
        return build("client", cfg, BuildContext(base_dir=CONFIG_PATH.parent))
    from openai import AsyncOpenAI

    location = cfg.get("location", "global")
    raw = AsyncOpenAI(base_url=vertex_base_url(cfg["project"], location), api_key="unused",
                      timeout=float(cfg.get("request_timeout_s", 60)),
                      max_retries=int(cfg.get("max_retries", 2)))
    defaults = {"reasoning_effort": cfg["reasoning_effort"]} if "reasoning_effort" in cfg else {}
    price = cfg.get("price_per_mtok")
    return vertex_client(cfg["model"], project=cfg["project"], location=location,
                         client=RequestDefaults(raw, **defaults),
                         price_per_mtok=(float(price[0]), float(price[1])) if price else None)


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text()) or {}


def _embedder(cfg: dict[str, Any], ctx: BuildContext) -> Embedder:
    try:
        return build("embedder", cfg["embedder"], ctx)
    except ImportError as exc:
        logger.warning("embedder %s unavailable (%s); using %s", cfg["embedder"].get("model"),
                       exc, cfg["fallback_embedder"]["type"])
        return build("embedder", cfg["fallback_embedder"], ctx)


def build_demo_harness(cfg: dict[str, Any], *,
                       controller: Literal["heuristic", "typesafe"] = "heuristic",
                       data_dir: Path = DATA_DIR, driver: Any = None, analyzer: Any = None,
                       embedder: Embedder | None = None) -> Harness:
    """The Ignite harness: one files backend per demo source. The keyword overrides let tests
    run offline; by default every role comes from `cfg`."""
    docs = load_documents(data_dir)
    return build_harness_for(cfg, {s: docs[s] for s in SOURCES}, SOURCE_DESCRIPTIONS,
                             controller=controller, driver=driver, analyzer=analyzer,
                             embedder=embedder)


def build_harness_for(cfg: dict[str, Any], docs: dict[str, list[Document]],
                      descriptions: dict[str, str], *,
                      controller: Literal["heuristic", "typesafe"] = "heuristic",
                      driver: Any = None, analyzer: Any = None,
                      embedder: Embedder | None = None) -> Harness:
    """A harness over `docs` (source name -> documents), one files backend per source, with the
    planner, judge, controller, budget and settings from `cfg`."""
    ctx = BuildContext(base_dir=CONFIG_PATH.parent)
    if driver is None or analyzer is None:
        client = make_client(cfg["driver"])
        driver = driver or ToolCallingDriver(client)
        judge_client = make_client(cfg["judge"]) if cfg.get("judge") else client
        analyzer = analyzer or LLMJudge(judge_client)
    decider = build("decider", cfg["typesafe"], ctx) if controller == "typesafe" else None
    embedder = SerialEmbedder(embedder or _embedder(cfg, ctx))
    backends = [FilesBackend.from_documents(source, source_docs, embedder=embedder,
                                            collection=source, description=descriptions[source])
                for source, source_docs in docs.items()]
    return Harness(backends, driver, embedders=[embedder], analyzer=analyzer, controller=decider,
                   mode="harness", budget=Budget(**cfg["budget"]),
                   settings=HarnessSettings(**cfg.get("settings", {})))
