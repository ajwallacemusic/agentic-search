"""Build the Ignite demo harness: one files backend per source, model roles from config.yaml.

The backends are built in Python because a YAML `files` backend loaded from a directory keeps
only file metadata (path, size, ...), and each paragraph needs its `title` as metadata."""

from __future__ import annotations

import asyncio
import atexit
import hashlib
import logging
import time
from pathlib import Path
from typing import Any, Literal

import yaml

from agentic_search.backends.files import FilesBackend
from agentic_search.config import BuildContext, build, resolve_env
from agentic_search.core.harness import Harness, HarnessSettings
from agentic_search.core.types import Budget, Document, TextPart
from agentic_search.embedders.base import Embedder
from agentic_search.models.driver import ToolCallingDriver
from agentic_search.models.llm_judge import LLMJudge
from agentic_search.models.vertex import vertex_base_url, vertex_client
from demo.ignite.corpus import DemoData

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent / "config.yaml"

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


class CachedEmbedder:
    """Keeps document embeddings in an .npz file, keyed by model id and text, so a restart does not
    re-embed the corpus (WANDS: ~4 min on a laptop). Queries and non-text items pass through."""

    def __init__(self, inner: Embedder, path: Path, *, save_every_s: float = 30.0):
        self.inner = inner
        self.path = Path(path)
        self.save_every_s = save_every_s
        self._vectors: dict[str, Any] = {}
        self._dirty = False
        self._saved_at = float("-inf")
        atexit.register(self.flush)
        if self.path.exists():
            import numpy as np

            with np.load(self.path) as f:
                self._vectors = dict(zip(f["keys"].tolist(), f["vectors"]))

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def _key(self, text: str) -> str:
        return hashlib.sha1(f"{self.inner.id}\n{text}".encode()).hexdigest()

    async def embed(self, items: list[Any], purpose: Any) -> list[Any]:
        if purpose != "document" or not all(isinstance(i, TextPart) for i in items):
            return await self.inner.embed(items, purpose)
        keys = [self._key(i.text) for i in items]
        missing = [i for i, k in enumerate(keys) if k not in self._vectors]
        if missing:
            fresh = await self.inner.embed([items[i] for i in missing], purpose)
            for i, vector in zip(missing, fresh):
                self._vectors[keys[i]] = vector
            self._dirty = True
            if time.monotonic() - self._saved_at >= self.save_every_s:
                self.flush()
        return [self._vectors[k] for k in keys]

    def flush(self) -> None:
        """Write new vectors to disk. Saves are throttled (a corpus is embedded in small batches,
        and each save rewrites the file); this also runs at exit."""
        import numpy as np

        if not self._dirty:
            return

        self.path.parent.mkdir(parents=True, exist_ok=True)
        keys = list(self._vectors)
        tmp = self.path.with_suffix(".tmp.npz")
        np.savez(tmp, keys=np.array(keys),
                 vectors=np.array([self._vectors[k] for k in keys], dtype=np.float32))
        tmp.replace(self.path)
        self._dirty, self._saved_at = False, time.monotonic()


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
                       data: DemoData, driver: Any = None, analyzer: Any = None,
                       embedder: Embedder | None = None) -> Harness:
    """The Ignite harness over one corpus of the demo data. The keyword overrides let tests run
    offline; by default every role comes from `cfg`."""
    return build_harness_for(cfg, data.documents, data.descriptions,
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
    if embedder is None:
        embedder = _embedder(cfg, ctx)
        if cfg.get("embedding_cache"):
            embedder = CachedEmbedder(embedder, CONFIG_PATH.parent / cfg["embedding_cache"])
    embedder = SerialEmbedder(embedder)
    backends = [FilesBackend.from_documents(source, source_docs, embedder=embedder,
                                            collection=source, description=descriptions[source])
                for source, source_docs in docs.items()]
    return Harness(backends, driver, embedders=[embedder], analyzer=analyzer, controller=decider,
                   mode="harness", budget=Budget(**cfg["budget"]),
                   settings=HarnessSettings(**cfg.get("settings", {})))
