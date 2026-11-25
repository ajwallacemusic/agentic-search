"""Build the Ignite demo harness: one files backend per source, model roles from config.yaml.

The backends are built in Python because a YAML `files` backend loaded from a directory keeps
only file metadata (path, size, ...), and each paragraph needs its `title` as metadata."""

from __future__ import annotations

import atexit
import hashlib
import logging
import os
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
from demo.ignite.corpus import DemoData
from demo.ignite.decision_judge import DecisionJudge
from demo.ignite.sid_swarm import SidSwarmDriver
from demo.ignite.swarm import Swarm, SwarmConfig

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent / "config.yaml"

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
        self._vectors = self._on_disk()

    def _on_disk(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        import numpy as np

        with np.load(self.path) as f:
            return dict(zip(f["keys"].tolist(), f["vectors"]))

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
        and each save rewrites the file); this also runs at exit. Vectors another cache on the same
        file saved meanwhile (one per corpus in `serve`) are kept."""
        import numpy as np

        if not self._dirty:
            return

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._vectors = {**self._on_disk(), **self._vectors}
        keys = list(self._vectors)
        tmp = self.path.with_suffix(".tmp.npz")
        np.savez(tmp, keys=np.array(keys),
                 vectors=np.array([self._vectors[k] for k in keys], dtype=np.float32))
        tmp.replace(self.path)
        self._dirty, self._saved_at = False, time.monotonic()


def make_client(cfg: dict[str, Any]) -> Any:
    """A model client from one `driver` / `judge` block of config.yaml (`*_env` keys resolved).
    `reasoning_effort`, `request_timeout_s` and `max_retries` are library client settings."""
    return build("client", resolve_env(cfg), BuildContext(base_dir=CONFIG_PATH.parent))


def make_judge(cfg: dict[str, Any] | None, fallback_client: Any) -> Any:
    """The judge from the `judge` block: `type: gemini_decision` is Gemini Decision 1 (one yes/no
    probability per hit, `decision_judge.py`); any other type is an LLM client behind LLMJudge.
    No block: the planner's client judges."""
    if not cfg:
        return LLMJudge(fallback_client)
    if cfg.get("type") == "gemini_decision":
        return DecisionJudge(model=cfg.get("model", "gemini-decision-1"),
                             key=os.environ.get(cfg.get("api_key_env", "GEMINI_API_KEY")) or None,
                             price_per_mtok_in=float(cfg.get("price_per_mtok", [0.10, 0.0])[0]),
                             request_timeout_s=float(cfg.get("request_timeout_s", 15)),
                             max_retries=int(cfg.get("max_retries", 1)))
    return LLMJudge(make_client(cfg))


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
        analyzer = analyzer or make_judge(cfg.get("judge"), client)
    decider = build("decider", cfg["typesafe"], ctx) if controller == "typesafe" else None
    if embedder is None:
        embedder = _embedder(cfg, ctx)
        if cfg.get("embedding_cache"):
            embedder = CachedEmbedder(embedder, CONFIG_PATH.parent / cfg["embedding_cache"])
    backends = [FilesBackend.from_documents(source, source_docs, embedder=embedder,
                                            collection=source, description=descriptions[source])
                for source, source_docs in docs.items()]
    return Harness(backends, driver, embedders=[embedder], analyzer=analyzer, controller=decider,
                   mode="harness", budget=Budget(**cfg["budget"]),
                   settings=HarnessSettings(**cfg.get("settings", {})))


def sid_swarm_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    """The demo config for SID-2 Swarm runs: its own budget (`sid_swarm.budget`; the swarm makes
    many calls at once, so the turn budget does not apply) and the judge off, so the swarm's own
    RRF order is scored as it is."""
    sid = cfg["sid_swarm"]
    return {**cfg, "budget": sid["budget"],
            "settings": {**cfg.get("settings", {}), "judge_delegate": False}}


def make_sid_driver(cfg: dict[str, Any], *, connect: Any = None) -> SidSwarmDriver:
    sid = cfg["sid_swarm"]
    return SidSwarmDriver(model=sid.get("model", "SID-2-Swarm"),
                          max_agents=int(sid.get("max_agents", 6)),
                          agent_max_turns=int(sid.get("agent_max_turns", 10)),
                          reward=sid.get("reward", "ndcg"),
                          price_per_mtok=tuple(sid.get("price_per_mtok", (0.50, 0.05, 2.00))),
                          key=os.environ.get(sid.get("api_key_env", "SID_API_KEY")) or None,
                          grace_s=float(sid.get("grace_s", 8.0)),
                          est_cost_per_action=float(sid.get("est_cost_per_action", 0.0)),
                          connect=connect)


def build_sid_harness_for(cfg: dict[str, Any], docs: dict[str, list[Document]],
                          descriptions: dict[str, str], *, connect: Any = None,
                          embedder: Embedder | None = None) -> Harness:
    """A harness whose model mode (`search(..., mode="model")`) is SID-2 Swarm: SID runs the
    agents, this harness runs their searches over the same files backends (`sid_swarm.py`)."""
    scfg = sid_swarm_cfg(cfg)
    return build_harness_for(scfg, docs, descriptions, driver=make_sid_driver(cfg, connect=connect),
                             analyzer=make_judge(cfg.get("judge"), None), embedder=embedder)


def build_demo_swarm(cfg: dict[str, Any], harness: Harness, descriptions: dict[str, str], *,
                     client: Any = None) -> Swarm:
    """The swarm over `harness`: the split call uses the planner model unless `client` is given;
    agent count, turns and RRF k come from the `swarm` block."""
    return Swarm(harness, client or make_client(cfg["driver"]),
                 SwarmConfig(**cfg.get("swarm", {})), descriptions)
