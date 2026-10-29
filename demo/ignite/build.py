"""Build the Ignite demo harness: one files backend per source, model roles from config.yaml.

The backends are built in Python because a YAML `files` backend loaded from a directory keeps
only file metadata (path, size, ...), and the deploy log needs filterable `date` and `service`."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

import yaml

from agentic_search.backends.files import FilesBackend
from agentic_search.config import BuildContext, build
from agentic_search.core.harness import Harness
from agentic_search.core.types import Budget
from agentic_search.embedders.base import Embedder
from agentic_search.models.driver import ToolCallingDriver
from agentic_search.models.llm_judge import LLMJudge
from demo.ignite.corpus import DATA_DIR, SOURCES, load_documents

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent / "config.yaml"

SOURCE_DESCRIPTIONS = {
    "incidents": ("Incident tickets: one per incident. Fields: opened (date), service, severity, "
                  "status. Text is the on-call write-up."),
    "deploys": ("Deploy log: one record per deploy. Fields: date, service, version, pr (PR "
                "number), author."),
    "pull_requests": ("Pull requests: title and description. Fields: number, merged (date), "
                      "service, author."),
    "dashboards": ("Weekly SLO reports per service: latency, availability, error budget. Fields: "
                   "service, week, week_start, week_end."),
    "runbooks": "Operational runbooks and how-to guides. Field: service.",
}


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
    """The keyword overrides let tests run offline; by default every role comes from `cfg`."""
    ctx = BuildContext(base_dir=CONFIG_PATH.parent)
    if driver is None or analyzer is None:
        client = build("client", cfg["driver"], ctx)
        driver = driver or ToolCallingDriver(client,
                                             max_calls_per_turn=int(cfg.get("max_calls_per_turn", 8)))
        analyzer = analyzer or LLMJudge(client)
    decider = build("decider", cfg["typesafe"], ctx) if controller == "typesafe" else None
    embedder = embedder or _embedder(cfg, ctx)
    docs = load_documents(data_dir)
    backends = [FilesBackend.from_documents(source, docs[source], embedder=embedder,
                                            description=SOURCE_DESCRIPTIONS[source])
                for source in SOURCES]
    return Harness(backends, driver, embedders=[embedder], analyzer=analyzer, controller=decider,
                   mode="harness", budget=Budget(**cfg["budget"]))
