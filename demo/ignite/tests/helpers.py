"""Offline harness for demo tests: scripted driver, keyword judge, hash embedder."""

from __future__ import annotations

from agentic_search.embedders.local import HashEmbedder
from agentic_search.testing import KeywordJudge, ScriptedDriver, call
from demo.ignite.build import build_demo_harness, load_config


def offline_harness(**kwargs):
    driver = ScriptedDriver([
        [call("lexical_search", source="incidents", text="search-api latency", limit=10)],
        [call("lexical_search", source="deploys", text="search-api reranker", limit=10)],
    ])
    return build_demo_harness(load_config(), driver=driver, analyzer=KeywordJudge(["search-api"]),
                              embedder=HashEmbedder(), **kwargs)


class BrokenDriver:
    """A driver whose plan() always raises, to test failed runs."""

    id = "broken-driver"
    supports_images = False

    async def plan(self, view, tools):
        raise RuntimeError("model unavailable")

    async def run_delegate(self, *args, **kwargs):
        raise NotImplementedError


def broken_harness():
    return build_demo_harness(load_config(), driver=BrokenDriver(),
                              analyzer=KeywordJudge(["search-api"]), embedder=HashEmbedder())
