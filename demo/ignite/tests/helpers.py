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
