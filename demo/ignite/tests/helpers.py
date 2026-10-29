"""Offline harness for demo tests: scripted driver, keyword judge, hash embedder."""

from __future__ import annotations

import tempfile
from functools import cache
from pathlib import Path

import yaml

from agentic_search.embedders.local import HashEmbedder
from agentic_search.testing import KeywordJudge, ScriptedDriver, call
from demo.ignite.build import build_demo_harness, load_config
from demo.ignite.corpus import DemoData, load_demo
from demo.ignite.tests.test_public_datasets import _musique

HERO_ID = "4hop1__0"


@cache
def demo_data() -> DemoData:
    """Demo data from a small MuSiQue-shaped fixture, with `hero` = 4hop1__0."""
    folder = Path(tempfile.mkdtemp(prefix="ignite-test-"))
    path, _ = _musique(folder)
    questions = folder / "questions.yaml"
    questions.write_text(yaml.safe_dump([{"id": "hero", "musique_id": HERO_ID},
                                         {"id": "two", "musique_id": "2hop__1"}]))
    return load_demo(path, questions)


def offline_harness(**kwargs):
    driver = ScriptedDriver([
        [call("lexical_search", source="wiki", text=f"{HERO_ID} text 0", limit=10)],
        [call("lexical_search", source="wiki", text=f"{HERO_ID} text 1", limit=10)],
    ])
    return build_demo_harness(load_config(), data=demo_data(), driver=driver,
                              analyzer=KeywordJudge([HERO_ID]), embedder=HashEmbedder(), **kwargs)


class BrokenDriver:
    """A driver whose plan() always raises, to test failed runs."""

    id = "broken-driver"
    supports_images = False

    async def plan(self, view, tools):
        raise RuntimeError("model unavailable")

    async def run_delegate(self, *args, **kwargs):
        raise NotImplementedError


def broken_harness():
    return build_demo_harness(load_config(), data=demo_data(), driver=BrokenDriver(),
                              analyzer=KeywordJudge([HERO_ID]), embedder=HashEmbedder())
