import json

from agentic_search.events import ToolCallStarted
from demo.ignite.corpus import load_questions
from demo.ignite.run_side_by_side import (
    format_event,
    gold_recall,
    run_one,
    save_path,
    summary_table,
)
from demo.ignite.tests.helpers import broken_harness, offline_harness

HERO = load_questions()["hero"]


def test_gold_recall_counts_doc_ids_in_top_10():
    ids = ["x"] * 9 + ["INC-4821", "PR-1203"]
    assert gold_recall(ids, ["INC-4821", "PR-1203"]) == 1


async def test_run_one_summarizes_harness_mode():
    lines: list[str] = []
    s = await run_one(offline_harness(), HERO, "harness", save_to=None, print_fn=lines.append)
    assert s.mode == "harness" and s.tool_calls >= 2 and s.turns >= 2 and s.gold_total == 4
    assert s.gold_recall >= 1  # INC-4821 is found on turn 0
    assert any("incidents" in line and "lexical_search" in line for line in lines)


async def test_save_writes_each_event_as_it_arrives(tmp_path):
    out = tmp_path / "r.jsonl"
    await run_one(offline_harness(), HERO, "retrieval", save_to=out, print_fn=lambda _: None)
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert rows[0]["type"] == "search_started" and rows[-1]["type"] == "search_finished"
    assert [r["seq"] for r in rows] == list(range(len(rows)))


async def test_failed_run_is_summarized(tmp_path):
    lines: list[str] = []
    out = tmp_path / "f.jsonl"
    s = await run_one(broken_harness(), HERO, "harness", save_to=out, print_fn=lines.append)
    assert s.stop_reason.startswith("failed")
    assert any("failed" in line for line in lines)
    assert json.loads(out.read_text().splitlines()[-1])["type"] == "search_failed"


def test_format_event_shows_source_and_tool():
    line = format_event(ToolCallStarted(
        search_id="s", seq=1, turn=0, at_ms=1.0, call_id="c", source="deploys",
        name="filter_search", arguments={"source": "deploys", "filter": {"op": "eq"}}))
    assert "deploys" in line and "filter_search" in line


def test_save_path_names_question_and_mode():
    assert save_path("hero", "harness", "20261029-1200").name == "20261029-1200-hero-harness.jsonl"


async def test_summary_table_has_both_modes():
    rows = [await run_one(offline_harness(), HERO, m, save_to=None, print_fn=lambda _: None)
            for m in ("retrieval", "harness")]
    table = summary_table(rows)
    assert "retrieval" in table and "harness" in table and "gold" in table
