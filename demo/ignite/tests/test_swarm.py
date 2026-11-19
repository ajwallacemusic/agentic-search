import pytest

from agentic_search.core.types import Budget
from demo.ignite.swarm import agent_budget, agent_query, parse_split, rrf, split_question


class RaisingClient:
    id = "raising"

    async def chat(self, system, messages, **kw):
        raise TimeoutError("slow")


def test_parse_split_reads_fenced_json():
    subs, fb = parse_split('```json\n{"sub_questions": ["a?", "b?", "c?"]}\n```', "Q", 4)
    assert subs == ["a?", "b?", "c?"] and fb is False


def test_parse_split_caps_and_dedupes():
    subs, _ = parse_split('{"sub_questions": ["a","a","b","c","d","e","f"]}', "Q", 4)
    assert subs == ["a", "b", "c", "d"]


@pytest.mark.parametrize("text", ["no json here", '{"sub_questions": []}', '{"sub_questions": ["only"]}'])
def test_parse_split_falls_back_to_the_question(text):
    assert parse_split(text, "Q", 4) == (["Q"], True)


async def test_split_question_falls_back_when_the_model_errors():
    split = await split_question(RaisingClient(), "Q", {"wiki": "d"}, 4)
    assert split.sub_questions == ["Q"] and split.fallback and split.usage.cost_usd == 0


def test_agent_budget_divides_the_harness_budget():
    total = Budget(max_turns=4, max_tool_calls=24, max_seconds=120, max_cost_usd=0.50)
    b = agent_budget(total, 4, split_cost=0.02, max_turns=2)
    assert (b.max_turns, b.max_tool_calls, b.max_seconds) == (2, 6, 120)
    assert b.max_cost_usd == pytest.approx(0.12)


def test_agent_budget_gives_at_least_one_call():
    assert agent_budget(Budget(max_tool_calls=3), 6, 0.0, 2).max_tool_calls == 1


def test_agent_budget_keeps_unlimited_cost_unlimited():
    assert agent_budget(Budget(max_cost_usd=None), 4, 0.02, 2).max_cost_usd is None


def test_agent_query_text():
    assert agent_query("Q", "P") == "Main question: Q. Your part: P."


def test_rrf_sums_reciprocal_ranks():
    out = rrf([["a", "b"], ["b", "c"]], k=60)
    assert [key for key, _, _ in out] == ["b", "a", "c"]
    assert out[0][1] == pytest.approx(1/62 + 1/61) and out[0][2] == [0, 1]


def test_rrf_ties_keep_first_seen_order():
    assert [k for k, _, _ in rrf([["x"], ["y"]], k=60)] == ["x", "y"]
