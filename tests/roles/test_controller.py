from agentic_search.core.hooks import Hooks
from agentic_search.core.state import SearchState
from agentic_search.core.types import Budget, Hit, Query, StopReason
from agentic_search.models.base import Action, TurnSummary
from agentic_search.roles.controller import Controller
from agentic_search.testing import FailingDecider, KeywordJudge, ScriptedController


def state(budget=None, history=(), relevant=None):
    s = SearchState(question=Query.of("q"), manifests={}, budget=budget or Budget())
    s.history = list(history)
    if relevant is not None:
        s.pool.add([Hit(doc_id="a", source="s")], turn=0, call_id="c")
        s.pool["s:a"].judged, s.pool["s:a"].p_relevant = True, 1.0 if relevant else 0.0
    return s


def turn(n_calls=1, n_errors=0, n_new=2, n_new_relevant=1, top=("s:a",)):
    return TurnSummary(turn=0, n_calls=n_calls, n_errors=n_errors, n_new=n_new,
                       n_new_relevant=n_new_relevant, top_keys=list(top))


def test_budget_stop_order():
    c = Controller(None)
    s = state(Budget(max_turns=1, max_tool_calls=1))
    assert c.budget_stop(s) is None
    s.usage.turns = 1
    assert c.budget_stop(s) is StopReason.BUDGET_TURNS
    s = state(Budget(max_cost_usd=0.1, max_tokens=10, max_seconds=None))
    s.usage.input_tokens = 10
    assert c.budget_stop(s) is StopReason.BUDGET_TOKENS
    s.usage.input_tokens, s.usage.cost_usd = 0, 0.2
    assert c.budget_stop(s) is StopReason.BUDGET_COST
    s = state(Budget(max_seconds=0.0))
    assert c.budget_stop(s) is StopReason.BUDGET_TIME


def test_heuristic_rules():
    c = Controller(None)
    assert c.heuristic(state(history=[turn(n_calls=2, n_errors=2)])).action is Action.REFINE
    assert c.heuristic(state(history=[turn(n_new_relevant=0)], relevant=True)).action is Action.STOP
    assert c.heuristic(state(history=[turn(n_new_relevant=0)], relevant=False)).action is Action.BROADEN
    assert c.heuristic(state(history=[turn(n_new=0, n_new_relevant=None)])).action is Action.STOP
    assert c.heuristic(state(history=[turn(), turn()])).action is Action.STOP  # top stable
    assert c.heuristic(state(history=[turn(top=("x",)), turn()])).action is Action.CONTINUE


def test_budget_remaining():
    s = state(Budget(max_turns=3, max_tool_calls=10, max_cost_usd=1.0))
    s.usage.turns, s.usage.tool_calls = 1, 4
    r = Controller(None).budget_remaining(s)
    assert r["turns"] == 2 and r["tool_calls"] == 6 and r["cost_usd"] == 1.0 and r["tokens"] is None


async def test_decide_uses_decider_and_traces():
    ctrl = ScriptedController([Action.BROADEN])
    s = state(history=[turn()])
    d = await Controller(ctrl).decide(s)
    assert d.action is Action.BROADEN
    assert ctrl.views[0].history[0].n_new == 2
    assert s.trace.of_type("decision")[0].data["by"] == "scripted-controller"


async def test_decide_falls_back_to_heuristic():
    s = state(history=[turn(n_new=0, n_new_relevant=None)])
    d = await Controller(FailingDecider()).decide(s)
    assert d.action is Action.STOP
    assert s.trace.of_type("decision_error")
    s2 = state(history=[turn(n_new=0, n_new_relevant=None)])
    assert (await Controller(KeywordJudge([])).decide(s2)).action is Action.STOP
    assert s2.trace.of_type("decision")[0].data["by"] == "heuristic"


async def test_decide_hook_raises_falls_back_to_heuristic():
    """When before_model_call hook raises, fall back to heuristic and record decision_error."""

    class PermissionHook(Hooks):
        async def before_model_call(self, model_id: str, payload):
            raise PermissionError("access denied")

    ctrl = ScriptedController([Action.BROADEN])
    s = state(history=[turn(n_new=0, n_new_relevant=None)])
    c = Controller(ctrl, hooks=PermissionHook())
    d = await c.decide(s)

    # Should fall back to heuristic (which returns STOP in this case)
    assert d.action is Action.STOP

    # Should have recorded decision_error
    errors = s.trace.of_type("decision_error")
    assert len(errors) > 0
    assert "PermissionError" in errors[0].data["error"]
    assert errors[0].data["decider"] == "scripted-controller"
