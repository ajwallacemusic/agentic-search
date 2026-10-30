from agentic_search import Harness
from agentic_search.core.harness import HarnessSettings
from agentic_search.core.hooks import Hooks
from agentic_search.core.state import SearchState
from agentic_search.core.types import Budget, Hit, Query, StopReason
from agentic_search.models.base import Action, Fact, Step, TurnSummary
from agentic_search.roles.controller import Controller
from agentic_search.testing import (
    FailingDecider,
    KeywordJudge,
    ScriptedController,
    ScriptedDriver,
    call,
)


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


def with_steps(s, *steps):
    s.steps = [Step(text=t, done=d) for t, d in steps]
    return s


def idle(**kw):
    return TurnSummary(turn=0, n_calls=1, n_errors=0, n_new=0, n_new_relevant=0, top_keys=[], **kw)


def test_fact_turn_continues_with_the_current_step():
    s = with_steps(state(history=[turn(n_new_relevant=0, n_new=0, top=())]), ("a", True), ("find b", False))
    s.history[-1].n_facts = 1
    d = Controller(None).heuristic(s)
    assert d.action is Action.CONTINUE and d.note == "step 2: find b"
    s.history[-1].n_facts, s.history[-1].n_steps_done = 0, 1
    assert Controller(None).heuristic(s).action is Action.CONTINUE


def test_open_step_turns_stop_into_continue():
    s = with_steps(state(history=[turn(n_new_relevant=0)], relevant=True), ("find a", False))
    d = Controller(None).heuristic(s)
    assert d.action is Action.CONTINUE and d.note == "step 1: find a"
    s = with_steps(state(history=[turn(n_new_relevant=0)], relevant=False), ("find a", False))
    assert Controller(None).heuristic(s).action is Action.CONTINUE
    s = with_steps(state(history=[turn(n_calls=2, n_errors=2)]), ("find a", False))
    assert Controller(None).heuristic(s).action is Action.REFINE


def test_two_idle_turns_stop_as_stuck():
    s = with_steps(state(history=[idle(), idle()], relevant=True), ("a", True), ("find b", False))
    d = Controller(None).heuristic(s)
    assert d.action is Action.STOP and d.note == "stuck on step 2"
    s.history[0].n_facts = 1
    assert Controller(None).heuristic(s).action is Action.CONTINUE
    s = with_steps(state(history=[idle()], relevant=True), ("find b", False))
    assert Controller(None).heuristic(s).action is Action.CONTINUE


def test_all_steps_done_uses_the_old_rules():
    s = with_steps(state(history=[turn(n_new_relevant=0)], relevant=True), ("a", True))
    assert Controller(None).heuristic(s).action is Action.STOP
    s = state(history=[turn(n_new_relevant=0)], relevant=False)
    assert Controller(None).heuristic(s).action is Action.BROADEN


async def test_steps_on_without_a_plan_behaves_like_off(docs_backend):
    async def go(steps):
        driver = ScriptedDriver([[call("lexical_search", source="docs", text="headache")]] * 3)
        h = Harness([docs_backend], driver, embedders=[docs_backend.embedder],
                    settings=HarnessSettings(steps=steps))
        res = await h.search("q", budget=Budget(max_turns=3))
        return res.stop_reason, res.usage.tool_calls, res.keys()
    assert await go(True) == await go(False)


async def test_budget_beats_an_open_step(docs_backend):
    driver = ScriptedDriver([[call("plan_steps", steps=["a", "b", "c"]),
                              call("lexical_search", source="docs", text="headache")],
                             [call("lexical_search", source="docs", text="fever")],
                             [call("lexical_search", source="docs", text="cough")]])
    h = Harness([docs_backend], driver, embedders=[docs_backend.embedder],
                settings=HarnessSettings(steps=True))
    res = await h.search("q", budget=Budget(max_turns=2))
    assert res.stop_reason is StopReason.BUDGET_TURNS


async def test_controller_view_carries_steps():
    ctrl = ScriptedController([Action.STOP])
    s = with_steps(state(history=[turn()]), ("a", False))
    s.notebook = [Fact(text="f", source_key="s:a", turn=0)]
    d = await Controller(ctrl).decide(s)
    assert d.action is Action.STOP
    assert ctrl.views[0].steps[0].text == "a" and ctrl.views[0].notebook[0].text == "f"


def test_opened_documents_or_a_new_plan_is_not_idle():
    for kw in ({"n_opened": 1}, {"plan_set": True}):
        s = with_steps(state(history=[idle(**kw), idle(**kw)], relevant=True), ("find b", False))
        assert Controller(None).heuristic(s).action is Action.CONTINUE


def test_all_error_turns_refine_instead_of_stuck():
    err = lambda: TurnSummary(turn=0, n_calls=2, n_errors=2, n_new=0, n_new_relevant=0)  # noqa: E731
    s = with_steps(state(history=[err(), err()]), ("find a", False))
    assert Controller(None).heuristic(s).action is Action.REFINE
