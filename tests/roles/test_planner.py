from agentic_search.core.hooks import Hooks
from agentic_search.core.state import SearchState
from agentic_search.core.types import Budget, Capability, Manifest, Query
from agentic_search.models.base import Action, Decision
from agentic_search.roles.planner import Planner, render_manifests
from agentic_search.testing import ScriptedDriver, call


def state():
    m = Manifest(source="docs", backend_type="files", capabilities={Capability.LEXICAL})
    s = SearchState(question=Query.of("q"), manifests={"docs": m}, budget=Budget())
    s.turn, s.digest = 2, "digest text"
    s.last_decision = Decision(action=Action.BROADEN, note="wider")
    return s


async def test_plan_truncates_renames_and_traces():
    d = ScriptedDriver([[call("a"), call("b"), call("c")]])
    s = state()
    res = await Planner(d, max_calls_per_turn=2).plan(s, [])
    assert [c.id for c in res.calls] == ["t2.c0", "t2.c1"]
    v = d.views[0]
    assert v.digest == "digest text" and v.directive.action is Action.BROADEN and v.max_calls == 2
    assert "`docs`" in v.manifest_summary
    ev = s.trace.of_type("plan")[0]
    assert ev.data["n_calls"] == 2 and ev.data["dropped"] == 1


async def test_hooks_can_rewrite_view():
    class Redact(Hooks):
        async def before_model_call(self, model_id, payload):
            return payload.model_copy(update={"digest": "[redacted]"})

    d = ScriptedDriver([[]])
    await Planner(d, hooks=Redact()).plan(state(), [])
    assert d.views[0].digest == "[redacted]"


def test_render_manifests_sorted():
    a = Manifest(source="a", backend_type="x", capabilities=set())
    b = Manifest(source="b", backend_type="x", capabilities=set())
    out = render_manifests({"b": b, "a": a})
    assert out.index("`a`") < out.index("`b`")
