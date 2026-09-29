import pytest

from agentic_search import Budget, Harness, Query
from agentic_search.backends.files import FilesBackend
from agentic_search.core.harness import HarnessError
from agentic_search.core.hooks import Hooks
from agentic_search.core.types import StopReason
from agentic_search.models.base import Action
from agentic_search.testing import (
    FailingDecider,
    KeywordJudge,
    ScriptedController,
    ScriptedDriver,
    call,
)


def lex(text, **kw):
    return call("lexical_search", source="docs", text=text, **kw)


def make(docs_backend, driver, **kw):
    return Harness([docs_backend], driver, embedders=[docs_backend.embedder], **kw)


async def test_harness_mode_loop(docs_backend):
    driver = ScriptedDriver([[lex("headache")], [lex("fever")]])
    h = make(docs_backend, driver, analyzer=KeywordJudge(["headache", "fever"]))
    res = await h.search("what treats headache?")
    assert set(res.keys()) == {"docs:d1", "docs:d4"}
    assert res.stop_reason is StopReason.CONTROLLER_STOP  # turn 1 found nothing new
    assert res.usage.turns == 2 and res.usage.tool_calls == 2
    assert all(r.judged and r.p_relevant == 1.0 for r in res.hits)
    turn1 = driver.views[1]
    assert "docs:d1" in turn1.digest and turn1.directive.action is Action.CONTINUE
    assert driver.views[0].digest.startswith("No searches")


async def test_retrieval_mode_single_pass(docs_backend):
    driver = ScriptedDriver([[lex("headache")], [lex("fever")]])
    res = await make(docs_backend, driver).search("q", mode="retrieval")
    assert res.stop_reason is StopReason.SINGLE_PASS and len(driver.views) == 1
    assert set(res.keys()) == {"docs:d1", "docs:d4"} and not res.hits[0].judged


async def test_budget_turns_and_tool_calls(docs_backend):
    judge = KeywordJudge(["headache"])
    d1 = ScriptedDriver([[lex("headache")], [lex("pain")]])
    r1 = await make(docs_backend, d1, analyzer=judge).search("q", budget=Budget(max_turns=1))
    assert r1.stop_reason is StopReason.BUDGET_TURNS
    d2 = ScriptedDriver([[lex("headache"), lex("pain")]])
    r2 = await make(docs_backend, d2, analyzer=judge).search("q", budget=Budget(max_tool_calls=1))
    assert r2.stop_reason is StopReason.BUDGET_TOOL_CALLS
    assert len(r2.trace.of_type("tool_call")) == 1


async def test_no_plan(docs_backend):
    res = await make(docs_backend, ScriptedDriver([])).search("q")
    assert res.stop_reason is StopReason.NO_PLAN and res.hits == []


async def test_controller_decider(docs_backend):
    driver = ScriptedDriver([[lex("headache")], [lex("pain")], [lex("castles")]])
    ctrl = ScriptedController([Action.CONTINUE, Action.STOP])
    res = await make(docs_backend, driver, controller=ctrl).search("q")
    assert res.stop_reason is StopReason.CONTROLLER_STOP and len(driver.views) == 2


async def test_model_mode_delegate(docs_backend):
    driver = ScriptedDriver(delegate_calls=[[lex("headache", id="a")]],
                            delegate_keys=["docs:d4", "docs:d1", "docs:nope"])
    res = await make(docs_backend, driver).search("q", mode="model")
    assert res.keys() == ["docs:d4", "docs:d1"]
    assert [h.score for h in res.hits] == [1.0, 0.5]
    assert res.stop_reason is StopReason.DELEGATE_DONE
    assert "docs:d1" in driver.delegate_outputs[0][0]
    assert "`docs`" in driver.delegate_context
    assert res.trace.of_type("delegate")[0].data["unknown_keys"] == ["docs:nope"]


async def test_model_mode_rerank_with_judge(docs_backend):
    driver = ScriptedDriver(delegate_calls=[[lex("headache", id="a")]],
                            delegate_keys=["docs:d4", "docs:d1"])
    res = await make(docs_backend, driver, analyzer=KeywordJudge(["aspirin"])).search("q", mode="model")
    assert res.keys() == ["docs:d1", "docs:d4"]


async def test_model_mode_budget(docs_backend):
    driver = ScriptedDriver(delegate_calls=[[lex("headache", id="a")], [lex("pain", id="b")]])
    res = await make(docs_backend, driver).search("q", mode="model", budget=Budget(max_tool_calls=1))
    assert driver.delegate_outputs[1][0].startswith("[budget]")
    assert res.stop_reason is StopReason.BUDGET_TOOL_CALLS


async def test_failing_judge_fails_open(docs_backend):
    driver = ScriptedDriver([[lex("headache")]])
    res = await make(docs_backend, driver, analyzer=FailingDecider()).search("q")
    assert res.hits and not any(h.judged for h in res.hits)
    assert res.trace.of_type("judge_error")
    assert res.stop_reason is StopReason.NO_PLAN


async def test_hooks_see_every_model_call(docs_backend):
    class Recording(Hooks):
        def __init__(self):
            self.models, self.events = [], []

        async def before_model_call(self, model_id, payload):
            self.models.append(model_id)
            return payload

        def on_trace_event(self, event):
            self.events.append(event.type)

    hooks = Recording()
    driver = ScriptedDriver([[lex("headache"), call("vector_search", source="docs",
                                                    field="embedding", hyde_text="fever")]])
    await make(docs_backend, driver, analyzer=KeywordJudge(["x"]), hooks=hooks).search("q")
    assert {"scripted-driver", "keyword-judge", "hash"} <= set(hooks.models)
    assert {"setup", "plan", "tool_call", "finalize"} <= set(hooks.events)


class BrokenBackend:
    name, backend_type = "broken", "x"

    def capabilities(self):
        return set()

    async def discover(self, detail="full", collection=None):
        raise ConnectionError("no route")

    async def execute(self, op):
        return []

    async def close(self):
        pass


async def test_setup_errors(docs_backend):
    h = Harness([docs_backend, BrokenBackend()], ScriptedDriver([]))
    await h.setup()
    assert "broken" in h.setup_errors and set(h.manifests) == {"docs"}
    with pytest.raises(HarnessError):
        await Harness([BrokenBackend()], ScriptedDriver([])).search("q")


async def test_sources_and_config_errors(docs_backend, medical_docs):
    h = make(docs_backend, ScriptedDriver([]))
    with pytest.raises(HarnessError):
        await h.search("q", sources=["nope"])
    with pytest.raises(HarnessError):
        await h.search("q", mode="bogus")
    with pytest.raises(HarnessError):
        Harness([docs_backend, FilesBackend.from_documents("docs", medical_docs)], ScriptedDriver([]))
    with pytest.raises(HarnessError):
        Harness([docs_backend], ScriptedDriver([]), annotations={"nope": {}})


async def test_annotations_applied_and_result_serializes(docs_backend):
    h = make(docs_backend, ScriptedDriver([[lex("headache")]]),
             annotations={"docs": {"description": "Drug notes"}})
    res = await h.search(Query.of("q"))
    assert h.manifests["docs"].description == "Drug notes"
    assert '"stop_reason"' in res.model_dump_json()
    async with h:
        pass
