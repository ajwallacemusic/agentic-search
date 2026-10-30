from agentic_search import Harness
from agentic_search.core.harness import HarnessSettings
from agentic_search.core.state import SearchState
from agentic_search.core.types import Budget
from agentic_search.roles.workspace import MAX_FACT_CHARS, MAX_STEPS, WORKSPACE_NAMES
from agentic_search.testing import ScriptedDriver, call


def lex(text, **kw):
    return call("lexical_search", source="docs", text=text, **kw)


def make(docs_backend, driver, steps=True):
    return Harness([docs_backend], driver, embedders=[docs_backend.embedder],
                   settings=HarnessSettings(steps=steps))


async def run(docs_backend, turns, **kw):
    """Run a search and return (driver, result, state-like captured via a spying driver)."""
    driver = ScriptedDriver(turns)
    h = make(docs_backend, driver, **kw)
    states: list[SearchState] = []
    orig = h._run_loop

    async def spy(state, *a, **k):
        states.append(state)
        return await orig(state, *a, **k)

    h._run_loop = spy
    res = await h.search("q", budget=Budget(max_turns=3))
    return driver, res, states[0]


async def test_steps_off_offers_no_workspace_tools(docs_backend):
    driver, _, _ = await run(docs_backend, [[lex("headache")]], steps=False)
    assert not {t.name for t in driver.tools_seen[0]} & WORKSPACE_NAMES
    assert driver.views[0].steps == []


async def test_steps_on_offers_workspace_tools(docs_backend):
    driver, _, _ = await run(docs_backend, [[lex("headache")]])
    assert WORKSPACE_NAMES <= {t.name for t in driver.tools_seen[0]}


async def test_plan_steps_sets_the_plan(docs_backend):
    _, _, s = await run(docs_backend, [
        [call("plan_steps", steps=["find a", "find b"]), lex("headache")],
        [call("plan_steps", steps=["other"]), lex("fever")]])
    assert [st.text for st in s.steps] == ["find a", "find b"]
    assert s.trace.of_type("workspace_dropped")[0].data["reason"] == "plan already set"


async def test_plan_steps_is_capped_and_empty_is_ignored(docs_backend):
    _, _, s = await run(docs_backend, [[call("plan_steps", steps=[]), lex("headache")]])
    assert s.steps == [] and s.trace.of_type("workspace_dropped")
    many = [f"s{i}" for i in range(MAX_STEPS + 4)]
    _, _, s = await run(docs_backend, [[call("plan_steps", steps=["", *many]), lex("headache")]])
    assert [st.text for st in s.steps] == many[:MAX_STEPS]


async def test_note_needs_a_known_source(docs_backend):
    _, _, s = await run(docs_backend, [
        [lex("headache")],
        [call("note", fact="nope", source_key="docs:zzz"),
         call("note", fact="real", source_key="docs:d1"), lex("fever")]])
    assert [(f.text, f.source_key, f.turn) for f in s.notebook] == [("real", "docs:d1", 1)]
    assert s.trace.of_type("workspace_dropped")[0].data["reason"] == "unknown source_key"


async def test_note_cannot_cite_a_document_the_same_turn_finds(docs_backend):
    _, _, s = await run(docs_backend, [[call("note", fact="x", source_key="docs:d1"),
                                        lex("headache")]])
    assert s.notebook == []


async def test_fact_text_is_trimmed_and_empty_dropped(docs_backend):
    _, _, s = await run(docs_backend, [
        [lex("headache")],
        [call("note", fact="  " + "x" * 400 + "  ", source_key="docs:d1"),
         call("note", fact="   ", source_key="docs:d1"), lex("fever")]])
    assert len(s.notebook) == 1 and s.notebook[0].text == "x" * MAX_FACT_CHARS


async def test_bad_step_done_is_ignored(docs_backend):
    _, _, s = await run(docs_backend, [[
        call("plan_steps", steps=["a", "b"]), call("step_done", step=0),
        call("step_done", step=3), call("step_done", step="1"), call("step_done", step=True),
        call("step_done", step=2), lex("headache")]])
    assert [st.done for st in s.steps] == [False, True]
    assert len(s.trace.of_type("workspace_dropped")) == 4


async def test_workspace_calls_do_not_count_as_tool_calls(docs_backend):
    _, res, s = await run(docs_backend, [[
        call("plan_steps", steps=["a"]), call("note", fact="f", source_key="docs:d1"),
        call("step_done", step=1), lex("headache")]])
    assert res.usage.tool_calls == 1
    assert s.history[0].n_calls == 1 and s.history[0].n_steps_done == 1


async def test_turn_with_only_workspace_calls_replans_once(docs_backend):
    driver, res, s = await run(docs_backend, [[call("plan_steps", steps=["a"])], [lex("headache")]])
    assert len(driver.views) >= 2 and driver.views[1].turn == 0
    assert driver.views[1].steps[0].text == "a"
    assert res.usage.tool_calls == 1 and s.history[0].turn == 0


async def test_workspace_only_turn_without_open_step_stops(docs_backend):
    driver, _, _ = await run(docs_backend, [[call("step_done", step=1)], [lex("headache")]])
    assert len(driver.views) == 1


async def test_delegate_mode_ignores_steps(docs_backend):
    driver = ScriptedDriver(delegate_calls=[[lex("headache")]])
    await make(docs_backend, driver).search("q", mode="model")
    assert not {t.name for t in driver.tools_seen[0]} & WORKSPACE_NAMES
