from agentic_search.core.hooks import SourcePolicy
from agentic_search.core.state import SearchState
from agentic_search.core.types import Budget, Hit, Query, TextPart
from agentic_search.roles.analyzer import Analyzer
from agentic_search.roles.executor import ExecResult
from agentic_search.testing import FailingDecider, KeywordJudge, ScriptedController, call


def setup_state():
    s = SearchState(question=Query.of("fever"), manifests={}, budget=Budget())
    hits = [Hit(doc_id="1", source="s", content=[TextPart(text="high fever")]),
            Hit(doc_id="2", source="s", content=[TextPart(text="castles")])]
    c = call("lexical_search", source="s", text="fever")
    new = s.pool.add(hits, turn=0, call_id=c.id)
    return s, [c], ExecResult(new_keys=new, hits_per_call={c.id: new})


async def test_judges_new_and_builds_digest():
    s, calls, res = setup_state()
    a = Analyzer(KeywordJudge(["fever"]))
    out = await a.analyze(s, calls, res, digest_model_id="driver")
    assert out.n_new == 2 and out.n_new_relevant == 1
    assert s.pool["s:1"].judged and s.pool["s:1"].p_relevant == 1.0
    assert "1 judged relevant" in out.digest
    assert "New relevant:" in out.digest and "s:1 p=1.00: high fever" in out.digest
    assert "Judged not relevant" in out.digest
    assert "2 hits, 2 new, 1 relevant" in out.digest
    assert len(s.trace.of_type("judge")) == 1


async def test_no_judge_leaves_unjudged():
    s, calls, res = setup_state()
    out = await Analyzer(None).analyze(s, calls, res, digest_model_id="driver")
    assert out.n_new_relevant is None
    assert s.pool["s:1"].unjudged_reason == "no judge configured"
    assert "New candidates (not judged)" in out.digest


async def test_failing_judge_fails_open():
    s, calls, res = setup_state()
    out = await Analyzer(FailingDecider()).analyze(s, calls, res, digest_model_id="d")
    assert out.n_new_relevant is None
    assert s.pool["s:1"].unjudged_reason.startswith("judge failed")
    assert s.trace.of_type("judge_error")[0].data["error"].startswith("RuntimeError")


async def test_decider_without_judge_support():
    s, calls, res = setup_state()
    a = Analyzer(ScriptedController([]))
    await a.analyze(s, calls, res, digest_model_id="d")
    assert "does not judge" in s.pool["s:1"].unjudged_reason


async def test_batching_and_policy():
    s, calls, res = setup_state()
    a = Analyzer(KeywordJudge(["fever"]), batch_size=1, policy=SourcePolicy({"s": {"judge-only"}}))
    out = await a.analyze(s, calls, res, digest_model_id="driver")
    assert len(s.trace.of_type("judge")) == 2
    # KeywordJudge is not allowed to see source s, so it sees empty content
    assert out.n_new_relevant == 0
    assert "withheld" in out.digest
    assert "no keyword" not in out.digest  # the judge rationale is withheld with the body


async def test_error_lines_in_digest():
    from agentic_search.core.types import ToolError
    s, calls, res = setup_state()
    bad = call("lexical_search", source="x", text="y")
    res.errors.append(ToolError(call_id=bad.id, kind="validation", message="unknown source 'x'"))
    out = await Analyzer(None).analyze(s, [*calls, bad], res, digest_model_id="d")
    assert "ERROR [validation] unknown source 'x'" in out.digest


async def test_hook_blocking_fails_open():
    from agentic_search.core.hooks import Hooks

    class BlockingHook(Hooks):
        async def before_model_call(self, model_id: str, payload):
            if model_id == "keyword-judge":
                raise PermissionError("judge access denied")
            return payload

    s, calls, res = setup_state()
    a = Analyzer(KeywordJudge(["fever"]), hooks=BlockingHook())
    # Should not raise; analyze must fail open
    out = await a.analyze(s, calls, res, digest_model_id="driver")
    assert out.n_new_relevant is None
    assert all(s.pool[k].unjudged_reason.startswith("judge failed") for k in res.new_keys)
    assert len(s.trace.of_type("judge_error")) == 1
    assert "PermissionError" in s.trace.of_type("judge_error")[0].data["error"]


async def test_judge_question_passes_through_hook():
    from agentic_search.core.hooks import Hooks
    from agentic_search.models.base import JudgeRequest

    class ReplaceQuestion(Hooks):
        async def before_model_call(self, model_id, payload):
            assert isinstance(payload, JudgeRequest)
            return payload.model_copy(update={"question": Query.of("REDACTED")})

    class RecordingJudge(KeywordJudge):
        def __init__(self):
            super().__init__(["fever"])
            self.questions = []

        async def judge(self, question, hits):
            self.questions.append(question.as_text())
            return await super().judge(question, hits)

    s, calls, res = setup_state()
    judge = RecordingJudge()
    out = await Analyzer(judge, hooks=ReplaceQuestion()).analyze(s, calls, res, digest_model_id="d")
    assert judge.questions == ["REDACTED"] and out.n_new_relevant == 1


def fetch_state(text: str = "high  fever\nand chills", *, source: str = "s"):
    """A turn whose only call fetched a document already in the pool (0 new)."""
    s = SearchState(question=Query.of("fever"), manifests={}, budget=Budget())
    hit = Hit(doc_id="1", source=source, content=[TextPart(text=text)])
    s.pool.add([hit], turn=0, call_id="earlier")
    c = call("fetch", source=source, doc_id="1")
    return s, [c], ExecResult(new_keys=[], hits_per_call={c.id: [hit.key]})


def test_fetched_known_document_is_shown_in_full_when_steps_on():
    s, calls, res = fetch_state()
    digest = Analyzer(None, show_opened=True).render_digest(s, calls, res, "driver")
    assert "1 hits, 0 new" in digest
    assert "Opened documents:\n- s:1: high fever and chills" in digest


def test_opened_section_absent_when_steps_off():
    s, calls, res = fetch_state()
    digest = Analyzer(None).render_digest(s, calls, res, "driver")
    assert "Opened documents" not in digest


def test_opened_section_ignores_non_fetch_calls():
    s, calls, res = setup_state()
    digest = Analyzer(None, show_opened=True).render_digest(s, calls, res, "driver")
    assert "Opened documents" not in digest


def test_opened_text_respects_source_policy():
    s, calls, res = fetch_state(source="secret")
    a = Analyzer(None, show_opened=True, policy=SourcePolicy({"secret": {"judge-only"}}))
    digest = a.render_digest(s, calls, res, "driver")
    assert "- secret:1: (content withheld by source policy)" in digest
    assert "chills" not in digest


def test_opened_text_is_cut_to_fetch_chars():
    s, calls, res = fetch_state("word " * 100)
    digest = Analyzer(None, show_opened=True, fetch_chars=20).render_digest(s, calls, res, "d")
    line = next(ln for ln in digest.splitlines() if ln.startswith("- s:1:"))
    assert line == "- s:1: word word word word…"


async def test_judge_sees_noted_facts():
    """Facts the planner noted (e.g. the anchor product's color) reach the judge with the question,
    so it can grade "same color as X" without having X in the batch."""
    from agentic_search.models.base import Fact

    seen: list[str] = []

    class Recording(KeywordJudge):
        async def judge(self, question, hits):
            seen.append(question.as_text())
            return await super().judge(question, hits)

    s, calls, res = setup_state()
    s.notebook.append(Fact(text="The anchor bed has color : gray.", source_key="s:9", turn=0))
    await Analyzer(Recording(["fever"])).analyze(s, calls, res, digest_model_id="d")
    assert seen and seen[0].startswith("fever")
    assert "The anchor bed has color : gray." in seen[0]


async def test_judge_question_unchanged_without_facts():
    seen: list[str] = []

    class Recording(KeywordJudge):
        async def judge(self, question, hits):
            seen.append(question.as_text())
            return await super().judge(question, hits)

    s, calls, res = setup_state()
    await Analyzer(Recording(["fever"])).analyze(s, calls, res, digest_model_id="d")
    assert seen == ["fever"]
