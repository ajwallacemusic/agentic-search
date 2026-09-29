from agentic_search.core.hooks import Hooks, SourcePolicy
from agentic_search.core.state import CandidatePool, SearchState, Trace, Usage
from agentic_search.core.types import Budget, Hit, ModelUsage, Query, TextPart


def hit(doc_id, source="s", text="t"):
    return Hit(doc_id=doc_id, source=source, content=[TextPart(text=text)], metadata={"m": 1})


def test_pool_add_dedupes_and_tracks_provenance():
    pool = CandidatePool()
    assert pool.add([hit("a"), hit("b")], turn=0, call_id="c1") == ["s:a", "s:b"]
    assert pool.add([hit("b"), hit("c")], turn=1, call_id="c2") == ["s:c"]
    assert len(pool) == 3 and "s:b" in pool
    prov = pool["s:b"].hit.provenance
    assert [(p.call_id, p.rank) for p in prov] == [("c1", 1), ("c2", 0)]
    assert pool["s:c"].first_turn == 1


def test_pool_rrf_and_blended_scores():
    pool = CandidatePool(rrf_k=0)
    pool.add([hit("a"), hit("b")], turn=0, call_id="c1")
    pool.add([hit("b")], turn=0, call_id="c2")
    assert pool.rrf("s:b") == 1 / 2 + 1 / 1
    scores = pool.scores(judge_weight=0.9)
    assert scores["s:b"] == 1.0  # unjudged: normalized rrf
    pool["s:a"].judged, pool["s:a"].p_relevant = True, 1.0
    pool["s:b"].judged, pool["s:b"].p_relevant = True, 0.0
    ranked = pool.ranked(judge_weight=0.9)
    assert [c.hit.doc_id for c, _ in ranked] == ["a", "b"]


def test_trace_listener_and_json():
    seen = []
    t = Trace()
    t.set_listener(seen.append)
    ev = t.add("plan", 0, duration_ms=1.5, n_calls=2)
    assert seen == [ev] and ev.data == {"n_calls": 2}
    assert t.of_type("plan") == [ev]
    assert '"plan"' in t.model_dump_json()


def test_usage_add_model():
    u = Usage()
    u.add_model(ModelUsage(input_tokens=3, output_tokens=4, cost_usd=0.1))
    assert u.total_tokens == 7 and u.cost_usd == 0.1


def test_search_state_defaults():
    s = SearchState(question=Query.of("q"), manifests={}, budget=Budget())
    assert s.turn == 0 and len(s.pool) == 0 and s.elapsed() >= 0
    assert s.digest.startswith("No searches")


async def test_default_hooks_pass_through():
    h = Hooks()
    assert await h.before_model_call("m", {"x": 1}) == {"x": 1}
    assert h.on_trace_event(None) is None


def test_source_policy_redacts():
    p = SourcePolicy({"phi": {"local-model"}})
    assert p.allows("public", "any") and p.allows("phi", "local-model")
    assert not p.allows("phi", "cloud-model")
    out = p.redact([hit("a", "phi"), hit("b", "public")], "cloud-model")
    assert out[0].content == [] and out[0].metadata["_redacted"] is True
    assert out[1].content != []
