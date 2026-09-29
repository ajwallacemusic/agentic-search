import pytest

from agentic_search.config import ConfigError, build_harness, load_harness, register, resolve_env
from agentic_search.models.cross_encoder import CrossEncoderJudge
from agentic_search.models.llm_judge import LLMJudge


def test_resolve_env(monkeypatch):
    monkeypatch.setenv("MY_KEY", "secret")
    assert resolve_env({"a": {"api_key_env": "MY_KEY"}, "b": [1]}) == {"a": {"api_key": "secret"}, "b": [1]}
    monkeypatch.delenv("MY_KEY")
    with pytest.raises(ConfigError, match="MY_KEY"):
        resolve_env({"api_key_env": "MY_KEY"})


def cfg(tmp_path):
    (tmp_path / "notes").mkdir(exist_ok=True)
    (tmp_path / "notes" / "a.md").write_text("aspirin headache")
    return {
        "embedders": [{"type": "hash", "id": "h", "dim": 32}],
        "backends": [{"name": "notes", "type": "files", "root": "notes", "embedder": "h",
                      "allowed_models": ["openai:local"]}],
        "driver": {"type": "openai_compat", "model": "local", "base_url": "http://localhost:8000/v1",
                   "max_calls_per_turn": 4},
        "analyzer": {"type": "cross_encoder", "model": "BAAI/bge-reranker-v2-m3"},
        "controller": {"type": "llm_judge", "client": {"type": "openai_compat", "model": "local",
                                                       "base_url": "http://localhost:8000/v1"}},
        "mode": "retrieval",
        "budget": {"max_turns": 2},
        "settings": {"judge_weight": 0.8},
    }


async def test_build_harness(tmp_path):
    h = build_harness(cfg(tmp_path), base_dir=tmp_path)
    assert h.driver.id == "openai:local" and h.driver.max_calls == 4
    assert isinstance(h.analyzer_decider, CrossEncoderJudge)
    assert isinstance(h.controller_decider, LLMJudge)
    assert h.mode == "retrieval" and h.budget.max_turns == 2 and h.settings.judge_weight == 0.8
    assert h.policy.allows("notes", "openai:local") and not h.policy.allows("notes", "other")
    manifests = await h.setup()
    assert manifests["notes"].resolve_collection(None).field("embedding").embedder_id == "h"


def test_errors(tmp_path):
    bad = cfg(tmp_path)
    bad["backends"][0]["type"] = "nope"
    with pytest.raises(ConfigError, match="unknown backend type 'nope'"):
        build_harness(bad, base_dir=tmp_path)
    no_driver = cfg(tmp_path)
    del no_driver["driver"]
    with pytest.raises(ConfigError, match="driver"):
        build_harness(no_driver, base_dir=tmp_path)
    missing_emb = cfg(tmp_path)
    missing_emb["backends"][0]["embedder"] = "zzz"
    with pytest.raises(ConfigError, match="zzz"):
        build_harness(missing_emb, base_dir=tmp_path)


async def test_load_yaml_with_annotations_file(tmp_path):
    import yaml
    c = cfg(tmp_path)
    c["annotations"] = "ann.yaml"
    (tmp_path / "ann.yaml").write_text(yaml.safe_dump({"notes": {"description": "Clinic notes"}}))
    (tmp_path / "h.yaml").write_text(yaml.safe_dump(c))
    h = load_harness(tmp_path / "h.yaml")
    await h.setup()
    assert h.manifests["notes"].description == "Clinic notes"


def test_register_custom_type(tmp_path):
    made = []
    register("embedder", "custom-test", lambda c, ctx: made.append(c) or __import__(
        "agentic_search.embedders.local", fromlist=["HashEmbedder"]).HashEmbedder(id="custom"))
    c = cfg(tmp_path)
    c["embedders"].append({"type": "custom-test", "x": 1})
    build_harness(c, base_dir=tmp_path)
    assert made == [{"type": "custom-test", "x": 1}]
    with pytest.raises(ValueError):
        register("gizmo", "x", lambda c, ctx: None)
