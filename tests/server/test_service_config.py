import textwrap

import pytest

from agentic_search.config import ConfigError
from agentic_search.core.secrets import scrub
from agentic_search.core.types import Budget
from agentic_search.server import AuthConfig, ProfileLimits, check_profiles, load_service
from agentic_search.server.models import BudgetOverride, budget_ceilings, resolve_budget

PROFILE = """
    backends: [{name: notes, type: files, root: docs, glob: "*.txt"}]
    driver: {type: openai_compat, model: local, base_url: "http://localhost:9/v1"}
"""


def write(tmp_path, text):
    (tmp_path / "docs").mkdir(exist_ok=True)
    (tmp_path / "docs" / "a.txt").write_text("aspirin treats headache")
    path = tmp_path / "service.yaml"
    path.write_text(textwrap.dedent(text))
    return path


def test_load_service(tmp_path):
    path = write(tmp_path, f"""
        service:
          auth: {{type: none}}
          default_profile: general
          max_concurrent_searches: 3
        profiles:
          general:{textwrap.indent(textwrap.dedent(PROFILE), "            ")}
          strict:{textwrap.indent(textwrap.dedent(PROFILE), "            ")}
            limits: {{max_budget: {{max_turns: 2}}, allow_include_content: false}}
    """)
    config, profiles = load_service(path)
    assert config.default_profile == "general" and config.max_concurrent_searches == 3
    assert set(profiles) == {"general", "strict"}
    assert profiles["strict"].limits.max_budget == {"max_turns": 2}
    assert not profiles["strict"].limits.allow_include_content
    assert profiles["general"].harness.backends["notes"]


@pytest.mark.parametrize("text, message", [
    ("service: {auth: {type: none}}\n", "profiles"),
    ("profiles: {a: {}}\n", "service"),
    ("service: {auth: {type: none}, bogus: 1}\nprofiles: {a: {}}\n", "service"),
])
def test_load_service_errors(tmp_path, text, message):
    with pytest.raises(ConfigError, match=message):
        load_service(write(tmp_path, text))


def test_profile_limits_validation():
    with pytest.raises(ValueError):
        ProfileLimits(max_budget={"max_wishes": 3})
    with pytest.raises(ValueError):
        ProfileLimits(max_budget={"max_turns": 0})


def test_resolve_budget():
    base = Budget(max_turns=4, max_cost_usd=None)
    got = resolve_budget(base, BudgetOverride(max_turns=9, max_tool_calls=5),
                         {"max_turns": 3, "max_cost_usd": 0.5})
    assert (got.max_turns, got.max_tool_calls, got.max_cost_usd) == (3, 5, 0.5)
    assert resolve_budget(base, None, {}) == base


def test_budget_ceilings_default_to_the_profile_budget():
    base = Budget(max_turns=4, max_tool_calls=32, max_tokens=None, max_cost_usd=None,
                  max_seconds=60.0)
    assert budget_ceilings(base, {}) == {"max_turns": 4, "max_tool_calls": 32, "max_seconds": 60.0}
    assert budget_ceilings(base, {"max_turns": 10, "max_cost_usd": 0.5}) == {
        "max_turns": 10, "max_tool_calls": 32, "max_seconds": 60.0, "max_cost_usd": 0.5}
    lowered = resolve_budget(base, BudgetOverride(max_turns=2, max_tokens=500),
                             budget_ceilings(base, {}))
    raised = resolve_budget(base, BudgetOverride(max_turns=99, max_seconds=600),
                            budget_ceilings(base, {}))
    assert (lowered.max_turns, lowered.max_tokens) == (2, 500)  # lowering is free
    assert (raised.max_turns, raised.max_seconds) == (4, 60.0)  # raising needs max_budget


class TestAuthConfig:
    """Tests for AuthConfig.keys() guard function."""

    def test_keys_parses_comma_separated_with_whitespace(self, monkeypatch):
        """Key parsing strips whitespace and skips empty tokens."""
        auth = AuthConfig(type="api_key", keys_env="MY_KEYS")
        monkeypatch.setenv("MY_KEYS", " key1 , ,key2 ")

        keys = auth.keys()

        assert keys == ["key1", "key2"]

    def test_keys_are_registered_for_scrubbing(self, monkeypatch):
        """Keys are registered and then scrubbed from text."""
        auth = AuthConfig(type="api_key", keys_env="MY_KEYS")
        monkeypatch.setenv("MY_KEYS", "secret_key_1,secret_key_2")

        keys = auth.keys()

        # After registration, scrub should redact the key (must be >= 4 chars to register)
        text_with_key = f"token {keys[0]} in message"
        scrubbed = scrub(text_with_key)
        assert keys[0] not in scrubbed
        assert "***" in scrubbed

    def test_unset_variable_raises_config_error(self, monkeypatch):
        """Unset environment variable raises ConfigError."""
        auth = AuthConfig(type="api_key", keys_env="NONEXISTENT_VAR")
        monkeypatch.delenv("NONEXISTENT_VAR", raising=False)

        with pytest.raises(ConfigError, match="NONEXISTENT_VAR"):
            auth.keys()

    def test_empty_variable_raises_config_error(self, monkeypatch):
        """Empty environment variable raises ConfigError."""
        auth = AuthConfig(type="api_key", keys_env="MY_EMPTY")
        monkeypatch.setenv("MY_EMPTY", "")

        with pytest.raises(ConfigError, match="MY_EMPTY"):
            auth.keys()

    def test_type_none_returns_empty_list(self):
        """With type=none, keys() returns empty list."""
        auth = AuthConfig(type="none")

        keys = auth.keys()

        assert keys == []


class TestCheckProfiles:
    """Tests for check_profiles validation function."""

    def test_no_profiles_raises_error(self):
        """Empty profiles dict raises ConfigError."""
        from agentic_search.server import ServiceConfig

        config = ServiceConfig(auth=AuthConfig(type="none"))

        with pytest.raises(ConfigError, match="needs at least one profile"):
            check_profiles(config, {})

    def test_missing_default_profile_raises_error(self):
        """Default profile not in profiles dict raises ConfigError."""
        from agentic_search.server import ServiceConfig

        config = ServiceConfig(
            auth=AuthConfig(type="none"),
            default_profile="missing"
        )
        profiles = {"exists": object()}

        with pytest.raises(ConfigError, match="default_profile"):
            check_profiles(config, profiles)

    def test_valid_profiles_pass(self):
        """Valid configuration passes check."""
        from agentic_search.server import ServiceConfig

        config = ServiceConfig(
            auth=AuthConfig(type="none"),
            default_profile="general"
        )
        profiles = {"general": object(), "strict": object()}

        # Should not raise
        check_profiles(config, profiles)

    def test_no_default_profile_with_profiles_passes(self):
        """When default_profile is None, check passes with any profiles."""
        from agentic_search.server import ServiceConfig

        config = ServiceConfig(auth=AuthConfig(type="none"))
        profiles = {"any": object()}

        # Should not raise
        check_profiles(config, profiles)
