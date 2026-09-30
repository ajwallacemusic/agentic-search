import textwrap

import pytest

from agentic_search.config import ConfigError
from agentic_search.core.types import Budget
from agentic_search.server import ProfileLimits, load_service
from agentic_search.server.models import BudgetOverride, resolve_budget

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
