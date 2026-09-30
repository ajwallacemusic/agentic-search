from pathlib import Path

from agentic_search.server.cli import main
from agentic_search.server.schema import FIXTURES_FILE, SCHEMA_FILE, event_schema, render_files

ROOT = Path(__file__).resolve().parents[2]


async def test_committed_contract_matches_models():
    """If this fails, run `uv run agentic-search export-schema --out schema` and commit."""
    rendered = await render_files()
    for name in (SCHEMA_FILE, FIXTURES_FILE):
        assert (ROOT / "schema" / name).read_text() == rendered[name], name


def test_schema_covers_every_event_type():
    schema = event_schema()
    mapping = schema["discriminator"]["mapping"]
    assert set(mapping) == {"search_started", "phase_started", "phase_finished",
                            "tool_call_started", "tool_call_finished", "results_updated",
                            "usage_updated", "search_finished", "search_failed"}


def test_cli_export_schema(tmp_path, capsys):
    assert main(["export-schema", "--out", str(tmp_path / "out")]) == 0
    printed = capsys.readouterr().out.split()
    assert len(printed) == 2 and all((tmp_path / "out").glob("*.json"))


def test_every_committed_fixture_frame_validates_against_committed_schema():
    """The fixtures are what goes over the wire; the schema must accept every frame."""
    import json

    from jsonschema import Draft202012Validator

    schema = json.loads((ROOT / "schema" / SCHEMA_FILE).read_text())
    validator = Draft202012Validator(schema)
    fixtures = json.loads((ROOT / "schema" / FIXTURES_FILE).read_text())
    errors = [(name, frame["id"], e.message)
              for name, frames in fixtures.items() for frame in frames
              for e in validator.iter_errors(frame["data"])]
    assert errors == []


def test_lean_fields_are_optional_and_described():
    defs = event_schema()["$defs"]
    assert "trace" not in defs["SearchResult"]["required"]
    assert "include_trace" in defs["SearchResult"]["properties"]["trace"]["description"]
    assert "content" not in defs["Hit"].get("required", [])
    assert "include_content" in defs["Hit"]["properties"]["content"]["description"]


def test_normalise_rounds_floats_to_six_decimals():
    from agentic_search.server.schema import _normalise

    got = _normalise({"score": 0.123456789, "hits": [{"p": 1.0000004}], "n": 3, "at_ms": 5.5})
    assert got == {"score": 0.123457, "hits": [{"p": 1.0}], "n": 3, "at_ms": 0.0}


def test_image_data_null_in_questions_is_described():
    data = event_schema()["$defs"]["ImagePart"]["properties"]["data"]
    assert "question" in data["description"] and "null" in data["description"]


def test_cli_serve_reports_config_errors(tmp_path, capsys):
    bad = tmp_path / "service.yaml"
    bad.write_text("- not\n- a mapping\n")
    assert main(["serve", "--config", str(bad)]) == 2
    err = capsys.readouterr().err
    assert err.startswith("agentic-search: ") and "mapping" in err


def test_cli_serve_passes_graceful_timeout(tmp_path, monkeypatch):
    import uvicorn

    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "a.txt").write_text("aspirin")
    cfg = tmp_path / "service.yaml"
    cfg.write_text("service: {auth: {type: none}}\n"
                   "profiles:\n  p:\n    backends: [{name: n, type: files, root: docs}]\n"
                   "    driver: {type: openai_compat, model: m, base_url: 'http://localhost:9/v1'}\n")
    seen = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: seen.append(kw))
    assert main(["serve", "--config", str(cfg)]) == 0
    assert main(["serve", "--config", str(cfg), "--graceful-timeout", "2.5"]) == 0
    assert [kw["timeout_graceful_shutdown"] for kw in seen] == [10.0, 2.5]
