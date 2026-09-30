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
