import json
import shutil
from pathlib import Path

from demo.ignite.check_data import check
from demo.ignite.corpus import DATA_DIR, load_documents, load_questions, load_records


def _copy_data(tmp_path: Path) -> Path:
    data = tmp_path / "data"
    shutil.copytree(DATA_DIR, data)
    return data


def _source_of(data: Path, doc_id: str) -> Path:
    for path in sorted(data.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip() and json.loads(line)["id"] == doc_id:
                return path
    raise KeyError(doc_id)


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _edit(data: Path, doc_id: str, *, metadata: dict | None = None, text: str | None = None,
          drop: list[str] = ()) -> None:
    path = _source_of(data, doc_id)
    rows = _rows(path)
    for r in rows:
        if r["id"] == doc_id:
            r["metadata"].update(metadata or {})
            for key in drop:
                r["metadata"].pop(key)
            if text is not None:
                r["text"] = text
    _write(path, rows)


def _remove(data: Path, doc_id: str) -> None:
    path = _source_of(data, doc_id)
    _write(path, [r for r in _rows(path) if r["id"] != doc_id])


def _append(data: Path, source: str, row: dict) -> None:
    path = data / f"{source}.jsonl"
    _write(path, _rows(path) + [row])


def _row(data: Path, doc_id: str) -> dict:
    return next(r for r in _rows(_source_of(data, doc_id)) if r["id"] == doc_id)


def test_real_data_passes():
    assert check() == []


def test_documents_carry_metadata_and_title():
    docs = {d.doc_id: d for d in load_documents()["deploys"]}
    assert docs["DEP-0913-07"].metadata["pr"] == 1187
    assert docs["DEP-0913-07"].metadata["title"]


def test_missing_gold_id_fails(tmp_path):
    data = _copy_data(tmp_path)
    _remove(data, "PR-1203")
    assert any("PR-1203" in p for p in check(data))


def test_duplicate_id_across_sources_fails(tmp_path):
    data = _copy_data(tmp_path)
    _append(data, "runbooks", _row(data, "INC-4821"))
    assert any("duplicate" in p and "INC-4821" in p for p in check(data))


def test_broken_deploy_link_fails(tmp_path):
    data = _copy_data(tmp_path)
    _edit(data, "DEP-0913-07", metadata={"pr": 1190})
    assert any("DEP-0913-07" in p for p in check(data))


def test_incident_leaking_root_cause_fails(tmp_path):
    data = _copy_data(tmp_path)
    _edit(data, "INC-4821", text=_row(data, "INC-4821")["text"] + " Suspect the reranker.")
    assert any("INC-4821" in p for p in check(data))


def test_dashboard_week_must_cover_fix_date(tmp_path):
    data = _copy_data(tmp_path)
    _edit(data, "DASH-W38-search", metadata={"week_start": "2026-09-21"})
    assert any("DASH-W38-search" in p for p in check(data))


def test_second_september_search_latency_incident_fails(tmp_path):
    data = _copy_data(tmp_path)
    _append(data, "incidents", {
        "id": "INC-9999", "title": "search-api slow", "text": "Elevated latency on search-api.",
        "metadata": {"opened": "2026-09-20", "service": "search-api", "severity": "SEV3",
                     "status": "resolved"}})
    assert any("only" in p for p in check(data))


def test_missing_required_metadata_fails(tmp_path):
    data = _copy_data(tmp_path)
    _edit(data, "DEP-0913-07", drop=["service"])
    assert any("service" in p for p in check(data))


def test_source_sizes_match_spec():
    sizes = {s: len(rows) for s, rows in load_records().items()}
    assert 25 <= sizes["incidents"] <= 35
    assert 50 <= sizes["deploys"] <= 70
    assert 20 <= sizes["pull_requests"] <= 30
    assert 10 <= sizes["dashboards"] <= 14
    assert 8 <= sizes["runbooks"] <= 12


def test_question_mix_matches_spec():
    qs = [q for q in load_questions().values() if q.id != "hero"]
    assert 8 <= len(qs) <= 12
    assert 3 <= sum(q.kind == "single_hop" for q in qs) <= 4
    assert 5 <= sum(q.kind == "multi_hop" for q in qs) <= 8
    assert all(len(q.gold) == 1 for q in qs if q.kind == "single_hop")
    assert all(2 <= len(q.gold) <= 4 for q in qs if q.kind == "multi_hop")


def test_check_reports_size_and_mix(tmp_path):
    data = _copy_data(tmp_path)
    (data / "runbooks.jsonl").write_text("")
    problems = check(data)
    assert any("runbooks" in p and "size" in p for p in problems)


def test_fix_pr_must_name_the_causing_pr(tmp_path):
    data = _copy_data(tmp_path)
    _edit(data, "PR-1203", text=_row(data, "PR-1203")["text"].replace("1187", "the rollout"))
    assert any("PR-1203" in p and "1187" in p for p in check(data))
