import csv

import pytest
import yaml

from demo.ignite.public.wands_multi import (
    MULTI_PATH,
    MultiQuestion,
    question_text,
    sample_wands_multi,
    select_questions,
)


def _exact(qid: str, n: int) -> list[str]:
    return sorted(f"wands-{qid}{i}" for i in range(n))


@pytest.fixture
def toy():
    classes = {"1": "Chairs", "2": "Chairs", "3": "Rugs", "4": "Lamps", "5": "Beds",
               "6": "Desks", "7": "Mirrors", "8": "Sofas", "9": "", "10": "Licensed Products",
               "11": "Tables", "12": "Benches", "13": "Clocks", "14": "Rugs"}
    queries = {q: (f"query {q}", c) for q, c in classes.items()}
    exact = {q: _exact(q, 2 if q == "12" else 4) for q in classes}
    return queries, exact


def test_question_text_template():
    assert question_text(["teak chair", "patio umbrella", "outdoor rug"]) == \
        "I'm shopping for a few things: teak chair; patio umbrella; outdoor rug."


def test_select_rules(toy):
    qs = select_questions(*toy, n=3, seed=7)
    assert len(qs) == 3
    assert [q.id for q in qs] == ["m01", "m02", "m03"]
    used = [p for q in qs for p in q.parts]
    assert len(used) == len(set(used))
    for q in qs:
        classes = [toy[0][p][1] for p in q.parts]
        assert len(set(classes)) == len(classes)
        assert "" not in classes and "Licensed Products" not in classes
        assert 3 <= len(q.parts) <= 4
        assert all(len(toy[1][p]) >= 3 for p in q.parts)
        assert q.gold == {p: toy[1][p] for p in q.parts}
        assert q.text == question_text([toy[0][p][0] for p in q.parts])


def test_select_is_deterministic(toy):
    assert select_questions(*toy, n=3, seed=7) == select_questions(*toy, n=3, seed=7)


def _write_tsv(path, header, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(header)
        w.writerows(rows)


def test_sample_loader_unions_graded_qrels(tmp_path):
    _write_tsv(tmp_path / "product.csv",
               ["product_id", "product_name", "product_class", "category hierarchy",
                "product_description", "product_features", "rating_count", "average_rating",
                "review_count"],
               [[p, f"item {p}", "X", "", "", "", "", "", ""] for p in ("P", "Q", "R")])
    _write_tsv(tmp_path / "label.csv", ["id", "query_id", "product_id", "label"],
               [["0", "A", "P", "Exact"], ["1", "A", "Q", "Exact"], ["2", "B", "P", "Partial"],
                ["3", "B", "R", "Exact"], ["4", "B", "Q", "Irrelevant"]])
    (tmp_path / "multi.yaml").write_text(yaml.safe_dump([
        {"id": "m01", "parts": ["A", "B"], "text": "t",
         "gold": {"A": ["wands-P", "wands-Q"], "B": ["wands-R"]}}]))
    s = sample_wands_multi(tmp_path, tmp_path / "multi.yaml")
    assert s.name == "wands-multi" and s.source == "products"
    assert s.qrels["m01"] == {"wands-P": 2, "wands-Q": 2, "wands-R": 2}
    assert s.parts["m01"] == {"A": ["wands-P", "wands-Q"], "B": ["wands-R"]}
    assert s.groups == {"all": ["m01"]}
    assert s.queries == {"m01": "t"}
    assert len(s.documents) == 3


def test_sample_loader_keeps_first_n(tmp_path):
    _write_tsv(tmp_path / "product.csv", ["product_id", "product_name", "product_class"], [])
    _write_tsv(tmp_path / "label.csv", ["id", "query_id", "product_id", "label"], [])
    (tmp_path / "multi.yaml").write_text(yaml.safe_dump([
        {"id": f"m0{i}", "parts": [], "text": "t", "gold": {}} for i in (1, 2, 3)]))
    assert sample_wands_multi(tmp_path, tmp_path / "multi.yaml", n=2).groups == \
        {"all": ["m01", "m02"]}


def test_frozen_yaml_matches_rules():
    if not MULTI_PATH.exists():
        pytest.skip("wands_multi.yaml not generated yet")
    qs = [MultiQuestion(**q) for q in yaml.safe_load(MULTI_PATH.read_text())]
    assert [q.id for q in qs] == [f"m{i:02d}" for i in range(1, 21)]
    used = [p for q in qs for p in q.parts]
    assert len(used) == len(set(used))
    assert all(3 <= len(q.parts) <= 4 and all(len(q.gold[p]) >= 3 for p in q.parts) for q in qs)
