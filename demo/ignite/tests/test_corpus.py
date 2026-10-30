import csv
import json

import pytest
import yaml

from demo.ignite.corpus import QUESTIONS_PATH, corpus_of, load_demo
from demo.ignite.public.datasets import para_id
from demo.ignite.tests.test_public_datasets import _musique
from demo.ignite.tests.test_wands_match import _folder


def _questions(tmp_path, picks):
    path = tmp_path / "questions.yaml"
    path.write_text(yaml.safe_dump([{"id": i, "set": s, "ref": r} for i, s, r in picks]))
    return path


def _wands(tmp_path):
    """The WANDS match product fixture plus query.csv and label.csv in the real shapes."""
    folder = _folder(tmp_path)
    with (folder / "query.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerows([["query_id", "query", "query_class"], ["0", "rustic coffee table", "x"],
                     ["1", "gray dresser", "y"]])
    with (folder / "label.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerows([["id", "query_id", "product_id", "label"], ["0", "0", "2", "Exact"],
                     ["1", "0", "3", "Partial"], ["2", "0", "4", "Irrelevant"],
                     ["3", "1", "13", "Exact"]])
    rules = tmp_path / "rules.yaml"
    rules.write_text(yaml.safe_dump([{"id": "match-01", "anchor_id": "1", "attribute": "style",
                                      "targets": ["Coffee Tables"],
                                      "text": "I have the winkleman teak end table."}]))
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps({"match-01": ["wands-2", "wands-3"]}))
    return folder, rules, labels


def _all(tmp_path, picks):
    path, _ = _musique(tmp_path)
    folder, rules, labels = _wands(tmp_path)
    return load_demo(musique_path=path, wands_folder=folder, match_path=rules, labels_path=labels,
                     questions_path=_questions(tmp_path, picks))


def test_questions_come_from_the_musique_sample(tmp_path):
    path, rows = _musique(tmp_path)
    data = load_demo(corpora={"musique"}, musique_path=path, questions_path=_questions(
        tmp_path, [("hero", "musique", "4hop1__0"), ("two", "musique", "2hop__1")]))["musique"]
    assert list(data.documents) == ["wiki"] and data.descriptions["wiki"]
    hero = data.questions["hero"]
    assert hero.text == "question 4hop1__0?" and hero.kind == "4hop"
    assert (hero.ref, hero.set, hero.corpus) == ("4hop1__0", "musique", "musique")
    row = next(r for r in rows if r["id"] == "4hop1__0")
    assert set(hero.gold) == {para_id(p["title"], p["paragraph_text"]) for p in row["paragraphs"]
                              if p["is_supporting"]}
    assert set(hero.gold) <= {d.doc_id for d in data.documents["wiki"]}
    assert list(data.questions) == ["hero", "two"]


def test_question_outside_the_sample_fails(tmp_path):
    path, _ = _musique(tmp_path)
    with pytest.raises(ValueError, match="4hop1__99"):
        load_demo(corpora={"musique"}, musique_path=path,
                  questions_path=_questions(tmp_path, [("hero", "musique", "4hop1__99")]))


def test_wands_question_gold_is_exact_labels(tmp_path):
    data = _all(tmp_path, [("w", "wands", "0")])["wands"]
    q = data.questions["w"]
    assert (q.text, q.corpus, q.gold) == ("rustic coffee table", "wands", ["wands-2"])
    assert list(data.documents) == ["products"] and len(data.documents["products"]) == 18


def test_match_question_gold_is_the_rule_labels(tmp_path):
    q = _all(tmp_path, [("m", "wands-match", "match-01")])["wands"].questions["m"]
    assert (q.corpus, q.kind, q.gold) == ("wands", "style", ["wands-2", "wands-3"])
    assert q.text == "I have the winkleman teak end table."


def test_only_requested_corpora_load(tmp_path):
    path, _ = _musique(tmp_path)
    out = load_demo(corpora={"musique"}, musique_path=path, wands_folder=tmp_path / "missing",
                    questions_path=_questions(tmp_path, [("hero", "musique", "4hop1__0"),
                                                         ("w", "wands", "0")]))
    assert list(out) == ["musique"] and list(out["musique"].questions) == ["hero"]


def test_duplicate_question_id_fails(tmp_path):
    with pytest.raises(ValueError, match="'hero'"):
        _all(tmp_path, [("hero", "musique", "4hop1__0"), ("hero", "wands-match", "match-01")])


def test_corpus_of_reads_the_questions_file(tmp_path):
    path = _questions(tmp_path, [("hero", "musique", "x"), ("m", "wands-match", "match-01")])
    assert corpus_of("m", path) == "wands" and corpus_of("hero", path) == "musique"
    with pytest.raises(KeyError):
        corpus_of("nope", path)


def test_shipped_questions_start_with_a_4hop_hero():
    picks = yaml.safe_load(QUESTIONS_PATH.read_text())
    assert picks[0]["id"] == "hero" and picks[0]["set"] == "musique"
    assert picks[0]["ref"].startswith("4hop")
    assert len({p["id"] for p in picks}) == len(picks)
