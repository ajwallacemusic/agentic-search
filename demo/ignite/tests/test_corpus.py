import pytest
import yaml

from demo.ignite.corpus import QUESTIONS_PATH, load_demo
from demo.ignite.public.datasets import para_id
from demo.ignite.tests.test_public_datasets import _musique


def _questions(tmp_path, picks):
    path = tmp_path / "questions.yaml"
    path.write_text(yaml.safe_dump([{"id": i, "musique_id": m} for i, m in picks]))
    return path


def test_questions_come_from_the_musique_sample(tmp_path):
    path, rows = _musique(tmp_path)
    data = load_demo(path, _questions(tmp_path, [("hero", "4hop1__0"), ("two", "2hop__1")]))
    assert list(data.documents) == ["wiki"] and data.descriptions["wiki"]
    hero = data.questions["hero"]
    assert hero.text == "question 4hop1__0?" and hero.kind == "4hop"
    assert hero.musique_id == "4hop1__0"
    row = next(r for r in rows if r["id"] == "4hop1__0")
    assert set(hero.gold) == {para_id(p["title"], p["paragraph_text"]) for p in row["paragraphs"]
                              if p["is_supporting"]}
    assert set(hero.gold) <= {d.doc_id for d in data.documents["wiki"]}
    assert list(data.questions) == ["hero", "two"]


def test_question_outside_the_sample_fails(tmp_path):
    path, _ = _musique(tmp_path)
    with pytest.raises(ValueError, match="4hop1__99"):
        load_demo(path, _questions(tmp_path, [("hero", "4hop1__99")]))


def test_shipped_questions_start_with_a_4hop_hero():
    picks = yaml.safe_load(QUESTIONS_PATH.read_text())
    assert picks[0]["id"] == "hero" and picks[0]["musique_id"].startswith("4hop")
    assert len({p["id"] for p in picks}) == len(picks)
