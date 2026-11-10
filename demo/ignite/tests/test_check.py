from demo.ignite.check import MatchProof, MusiqueProof, RatedProof, load_proofs, wands_labels
from demo.ignite.corpus import Question
from demo.ignite.public.wands_match import MatchRule, load_products
from demo.ignite.tests.helpers import demo_data
from demo.ignite.tests.test_public_datasets import _wands
from demo.ignite.tests.test_wands_match import _folder


def _q(**kw):
    base = {"id": "q", "text": "t", "set": "wands", "corpus": "wands", "kind": "query",
            "gold": [], "ref": "0"}
    return Question(**{**base, **kw})


def test_musique_right_is_a_listed_paragraph_and_marks_the_answer():
    hero = demo_data().questions["hero"]
    p = MusiqueProof(hero)
    assert f"{len(hero.gold)} paragraphs" in p.rule() and hero.answers[0] in p.rule()
    assert p.check(hero.gold[0])["verdict"] == "right"
    wrong = p.check("not-a-gold-id")
    assert wrong["verdict"] == "wrong" and wrong["marks"] == hero.answers[:1]


def test_wands_labels_keep_every_grade_and_the_higher_of_two(tmp_path):
    folder = _wands(tmp_path)
    with (folder / "label.csv").open("a") as f:
        f.write("6\t0\t3\tExact\n7\t0\t0\tIrrelevant\n")  # 0/3: Partial then Exact; 0/0 stays Exact
    labels = wands_labels(folder, {"0"})
    assert labels == {"0": {"wands-0": "Exact", "wands-3": "Exact", "wands-1": "Irrelevant"}}


def test_rated_proof_names_the_rater_label():
    p = RatedProof(_q(text="wood bed", gold=["wands-0"]),
                   {"wands-0": "Exact", "wands-3": "Partial", "wands-1": "Irrelevant"})
    assert "“wood bed”" in p.rule() and "1 products are Exact" in p.rule()
    assert [p.check(d)["verdict"] for d in ("wands-0", "wands-3", "wands-1", "wands-9")] == \
        ["right", "partial", "wrong", "unrated"]


def _match(tmp_path):
    rule = MatchRule(id="m", anchor_id="1", attribute="style",
                     targets=["Coffee Tables", "TV Stands"])
    return MatchProof(_q(set="wands-match", kind="style", ref="m"), rule,
                      load_products(_folder(tmp_path)))


def test_match_proof_shows_type_and_value(tmp_path):
    p = _match(tmp_path)
    assert "Coffee Tables or TV Stands" in p.rule() and "“rustic”" in p.rule()
    assert "winkleman teak end table (product 1)" in p.rule()
    right = p.check("wands-2")
    assert right["verdict"] == "right" and right["target"] == "Coffee Tables"
    assert "style : rustic" in right["marks"]
    wrong_value = p.check("wands-4")
    assert wrong_value["verdict"] == "wrong" and "✗ (need rustic)" in wrong_value["why"]
    wrong_type = p.check("wands-9")
    assert wrong_type["verdict"] == "wrong" and "End Tables ✗" in wrong_type["why"]
    assert "already own" in p.check("wands-1")["why"]
    assert p.check("wands-999")["verdict"] == "unrated"


def test_match_coverage_counts_right_results_per_target(tmp_path):
    p = _match(tmp_path)
    assert p.coverage(["wands-2", "wands-3", "wands-4", "wands-9"]) == \
        [["Coffee Tables", 2], ["TV Stands", 0]]


def test_load_proofs_for_musique_reads_no_wands_files():
    proofs = load_proofs(demo_data().questions, wands_folder=None)
    assert {type(p) for p in proofs.values()} == {MusiqueProof}
