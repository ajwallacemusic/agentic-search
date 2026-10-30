import csv

import pytest

from demo.ignite.public.wands_match import (
    MatchRule,
    anchor_value,
    gold_for,
    load_products,
    parse_features,
    select_rules,
)

COLUMNS = ["product_id", "product_name", "product_class", "category hierarchy",
           "product_description", "product_features", "rating_count", "average_rating",
           "review_count"]
LIVING = "Furniture / Living Room Furniture / Tables"
BEDROOM = "Furniture / Bedroom Furniture / Storage"
PRODUCTS = [
    ("1", "winkleman teak end table", "End Tables", LIVING, "style : rustic|dswoodtone : light wood"),
    ("2", "harlow coffee table", "Coffee Tables", LIVING, "style : rustic"),
    ("3", "brook coffee table", "Coffee Tables", LIVING, "style : Modern|style : Rustic"),
    ("4", "mesa coffee table", "Coffee Tables", LIVING, "style : modern"),
    ("5", "kora coffee table", "Coffee Tables", LIVING, "style : modern"),
    ("6", "aldo tv stand", "TV Stands", LIVING, "style : rustic"),
    ("7", "tess tv stand", "TV Stands", LIVING, "style : modern"),
    ("8", "ovi tv stand", "TV Stands", LIVING, "style : Rustic"),
    ("9", "rustic pine side table", "End Tables", LIVING, "style : rustic"),
    ("10", "duo end table", "End Tables", LIVING, "style : modern|style : rustic"),
    ("11", "ivy nightstand", "Nightstands", BEDROOM, "dswoodtone : Light  Wood|color : gray"),
    ("12", "ivy nightstand", "Nightstands", BEDROOM, "dswoodtone : dark wood"),
    ("13", "pell dresser", "Dressers", BEDROOM, "dswoodtone : light wood|color : gray"),
    ("14", "rho dresser", "Dressers", BEDROOM, "dswoodtone : dark wood|color : white"),
    ("15", "sami dresser", "Dressers", BEDROOM, "dswoodtone : light wood|color : white"),
    ("16", "lund nightstand", "Nightstands", BEDROOM, "dswoodtone : light wood|color : white"),
    ("17", "kaya nightstand", "Nightstands", BEDROOM, "dswoodtone : dark wood|color : gray"),
    ("18", "noor shower head", "Shower Heads", "Home Improvement / Bath", "color : gray"),
]


def _folder(tmp_path):
    with (tmp_path / "product.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(COLUMNS)
        for pid, name, klass, cat, feats in PRODUCTS:
            w.writerow([pid, name, klass, cat, "", feats, "", "", ""])
    return tmp_path


@pytest.fixture
def products(tmp_path):
    return load_products(_folder(tmp_path))


def test_parse_features_splits_keys_and_values():
    assert parse_features("dswoodtone : Light Wood|color:gray|color : Gray|style : rustic") == {
        "dswoodtone": ["light wood"], "color": ["gray"], "style": ["rustic"]}


def test_values_normalize_case_and_spaces(products):
    rule = MatchRule(id="m", anchor_id="13", attribute="wood_tone", targets=["Nightstands"])
    assert "wands-11" in gold_for(rule, products)  # "Light  Wood" == "light wood"


def test_target_matches_on_any_value(products):
    rule = MatchRule(id="m", anchor_id="1", attribute="style", targets=["Coffee Tables"])
    assert gold_for(rule, products) == ["wands-2", "wands-3"]


def test_anchor_with_two_values_is_rejected(products):
    with pytest.raises(ValueError):
        anchor_value(MatchRule(id="m", anchor_id="10", attribute="style", targets=["TV Stands"]),
                     products)


def test_gold_is_target_class_and_same_value_and_never_the_anchor(products):
    rule = MatchRule(id="m", anchor_id="1", attribute="style",
                     targets=["End Tables", "TV Stands"])
    assert gold_for(rule, products) == ["wands-10", "wands-6", "wands-8", "wands-9"]


def test_select_rules_follows_the_rules(products):
    rules = select_rules(products, n=4, min_gold=2, max_share=0.75)
    assert rules
    names = [p.name for p in products.values()]
    for rule in rules:
        anchor = products[rule.anchor_id]
        value = anchor_value(rule, products)
        assert names.count(anchor.name) == 1
        assert value not in anchor.name
        assert anchor.product_class not in rule.targets and 1 <= len(rule.targets) <= 2
        assert all(anchor.area in {p.area for p in products.values() if t in p.classes}
                   for t in rule.targets)
        gold = gold_for(rule, products)
        assert len(gold) >= 2
        for target in rule.targets:
            in_class = [p for p in products.values() if target in p.classes
                        and p.features.get(rule.feature_key)]
            matching = [p for p in in_class if f"wands-{p.product_id}" in gold]
            assert len(matching) <= 0.75 * len(in_class)
        assert rule.text == ""


def test_select_rules_is_deterministic(products):
    assert select_rules(products, n=4, min_gold=2, max_share=0.75) == \
        select_rules(products, n=4, min_gold=2, max_share=0.75)


# ---- the committed question set (needs the WANDS cache) ------------------------------------

def _real_products():
    from demo.ignite.public.fetch import CACHE_DIR
    folder = CACHE_DIR / "wands"
    if not (folder / "product.csv").exists():
        pytest.skip("WANDS not downloaded (run eval_public --dataset wands once)")
    return load_products(folder)


def test_rule_ids_are_unique():
    from demo.ignite.public.wands_match import load_rules
    ids = [r.id for r in load_rules()]
    assert ids and len(ids) == len(set(ids))


def test_question_text_names_anchor_and_hides_value():
    from demo.ignite.public.wands_match import load_rules, norm
    products = _real_products()
    for rule in load_rules():
        text = norm(rule.text)
        assert products[rule.anchor_id].name in text, rule.id
        assert anchor_value(rule, products) not in text, rule.id


def test_committed_labels_match_the_rules():
    import json

    from demo.ignite.public.wands_match import LABELS_PATH, build_labels, load_rules
    assert build_labels(load_rules(), _real_products()) == json.loads(LABELS_PATH.read_text())


def test_sample_wands_match_groups_by_attribute(tmp_path):
    import json

    import yaml

    from demo.ignite.public.wands_match import sample_wands_match
    folder = _folder(tmp_path)
    rules = [MatchRule(id="m1", anchor_id="1", attribute="style", targets=["Coffee Tables"],
                       text="I have the winkleman teak end table. A coffee table to match?"),
             MatchRule(id="m2", anchor_id="13", attribute="wood_tone", targets=["Nightstands"],
                       text="I have the pell dresser. A nightstand to match?")]
    (tmp_path / "rules.yaml").write_text(yaml.safe_dump([r.model_dump() for r in rules]))
    (tmp_path / "labels.json").write_text(json.dumps({"m1": ["wands-2", "wands-3"],
                                                      "m2": ["wands-11", "wands-16"]}))
    s = sample_wands_match(folder, tmp_path / "rules.yaml", tmp_path / "labels.json")
    assert s.name == "wands-match" and s.source == "products"
    assert s.groups == {"style": ["m1"], "wood_tone": ["m2"]}
    assert s.queries["m1"] == rules[0].text
    assert s.qrels == {"m1": {"wands-2": 1, "wands-3": 1}, "m2": {"wands-11": 1, "wands-16": 1}}
    assert len(s.documents) == len(PRODUCTS)
