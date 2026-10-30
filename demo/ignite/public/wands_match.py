"""WANDS match: our "look it up, then match it" questions over the WANDS catalogue.

Each question names a product the shopper owns (the anchor) and asks for other product types that
match it on one attribute (wood tone, color or style). Gold comes from a rule over WANDS fields,
never from a model: a product is gold if it is in a target class and any of its values for the
attribute equals the anchor's single value.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from demo.ignite.public.datasets import _tsv

ATTRIBUTES: dict[str, str] = {"wood_tone": "dswoodtone", "color": "color", "style": "style"}


def norm(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().lower()


def parse_features(text: str) -> dict[str, list[str]]:
    """`key : value|key : value` -> key (lower case, no spaces) -> distinct normalized values."""
    out: dict[str, list[str]] = {}
    for pair in (text or "").split("|"):
        if ":" not in pair:
            continue
        key, value = pair.split(":", 1)
        key, value = re.sub(r"\s+", "", key).lower(), norm(value)
        if key and value and value not in out.setdefault(key, []):
            out[key].append(value)
    return out


class MatchRule(BaseModel):
    id: str
    anchor_id: str
    attribute: Literal["wood_tone", "color", "style"]
    targets: list[str]
    text: str = ""

    @property
    def feature_key(self) -> str:
        return ATTRIBUTES[self.attribute]


@dataclass
class Product:
    product_id: str
    name: str
    product_class: str
    area: str  # first two levels of the category path, e.g. "Furniture / Bedroom Furniture"
    features: dict[str, list[str]] = field(default_factory=dict)

    @property
    def classes(self) -> set[str]:
        """WANDS joins several classes with "|"."""
        return {c.strip() for c in self.product_class.split("|") if c.strip()}


def load_products(folder: Path) -> dict[str, Product]:
    out: dict[str, Product] = {}
    for row in _tsv(Path(folder) / "product.csv"):
        path = [p.strip() for p in (row.get("category hierarchy") or "").split("/")]
        out[row["product_id"]] = Product(
            product_id=row["product_id"], name=norm(row.get("product_name") or ""),
            product_class=(row.get("product_class") or "").strip(),
            area=" / ".join(path[:2]) if len(path) >= 2 and all(path[:2]) else "",
            features=parse_features(row.get("product_features") or ""))
    return out


def anchor_value(rule: MatchRule, products: dict[str, Product]) -> str:
    values = products[rule.anchor_id].features.get(rule.feature_key, [])
    if len(values) != 1:
        raise ValueError(f"rule {rule.id}: anchor {rule.anchor_id} has {len(values)} "
                         f"{rule.attribute} values, needs exactly one")
    return values[0]


def _matches(p: Product, targets: set[str], key: str, value: str) -> bool:
    return bool(p.classes & targets) and value in p.features.get(key, [])


def gold_for(rule: MatchRule, products: dict[str, Product]) -> list[str]:
    value, targets = anchor_value(rule, products), set(rule.targets)
    return sorted(f"wands-{pid}" for pid, p in products.items()
                  if pid != rule.anchor_id and _matches(p, targets, rule.feature_key, value))


def build_labels(rules: list[MatchRule], products: dict[str, Product]) -> dict[str, list[str]]:
    return {r.id: gold_for(r, products) for r in rules}


def select_rules(products: dict[str, Product], *, n: int = 20, seed: int = 7, min_gold: int = 5,
                 max_share: float = 0.5) -> list[MatchRule]:
    """Seeded rules, round-robin over the attributes. An anchor has one value for the attribute,
    a name unique in the catalogue, the value not in its name, and a category area. Targets are 1
    or 2 other classes seen in the anchor's area where the value picks out at most `max_share` of the
    products that list the attribute; together they give at least `min_gold` gold products."""
    rng = random.Random(seed)
    name_count: dict[str, int] = {}
    for p in products.values():
        name_count[p.name] = name_count.get(p.name, 0) + 1

    pools: dict[str, list[str]] = {}
    for attr, key in ATTRIBUTES.items():
        pool = sorted(pid for pid, p in products.items()
                      if len(p.features.get(key, [])) == 1 and name_count[p.name] == 1
                      and p.area and p.classes and p.features[key][0] not in p.name)
        rng.shuffle(pool)
        pools[attr] = pool

    by_area: dict[str, set[str]] = {}  # area -> classes seen in it
    by_class: dict[str, list[Product]] = {}
    for p in products.values():
        for c in p.classes:
            by_class.setdefault(c, []).append(p)
            if p.area:
                by_area.setdefault(p.area, set()).add(c)

    rules: list[MatchRule] = []
    used: set[str] = set()
    attrs = list(ATTRIBUTES)
    while len(rules) < n and any(pools.values()):
        for attr in attrs:
            if len(rules) >= n:
                break
            key = ATTRIBUTES[attr]
            while pools[attr]:
                pid = pools[attr].pop()
                if pid in used:
                    continue
                anchor = products[pid]
                value = anchor.features[key][0]
                eligible = []
                for klass in sorted(by_area[anchor.area] - anchor.classes):
                    rows = [p for p in by_class[klass] if p.features.get(key)]
                    hits = sum(value in p.features[key] for p in rows if p is not anchor)
                    if hits and hits <= max_share * len(rows):
                        eligible.append((klass, hits))
                if not eligible:
                    continue
                k = min(rng.choice([1, 2]), len(eligible))
                picked = sorted(rng.sample(eligible, k))
                if sum(h for _, h in picked) < min_gold:
                    continue
                used.add(pid)
                rules.append(MatchRule(id=f"match-{len(rules) + 1:02d}", anchor_id=pid,
                                       attribute=attr, targets=[c for c, _ in picked]))
                break
    return rules
