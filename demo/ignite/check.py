"""Check it yourself: why each result is marked right or not, in plain words.

The viewer's ★ marks come from each dataset's own labels. This module states the rule for each
question and, for any result, the label or the catalogue fields that decide it, so a viewer can
open a result and see the proof instead of trusting the mark.

- musique: the paragraphs the MuSiQue authors list as needed to answer.
- wands: Wayfair's own raters (Exact, Partial, Irrelevant); only Exact counts as right.
- wands-match: a fixed rule over catalogue fields (product type and one attribute), no model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from demo.ignite.corpus import Question
from demo.ignite.public.datasets import _tsv
from demo.ignite.public.fetch import fetch
from demo.ignite.public.wands_match import (
    MATCH_PATH,
    MatchRule,
    Product,
    anchor_value,
    load_products,
    load_rules,
)

ATTRIBUTE_WORDS = {"wood_tone": "wood tone", "color": "color", "style": "style"}


class Proof(Protocol):
    def rule(self) -> str: ...
    def check(self, doc_id: str) -> dict[str, Any]: ...


def _result(verdict: str, why: str, marks: list[str] | None = None,
            target: str | None = None) -> dict[str, Any]:
    """verdict: right | partial | wrong | unrated. marks: text to highlight in the document."""
    return {"verdict": verdict, "why": why, "marks": marks or [], "target": target}


@dataclass
class MusiqueProof:
    q: Question

    def rule(self) -> str:
        answer = f" The true answer is “{self.q.answers[0]}”." if self.q.answers else ""
        return (f"★ right = one of the {len(self.q.gold)} paragraphs the MuSiQue authors list as "
                f"needed to answer this question.{answer}")

    def check(self, doc_id: str) -> dict[str, Any]:
        marks = self.q.answers[:1]
        if doc_id in self.q.gold:
            return _result("right", "MuSiQue lists this paragraph as one needed to answer.", marks)
        return _result("wrong", "Not one of the paragraphs MuSiQue lists as needed.", marks)


@dataclass
class RatedProof:
    q: Question
    labels: dict[str, str]  # doc id -> Exact | Partial | Irrelevant

    def rule(self) -> str:
        return (f"★ right = Wayfair's own raters marked the product Exact for “{self.q.text}”. "
                f"{len(self.q.gold)} products are Exact. Partial does not count.")

    def check(self, doc_id: str) -> dict[str, Any]:
        label = self.labels.get(doc_id)
        if label == "Exact":
            return _result("right", "Wayfair raters: Exact match for this query.")
        if label == "Partial":
            return _result("partial", "Wayfair raters: Partial match. Only Exact counts as right.")
        if label == "Irrelevant":
            return _result("wrong", "Wayfair raters: Irrelevant for this query.")
        return _result("unrated", "Wayfair raters did not rate this product for this query. "
                                  "It counts as not right.")


@dataclass
class MatchProof:
    q: Question
    match: MatchRule
    products: dict[str, Product]

    @property
    def anchor(self) -> Product:
        return self.products[self.match.anchor_id]

    @property
    def value(self) -> str:
        return anchor_value(self.match, self.products)

    def rule(self) -> str:
        word = ATTRIBUTE_WORDS[self.match.attribute]
        return (f"★ right = a product of type {' or '.join(self.match.targets)} whose catalogue "
                f"{word} is “{self.value}”. That is the {word} the catalogue lists for the "
                f"{self.anchor.name} (product {self.match.anchor_id}). A fixed rule over "
                "Wayfair's own fields decides it. No model does.")

    def check(self, doc_id: str) -> dict[str, Any]:
        pid = doc_id.removeprefix("wands-")
        p = self.products.get(pid)
        if p is None:
            return _result("unrated", "Not in the catalogue.")
        key, word, targets = self.match.feature_key, ATTRIBUTE_WORDS[self.match.attribute], \
            set(self.match.targets)
        values = p.features.get(key, [])
        marks = [f"{key} : {v}" for v in values]
        if pid == self.match.anchor_id:
            return _result("wrong", "This is the product you already own. It does not count.",
                           marks)
        hit_types = sorted(p.classes & targets)
        type_part = (f"type {', '.join(sorted(p.classes)) or 'not listed'} "
                     + ("✓" if hit_types else f"✗ (need {' or '.join(self.match.targets)})"))
        value_part = (f"{word} {', '.join(values) or 'not listed'} "
                      + ("✓" if self.value in values else f"✗ (need {self.value})"))
        ok = bool(hit_types) and self.value in values
        return _result("right" if ok else "wrong", f"Catalogue: {type_part} · {value_part}.",
                       marks + sorted(p.classes), hit_types[0] if ok else None)

    def coverage(self, doc_ids: list[str]) -> list[list[Any]]:
        """[target type, number right of that type] for each target, so "9 faucets and no towel
        bar" shows up next to a 9 of 10 score."""
        got = [self.check(d)["target"] for d in doc_ids]
        return [[t, got.count(t)] for t in self.match.targets]


def wands_labels(folder: Path, refs: set[str]) -> dict[str, dict[str, str]]:
    """WANDS query id -> doc id -> raw rater label, for the given queries only."""
    out: dict[str, dict[str, str]] = {}
    rank = {"Irrelevant": 0, "Partial": 1, "Exact": 2}
    for row in _tsv(Path(folder) / "label.csv"):
        if row["query_id"] in refs:
            q = out.setdefault(row["query_id"], {})
            key = f"wands-{row['product_id']}"
            # 14 pairs carry two labels; keep the higher one, as sample_wands does
            if rank.get(row["label"], -1) > rank.get(q.get(key, ""), -1):
                q[key] = row["label"]
    return out


def load_proofs(questions: dict[str, Question], *, wands_folder: Path | None = None,
                match_path: Path = MATCH_PATH) -> dict[str, Proof]:
    """Question id -> its proof. Reads the WANDS files only when a WANDS question is present."""
    out: dict[str, Proof] = {}
    wands = [q for q in questions.values() if q.corpus == "wands"]
    folder = (wands_folder or fetch("wands")) if wands else None
    labels = wands_labels(folder, {q.ref for q in wands if q.set == "wands"}) if folder else {}
    matches = [q for q in wands if q.set == "wands-match"]
    products = load_products(folder) if folder and matches else {}
    rules = {r.id: r for r in load_rules(match_path)} if matches else {}
    for q in questions.values():
        if q.set == "musique":
            out[q.id] = MusiqueProof(q)
        elif q.set == "wands":
            out[q.id] = RatedProof(q, labels.get(q.ref, {}))
        else:
            out[q.id] = MatchProof(q, rules[q.ref], products)
    return out
