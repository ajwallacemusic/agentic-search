"""WANDS multi: shopping tasks whose parts do not depend on each other.

Each question joins 3 or 4 real WANDS shopper queries from different product classes. Gold for a
part is that query's human-labelled Exact products; nothing is made by a model. The set is picked
once by a seeded rule and committed (`wands_multi.yaml`), so it cannot be tuned after a run.

Build: uv run python -m demo.ignite.public.wands_multi --seed 7 --n 20 > demo/ignite/wands_multi.yaml
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import yaml
from pydantic import BaseModel

from demo.ignite.public.datasets import WANDS_DESCRIPTION, WANDS_GAIN, PublicSample, _product_doc, _tsv
from demo.ignite.public.wands_match import NOT_PRODUCT_TYPES

DEMO_DIR = Path(__file__).parent.parent
MULTI_PATH = DEMO_DIR / "wands_multi.yaml"


class MultiQuestion(BaseModel):
    id: str
    parts: list[str]
    text: str
    gold: dict[str, list[str]]


def question_text(part_texts: list[str]) -> str:
    return "I'm shopping for a few things: " + "; ".join(part_texts) + "."


def select_questions(queries: dict[str, tuple[str, str]], exact: dict[str, list[str]], *,
                     n: int = 20, seed: int = 7, min_exact: int = 3) -> list[MultiQuestion]:
    """`queries`: query id -> (text, query_class); `exact`: query id -> Exact doc ids.
    Parts in a question have distinct non-empty classes; no query is used twice."""
    rng = random.Random(seed)
    pool = sorted((q for q, (_, klass) in queries.items()
                   if klass.strip() and klass not in NOT_PRODUCT_TYPES
                   and len(exact.get(q, [])) >= min_exact), key=int_or_str)
    rng.shuffle(pool)
    out: list[MultiQuestion] = []
    while len(out) < n:
        size = rng.choice([3, 4])
        parts: list[str] = []
        for q in pool:
            if len(parts) == size:
                break
            if queries[q][1] not in {queries[p][1] for p in parts}:
                parts.append(q)
        if len(parts) < 3:
            break
        pool = [q for q in pool if q not in parts]
        out.append(MultiQuestion(id=f"m{len(out) + 1:02d}", parts=parts,
                                 text=question_text([queries[p][0] for p in parts]),
                                 gold={p: sorted(exact[p]) for p in parts}))
    return out


def int_or_str(q: str) -> tuple[int, str]:
    return (int(q), "") if q.isdigit() else (0, q)


def _graded(folder: Path) -> dict[str, dict[str, int]]:
    graded: dict[str, dict[str, int]] = {}
    for row in _tsv(Path(folder) / "label.csv"):
        gain = WANDS_GAIN.get(row["label"])
        if gain:
            q = graded.setdefault(row["query_id"], {})
            key = f"wands-{row['product_id']}"
            q[key] = max(q.get(key, 0), gain)
    return graded


def sample_wands_multi(folder: Path, path: Path = MULTI_PATH, *,
                       n: int | None = None) -> PublicSample:
    """The committed multi questions over the full catalogue. qrels = the union of the parts'
    graded labels (Exact 2, Partial 1; the higher grade wins); `parts` = per-part Exact gold."""
    qs = [MultiQuestion(**q) for q in yaml.safe_load(Path(path).read_text()) or []]
    if n is not None:
        qs = qs[:n]
    graded = _graded(folder)
    qrels: dict[str, dict[str, int]] = {}
    for q in qs:
        merged: dict[str, int] = {}
        for p in q.parts:
            for doc, gain in graded.get(p, {}).items():
                merged[doc] = max(merged.get(doc, 0), gain)
        qrels[q.id] = merged
    return PublicSample(
        name="wands-multi", source="products", description=WANDS_DESCRIPTION,
        documents=[_product_doc(r) for r in _tsv(Path(folder) / "product.csv")],
        queries={q.id: q.text for q in qs}, qrels=qrels, groups={"all": [q.id for q in qs]},
        parts={q.id: q.gold for q in qs})


def _dump(qs: list[MultiQuestion]) -> str:
    class Flow(list):
        pass

    yaml.SafeDumper.add_representer(
        Flow, lambda d, v: d.represent_sequence("tag:yaml.org,2002:seq", v, flow_style=True))
    rows = [{"id": q.id, "text": q.text, "parts": Flow(q.parts),
             "gold": {p: Flow(g) for p, g in q.gold.items()}} for q in qs]
    return yaml.safe_dump(rows, sort_keys=False, width=100)


def main() -> None:
    from demo.ignite.public.fetch import fetch

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()
    folder = fetch("wands")
    queries = {r["query_id"]: (r["query"].strip(), (r.get("query_class") or "").strip())
               for r in _tsv(folder / "query.csv")}
    exact = {q: [d for d, g in labels.items() if g == 2] for q, labels in _graded(folder).items()}
    print(_dump(select_questions(queries, exact, n=args.n, seed=args.seed)), end="")


if __name__ == "__main__":
    main()
