"""Seeded samples of the public eval datasets, as documents + queries + graded qrels."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

from agentic_search.core.types import Document, TextPart, text_of
from agentic_search.eval.datasets import BeirDataset


@dataclass
class PublicSample:
    name: str
    source: str
    description: str
    documents: list[Document]
    queries: dict[str, str]
    qrels: dict[str, dict[str, int]]
    groups: dict[str, list[str]]

    def to_beir(self) -> BeirDataset:
        corpus = {d.doc_id: {"title": str(d.metadata.get("title", "")), "text": text_of(d.content)}
                  for d in self.documents}
        return BeirDataset(name=self.name, corpus=corpus, queries=dict(self.queries),
                           qrels={q: dict(g) for q, g in self.qrels.items()})


# ---- MuSiQue ------------------------------------------------------------------------------

def para_id(title: str, text: str) -> str:
    """One id per distinct paragraph: the same paragraph appears under many questions."""
    return "msq-" + hashlib.sha1(f"{title}\n{text}".encode()).hexdigest()[:12]


def _hop_group(qid: str) -> str:
    return qid.split("__")[0][:4]  # "2hop", "3hop1" -> "3hop", "4hop3" -> "4hop"


def _para_doc(p: dict) -> Document:
    return Document(doc_id=para_id(p["title"], p["paragraph_text"]),
                    content=[TextPart(text=f"{p['title']}\n\n{p['paragraph_text']}")],
                    metadata={"title": p["title"]})


def sample_musique(path: Path, *, per_hop: int = 20, corpus_size: int = 5000,
                   seed: int = 7) -> PublicSample:
    """`per_hop` questions from each of 2-, 3- and 4-hop. The corpus holds every paragraph of
    the sampled questions, then paragraphs of other questions as distractors up to
    `corpus_size` (the cap never drops a sampled question's paragraph)."""
    with Path(path).open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    rng = random.Random(seed)
    by_group: dict[str, list[str]] = {}
    for r in rows:
        by_group.setdefault(_hop_group(r["id"]), []).append(r["id"])
    groups = {g: sorted(rng.sample(sorted(ids), min(per_hop, len(ids))))
              for g, ids in sorted(by_group.items())}
    chosen = {q for qs in groups.values() for q in qs}

    docs: dict[str, Document] = {}
    queries: dict[str, str] = {}
    qrels: dict[str, dict[str, int]] = {}
    for r in rows:
        if r["id"] not in chosen:
            continue
        queries[r["id"]] = r["question"]
        qrels[r["id"]] = {}
        for p in r["paragraphs"]:
            doc = _para_doc(p)
            docs.setdefault(doc.doc_id, doc)
            if p["is_supporting"]:
                qrels[r["id"]][doc.doc_id] = 1

    distractors: dict[str, Document] = {}
    for r in rows:
        if r["id"] in chosen:
            continue
        for p in r["paragraphs"]:
            doc = _para_doc(p)
            if doc.doc_id not in docs:
                distractors.setdefault(doc.doc_id, doc)
    extra = sorted(distractors)
    rng.shuffle(extra)
    for doc_id in extra[: max(0, corpus_size - len(docs))]:
        docs[doc_id] = distractors[doc_id]

    return PublicSample(
        name="musique", source="wiki",
        description="Wikipedia paragraphs, one per item: title and text. Field: title.",
        documents=list(docs.values()), queries=queries, qrels=qrels, groups=groups)


# ---- WANDS --------------------------------------------------------------------------------

WANDS_GAIN = {"Exact": 2, "Partial": 1}


def _tsv(path: Path) -> list[dict[str, str]]:
    csv.field_size_limit(sys.maxsize)
    with Path(path).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _number(value: str | None) -> float | int | None:
    if value is None or not value.strip():
        return None
    try:
        n = float(value)
    except ValueError:
        return None
    return int(n) if n.is_integer() and "." not in value else n


def _product_doc(row: dict[str, str]) -> Document:
    name = (row.get("product_name") or "").strip()
    klass = (row.get("product_class") or "").strip()
    category = (row.get("category hierarchy") or "").strip()
    parts = [name, " · ".join(x for x in (klass, category) if x),
             (row.get("product_description") or "").strip(),
             (row.get("product_features") or "").strip()]
    metadata: dict[str, object] = {"title": name, "product_class": klass, "category": category}
    for key in ("rating_count", "average_rating"):
        n = _number(row.get(key))
        if n is not None:
            metadata[key] = n
    return Document(doc_id=f"wands-{row['product_id']}",
                    content=[TextPart(text="\n\n".join(p for p in parts if p))],
                    metadata=metadata)


def sample_wands(folder: Path, *, n: int = 60, seed: int = 7) -> PublicSample:
    """`n` queries that have at least one Exact product; the whole catalogue as the corpus;
    Exact = 2 and Partial = 1 (graded, as WANDS is usually scored)."""
    folder = Path(folder)
    labels = _tsv(folder / "label.csv")
    graded: dict[str, dict[str, int]] = {}
    for row in labels:
        gain = WANDS_GAIN.get(row["label"])
        if gain:
            graded.setdefault(row["query_id"], {})[f"wands-{row['product_id']}"] = gain
    with_exact = sorted((q for q, g in graded.items() if 2 in g.values()), key=int)
    chosen = sorted(random.Random(seed).sample(with_exact, min(n, len(with_exact))), key=int)
    texts = {r["query_id"]: r["query"] for r in _tsv(folder / "query.csv")}
    docs = [_product_doc(r) for r in _tsv(folder / "product.csv")]
    return PublicSample(
        name="wands", source="products",
        description=("Wayfair product catalogue, one product per item: name, class, category "
                     "path, description, features. Fields: title, product_class, category, "
                     "rating_count, average_rating."),
        documents=docs, queries={q: texts[q] for q in chosen},
        qrels={q: graded[q] for q in chosen}, groups={"all": chosen})
