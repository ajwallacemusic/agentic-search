"""The Ignite demo data: one corpus per public dataset, and the demo questions picked from three
question sets in questions.yaml.

- musique: the seeded MuSiQue sample as one `wiki` source; questions by MuSiQue id.
- wands: the WANDS catalogue as one `products` source. Two question sets run on it: `wands` (the
  published queries, by WANDS query id, from the seeded sample of 60) and `wands-match` (our
  rule-labeled questions, by rule id, from wands_match.yaml).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

from agentic_search.core.types import Document
from demo.ignite.public.datasets import sample_musique, sample_wands
from demo.ignite.public.fetch import fetch
from demo.ignite.public.wands_match import LABELS_PATH, MATCH_PATH, load_rules

QUESTIONS_PATH = Path(__file__).parent / "questions.yaml"
MUSIQUE_FILE = "musique_ans_v1.0_dev.jsonl"
CORPUS_OF = {"musique": "musique", "wands": "wands", "wands-match": "wands"}


class Question(BaseModel):
    id: str
    text: str
    set: Literal["musique", "wands", "wands-match"]
    corpus: Literal["musique", "wands"]
    kind: str  # musique: hop group; wands: "query"; wands-match: the attribute
    gold: list[str]
    ref: str  # MuSiQue id, WANDS query id or WANDS match rule id
    answers: list[str] = []  # MuSiQue: the true answer, then its aliases; empty for WANDS


@dataclass
class DemoData:
    documents: dict[str, list[Document]]
    descriptions: dict[str, str]
    questions: dict[str, Question]


def _picks(questions_path: Path) -> list[dict]:
    picks = yaml.safe_load(Path(questions_path).read_text()) or []
    seen: set[str] = set()
    for p in picks:
        if p["id"] in seen:
            raise ValueError(f"question id {p['id']!r} is used twice in {questions_path}")
        seen.add(p["id"])
    return picks


def corpus_of(question_id: str, questions_path: Path = QUESTIONS_PATH) -> str:
    """The corpus a question runs on, read from the questions file without loading any data."""
    for p in _picks(questions_path):
        if p["id"] == question_id:
            return CORPUS_OF[p["set"]]
    raise KeyError(question_id)


def _missing(pick: dict) -> ValueError:
    return ValueError(f"question {pick['id']!r}: {pick['set']} id {pick['ref']!r} is not in the "
                      "sample")


def _musique(path: Path | None, picks: list[dict]) -> DemoData:
    """The same sample the public eval scores (default settings, seed 7), so every question's
    supporting paragraphs are in the corpus."""
    sample = sample_musique(path or fetch("musique") / MUSIQUE_FILE)
    group_of = {qid: g for g, qids in sample.groups.items() for qid in qids}
    questions = {}
    for p in picks:
        if p["ref"] not in sample.queries:
            raise _missing(p)
        questions[p["id"]] = Question(id=p["id"], text=sample.queries[p["ref"]], set="musique",
                                      corpus="musique", kind=group_of[p["ref"]],
                                      gold=sorted(sample.qrels[p["ref"]]), ref=p["ref"],
                                      answers=sample.answers.get(p["ref"], []))
    return DemoData(documents={sample.source: sample.documents},
                    descriptions={sample.source: sample.description}, questions=questions)


def _wands(folder: Path | None, picks: list[dict], match_path: Path,
           labels_path: Path) -> DemoData:
    sample = sample_wands(folder or fetch("wands"))
    rules = {r.id: r for r in load_rules(match_path)} if any(
        p["set"] == "wands-match" for p in picks) else {}
    labels = json.loads(Path(labels_path).read_text()) if rules else {}
    questions = {}
    for p in picks:
        ref = p["ref"]
        if p["set"] == "wands":
            if ref not in sample.queries:
                raise _missing(p)
            q = Question(id=p["id"], text=sample.queries[ref], set="wands", corpus="wands",
                         kind="query", ref=ref,
                         gold=sorted(d for d, g in sample.qrels[ref].items() if g == 2))
        else:
            if ref not in rules:
                raise _missing(p)
            q = Question(id=p["id"], text=rules[ref].text, set="wands-match", corpus="wands",
                         kind=rules[ref].attribute, gold=sorted(labels[ref]), ref=ref)
        questions[p["id"]] = q
    return DemoData(documents={sample.source: sample.documents},
                    descriptions={sample.source: sample.description}, questions=questions)


def load_demo(*, corpora: set[str] | None = None, musique_path: Path | None = None,
              wands_folder: Path | None = None, questions_path: Path = QUESTIONS_PATH,
              match_path: Path = MATCH_PATH, labels_path: Path = LABELS_PATH
              ) -> dict[str, DemoData]:
    """Corpus name -> its documents and questions. Loads only `corpora` (default: all), fetching
    each dataset into the cache on first use."""
    picks = _picks(questions_path)
    wanted = corpora or {"musique", "wands"}
    by_corpus = {c: [p for p in picks if CORPUS_OF[p["set"]] == c] for c in ("musique", "wands")}
    out: dict[str, DemoData] = {}
    if "musique" in wanted:
        out["musique"] = _musique(musique_path, by_corpus["musique"])
    if "wands" in wanted:
        out["wands"] = _wands(wands_folder, by_corpus["wands"], match_path, labels_path)
    return out
