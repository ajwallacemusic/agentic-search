"""The Ignite demo data: the seeded MuSiQue sample as one `wiki` source, plus the demo questions
picked from it by MuSiQue id in questions.yaml."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel

from agentic_search.core.types import Document
from demo.ignite.public.datasets import sample_musique
from demo.ignite.public.fetch import fetch

QUESTIONS_PATH = Path(__file__).parent / "questions.yaml"
MUSIQUE_FILE = "musique_ans_v1.0_dev.jsonl"


class Question(BaseModel):
    id: str
    text: str
    kind: str  # MuSiQue hop group: "2hop", "3hop" or "4hop"
    gold: list[str]
    musique_id: str


@dataclass
class DemoData:
    documents: dict[str, list[Document]]
    descriptions: dict[str, str]
    questions: dict[str, Question]


def load_demo(musique_path: Path | None = None,
              questions_path: Path = QUESTIONS_PATH) -> DemoData:
    """The same sample the public eval scores (default settings, seed 7), so every demo question's
    supporting paragraphs are in the corpus. Fetches MuSiQue into the cache on first use."""
    sample = sample_musique(musique_path or fetch("musique") / MUSIQUE_FILE)
    group_of = {qid: g for g, qids in sample.groups.items() for qid in qids}
    questions: dict[str, Question] = {}
    for pick in yaml.safe_load(Path(questions_path).read_text()) or []:
        mid = pick["musique_id"]
        if mid not in sample.queries:
            raise ValueError(f"question {pick['id']!r}: MuSiQue id {mid!r} is not in the sample")
        questions[pick["id"]] = Question(id=pick["id"], text=sample.queries[mid],
                                         kind=group_of[mid], gold=sorted(sample.qrels[mid]),
                                         musique_id=mid)
    return DemoData(documents={sample.source: sample.documents},
                    descriptions={sample.source: sample.description}, questions=questions)
