"""The Ignite demo corpus: one JSONL file per source, plus the demo questions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

from agentic_search.core.types import Document, TextPart

SOURCES: tuple[str, ...] = ("incidents", "deploys", "pull_requests", "dashboards", "runbooks")
DATA_DIR = Path(__file__).parent / "data"


class Question(BaseModel):
    id: str
    text: str
    kind: Literal["single_hop", "multi_hop"]
    gold: list[str]


def load_records(data_dir: Path = DATA_DIR) -> dict[str, list[dict]]:
    """Source name -> raw rows `{"id", "title", "text", "metadata"}`."""
    out: dict[str, list[dict]] = {}
    for source in SOURCES:
        path = Path(data_dir) / f"{source}.jsonl"
        with path.open(encoding="utf-8") as f:
            out[source] = [json.loads(line) for line in f if line.strip()]
    return out


def load_documents(data_dir: Path = DATA_DIR) -> dict[str, list[Document]]:
    return {
        source: [Document(doc_id=r["id"], content=[TextPart(text=f"{r['title']}\n\n{r['text']}")],
                          metadata={**r["metadata"], "title": r["title"]})
                 for r in rows]
        for source, rows in load_records(data_dir).items()
    }


def load_questions(data_dir: Path = DATA_DIR) -> dict[str, Question]:
    raw = yaml.safe_load((Path(data_dir) / "questions.yaml").read_text()) or []
    return {q.id: q for q in (Question(**r) for r in raw)}
