"""BEIR-format datasets: corpus.jsonl, queries.jsonl, qrels/<split>.tsv."""

from __future__ import annotations

import csv
import json
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from agentic_search.core.types import Document, TextPart

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"


@dataclass
class BeirDataset:
    name: str
    corpus: dict[str, dict[str, str]]
    queries: dict[str, str]
    qrels: dict[str, dict[str, int]]

    def documents(self) -> list[Document]:
        docs = []
        for doc_id, row in self.corpus.items():
            title, text = row.get("title", ""), row.get("text", "")
            docs.append(Document(doc_id=doc_id, content=[TextPart(text=f"{title}\n{text}".strip())],
                                 metadata={"title": title} if title else {}))
        return docs


def _jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_beir(path: str | Path, split: str = "test") -> BeirDataset:
    path = Path(path)
    corpus = {str(r["_id"]): {"title": r.get("title") or "", "text": r.get("text") or ""}
              for r in _jsonl(path / "corpus.jsonl")}
    qrels: dict[str, dict[str, int]] = {}
    with (path / "qrels" / f"{split}.tsv").open(encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)  # header
        for row in reader:
            if len(row) >= 3:
                qrels.setdefault(row[0], {})[row[1]] = int(row[2])
    queries = {str(r["_id"]): r["text"] for r in _jsonl(path / "queries.jsonl")
               if str(r["_id"]) in qrels}
    return BeirDataset(name=path.name, corpus=corpus, queries=queries, qrels=qrels)


def download_beir(name: str, dest_dir: str | Path) -> Path:
    dest = Path(dest_dir)
    target = dest / name
    if (target / "corpus.jsonl").exists():
        return target
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / f"{name}.zip"
    urllib.request.urlretrieve(BEIR_URL.format(name=name), archive)
    with zipfile.ZipFile(archive) as z:
        z.extractall(dest)
    archive.unlink()
    return target
