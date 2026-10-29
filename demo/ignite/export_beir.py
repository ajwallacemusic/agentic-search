"""Export the Ignite corpus and questions in BEIR format: corpus.jsonl, queries.jsonl,
qrels/test.tsv (one row per gold id, score 1).

Run: uv run python -m demo.ignite.export_beir
"""

from __future__ import annotations

import json
from pathlib import Path

from demo.ignite.corpus import DATA_DIR, load_questions, load_records

BEIR_DIR = Path(__file__).parent / "beir"


def export_beir(out_dir: Path = BEIR_DIR, data_dir: Path = DATA_DIR) -> Path:
    out_dir = Path(out_dir)
    (out_dir / "qrels").mkdir(parents=True, exist_ok=True)
    rows = [r for source_rows in load_records(data_dir).values() for r in source_rows]
    (out_dir / "corpus.jsonl").write_text("".join(
        json.dumps({"_id": r["id"], "title": r["title"], "text": r["text"]}) + "\n" for r in rows))
    questions = load_questions(data_dir).values()
    (out_dir / "queries.jsonl").write_text("".join(
        json.dumps({"_id": q.id, "text": q.text}) + "\n" for q in questions))
    qrels = ["query-id\tcorpus-id\tscore"] + [f"{q.id}\t{g}\t1" for q in questions for g in q.gold]
    (out_dir / "qrels" / "test.tsv").write_text("\n".join(qrels) + "\n")
    return out_dir


if __name__ == "__main__":
    print(export_beir())
