"""Check the Ignite corpus: unique ids, gold ids exist, required metadata, and the hero chain.

Run: uv run python -m demo.ignite.check_data
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import yaml

from demo.ignite.corpus import DATA_DIR, Question, load_records

REQUIRED = {
    "incidents": ("opened", "service", "severity", "status"),
    "deploys": ("date", "service", "version", "pr", "author"),
    "pull_requests": ("number", "merged", "service"),
    "dashboards": ("service", "week", "week_start", "week_end"),
    "runbooks": ("service",),
}

HERO_CHAIN = ("INC-4821", "DEP-0913-07", "PR-1203", "DASH-W38-search")
FIX_DATE = "2026-09-18"


def _hero_rules(docs: dict[str, dict], incidents: list[dict]) -> list[str]:
    problems: list[str] = []

    def rule(doc_id: str, ok: bool, why: str) -> None:
        if not ok:
            problems.append(f"{doc_id}: {why}")

    for doc_id in HERO_CHAIN + ("PR-1187",):
        if doc_id not in docs:
            problems.append(f"{doc_id}: hero chain document is missing")
    if problems:
        return problems
    inc, dep = docs["INC-4821"], docs["DEP-0913-07"]
    fix, dash = docs["PR-1203"], docs["DASH-W38-search"]
    text = {k: v["text"].lower() for k, v in docs.items()}

    rule("INC-4821", inc["metadata"].get("service") == "search-api", "service must be search-api")
    rule("INC-4821", inc["metadata"].get("opened") == "2026-09-14", "opened must be 2026-09-14")
    for word in ("reranker", "1187", "1203"):
        rule("INC-4821", word not in text["INC-4821"], f"text must not mention {word!r}")

    m = dep["metadata"]
    rule("DEP-0913-07", m.get("service") == "search-api", "service must be search-api")
    rule("DEP-0913-07", m.get("date") == "2026-09-13", "date must be 2026-09-13")
    rule("DEP-0913-07", m.get("version") == "4.12", "version must be 4.12")
    rule("DEP-0913-07", m.get("pr") == 1187, "pr must be 1187 (link to PR-1187)")
    rule("DEP-0913-07", "reranker v2" in text["DEP-0913-07"], "text must mention 'reranker v2'")

    rule("PR-1187", docs["PR-1187"]["metadata"].get("number") == 1187, "number must be 1187")

    m = fix["metadata"]
    rule("PR-1203", m.get("number") == 1203, "number must be 1203")
    rule("PR-1203", m.get("merged") == FIX_DATE, f"merged must be {FIX_DATE}")
    rule("PR-1203", "inc-4821" in text["PR-1203"], "text must reference INC-4821")
    rule("PR-1203", "reranker v2" in text["PR-1203"], "text must mention 'reranker v2'")

    m = dash["metadata"]
    rule("DASH-W38-search", m.get("service") == "search-api", "service must be search-api")
    rule("DASH-W38-search", str(m.get("week_start")) <= FIX_DATE <= str(m.get("week_end")),
         f"week must cover the fix date {FIX_DATE}")
    for word in ("reranker", "inc-4821"):
        rule("DASH-W38-search", word not in text["DASH-W38-search"],
             f"text must not mention {word!r}")

    september = [r["id"] for r in incidents
                 if r["metadata"].get("service") == "search-api"
                 and str(r["metadata"].get("opened", "")).startswith("2026-09")
                 and "latency" in r["text"].lower()]
    if september != ["INC-4821"]:
        problems.append(f"INC-4821 must be the only September search-api latency incident; "
                        f"found {september}")
    return problems


def check(data_dir: Path = DATA_DIR) -> list[str]:
    """Problems with the corpus; an empty list means it passes."""
    problems: list[str] = []
    records = load_records(data_dir)
    counts = Counter(r["id"] for rows in records.values() for r in rows)
    problems += [f"duplicate id {i!r} ({n} times)" for i, n in counts.items() if n > 1]

    docs: dict[str, dict] = {}
    for source, rows in records.items():
        for r in rows:
            docs.setdefault(r["id"], r)
            missing = [k for k in ("id", "title", "text", "metadata") if k not in r]
            if missing:
                problems.append(f"{r.get('id')}: missing {missing}")
                continue
            for key in REQUIRED[source]:
                if key not in r["metadata"]:
                    problems.append(f"{r['id']} ({source}): missing metadata {key!r}")

    raw = yaml.safe_load((Path(data_dir) / "questions.yaml").read_text()) or []
    ids = Counter(q.get("id") for q in raw)
    problems += [f"duplicate question id {i!r}" for i, n in ids.items() if n > 1]
    for q in raw:
        try:
            question = Question(**q)
        except ValueError as exc:
            problems.append(f"question {q.get('id')!r} is invalid: {exc}")
            continue
        for gold in question.gold:
            if gold not in docs:
                problems.append(f"question {question.id!r}: gold id {gold!r} does not exist")

    problems += _hero_rules(docs, records["incidents"])
    return problems


def main() -> int:
    problems = check()
    for p in problems:
        print(p)
    if problems:
        return 1
    n_docs = sum(len(rows) for rows in load_records().values())
    n_questions = len(yaml.safe_load((DATA_DIR / "questions.yaml").read_text()) or [])
    print(f"ok: {n_docs} docs, {n_questions} questions")
    return 0


if __name__ == "__main__":
    sys.exit(main())
