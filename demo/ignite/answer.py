"""The demo's answer line: one model call reads a mode's top documents and answers.

The library returns ranked documents, not answers. On stage, an answer read from each mode's own
top 10 makes the difference visible: the same reader, given different evidence. Not scored by the
evals; for MuSiQue the viewer also shows whether it matches the dataset's answer.

The reader writes short steps (one fact per line, with its document number), then a final
`Answer:` line. MuSiQue often states a hop loosely ("from" South Carolina for "born in" it,
"part of the Columbia metro area" for "shares a border with Columbia"), so the reader is told to
accept the closest fact; a strict reader said "not found" on harness runs that had every gold
document (tuning-log.md, round 12)."""

from __future__ import annotations

import re
from typing import Any

from agentic_search.models.llm import ChatMessage

ANSWER_SYSTEM = (
    "Answer the question using only the numbered documents. Questions may need several steps: "
    "find each fact in a document, then use it to find the next. Documents often state a step "
    "loosely (e.g. 'from' a place for 'born in' it, or 'part of the city's metro area' for "
    "'next to' it); accept the closest fact a document gives. Write your steps briefly, one line "
    "each, naming the document number for each fact. If the question asks for products, name up "
    "to three matching products from the documents. Then end with one final line: "
    "'Answer: <short answer>'. Only if no document gives any fact for a step, end with "
    "'Answer: Not found in these documents.' Do not use outside knowledge. Notes the search "
    "took (facts it read in documents on the way) may be used like documents; cite them as "
    "[note].")
NOT_FOUND = "Not found in these documents."
MAX_DOC_CHARS = 1200
_ANSWER_LINE = re.compile(r"^\s*\**answer\**\s*:\s*\**\s*(.+?)\s*\**\s*$", re.IGNORECASE | re.MULTILINE)


def parse_reply(text: str) -> tuple[str, list[str]]:
    """(final answer, step lines). Without an `Answer:` line the reply was cut short or off
    format; it then counts as not found rather than showing half a sentence."""
    matches = list(_ANSWER_LINE.finditer(text))
    if not matches:
        return NOT_FOUND, [ln.strip() for ln in text.splitlines() if ln.strip()]
    last = matches[-1]
    steps = [ln.strip() for ln in text[:last.start()].splitlines() if ln.strip()]
    return last.group(1).strip(), steps


async def answer_from(client: Any, question: str, docs: list[tuple[str, str]],
                      notes: list[str] | None = None) -> tuple[str, list[str]]:
    """`docs` = (title, text) pairs, best first; `notes` = facts the mode noted while searching
    (the harness's notebook; none for retrieval). A "find X, then match it" question needs the
    note about X, since X itself is rarely in the top 10. Returns (answer, steps)."""
    if not docs:
        return "No documents to answer from.", []
    body = "\n\n".join(f"[{i}] {title}\n{text[:MAX_DOC_CHARS]}"
                       for i, (title, text) in enumerate(docs, 1))
    prompt = f"Question: {question}\n\nDocuments:\n{body}"
    if notes:
        prompt += "\n\nNotes the search took:\n" + "\n".join(f"- {n}" for n in notes[:20])
    resp = await client.chat(ANSWER_SYSTEM, [ChatMessage(role="user", content=prompt)],
                             max_tokens=1500)
    return parse_reply(resp.text)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def is_correct(answer: str, expected: list[str]) -> bool | None:
    """True if any expected answer appears in `answer` (case and punctuation ignored); None when
    there is no known answer to check against."""
    if not expected:
        return None
    got = f" {_norm(answer)} "
    return any(f" {_norm(e)} " in got for e in expected if _norm(e))
