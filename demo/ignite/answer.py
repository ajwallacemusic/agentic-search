"""The demo's answer line: one fast model call reads a mode's top documents and answers.

The library returns ranked documents, not answers. On stage, an answer read from each mode's own
top 10 makes the difference visible: the same reader, given different evidence. Not scored by the
evals; for MuSiQue the viewer also shows whether it matches the dataset's answer."""

from __future__ import annotations

import re
from typing import Any

from agentic_search.models.llm import ChatMessage

ANSWER_SYSTEM = (
    "Answer the question using only the numbered documents given. Do not use outside knowledge. "
    "Reply in one short sentence. If the question asks for products, name up to three matching "
    "products from the documents. If the documents do not contain the answer, reply exactly: "
    "Not found in these documents.")
MAX_DOC_CHARS = 1200


async def answer_from(client: Any, question: str, docs: list[tuple[str, str]]) -> str:
    """`docs` = (title, text) pairs, best first."""
    if not docs:
        return "No documents to answer from."
    body = "\n\n".join(f"[{i}] {title}\n{text[:MAX_DOC_CHARS]}"
                       for i, (title, text) in enumerate(docs, 1))
    prompt = f"Question: {question}\n\nDocuments:\n{body}"
    resp = await client.chat(ANSWER_SYSTEM, [ChatMessage(role="user", content=prompt)],
                             max_tokens=200)
    return resp.text.strip()


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def is_correct(answer: str, expected: list[str]) -> bool | None:
    """True if any expected answer appears in `answer` (case and punctuation ignored); None when
    there is no known answer to check against."""
    if not expected:
        return None
    got = f" {_norm(answer)} "
    return any(f" {_norm(e)} " in got for e in expected if _norm(e))
