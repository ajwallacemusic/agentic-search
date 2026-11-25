"""SID-2 Swarm as a model-mode (delegate) driver for the Ignite demo.

SID's server runs every agent of the swarm: it forks agents, passes their messages and decides
when they wait (the fork / message_agents / wait / list_agents tools are SID's; we must not
define them). We run the search tools the agents call and fuse their reports. This follows SID's
reference implementation (https://platform.sid.ai/docs/sid-2-swarm/reference-implementation):

- one `response.create` over a WebSocket to `/v1/responses` with header
  `OpenAI-Beta: responses_multi_agent=v1` and `multi_agent: {enabled, max_agents, ...}`;
- each `function_call` (search / text_search / read / report_helpful_ids) is answered with a
  `response.inject`; a valid report ends that agent (`end_agent: true`);
- a fork gives each new agent a fork of its parent's `DocumentCache` (sid-sdk), which renders
  results in the format SID-2 was trained on and remembers what each agent has seen;
- at `response.completed` the reports are fused with RRF (k=60); SID does not rank.

Searches go through the harness runtime (`call_hits`, library model mode), so they land in the
pool, the trace and the tool-call/cost/time budgets like any other mode. `search` is our
`vector_search` (hyde_text = the query), `text_search` is `lexical_search`. `read` is served from
the agent's cache: a document can be read once any agent's search returned it.

Usage: SID's docs do not say which event carries token counts. Any `usage` object found on an
event is added up; if none arrives, the cost reads 0 and the result note says so.
Key: `SID_API_KEY`, else the macOS Keychain item `sid-api-key`.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import subprocess
from collections.abc import AsyncIterator, Callable
from typing import Any

from agentic_search.core.types import Budget, Hit, ModelUsage, Query, text_of
from agentic_search.models.base import (
    DelegateResult,
    PlannerView,
    PlanResult,
    ToolCall,
    ToolRuntime,
    ToolSpec,
)

BASE_URL = "https://api.sid-1.com/v1"
MODEL = "SID-2-Swarm"
SNIPPET_FIELD = "content"

INSTRUCTIONS = """You are an expert research assistant that is given a question and must use the provided search tools to find all documents needed to answer the question.

Steps:
1. Reflect on what information is needed to answer the question and use the search tools to find documents. Each document has an id.
2. Repeat step 1 until all documents necessary and sufficient to answer the question have been found. Take as many turns and searches as needed – you can make multiple searches per turn! Most questions will require multiple turns. Many will need more.
3. Use the report_helpful_ids tool to report the most helpful document ids. List the most helpful document ids first (important!).

The episode ends once report_helpful_ids is called."""

LAST_TURN_MESSAGE = (
    "This is your last turn: you must call report_helpful_ids now (an empty list is allowed). "
    "Anything you have found but not reported is lost when this turn ends."
)

BUDGET_MESSAGE = ("[budget] The search budget is used up: call report_helpful_ids now with "
                  "the most helpful ids you have found.")

_SNIPPET_SIZE = {"type": "integer", "description": (
    "Words shown per result snippet (default 50). Must be a positive integer. "
    "Pass a larger value to see more of each document.")}
_RANGES = ("Result ids may be ranged references 'id#start:end' naming the exact character span "
           "of the document shown; stretches you have already been shown are replaced by "
           "'[seen: \"#start:end\"]' markers instead of repeating, and doc_length gives the "
           "document's total character count.")

# The SID-2 tool schemas, unchanged from the reference implementation.
TOOLS: list[dict[str, Any]] = [
    {"type": "function", "name": "search",
     "description": ("Semantic (vector) search over the dataset. Best for conceptual or paraphrased "
                     "queries: a long, descriptive query of what you want beats a few keywords; use "
                     "text_search when you know the distinctive words a match would contain. "
                     + _RANGES),
     "parameters": {"type": "object", "properties": {
         "query": {"type": "string", "description": "The query to search the dataset with"},
         "limit": {"type": "integer", "description": "The number of results to return"},
         "snippet_size": _SNIPPET_SIZE}, "required": ["query"]}},
    {"type": "function", "name": "text_search",
     "description": ("Full-text keyword search over the dataset, ranked by BM25. Provide the terms "
                     "you want matched as plain words separated by spaces: the distinctive content "
                     "words a matching document would contain. " + _RANGES),
     "parameters": {"type": "object", "properties": {
         "query": {"type": "string",
                   "description": "Plain space-separated content words to match (BM25)."},
         "limit": {"type": "integer", "description": "The number of results to return"},
         "snippet_size": _SNIPPET_SIZE}, "required": ["query"]}},
    {"type": "function", "name": "read",
     "description": ("Search results show only short snippets. Read up to 10,000 characters of a "
                     "document per call. A bare ID starts at character 0; a ranged reference "
                     "starts at its requested offset. Use further ranged reads to continue."),
     "parameters": {"type": "object", "properties": {"id": {"type": "string", "description": (
         "The id of the document to read. Also accepts a ranged reference 'id#start:end' "
         "(character offsets, e.g. '123456#800:1600') to read only that part of the document's "
         "text; a range running past the end of the document is trimmed to the part that "
         "overlaps it.")}}, "required": ["id"]}},
    {"type": "function", "name": "report_helpful_ids",
     "description": ("Report the most helpful document ids from the search results. This is "
                     "required in the last step. Each id may be reported at most once; repeated "
                     "ids are penalized. Ids may be whole documents ('123456') or ranged "
                     "references ('123456#800:1600'); a ranged reference counts as its whole "
                     "document, so there is no need to strip the '#start:end' part."),
     "parameters": {"type": "object", "properties": {"ids": {
         "type": "array", "description": "The ids of the most helpful documents from the search results.",
         "items": {"type": "string"}}}, "required": ["ids"]}},
]

TOOL_EXAMPLES = [
    {"caption": "Example: a semantic search with a long, descriptive query:", "name": "search",
     "arguments": {"query": "the county that contains the capital of the state where the "
                            "officer was born", "limit": 5}},
    {"caption": "Example: a keyword search:", "name": "text_search",
     "arguments": {"query": "teak patio dining table natural finish", "limit": 5}},
    {"caption": "Example: reading one character range:", "name": "read",
     "arguments": {"id": "placeholder_1#800:1600"}},
    {"caption": "Example: ending the episode with a report:", "name": "report_helpful_ids",
     "arguments": {"ids": ["placeholder_1", "placeholder_2", "placeholder_3"]}},
]


class ToolMistake(Exception):
    """A mistake the model made (bad id, bad arguments): sent back to it as the tool output."""


def api_key(env: str = "SID_API_KEY", keychain_item: str = "sid-api-key") -> str:
    key = os.environ.get(env)
    if key:
        return key
    try:
        return subprocess.check_output(["security", "find-generic-password", "-s", keychain_item,
                                        "-w"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"no SID API key: set {env} or add Keychain item {keychain_item}") \
            from exc


def rrf(rankings: list[list[str]], k: int = 60) -> list[str]:
    """SID's fusion: score 1/(k + rank + 1), rank from 0; empty reports abstain."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, key in enumerate(ranking):
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda key: (-scores[key], key))


def _usage_in(event: dict[str, Any]) -> list[dict[str, Any]]:
    """Every `usage` object on an event (top level, response, item); SID does not document which."""
    found = []
    for holder in (event, event.get("response") or {}, event.get("item") or {}):
        u = holder.get("usage") if isinstance(holder, dict) else None
        if isinstance(u, dict):
            found.append(u)
    return found


def _source_of(tools: list[ToolSpec]) -> str:
    for t in tools:
        if t.name == "lexical_search":
            enum = t.parameters.get("properties", {}).get("source", {}).get("enum") or []
            if enum:
                return str(enum[0])
    raise ValueError("SID swarm needs a source with lexical search")


@contextlib.asynccontextmanager
async def openai_connect(key: str, base_url: str = BASE_URL) -> AsyncIterator[Any]:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url=base_url, api_key=key)
    async with client.beta.responses.connect(
            extra_headers={"OpenAI-Beta": "responses_multi_agent=v1"}) as conn:
        yield conn


class SidSwarmDriver:
    supports_images = False

    def __init__(self, *, model: str = MODEL, max_agents: int = 6, agent_max_turns: int = 10,
                 reward: str = "ndcg", default_limit: int = 5, vector_field: str = "embedding",
                 price_per_mtok: tuple[float, float, float] = (0.50, 0.05, 2.00),
                 key: str | None = None, base_url: str = BASE_URL,
                 connect: Callable[[], Any] | None = None, poll_s: float = 0.5,
                 grace_s: float = 8.0, est_cost_per_action: float = 0.0):
        if not 1 <= max_agents <= 24:
            raise ValueError("max_agents must be 1..24")
        self.model, self.max_agents, self.agent_max_turns = model, max_agents, agent_max_turns
        self.reward, self.default_limit, self.vector_field = reward, default_limit, vector_field
        self.price_in, self.price_cached, self.price_out = price_per_mtok
        self._key, self.base_url, self._connect, self.poll_s = key, base_url, connect, poll_s
        self.grace_s = grace_s  # after the budget runs out: refuse searches, take reports
        # SID-2 Swarm sends no token counts (checked live 2026-11-25; usage is only on the
        # dashboard), so cost is estimated per model action (tool or multi-agent call).
        self.est_cost_per_action = est_cost_per_action
        self.id = f"sid:{model}"
        self.events: list[dict[str, Any]] = []  # every server event of the last run, for checks

    async def plan(self, view: PlannerView, tools: list[ToolSpec]) -> PlanResult:
        raise NotImplementedError("SID-2 Swarm runs only in model mode")

    def cost(self, u: ModelUsage, cached: int) -> float:
        fresh = max(0, u.input_tokens - cached)
        return (fresh * self.price_in + cached * self.price_cached
                + u.output_tokens * self.price_out) / 1e6

    async def run_delegate(self, question: Query, tools: list[ToolSpec], runtime: ToolRuntime,
                           budget: Budget, context: str = "") -> DelegateResult:
        from sid import DocumentCache

        run = _Run(self, runtime, _source_of(tools), DocumentCache(range_mode="lenient"))
        connect = self._connect or (lambda: openai_connect(self._key or api_key(), self.base_url))
        self.events = []
        async with connect() as conn:
            run.conn = conn
            await run.send({
                "type": "response.create", "model": self.model, "input": question.as_text(),
                "instructions": INSTRUCTIONS, "tools": TOOLS, "max_output_tokens": 4096,
                "sid": {"tool_examples": TOOL_EXAMPLES, "reward": self.reward},
                "multi_agent": {"enabled": True, "max_agents": self.max_agents,
                                "agent_max_turns": self.agent_max_turns,
                                "last_turn_message": LAST_TURN_MESSAGE}})
            try:
                await run.loop()
            finally:
                for task in run.tasks:
                    task.cancel()
        ranked = rrf([r for r in run.reports.values() if r])
        notes = [f"{len(run.caches)} agents, {run.calls} tool calls, "
                 f"{sum(1 for r in run.reports.values() if r)} reports"]
        if run.stopped:
            notes.append(f"stopped early: {run.stopped}")
        if not run.saw_usage and self.est_cost_per_action:
            notes.append(f"cost estimated: {run.actions} model actions x "
                         f"${self.est_cost_per_action} (SID sends no usage)")
        elif not run.saw_usage:
            notes.append("SID reported no token usage; cost reads 0")
        return DelegateResult(ranked_keys=ranked, usage=run.usage, note="; ".join(notes))


class _Run:
    """One swarm run: answers tool calls concurrently, keeps one cache per agent path."""

    def __init__(self, driver: SidSwarmDriver, runtime: ToolRuntime, source: str, root_cache: Any):
        self.d, self.runtime, self.source = driver, runtime, source
        self.conn: Any = None
        self.response_id: str | None = None
        self.caches: dict[str, Any] = {"/root": root_cache}
        self.reports: dict[str, list[str]] = {}
        self.rendered: dict[str, asyncio.Future[None]] = {}
        self.tasks: set[asyncio.Task[None]] = set()
        self.failed: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self.lock = asyncio.Lock()
        self.calls = 0
        self.usage = ModelUsage()
        self.cached = 0
        self.saw_usage = False
        self.actions = 0
        self.stopped: str | None = None

    async def send(self, message: dict[str, Any]) -> None:
        async with self.lock:
            await self.conn.send(message)

    async def loop(self) -> None:
        receive: asyncio.Future[Any] | None = None
        try:
            grace_until: float | None = None
            while True:
                if grace_until is None and self.runtime.budget_exhausted():
                    self.stopped = "budget"
                    grace_until = asyncio.get_running_loop().time() + self.d.grace_s
                if grace_until is not None and asyncio.get_running_loop().time() >= grace_until:
                    return
                if receive is None:
                    receive = asyncio.ensure_future(self.conn.recv_bytes())
                done, _ = await asyncio.wait({receive, self.failed}, timeout=self.d.poll_s,
                                             return_when=asyncio.FIRST_COMPLETED)
                if self.failed.done():
                    self.failed.result()
                if receive not in done:
                    continue
                raw, receive = receive.result(), None
                event = json.loads(raw)
                if self.handle(event):
                    return
        finally:
            if receive is not None:
                receive.cancel()

    def handle(self, event: dict[str, Any]) -> bool:
        """Act on one server event; True when the swarm is done."""
        self.d.events.append(event)
        self.add_usage(event)
        kind = event.get("type")
        agent = (event.get("agent") or {}).get("agent_name") or "/root"
        item = event.get("item") or {}
        if kind == "response.created":
            self.response_id = event["response"]["id"]
        elif kind == "response.output_item.done" and item.get("type") == "function_call":
            self.estimate(1)
            self.answer_later(agent, item)
        elif kind == "response.output_item.done" and item.get("type") == "multi_agent_call":
            self.estimate(1)
        elif kind == "response.output_item.done" and item.get("type") == "multi_agent_call_output":
            children = json.loads(item["output"][0]["text"])["agents"]
            for path, child in zip(children, self.caches[agent].fork(len(children)), strict=True):
                self.caches[path] = child
        elif kind == "response.completed":
            return True
        elif kind in ("response.failed", "error"):
            err = (event.get("response") or {}).get("error") or event.get("error")
            raise RuntimeError(f"SID swarm failed: {err}")
        return False

    def estimate(self, n_actions: int) -> None:
        """Estimated cost of model actions, when SID sends no usage (see est_cost_per_action)."""
        self.actions += n_actions
        if self.saw_usage or not self.d.est_cost_per_action:
            return
        step = ModelUsage(cost_usd=n_actions * self.d.est_cost_per_action)
        self.usage = self.usage.plus(step)
        self.runtime.report_usage(step)

    def add_usage(self, event: dict[str, Any]) -> None:
        for u in _usage_in(event):
            self.saw_usage = True
            n_in, n_out = int(u.get("input_tokens", 0)), int(u.get("output_tokens", 0))
            cached = int((u.get("input_tokens_details") or {}).get("cached_tokens", 0))
            step = ModelUsage(input_tokens=n_in, output_tokens=n_out)
            step = step.model_copy(update={"cost_usd": self.d.cost(step, cached)})
            self.usage, self.cached = self.usage.plus(step), self.cached + cached
            self.runtime.report_usage(step)

    def answer_later(self, agent: str, item: dict[str, Any]) -> None:
        previous = self.rendered.get(agent)
        rendered: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self.rendered[agent] = rendered
        task = asyncio.ensure_future(self.answer(agent, item, previous, rendered))
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    async def answer(self, agent: str, item: dict[str, Any],
                     previous: asyncio.Future[None] | None, rendered: asyncio.Future[None]) -> None:
        try:
            name = item.get("name")
            out: dict[str, Any] = {"type": "function_call_output", "call_id": item["call_id"]}
            try:
                fetched: tuple[str, int, list[Hit]] | None = None
                error: ToolMistake | None = None
                try:
                    args = json.loads(item.get("arguments") or "{}")
                    if name in ("search", "text_search"):
                        fetched = await self.search(name, args, item["call_id"])
                except ToolMistake as e:
                    error = e
                except json.JSONDecodeError:
                    error, args = ToolMistake("arguments were not valid JSON"), {}
                if previous is not None:
                    await previous  # this agent's earlier calls render first (seen-ledger)
                cache = self.caches[agent]
                try:
                    if error is not None:
                        raise error
                    if name == "report_helpful_ids":
                        self.reports[agent] = report(cache, args.get("ids"))
                        out["end_agent"] = True
                    elif name == "read":
                        out["output"] = read(cache, args.get("id"))
                    elif fetched is not None:
                        out["output"] = render(cache, *fetched)
                    else:
                        raise ToolMistake(f"unknown tool {name!r}")
                except ToolMistake as e:
                    out["output"] = str(e)
            finally:
                if not rendered.done():
                    rendered.set_result(None)
            self.calls += 1
            await self.send({"type": "response.inject", "response_id": self.response_id,
                             "input": [out]})
        except Exception as e:  # the backend or the socket failed: fail the swarm
            if not self.failed.done():
                self.failed.set_exception(e)

    async def search(self, name: str, args: dict[str, Any], call_id: str) -> tuple[str, int, list[Hit]]:
        query = args.get("query", "")
        if not isinstance(query, str) or not query:
            raise ToolMistake("search requires a non-empty `query`. To fetch a document by id, "
                              "use the `read` tool.")
        limit, size = args.get("limit", self.d.default_limit), args.get("snippet_size", 50)
        if not isinstance(limit, int) or not isinstance(size, int) or limit <= 0 or size <= 0:
            raise ToolMistake(f"Invalid {name} arguments: limit and snippet_size must be "
                              "positive integers")
        if name == "search":
            call = ToolCall(id=call_id, name="vector_search", arguments={
                "source": self.source, "field": self.d.vector_field,
                "hyde_text": query, "limit": limit})
        else:
            call = ToolCall(id=call_id, name="lexical_search", arguments={
                "source": self.source, "text": query, "limit": limit})
        [outcome] = await self.runtime.call_hits([call], count_turn=False)
        if outcome.error is not None and outcome.text.startswith("[budget]"):
            raise ToolMistake(BUDGET_MESSAGE)
        if outcome.error is not None:
            raise ToolMistake(outcome.text)
        return query, size, outcome.hits


def _document(hit: Hit) -> dict[str, Any]:
    doc: dict[str, Any] = {SNIPPET_FIELD: text_of(hit.content)}
    title = hit.metadata.get("title")
    if isinstance(title, str) and title:
        doc["title"] = title
    return doc


def render(cache: Any, query: str, snippet_size: int, hits: list[Hit]) -> str:
    """The reference `render_results`, with our hits: data id = the hit key."""
    out = []
    for hit in hits:
        if hit.key not in cache:
            cache.add_document(hit.key, _document(hit))
        view = cache.apply_snippet(hit.key, SNIPPET_FIELD, query, snippet_size=snippet_size)
        out.append(view.render_xml())
        cache.update_seen(view)
    return "\n".join(out) if out else "No results found. Try a different query."


def _parse(reference: Any) -> tuple[str, tuple[int, int] | None]:
    from sid import parse_rendered_model_facing_id

    try:
        return parse_rendered_model_facing_id(reference)
    except (ValueError, TypeError) as e:
        raise ToolMistake(str(e)) from None


def read(cache: Any, reference: Any) -> str:
    """The reference `read`: up to 10,000 characters of a document some search returned."""
    from sid import InvalidCharacterRange

    mid, span = _parse(reference)
    if not cache.contains_model_facing_id(mid):
        raise ToolMistake(f"Invalid document id '{mid}'. You can only read documents returned by "
                          "search. Use 'search' or 'text_search' first.")
    data_id = cache.to_data_id(mid)
    if span is not None:
        try:
            start, end = cache.resolve_char_range(data_id, SNIPPET_FIELD, span)
        except InvalidCharacterRange as e:
            raise ToolMistake(f"{e}. Re-read '{mid}' without a range.") from None
        span = (start, min(end, start + 10_000))
    elif len(cache.get_document(data_id)[SNIPPET_FIELD]) > 10_000:
        span = (0, 10_000)
    view = cache.get_single_span_document_view(data_id, SNIPPET_FIELD, snippet_display_span=span)
    text = view.render_xml()
    cache.update_seen(view)
    return str(text)


def report(cache: Any, ids: Any) -> list[str]:
    """The reference `report`: deduplicated, as our hit keys; unknown ids dropped."""
    if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
        raise ToolMistake("ids must be a list of strings")
    seen: list[str] = []
    for ref in ids:
        mid, _ = _parse(ref)
        if mid not in seen:
            seen.append(mid)
    return [cache.to_data_id(m) for m in seen if cache.contains_model_facing_id(m)]


__all__ = ["SidSwarmDriver", "rrf"]
