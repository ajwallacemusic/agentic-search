"""Workspace tools: the planner's step plan and notebook, applied to the search state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agentic_search.core.state import SearchState
from agentic_search.events import FactNoted, StepsUpdated
from agentic_search.models.base import WORKSPACE_NAMES, Fact, Step, ToolCall, ToolSpec

MAX_STEPS = 6
MAX_FACT_CHARS = 300
MAX_FACTS_PER_TURN = 8
MAX_NOTEBOOK = 30

WORKSPACE_TOOLS: list[ToolSpec] = [
    ToolSpec(
        name="plan_steps",
        description=("Set the plan for a question that needs several steps, where a later search "
                     "depends on something read earlier. Call once, early. Skip it for one-step "
                     "questions."),
        parameters={"type": "object",
                    "properties": {"steps": {"type": "array", "items": {"type": "string"}}},
                    "required": ["steps"]}),
    ToolSpec(
        name="note",
        description=("Record a fact read in a document, for use in later searches. source_key is "
                     "the document's `source:doc_id`."),
        parameters={"type": "object",
                    "properties": {"fact": {"type": "string"}, "source_key": {"type": "string"}},
                    "required": ["fact", "source_key"]}),
    ToolSpec(
        name="step_done",
        description="Mark step N (1-based) as done.",
        parameters={"type": "object", "properties": {"step": {"type": "integer"}},
                    "required": ["step"]}),
]


@dataclass
class WorkspaceUpdate:
    n_facts: int = 0
    n_steps_done: int = 0
    changed: bool = False
    plan_set: bool = False


def apply_workspace(state: SearchState,
                    calls: list[ToolCall]) -> tuple[list[ToolCall], WorkspaceUpdate]:
    """Apply workspace calls to `state`; return the other calls unchanged and in order."""
    update = WorkspaceUpdate()
    steps_changed = False
    rest: list[ToolCall] = []

    def drop(c: ToolCall, reason: str) -> None:
        state.trace.add("workspace_dropped", state.turn, call=c.name, reason=reason)

    for c in calls:
        if c.name not in WORKSPACE_NAMES:
            rest.append(c)
        elif c.name == "plan_steps":
            raw = c.arguments.get("steps")
            if state.steps:
                drop(c, "plan already set")
                continue
            texts = [s.strip() for s in raw if isinstance(s, str) and s.strip()] \
                if isinstance(raw, list) else []
            if not texts:
                drop(c, "no usable steps")
                continue
            state.steps = [Step(text=t) for t in texts[:MAX_STEPS]]
            update.changed = steps_changed = update.plan_set = True
        elif c.name == "note":
            text, key = c.arguments.get("fact"), c.arguments.get("source_key")
            if not isinstance(text, str) or not text.strip():
                drop(c, "empty fact")
            elif not isinstance(key, str) or key not in state.pool:
                drop(c, "unknown source_key")
            elif any(f.text == text.strip()[:MAX_FACT_CHARS] and f.source_key == key
                     for f in state.notebook):
                drop(c, "duplicate fact")
            elif update.n_facts >= MAX_FACTS_PER_TURN:
                drop(c, "too many facts this turn")
            elif len(state.notebook) >= MAX_NOTEBOOK:
                drop(c, "notebook full")
            else:
                state.notebook.append(Fact(text=text.strip()[:MAX_FACT_CHARS], source_key=key,
                                           turn=state.turn))
                update.n_facts += 1
                update.changed = True
                state.emitter.emit(FactNoted, turn=state.turn, fact=state.notebook[-1].text,
                                   source_key=key)
        else:  # step_done
            n: Any = c.arguments.get("step")
            if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= len(state.steps):
                drop(c, "step out of range")
            elif state.steps[n - 1].done:
                drop(c, "step already done")
            else:
                state.steps[n - 1].done = True
                update.n_steps_done += 1
                update.changed = steps_changed = True
    if steps_changed:
        state.emitter.emit(StepsUpdated, turn=state.turn,
                           steps=[st.model_copy() for st in state.steps],
                           current=state.current_step())
    return rest, update
