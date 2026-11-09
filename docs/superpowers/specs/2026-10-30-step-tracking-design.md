# Step Tracking for the Search Loop

**Status:** approved by AJ in chat, 2026-10-30
**Kind:** library feature (`src/agentic_search/`), on its own branch and PR. The Ignite demo picks it up afterwards.
**Found by:** the Ignite WANDS match questions (`docs/superpowers/specs/2026-10-30-ignite-wands-design.md`). Evidence is in §2.

## 1. What and why

Some questions need two or more steps where a later step depends on something *read* in an earlier one:

- "I have the Schatzi rainbow wave coffee table. Find a bookcase with the same wood tone." First find the table and read `dswoodtone : light wood`. Then find light-wood bookcases.
- "Find the 2024 flood policy, then the claims that fall under its exclusions." First find the policy and read its exclusions. Then search claims for them. (AJ's real-world case. Claims also carry the policy number as a field, which is a known link; the exclusions are a learned link.)

Today the loop cannot do this. It values only documents that answer the question, so the stepping-stone document (the coffee table, the policy) is judged "not relevant", its facts never reach the planner, and the loop loses its place.

The feature: when a question needs steps, the loop keeps a short **step plan** and a **notebook of learned facts**, separate from the **results it returns**. Each round knows which step it is on.

## 2. Evidence (2026-10-30, WANDS match, 20 questions)

- Harness P@10 0.050, retrieval 0.005 (`demo/ignite/results-wands-match.md` on branch `demo/ignite-wands`).
- Trace, bookcase question: round 1 finds the coffee table; the judge marks it not relevant; controller says *broaden*. Round 2 fetches the table; the digest shows "1 hits, 0 new" and none of its text; the built-in stop rule stops ("no new relevant results").
- Planner digests show each hit as its first 240 characters (`Analyzer.snippet_chars`, not configurable). WANDS features come at the end of the text, so the wood tone is never shown.
- Throwaway spike (runtime patches, no files changed) that (a) kept a "facts from documents you opened" section in every digest, (b) showed fetched documents in full, (c) raised snippets to 800 chars, plus a stop rule that runs at least 3 rounds: P@10 0.03 → 0.10 on 6 questions. On the dining-table question the planner *did* read "medium wood" and regex-searched `dswoodtone : medium wood` (without a class filter); the judge called the results not relevant, the controller said *broaden*, and the planner went back to searching for the table. On the bookcase question it ignored the facts and searched "Schatzi" four more times.
- Conclusion: showing more text is necessary but not enough. The missing piece is a sense of "which step am I on", shared by planner, judge and controller.

Why MuSiQue hid this: its gold set includes every supporting paragraph, so stepping stones *are* answers and get judged relevant.

## 3. Goals

1. **Step plan.** For a question that needs steps, the loop holds an ordered list of steps (a sentence each), marks the current one, and moves on when it is done. A one-step question has no plan and runs as today.
2. **Notebook.** Facts learned from documents (for example `wood tone: light wood`, `exclusions: storm surge, sewer backup`), each with the document it came from. Kept across rounds. Shown to the planner every round. Not part of the returned results.
3. **Reading shows content.** When the agent fetches a document, the planner sees its text (bounded), whether or not the document is new or relevant.
4. **Stepping stones are not noise.** A document that served a step (it gave a fact) is not treated as a failure, and does not by itself trigger *broaden*.
5. **Progress includes steps.** Finishing a step or learning a fact counts as progress for the stop rule. The loop does not stop while a step is open and budget remains, unless it is stuck.
6. **Known and learned links mix.** Any step may use any existing tool, including `filter_search` and `traverse` for known links. No new graph work.
7. **Visible.** Steps, the current step and learned facts appear as events, in the trace, in the TypeScript client and in the Ignite viewer, so a room can watch "step 2 of 3: bookcases with light wood".
8. **Snippet length is a setting** (today fixed at 240).

## 4. Non-goals

- No new search tools, and no word boosting (raised separately as its own library idea).
- No change to retrieval mode (one planned pass) or to `bm25`.
- No changes to the eval datasets or their labels.
- No per-step budgets. The existing budget covers the whole search.

## 5. Anti-goals (a failure even if it works)

- **Plain questions pay for it.** A one-step question must not get slower, pricier or worse. Target: no extra model call when there is no plan (the plan comes back with the first planning call, not from a separate call), and published WANDS cost per query within 10% of today.
- **Tuned to our questions.** No prompt or rule may mention WANDS, wood tone, features or any dataset. The WANDS match question set stays fixed.
- **Silent behaviour change for existing users.** The feature is off by default (a `HarnessSettings` field). The Ignite demo turns it on.
- **Facts the documents don't say.** Every notebook fact names the document it came from; a fact with no source is dropped.

## 6. Acceptance

Measured with the Ignite eval (`demo/ignite/public/eval_public.py`), one run per query, same models and settings as today, feature on vs off:

| Set | Must |
|---|---|
| WANDS match (20) | harness P@10 clearly up. Goal: ≥ 0.30 (today 0.05). If it falls short, report and come back to AJ; do not change the questions. |
| MuSiQue (60) | harness recall@10 not worse beyond noise (today 0.792; a drop of more than ~0.03 is a finding). |
| Published WANDS (60) | harness nDCG@10 not worse beyond noise (today 0.776) and cost per query within 10% (today $0.0273). |

Plus: library tests (`uv run pytest -q`) and lint stay clean; new behaviour covered offline with the scripted driver and keyword judge in `agentic_search.testing`; event changes reflected in `clients/typescript`.

## 7. Decided (as AJ decided it)

- A real library feature, not a demo trick.
- Separate library branch and PR (`feat/step-tracking`); the demo branch `demo/ignite-wands` resumes on top of it afterwards.
- Graph search handles known links; step tracking handles learned links; a question may need both.

## 8. My calls (not discussed with AJ)

- Off by default, on in the Ignite demo.
- No separate "split the question" model call; the planner returns the step plan (or none) in its first planning output.
- Each notebook fact must cite a document.
- Acceptance targets in §6, including the 0.30 goal and the 10% cost bound.
- The spike-era demo changes (a min-turns stop rule and a WANDS description tip) are stashed on the demo branch (`git stash list`: "ignite-wands: min-turns stop rule …") and not used. This feature replaces them. Dropping the stash is AJ's call.

## 9. Settled after the builder check (my calls)

1. **What "relevant" means stays the same.** The judge's relevance score keeps meaning "relevant to the whole question", and only that score decides the returned results. Usefulness to a step is a separate signal, used for the notebook and for progress. So a stepping stone is returned only if it is also relevant to the whole question: the coffee table is not (it is not a bookcase); a MuSiQue bridge paragraph usually is (as today). This keeps both acceptance rows reachable.
2. **Loop mode only.** Step tracking applies to the planned loop. In delegate mode the setting has no effect, and the docs say so.
3. **Protocols change only by optional additions.** New fields on `PlannerView`, `PlanResult` and `ControllerView` get defaults that keep today's behaviour; `Decider.judge` does not change. Fact extraction sits with the planner (it already sees the digest), so non-LLM judges (cross-encoder, TypeSafe) and custom drivers keep working. A driver that returns no plan simply runs without steps.
4. **Where acceptance is measured.** The library PR is complete when the offline tests pass. Before it merges, the demo branch is rebased onto `feat/step-tracking` and the three evals in §6 run there (they need Vertex and cost about $4). The merge into `main` waits for those numbers and for AJ. The Ignite viewer part of goal 7 is built on the demo branch, not in the library PR.

## 10. Left to the builder

How the plan and notebook are stored in the search state; the exact prompt wording; who extracts facts (planner or judge); how the judge and controller use the current step; event names and fields; the snippet and fetch-text limits; the order of the work.
