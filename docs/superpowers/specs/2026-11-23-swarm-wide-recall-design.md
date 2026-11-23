# Swarm validation, part A: questions where the work splits

Date: 2026-11-23. Branch: `demo/ignite-swarm` (builds on the swarm mode from
`2026-11-19-ignite-swarm-design.md`). Part B (agents that talk, a shared board) is a separate
design, written after this one runs.

## Why

The 2026-11-19 swarm lost on every set we have (tuning log, round 15). All three sets are chain
questions: step 2 needs step 1's answer. SID's report says swarms win on wide "find every item
that fits" searches and gain little on chain sets (HotpotQA +0.05). So we have not yet tested the
case a swarm should win.

AJ, 2026-11-23: "I want to validate that swarm can work." This is a fair test, not a stage
feature. The question: **on questions whose parts do not depend on each other, does our swarm
beat the single harness loop at the same budget?**

## Goals

- A new eval set, `wands-multi`: about 20 questions. Each joins 3–4 real WANDS shopper queries
  from different product classes into one shopping task.
- Run `retrieval`, `harness`, `swarm` (fair budget) and `swarm-4x` (4× budget) on it, one mode
  at a time, and write `results-wands-multi.md` plus a tuning-log round.
- Give a plain verdict against the pass mark below.

## Non-goals

- No viewer or stage change. If the swarm wins, putting it on stage is a later step.
- No change to the swarm algorithm (split, fan-out, RRF stay as built). That is part B.
- No change to the library (`src/agentic_search/`).

## Anti-goals (a failure even if it works)

- **Gold made by a model.** Gold must come from the WANDS human labels only.
- **A set tuned until the swarm wins.** The questions are built once, by a script with a fixed
  seed, before any mode runs. No re-rolling the seed after seeing results.
- **Hiding a loss.** If the swarm loses here too, the results file says so plainly.
- Breaking the existing sets, modes or replays.

## Approaches considered

1. **New data where the work splits (this doc).** Cheap (1–2 days, about $5–10), and it tests
   SID's main claim with no change to the swarm.
2. **Agents that talk to each other (part B).** 3–5 days; SID only got it to work with training.
3. **More budget only.** Tests "more spend wins", not the swarm idea. Folded into 1 as the
   `swarm-4x` control.

AJ picked both 1 and 2 on 2026-11-23, in the order 1 then 2.

## Design

### The data

- Source: the WANDS files already in `demo/ignite/public/.cache/wands/` (`query.csv`: 480
  queries with `query_class`; `label.csv`: human labels Exact / Partial / Irrelevant;
  `product.csv`). The loaders are in `demo/ignite/public/datasets.py`.
- A **part** is one WANDS query. A **question** joins 3–4 parts.
- Rules for picking parts (our call):
  - Every part has a different `query_class`, and no class appears in the excluded list
    `NOT_PRODUCT_TYPES` from `wands_match.py`.
  - Every part has at least 3 Exact products. Otherwise one part could be impossible.
  - No WANDS query is used in more than one question.
  - Fixed seed. The script writes `demo/ignite/wands_multi.yaml` (questions, part query ids,
    gold ids per part), which is committed. The YAML is the frozen set.
- Wording (our call): a fixed template with no model. Example: "I'm shopping for a few things:
  a teak outdoor chair; a patio umbrella; an outdoor rug." The parts are the WANDS query texts,
  unchanged. This makes the split easy on purpose. That is the case under test.
- **Gold** for a question = the union of the parts' Exact products. Each gold product keeps its
  part id, so we can score by part.

### Scores

- **Part coverage@10** (headline): the share of a question's parts with at least one Exact
  product in the top 10. With 4 parts: 0, 0.25, 0.5, 0.75 or 1.
- **Part recall@30** (our call, after the builder check): many WANDS queries have more than 30
  Exact products (median about 42), so plain recall@30 would mostly measure the biggest part.
  Instead, score each part on its own and take the mean over the parts. A part's score = its
  Exact products in the top 30 ÷ min(its Exact count, ⌊30 ÷ number of parts⌋). Cap each part's
  score at 1. This needs a 30-item result list from every mode. All modes get the same `top_k`.
  `eval.py` passes `k=10, recall_k=10` today; the new set needs 30 (the library allows it).
- A product that is Exact for more than one part counts for each of those parts.
- Also report nDCG@10 (Exact = 2, Partial = 1, as in the `wands` set), cost, seconds and tool
  calls per question, as in the other results files.

### Modes and budget

- `retrieval`, `harness`, `swarm`: as in round 15. Budget is `config.yaml`'s `budget` (24 tool
  calls, $0.50, 120 s). For the swarm this total is shared by all agents.
- `swarm-4x`: the same swarm with a total of 96 tool calls and $2.00. Time stays at 120 s, and
  agent turns stay at 2. Eval-only. It is a control: it shows whether extra spend explains a win.
- Run one mode at a time and one set at a time, with swarm agents at concurrency 1 as in round 15.

### Pass mark (set before the run)

Gap = `swarm` part coverage@10 − `harness` part coverage@10 (absolute). Check in this order
(our call, after the builder check):

1. **The swarm works:** gap ≥ 0.10, and `swarm` part recall@30 is not lower than `harness`.
   (If `swarm-4x` also passes, the verdict stays "works"; report the 4× numbers next to it.)
2. **Only spend works:** `swarm` does not pass rule 1, but `swarm-4x` does (same rule, with
   `swarm-4x` in place of `swarm`).
3. **Unclear:** gap ≥ 0.05, but rule 1 failed (the gap is 0.05–0.10, or the gap is ≥ 0.10 but
   part recall@30 is lower). Say whether 20 more questions are worth the cost.
4. **The swarm does not work here:** gap < 0.05.

With 20 questions this bar is rough. The results file says so.

### Code (left to the builder where not named)

- `demo/ignite/public/wands_multi.py`: builds and loads the set. It reuses `datasets.py`
  helpers in the same way `wands_match.py` does.
- `demo/ignite/public/eval_public.py`: a `wands-multi` dataset and the `swarm-4x` mode. The new
  scores live with the existing metric code.
- Offline tests: picking rules (distinct classes, at least 3 Exact, no reuse, same output for
  the same seed), gold union and part ids, both scores on hand-made hit lists, and the 4× budget
  maths.

### Spend and timing

- Expected: about $5–10 for all four modes. Log the real spend in the tuning log.
- Finish before 2026-12-01, when the Argon price is reported to double.

## Left to the builder

- Exact template wording, and whether a question has 3 or 4 parts (a mix is fine).
- Where the scores sit in the code, and the results file layout (follow `results-wands.md`).
- Whether `top_k` 30 needs a library-side check (the library is not changed. If 30 is not
  allowed, stop and report).
