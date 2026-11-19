# Ignite demo: a third "swarm" lane

Date: 2026-11-19. Branch: `demo/ignite-swarm`. Builds on
`2026-10-29-ignite-demo-design.md` and `2026-10-30-ignite-wands-design.md`.

## Why

SID released SID-2 Swarm (https://www.sid.ai/models, technical report at
https://www.sid.ai/research/sid-2-technical-report): a team of search agents that split one
question, search at the same time in separate contexts, and merge their reported documents with
RRF (reciprocal rank fusion). SID trains the teamwork with RL. Their report says that when the
agents stop talking to each other, the swarm scores about the same as RRF over independent runs.

We want two things (AJ, 2026-11-19):

1. **Show it on stage** as a third lane next to `retrieval` and `harness` in the Ignite viewer.
2. **Measure it**: does a prompt-built swarm beat the single harness loop on our three question
   sets (`musique`, `wands`, `wands-match`)?

We do not train anything. This is the untrained version of the idea: split, fan out, fuse.

## Goals

- A `swarm` mode: one planner call splits the question into sub-questions; one harness loop per
  sub-question, all at the same time; RRF merges their ranked hits into one top 10.
- A third viewer lane that shows it live: the sub-questions, each agent's progress, the merged
  top 10, time and cost. Replay works for it like the other two lanes.
- A `swarm` row in the eval results for all three sets, next to `bm25`, `retrieval`, `harness`.
- Time budget raised from 60 s to 120 s for **all** modes (AJ, 2026-11-19), so the comparison
  stays fair.

## Non-goals

- No agent-to-agent messages, shared board, or coordinator. That was option 2; not now.
- No change to the library (`src/agentic_search/`). The swarm lives in the demo folder.
- No training and no new model provider.

## Anti-goals (a failure even if it works)

- Breaking or slowing the existing `retrieval` and `harness` lanes or their replays.
- An external URL in the viewer (`test_no_external_urls`: the demo must run offline).
- Hiding a bad result. If the swarm loses (it may, on chain questions), the results file and the
  tuning log say so plainly. That is a fine talk point: a swarm splits work that runs side by
  side, but MuSiQue hops and WANDS match steps are chains, where step 2 needs step 1's answer.
- Gold marks visible before the lane finishes (same rule as the other lanes).

## Approaches considered

1. **Split the question (picked).** One planner call cuts the question into 3–6 sub-questions;
   each agent runs the harness loop on one. Closest to SID's "divide and conquer" and gives the
   stage lane something to show. Costs one extra call; on chain questions the split may guess
   later steps wrong.
2. **Same question, different tools.** Every agent gets the full question but a different tool
   or style (keyword only, vector only, ...). No extra call, but it is really just fusion and
   does not look like a team.
3. **Coordinated swarm with a shared board** (from the first discussion). Closer to SID, 3–5
   days, may not beat 1. Deferred: do 1 first and see if the idea helps at all.

AJ picked 1 on 2026-11-19.

## Design

### Constraints from today's code

- The viewer posts to the library server's `/v1/search/stream` with `mode` = `retrieval` | `harness`
  and reads the SSE event stream (`viewer/index.html`, `requestBody`, `MODES`). `swarm` is not a
  library mode, so it needs its own demo endpoint.
- Planner is `gemini-4-argon` (slow; 45 s request timeout). Harness runs take 45–55 s today.
  On 2026-11-10, running work in parallel on the same model slowed calls and dropped scores.
  So each swarm agent gets a smaller budget than a single harness run.
- `config.yaml` `budget.max_seconds: 60`; viewer shows "⚠ Past 60 s" (`overtime`, line ~418/699).

### Flow

```
question
  └─ split (1 planner call) → sub-questions s1..sN   (N from 3 to 6; config default max 4)
       ├─ agent 1: harness loop on s1 ─┐
       ├─ agent 2: harness loop on s2 ─┤  all at the same time
       └─ agent N: harness loop on sN ─┘
            └─ RRF over the N ranked hit lists → merged top 10 + all agents' noted facts
```

- **Split.** One call to the planner model with the question and source descriptions. Returns
  JSON: a list of short sub-questions. If the call fails or returns fewer than 2, fall back to
  one agent on the original question (the lane says so).
- **Agent query.** Each agent searches with the text "Main question: <q>. Your part: <s_i>.", so
  the planner and judge know the bigger goal but focus on one part.
- **Agent budget: fair spend (AJ, 2026-11-19).** The whole swarm gets the same total budget as
  one harness run (`budget` in `config.yaml`: 24 tool calls, $0.50, 120 s). The split call's
  cost comes out first; the rest is divided evenly over the N agents (tool calls 24 / N rounded
  down, at least 1; cost (0.50 − split cost) / N). So a swarm win means it used the budget
  better, not that it spent more. Turns per agent: `max_turns: 2` (our call: turns are about
  time, and four 4-turn agents would not fit 120 s). New config block `swarm:` with
  `max_agents` (default 4), `agent_max_turns` (default 2), `rrf_k` (default 60). The whole swarm
  run is cut at the shared 120 s; agents still running are cancelled and their hits so far are
  kept (from the stream's snapshot events). If the library cannot give one search its own
  budget on a shared `Harness`, build one `Harness` per agent with the divided budget.
- **Merge.** Standard RRF: score(doc) = Σ 1 / (rrf_k + rank_i(doc)) over agents that returned
  it. Ties keep first-seen order. Notes = all agents' noted facts, in agent order (feeds the
  answer box, like the harness lane).
- **Usage.** Cost and tokens = split call + sum of agents. Time = wall clock. Turns reported as
  the max over agents (and per agent in the lane).

### Code

- `demo/ignite/swarm.py`: the split call, the fan-out over one shared `Harness` (concurrent
  `harness.stream`/`search` calls), RRF, and an async event stream for the lane.
- `demo/ignite/serve.py`: new SSE endpoint `POST /ignite/swarm` with the same request body as
  `/v1/search/stream` (`profile`, `question`, `snapshot_k`, `include_trace`). Events: a `split` event
  (sub-questions), each agent's library events wrapped with its agent index, a final `done` event
  with the merged hits, notes and usage.
- `demo/ignite/eval.py` / `public/eval_public.py`: a `swarm` mode. A small runner object with
  the same `search` shape `TurnRecorder` and `run_eval` expect, returning merged hits and usage.
- `demo/ignite/run_side_by_side.py`: can run and `--save` the swarm too, for replay files.
- `config.yaml`: `budget.max_seconds: 120`; new `swarm:` block.

### Viewer lane

Third column, same look as the other two (Egen tokens). Rows are agents, not tools:

```
┌ Swarm ───────────────────────────── 34.2 s ┐
│ split → 4 agents → merge (RRF)             │
│ agent  sub-question             turns hits │
│ 1      Where was A. Deveaux…    ●●   12    │
│ 2      Capital of that state…   ●○   9     │
│ 3      City bordering it…       ●●   14    │
│ 4      County of that city…     ●    7     │
│ merged top 10 ...  (★ gold after finish)   │
│ cost $0.21                                 │
└────────────────────────────────────────────┘
```

- Overtime line moves from 60 s to 120 s for all lanes.
- Answer box and "check it yourself" work on the merged top 10, as for the other lanes.
- The three columns must still fit the stage screen. The builder may make columns narrower or
  stack the swarm lane under the others if three do not fit; say which in the tuning log.

### Testing

- Offline unit tests with fake models (like `tests/helpers.py`): split parsing and fallback,
  RRF math, budget cut, event order, endpoint shape, eval runner shape.
- `viewer.test.mjs`: the swarm lane renders from a saved event file; no external URLs.
- Live: run each eval set once, **one set at a time** (not in parallel). Then rehearse the
  heroes (`hero`, `awning`, `trays`, `match-hero`) live in the viewer, 2× each, and save backup
  replay files.

### Success

- The swarm lane runs live on the four hero questions inside 120 s, and replays work.
- `results-musique.md`, `results-wands.md`, `results-wands-match.md` each have a `swarm` row,
  re-run with 120 s for every mode. The new numbers **replace** the 60 s numbers (AJ,
  2026-11-19). The harness numbers will move too, so update the README tables and the
  talk numbers that quote them (for example WANDS match harness P@10 0.515), and say in the
  tuning log that the budget changed.
- The verdict banner at the top of the viewer covers all three modes (our call). Old two-lane
  replay files still load; the swarm column then shows "no saved file for this mode".
- A tuning-log round says what the swarm won and lost, and what it cost.

### Spend and timing (our call)

- Log the dollar spend of each eval run in the tuning log.
- Aim to finish the live runs before 2026-12-01: the Argon price is reported to double then.

## Left to the builder

- Exact split prompt wording and JSON schema.
- How agent events are shaped on the wire, as long as the viewer can show the lane above.
- Column layout details for three lanes.
- Whether the eval runs agents with lower concurrency to avoid the shared-model slowdown; if so,
  say so in the results file.
