# cand_r037_0_spatial-grounding — middleware pipeline with a code-planner decision stage

## axis:
Component graph / decision locus (code). The parent (r029) makes one model call per step on a prompt
grounded by a neighbour-ledger table, then applies a single veto. This candidate follows the steer
toward r018's structure: an ordered MIDDLEWARE STACK of peer stages that share only a per-step
`StepContext`. Each stage has `pre(ctx)` (may answer before the model) and `post(ctx, action)` (may
rewrite a proposal). What is new relative to r018 is that the innermost stage, **NavStage**, is a
BFS planner that owns movement. The model is consulted only when no stage answers.

## change:
- `tools/context.py` (ported from r018 and extended): observation → StepContext, holding the cursor,
  the `map:` cells (row = cursor.y), the message, the adjacent-noun prose and the legal set. It adds
  `shared` (harness-owned cross-step dict), `blocked`, `unblock`, `chart`, `source`, and a `lethal(d)`
  method (a lava/water noun, or a `}` drawn or charted).
- `tools/stages.py` (new). Stack, outer to inner:
  - **Legal**: audits the vocabulary.
  - **Modal**: answers `[yn]` with `y` and `--More--` with `more`, with no model call.
  - **Door**: on `This door is locked.`, kicks as a two-step verb (`kick`, then the bearing when
    `In what direction?` appears) and keeps that edge unconvicted.
  - **Bump**: a (square, dir) edge is booked ONLY on a refusal message (`It's a wall`, `solid stone`,
    `in vain`, `bars`, diagonal) or an empty message. It is never booked on a swing, a creature
    target, or an open prompt. Three unexplained no-moves in a row also book it.
  - **Hazard**: vetoes lava and water.
  - **Foe**: a letter glyph on an adjacent square gets a swing, with no retreat.
  - **Nav**: keeps a per-level chart of remembered glyphs and stood-on squares. It sets a `bars` flag
    on Boxoban's bars message. Every step it re-elects a goal: a drawn `>` if reachable, else the
    nearest unstood square bordering unseen space, else the nearest unstood square. It emits the
    first BFS hop, respecting every convicted edge on every hop. Order is distance, then (y, x), a
    total order. It emits `down` when standing on the charted `>`. When nothing reachable is
    unexplored, it `search`es up to 10 times per square before handing off to the model.
  - **Orbit**: applies only to model proposals.
- `agent.py:PipelineHarness.act` runs ALL pre hooks so every stateful stage observes every frame.
  The first legal answer wins and its stage becomes `ctx.source`. If there is no answer, it calls the
  model (reply snapped to the earliest legal mention). Post hooks then run inner to outer, and the
  provenance trail is written to `response.reasoning`. Stage exceptions are caught per stage.
- `build_agent` is unchanged.
- Prompting keeps the naive transcript plus priors plus a "Checks on this frame" note block. The
  ledger table was removed.
- AGENT.md and memory.md were rewritten to describe the new division of labour.

## why:
The r029 diagnosis names the binding constraint: the no-reasoning model ignores the fact table
(63.6% of picks contradict it). The planner-locus lines (r001, r036) score 0.35–0.375, against the
parent's 0.150. It also names four HARNESS-BUGs, each of which now maps to one stage:
- swings booked as failed moves → a message-classified Bump;
- the 46-frame `[yn]` lock → Modal;
- oscillation with no swing beside a rat → Foe;
- a blank `Reasoning` column → the trail.

Replay cost falls sharply as well. Model calls happen only on frames no stage answers, and on a
synthetic 15x7 maze smoke test the pipeline reached `>` in 27 steps with 0 model calls. Predicted
gains are on MazeWalk and Corridor-R3 (orbits become BFS exploration). Boxoban and Quest are not
expected to move.

## module map / levers
| file | role | lever |
|---|---|---|
| agent.py | pipeline driver: all-pre, first-answer, post, provenance | code |
| tools/context.py | observation → StepContext (+ shared state fields) | code |
| tools/stages.py | 8 stages + `default_stack()` order; Nav = the planner | code |
| tools/action_io.py | legal-set parse, whole-word mentions, `snap` | code |
| tools/prompting.py | priors + stage notes assembly (model fallback only) | code |
| AGENT.md / memory.md / rules.md / skills/ | prose for fallback frames | prose |

## files:
agent.py, tools/context.py (new, from r018), tools/stages.py (new), tools/action_io.py (+snap),
tools/prompting.py, AGENT.md, memory.md, NOTES.md
