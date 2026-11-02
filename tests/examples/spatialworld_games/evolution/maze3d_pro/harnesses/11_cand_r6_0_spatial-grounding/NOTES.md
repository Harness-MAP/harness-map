# maze3d_pro rich harness — module map & levers

Rich harness for the SpatialWorld multi-floor 3D maze (maze3d_pro). The frozen agent
(Qwen3-VL-30B-A3B-Instruct) and the SpatialWorld engine are untouched; only this harness is
optimized. `build_agent(evaluator)` is the entrypoint the extended evaluator installs as
`ai_model_func`; the engine `Action` contract and the maze `success` scoring stay byte-compatible
with the prompt-only baseline. The parent genome carried forward here is the arm's elite
(`prop_r005_064a17/cand_r005_0_spatial-grounding`), not the 0.08 staged anchor the loop seeds
`$OUT_DIR` with — the anchor's pass-through would have thrown away the chart, the flank ladder and
the reckoning that this iteration's change is built on.

## axis
**The ROTATION, priced in decisions** (code lever: `tools/action.py`'s engine-envelope algebra +
`tools/plan.py`'s route rungs). Every ancestor in this lineage bought a multi-cell translation for
one decision through `granularity` and then paid a full decision for each 90 degrees of turning.
A rotation has the same knob and nobody had used it. This iteration gives the harness a rotation
*planner*: it decides which way to rotate and how far to rotate in one decision, where the parent
decided only "turn left or turn right, once".

## module map
| file | role |
|------|------|
| agent.py | ENTRYPOINT. `build_agent(evaluator)` returns the async per-step fn and opens ONE `plan.Pilot` for the episode; `compose_system_prompt()` recomposes the prompt from the .md priors. Unchanged this iteration. |
| tools/sight.py | ONE FRAME READ AS GEOMETRY: the corridor's length and kind down column 400, and the FLANK LADDER — the two neighbours of every corridor cell the ray proves open. Censused exact over 5624 poses / 24 938 band readings. Unchanged this iteration bar one `break`-free rewrite of the ladder prefix. |
| tools/plan.py | the chart those readings fill, the route over it, the ROTATION PLANNER, and the governor. `MazeChart` keeps PLAIN (a ray went through) strictly apart from PASSABLE-ONLY (a flank showed it empty, so it could still be the exit); that distinction is what the floor-exhaustion proof rests on. |
| tools/action.py | the key table, the engine `Action` metadata envelope, the bearing algebra and NEW the rotation algebra. |
| tools/vlm.py | the perceive -> measure -> state -> ask -> govern step around `evaluator.baseline_step`. Unchanged. |
| tools/frame.py | frame capture via the evaluator's own screenshot. Unchanged. |
| AGENT.md | role framing + the perception facts MEASURED on this build; NEW the one-cell limit of sideways vision. |
| rules.md | the legal action set, the granularity contract, the real climb precondition; NEW what a decision costs. |
| skills/navigation.md | the navigation priority order; NEW keep turning the same way. |
| memory.md | the per-step scratchpad TEMPLATE, rendered live by `Pilot.render_memory`. |

## change
Three things, all inside the same mechanism.

1. `tools/action.py` gains the rotation algebra — `pivot_to` (the one rotation onto a bearing, short
   way round), `pivot_path` (the headings it passes THROUGH), `heading_after`, `sweep_arc` — and
   `retarget` now stamps `metadata["repeat"]` on a turn. **Measured on the real
   `PygameInputSource`:** `turn_right` with `repeat` 1/2/3 and no `granularity` walks
   NORTH -> EAST / SOUTH / WEST in ONE `execute_action`, while `move_forward` with
   `granularity: large` and `repeat: 3` still advances a single stride (the adapter forces `repeat`
   to 1 whenever a granularity is present). The episode budget is the engine loop's own
   `step_count`, one iteration per `ai_model_func` call, so a rotation spent in one call costs ONE
   decision whatever its arc. No `granularity` key is written for a turn, so the response contract
   ("granularity iff a translation") holds — 4626/4626 emitted actions on the panel below.
2. `tools/plan.py` gains `pivot_step` + `blind_arc` + `ray_charted`. The harness sees a frame only
   at the heading it STOPS on, so spending an arc whole forfeits exactly the looks from its interior
   headings. `ray_charted` is the test that makes the compression free rather than a gamble: a look
   down a bearing is worthless only when every cell the ray can reach is already PROVEN PLAIN up to
   a charted blocker — an uncharted cell, or one known only from a flank band (passable, and so
   possibly the exit, since the exit paints no side face), and the arc is cut short so the next
   frame arrives from there. `PIVOT_MAX = 1` reproduces the ancestor exactly.
3. `look_bearing` replaces "the nearest bearing still unsettled" with ONE consistent rotation
   through all of them. Rotating one way passes every bearing in three quarters; taking the nearest
   pending bearing each time can double back and spend four.

Two safety changes came out of reading the losing trace rather than from the axis. `last_resort`
closes a hole the parent has had all along: when no exit is charted, nothing unseen is left, and
every stair's remaining climb leads to a floor already charted out, `plan()` used to return None and
the harness spent the rest of the episode executing whatever the frozen model happened to say — 12
decisions of the arm's one losing level, and 190 decisions of pure wandering on one lever
configuration below. It now drops the liveness test (the only guess in the stair rung) and walks
back through the nearest door with an untried climb. And `best_stairs` ranks the charted stairs
before routing to one, so the harness stops walking to the nearest stair when that stair is the one
it came up and has nothing left behind it.

## why (the gradient)
The brief was a SEARCH-SPACE STEER toward `cand_r002_1_action-planning` / `cand_r001_0_spatial-grounding`,
naming the **token-level implementation** as the productive difference. I read that as the direction
to vary and went looking for where the decisions actually go, with a rig rather than a guess.

**First I tried to widen the perception, and the frame refuted it.** The sibling lineage's "cone
lattice snap" reads every column of the frame as its own ray and charts the whole visible cone. I
built the band arithmetic for it and censused it against ground truth over **every pose of all 25
levels, both floors, four bearings (5624 poses)**. The lateral-1 family reproduced the shipped flank
ladder exactly (0 wrong at every corridor depth, residual |dp| 0.002 -> 0.27 at depth 9), which
validates the geometry — and **every band at lateral 2 or beyond was vetoed 100% of the time.** A
histogram of what the renderer actually paints settles it: over 462 622 column readings the implied
lateral offset of the stop **never once exceeds 0.555**, clustering uniformly over [0, 0.5] (front
faces) with a 39% spike at exactly 0.5 (the side faces of the lateral-1 cells). **This build paints
no cone. There is nothing two cells to the side to decode.** Replacing the ladder's
prefix-of-certainties with per-band verification off the slab's own top edge buys 25 205 readings
against 24 938 — **+1.1%**. The decode is saturated; `tools/sight.py` is unchanged this iteration.

**So the binding constraint is throughput, and the rig says where.** Over the 823 decisions the
parent spends on the six-tape panel: **402 strides (48.8%), 309 look-turns (37.5%), 87 route turns
(10.6%), 25 climbs (3.0%)**. Turns are 48% of the budget and the look-turn is the single largest
line item. Compressing every run of same-direction turns into one decision bounds the prize at
**100 of 823 (12.2%)**.

**And the one losing level is a near miss, measured exactly.** With the cap lifted, L25 needs **112**
decisions against a budget of 100 — the other 24 levels need 3 to 71. The loss is not a navigation
failure: at decision 77 the transit floor is charted out (163 cells, 0 frontier, 0 conflicts, pose
intact) and the harness is walking back to a stair it never reaches. The whole question was whether
the rotation is worth 12 decisions.

## measured (offline rig, real engine and real action path, stub proposer in place of the VLM)
`eval_runner`'s own seam: real `Maze3DProGame`, real `PygameInputSource.execute_action`, real
`build_conversation_history`, real 29-turn history, `harness.build_agent` installed exactly as the
loop installs it, levers overridden AFTER `build_agent` (it purges `sys.modules["tools*"]`). Six
degenerate proposers x 25 levels = 150 paired episodes per arm. **The control reproduces the
parent's published 144/150 at mean 32.92 decisions to the decimal**, which is what makes the deltas
below readable.

| configuration | wins | mean decisions | dead reckoning | contract |
|---|---|---|---|---|
| CONTROL — `PIVOT_MAX=1, SWEEP=False` (the parent) | 144/150 | 32.92 | 0/4938 wrong | 4938/4938 |
| arc <= 2 (half turn) | 144/150 | **30.84** | 0/4626 | 4626/4626 |
| **SHIPPED** — arc <= 3 | **144/150** | **30.84** | **0/4626** | **4626/4626** |
| SHIPPED minus the consistent sweep | 144/150 | 30.92 | 0/4638 | 4638/4638 |

**−2.08 decisions an episode, −6.3%, with the win set unchanged, zero map conflicts and zero
reckoning errors in 4626 audited decisions.** The `repeat` histogram shifts from {1: 2877, 2: 909}
to {1: 2251, 2: 1260}: 351 half-turns now ride one decision. **`PIVOT_MAX` is SATURATED, not tuned**
— `pivot_to` never asks for more than a half turn, so 3 is byte-identical to 2 on all 150 episodes.
The consistent sweep is worth 0.08 of that and is kept because it is part of the same decision and
never negative.

**Cross-checked END TO END through the REAL evaluator**, where only the model's text is stubbed:
the real `Maze3DProOpenAIEvaluator`, its own `openai_ai_model` as `baseline_step`, the real
`build_conversation_history` and 29-turn history, the real JSON parser and its fallback, the real
`PygameInputSource` and `EvaluationEngine`, and the scored `success` field. Six levels x three stub
arms (`forward`, `spin`, and one that never emits valid JSON at all, so every response falls through
to `default_move_forward`): **18/18 wins on each arm.** Paired VLM calls, parent -> shipped:

| level | parent | shipped | game's own `steps_taken` (shipped) |
|---|---|---|---|
| L01 | 3 | 3 | 3 |
| L02 | 16 | **14** | 16 |
| L13 | 22 | **18** | 20 |
| L17 | 8 | **7** | 8 |
| L21 | 30 | **28** | 30 |
| L24 | 71 | **67** | 75 |
| total | 150 | **137 (−8.7%)** | 157 |

The offline rig's decision counts above match this seam **to the unit** on all six levels, which is
why the panel numbers can be read as real. **Note the last column:** the game's own `steps_taken`
counter advances once per quarter turn, while the episode budget is the engine loop's `step_count`
(one iteration per `ai_model_func` call). So `eval_result.json` will now report
`steps_taken > vlm_calls` — 157 against 137 here — where every ancestor reported them equal. That
divergence is telemetry, not a budget: `success` is the scored field, `vlm_calls` is the decisions
actually spent, and the loop break is on `step_count`.

Per-level decisions NEEDED with the cap lifted, parent -> shipped: the total falls **835 -> 778**
and no level regresses. L02 16->14, L05 36->32, L06 4->3, L09 27->26, L10 26->24, L11 35->32,
L13 22->18, L14 58->53, L15 16->14, L17 8->7, L19 65->59, L20 24->21, L21 30->28, L23 45->39,
L24 71->67, **L25 112->107**. L25 is still over the 100 cap on this rig — the rotation closed 5 of
its 12-decision shortfall, not all 12 — and it remains the one loss.

## REFUTED this iteration, measured, do not rebuild
- **The off-axis cone.** The renderer paints nothing beyond lateral 0.555 (462 622 column readings).
  The cone decoder is not wrong here, it is *empty*: 100% of its claims are vetoed.
- **Per-band verification instead of the ladder's prefix rule.** +267 readings of 24 938 (+1.1%).
- **Planning the route in decisions instead of cells.** A pose-state BFS over `(cell, heading)`
  whose edges ARE the decisions (rotate 1-2 quarters, stride 1-3 charted cells) is the obvious
  companion to this axis and it **COSTS** 0.51 decisions an episode; routing to a fixed cell-chosen
  target with the same search costs 0.16. A 1-wide maze has one path between any two cells, so
  there is no turn-for-cell trade left for the search to find. Removed, not left switched off.
- **Ending the floor sweep near the door.** Ranking equally-near frontier targets by distance from
  the stair the sweep will depart through is INERT — ties at equal cell distance are too rare to
  carry it — and it cost a decision on L10. Removed.
- **Clipping a stride so it stops ON a cell that still owes a sideways look.** Exactly inert: the
  frontier-nearest objective already stops at every opening's mouth. Removed.
- **Learning the ceiling** (`CEILING_LEARNS`, left in the file switched OFF with its measurement).
  Ranking a floor change by whether the harness has stood on the floor it leads to really does kill
  every refused climb — 5 of 25 levels, 17 decisions, and SHIPPED-with-it reaches 761 — but **on its
  own it costs SIX WINS**, because reordering the climbs walks the per-visit epoch model into a
  state where its epoch and the engine's floor disagree and all four climbs of a two-stair floor are
  spent, after which the chart offers nothing at all and L23 becomes unwinnable at ANY budget (500
  decisions, still lost). A 0.6-decision gain is not worth a lever that can do that. Fix the epoch
  model first — see "what is still open".
- **Remembering per-FLOOR which climb direction the retirement test disproved.** A strictly worse
  version of the same idea: the twin test's false positives are contained to one (cell, direction)
  by `chart.tried`, and promoting one of them to a floor-wide veto is what made L23 unwinnable.

## a trap worth one line
`CLIMB_ACTIONS` is a frozenset and the order the two climbs are tried in is **load-bearing and not
alphabetical**. Refactoring `for name in ("climb_up", "climb_down")` into
`sorted(frozenset(CLIMB_ACTIONS) - spent)` silently puts `climb_down` first and costs **seven wins
and 6.3 decisions an episode**. `tools/action.py` now exports `CLIMB_ORDER` for this; iterate that,
never `sorted(CLIMB_ACTIONS)`.

## evolvable levers (axes)
- prose: AGENT.md / rules.md / skills/*.md / memory.md — what the agent is told.
- code: agent.py / tools/*.py — the mechanism.
- knobs, all CLASS attributes of `plan.Pilot` (a rig must set them AFTER `build_agent`):
  `PIVOT_MAX` (saturated at 2), `SWEEP`, `CEILING_LEARNS` (refuted, off), `OBEY_MODE`,
  `FLOOR_GATE`/`STALE_LIMIT`, `STRIDE_MAX`, `MAX_CLIMBS`, `TWIN_MATCHES`/`TWIN_WINDOW`,
  `STALL_LIMIT`, `LADDER`.
- `Sight.LADDER` / `Sight.MAX_DEPTH` / `Sight.SIDE_ROWS` — the decoder's own reach.

## what is still open
- **The epoch model is the next axis, and it is now the one with evidence behind it.** A floor is
  mapped per VISIT, so returning to a floor by a different stair starts a fresh map in a fresh
  coordinate frame and the knowledge does not re-join. Every level that needs two climbs pays for
  it, it is what makes `CEILING_LEARNS` unsafe, and it is why `last_resort` had to exist. The pose
  frame is already shared (a climb preserves (x, y), and all 60 stair tiles in these 25 levels are
  vertically paired), so one map per reckoned FLOOR is reachable — the only real work is holding a
  climb's looks in a buffer until the retirement test has decided which floor they belong to.
- L25 needs 107 decisions and has 100. Its transit floor costs 75 of them to chart out 163 cells at
  0.89 decisions per passable cell, which is close to the floor of this chart design; the remaining
  five have to come from somewhere other than the route or the decode, both of which are now
  measured out.
- A stride still cannot carry more than three cells: a translation with no `granularity` at all has
  its `repeat` forced to 1 as well (measured), so there is no envelope trick left on that side.

## anti-cheat
Decide from the observation only. Every field in the injected block is derived from the pixels the
model is already shown plus the agent's own action history. An AST scan of the whole genome finds
**no** access to `game`, `engine`, `input_source`, `tower_data`, `all_exits`, `px`/`py`/`pz`/`pDir`,
`game_won`, `distance_to_exit`, `get_state`, `maze_file` or `total_floors`, and no import of the
engine, the adapter or pygame. The harness reads exactly `baseline_step`,
`capture_game_screenshot`, `current_prompt`, `conversation_history` and `max_steps` off the
evaluator. `evaluator.baseline_step` is called on EVERY step, so the frozen model chooses every
decision and the VLM-call counter can never void the run. The ground-truth level data used in the
censuses above lives only in the offline rig, which `agent.py` never imports.

## files
tools/action.py, tools/plan.py, tools/sight.py, AGENT.md, rules.md, skills/navigation.md, NOTES.md
