# maze3d_pro rich harness — module map & levers

Rich harness for the SpatialWorld multi-floor 3D maze (maze3d_pro), frozen agent
Qwen3-VL-30B-A3B-Instruct. `build_agent(evaluator)` is the entrypoint the extended evaluator
installs as `ai_model_func`; the engine `Action` contract, the legal action set, the JSON response
format and the maze `success` scoring stay byte-compatible with the prompt-only baseline, and
`evaluator.baseline_step` still makes every VLM call, parse and fallback. Parent:
`anchor_maze3d_pro` (the staged rich anchor — a pass-through that holds no state and reads no
pixels; 2/25 live, 8/150 on the panel below).

## axis

**code / spatial-grounding — PERCEPTION WIDTH: the flank LADDER, charted.**

Every ancestor in this lineage that read the pixels at all read the two cells beside the **body** —
the pair of wide quads at the screen edges — and nothing else sideways. This one reads the two
neighbours of **every corridor cell the forward ray proves open**, and files them all into the
dead-reckoned chart. One look therefore classifies 4.43 cells instead of 2, a cell crossed by a
stride arrives already surrounded by settled map, and the frontier collapses from "every corridor
cell I walked through" onto the openings that are genuinely unexplored.

The steer asked to move toward the `cand_r002_1_action-planning` / `cand_r003_0_spatial-grounding`
region and named the productive difference as the **token-level** half of φ, so this iteration lands
in that region's component graph — the same six modules those two ship (`agent.py` +
`tools/{__init__,action,frame,plan,sight,vlm}.py`), nothing added, split or rewired — and differs
inside the components, which is where the steer pointed: the ruler the decode needs lives as class
attributes of the class that uses it, the decision helpers are module-scope functions over a
`Pilot` rather than methods of it, and the measured constants are integers in the frame's own pixels
instead of fractions. Measured move, in the steered direction on every token feature:

| f_AST / f_num | parent anchor | **this** | r002_1 | r003_0 |
|---|---|---|---|---|
| components    | 7      | **30**    | 29    | 21    |
| cf_mean       | 1.43   | **7.43**  | 4.90  | 9.10  |
| cf_max        | 4      | **78**    | 45    | 60    |
| cf_std        | 1.40   | **14.76** | 8.11  | 16.61 |
| num_per_comp  | 0.29   | **7.07**  | 3.45  | 9.33  |
| logmag_mean   | 0.347  | **0.952** | 0.643 | 0.852 |
| logmag_std    | 0.347  | **1.384** | 0.948 | 1.286 |

The structure block stays in the region as well (37 nodes / 35 edges against r002_1's 36 / 33), which
is what "differ at the token level, not in the wiring" means in practice.

## change

### 1. `tools/sight.py` (NEW) — the frame, read as geometry

Nothing here is read off the draw code (the proposer sandbox blocks the engine source, and a census
is the stronger warrant anyway: it would catch a renderer change, which a transcribed constant would
not). The game was driven to **every walkable cell of all 25 levels, both floors, all four bearings
— 5624 poses** — each frame decoded by the shipped module and scored against where the body could
actually go.

* **Distance ahead = the HEIGHT of the paint down the centre column.** Whatever blocks the corridor
  is centred on the horizon and scales as 1/distance: the run down column 400 that is neither
  ceiling `(15,15,35)` nor ground `(50,60,65)` measures 401/403 px at one cell, 201/203 at two,
  133/135 at three, … 37 at eleven — disjoint rungs, each `FACE_SPAN / (rows - 1)` on the nose.
* **Kind = the colour of that run**, voted over the whole run rather than sampled: `red == 0` is the
  teal stair face, `blue == 0` the single un-fogged gold `(255,200,0)` exit, anything else a brown
  wall. A single sampled pixel is what reads a distant stair as a wall; the vote does not.
* **THE LADDER.** The neighbours of the corridor cell `k` steps ahead are painted into the band from
  `SIDE_SPAN // k` to `SIDE_SPAN // (k+1)` off the centre line — 100..250, 250..300, 300..325,
  325..340, 340..350 for k = 0..4 on the left, mirrored right. The bands are disjoint and ordered,
  so a face at one distance cannot be read as a face at another, and the band MIDPOINT at the
  horizon dodges both the 1 px face outline and the stair's 3 px frame. A wall beside the corridor
  is `(160,85,60)` fogged, a stair `(0,100,120)` fogged, and an **empty neighbour paints nothing**,
  so bare ground or ceiling shows through.
* **The occlusion gate is `k <= depth - 1`**, where `depth` is the nearest FRONT face — *not* the
  walkable run. The band for `k = depth - 1` ends exactly where that face begins, so the gated bands
  are precisely the ones nothing can cover. Gating on the walkable run instead reads a front face as
  a side one wherever the blocker is itself steppable (a stair or the exit); the gate removes that
  entirely. Three rows are sampled per band and unanimity is required, so a clipped band is a
  refusal rather than a guess.

**Census result: forward ray `(kind, distance)` 5624/5624 exact; flank ladder 24938/24938 exact, 0
wrong and 0 gated band refused; the stride never over-reaches the engine's own walkable run (0 of
5624). Yield 4.43 cells classified per look against 2 for the one-cell ancestors.**

### 2. `tools/plan.py` (NEW) — the chart those readings fill

`MazeChart` holds one occupancy map per EPOCH (a namespace per floor entered) in a single reckoned
frame, and keeps two kinds of "open" strictly apart, which is the whole soundness argument:

* **PLAIN** — a cell a ray passed *through*, or one the body stood on. The ray stops at the first
  cell that is not plain floor (a wall, a stair **and the exit** all terminate it), so every cell it
  passes through is proven not to be the goal.
* **PASSABLE-ONLY** — a cell a FLANK band showed empty. The renderer paints the exit only as a front
  face, so an exit cell beside the corridor paints nothing and reads exactly like open floor. Such a
  cell is enterable and nothing more.

A cell is SETTLED when it is a wall, a stair, the exit, or proven plain, so "no reachable frontier
left and no exit charted" **proves** the exit is not on this floor — and that is what makes the floor
change free. The ladder therefore does not weaken the proof: it settles WALLS and STAIRS from a
distance (which is what kills the phantom frontier behind a stride) while a flank-read opening stays
unsettled until a ray goes down it.

`Pilot` reckons the pose (a turn is never refused; `sight.reach` states a translation's travel before
it is spent), files each look, and routes by BFS over charted-passable cells in the one order this
lineage has priced: **a charted exit → a frontier → only once the floor is exhausted, a stair.** A
floor change is not observable — a real climb can leave the frame byte-identical — so it is committed
optimistically into a fresh epoch and settled afterwards from per-pose digests: one mismatch confirms
it, two matches retire the epoch as a refused climb and restore the map underneath. All 60 stair
tiles in the 25 levels have a matching tile on the other floor (60/60), so charting the landing cell
as a stair is a measurement, not an assumption.

**`clamp_translation` is the new reckoning guarantee.** An obeyed forward stride is shortened to the
part the frame accounts for and a backward one to a single cell, so the travel is exactly known
before the move and settled exactly after it. It can only shorten a stride that would carry the body
*past* the stair or exit terminating the ray — the one place a longer stride is unwanted anyway,
since a climb works only from on top of the tile. It is **measured inert on the six-tape panel**
(144/150 and the identical override/decision means with and without it), so it is a soundness
guarantee for proposals this panel's tapes never make, not a source of the gain.

### 3. The prose priors, rewritten to the facts this build has

`AGENT.md` drops the HUD that this build does not draw (no `POS`/`FLOOR`/`DIR`, no `[Q] UP` /
`[E] DOWN`, no text of any kind) and states the nested-rectangle corridor, the three materials, the
exit-only-ahead rule, the never-drawn tile underfoot, and the side faces the ladder reads. `rules.md`
keeps the legal set and the granularity contract byte-identical and corrects the climb precondition
to its real one (a stair underfoot, which the MEASURED STATE block states), plus stride economics as
economics. `skills/navigation.md` is a priority order over the measured block and says explicitly
that a wall ahead means TURN — `move_backward` keeps the bearing, so stepping back lands you in the
previous cell still facing the wall and the next forward walks straight back into it, a two-cycle
this lineage has measured. `memory.md` is the live runtime template rendered into the system turn.

## why (the gradient)

`diagnosis.md` for the parent anchor: primary **2/25**, 23 of 23 losses absorbed into a terminal
periodic orbit, 0 climbs in 2340 decisions, and §5.1/§5.2/§5.3/§5.6 marked CONFIRMED — measure the
outcome, delete the HUD prose, wire the per-turn channel, read the corridor exactly. Those are
reproduced here rather than re-derived.

The new part answers the **first open item in the best sibling's own notes**: *"Clock, not routing, is
now the binding constraint. The cost is one decision per corridor cell, because a lateral band only
reads the cell the agent is in: a corridor's side cells cannot be settled from a distance, so
completeness costs a visit. Anything that settles a cell's neighbours without standing on it buys
levels directly. Price it on decisions-per-settled-cell."* That is exactly what the ladder does, and
it is priced on that metric below as well as on `success`.

## verification

**Rig.** A closed-loop stub panel: the REAL evaluator, game, renderer, `build_conversation_history`,
29-turn image history, `extract_action_info_from_response`, parse fallback, `PygameInputSource`,
`EvaluationEngine` and `success` field — only the model's TEXT is stubbed, with six context-free
tapes (always-forward-small, always-forward-large, always-turn-right, always-climb-up, an `L,F,F,U`
cycle and an `F,F,R,D,F` cycle). Deterministic, no network, 150 episodes in ~3 min. Its two traps,
both hit: the evaluator resolves the level file **relative to the SpatialWorld root** (a rig that
does not `chdir` there reports `total_floors == 1` and zero exits while still "completing"), and
`build_agent` purges `sys.modules["tools*"]`, so **every lever override must be applied AFTER the
call** or the control silently measures the shipped config.

| harness | panel (25 levels x 6 tapes) | mean decisions |
|---|---|---|
| `anchor_maze3d_pro` (the parent) | **8 / 150** | 95.4 |
| this harness with `Pilot.LADDER = 0` (the one-cell ancestor) | **132 / 150** | 49.3 |
| `cand_r004_1_action-planning` (the best sibling, same rig) | **132 / 150** | 49.3 |
| **this harness as shipped** | **144 / 150** | **32.9** |

`LADDER = 0` reproduces the best sibling's total **and its mean decision count to the first
decimal**, on the identical win set `{1..18, 20..23}` — so the ablation is a true control and the
only thing this candidate's code adds is the ladder. With it: **+12 / −0 of 150**, the won set is
`{1..24}` on **all six tapes** (so the mechanism is invariant to what the proposer says, which is the
property that matters against an unseeded `temperature=1.0, top_p=0.9` sampler), and the flips are
**L19 and L24 on every arm**. The single remaining loss is L25 on every arm.

**Throughput, the metric the gradient asked for** (means over all 150 episodes):

| | LADDER = 0 | shipped |
|---|---|---|
| decisions per settled cell | 0.585 | **0.353** |
| cells settled per cell stood on | 3.24 | **6.49** |
| cells stood on | 28.6 | **14.7** |
| cells settled | 80.0 | **88.7** |
| decisions on the 22 levels both win | 42.4 | **26.7** |

**Soundness and cost.** Over all 150 episodes: map conflicts **0**, episodes where the reckoned pose
was dropped **0**, blocked translations **0**. The decode costs **6.5 ms per decision** (the frame
md5 is most of it) against a ~2000 ms model call. Blinding the decoder so every frame reads
unreadable gives **0/50 with no exception on two tapes — exactly the anchor's score on those tapes**:
the mechanism goes inert, never harmful, if the renderer ever stops drawing side faces.

**LIVE, real Qwen3-VL-30B-A3B-Instruct through `eval_runner.py` with `ANTICHEAT_GUARD=1`** —
`oracle_access` null, `used_rich_harness` true, exactly one VLM call per decision on every episode
(`vlm_calls == steps_taken`). Five levels: the two the ladder flips on the rig, the rig's one
remaining loss, and the two regression traps this lineage has lost before.

| level | anchor live | `cand_r004_1` live (published) | this harness live |
|---|---|---|---|
| L19 | loss | loss, 100 steps | **WIN, 65 steps, floor 2** |
| L24 | loss | loss, 100 steps | **WIN, 71 steps** |
| L25 | loss | — | loss, 100 steps, distance 9 |
| L13 | loss (start-floor exit) | WIN, 27 steps | **WIN, 22 steps** |
| L02 | loss, floor 1 | WIN, 18 steps | **WIN, 16 steps, floor 2** |

Both levels the best sibling reports as its only live losses convert, and neither regression trap
moves. These are single episodes against an unseeded sampler — the deterministic six-tape panel is
the load-bearing evidence; the live runs are the confirmation that it transfers.

## module map

| file | role |
|------|------|
| agent.py | ENTRYPOINT. `build_agent(evaluator)` returns the async per-step fn and opens ONE `tools.plan.Pilot` per episode, threading it into `tools.vlm.decide`; `compose_system_prompt()` recomposes the prior from the .md files. `evaluator.game` does **not** exist yet at this point — `run_evaluation` builds it after the call — so nothing may touch engine state here, and nothing does. |
| **tools/sight.py** | **NEW.** `Sight`: one frame read whole — the ruler as class attributes, the corridor's distance from the centre-column run height, its kind from a colour vote over that run, the FLANK LADDER with its occlusion gate, the frame digest, and the lines that state all of it in words. |
| **tools/plan.py** | **NEW.** `MazeChart` (per-epoch occupancy with PLAIN vs PASSABLE-ONLY provenance, frontier test, BFS), `Pilot` (reckoning, filing, the climb/epoch bookkeeping with the digest retirement test, the governor, the per-turn block), and the decision helpers as module-scope functions over a pilot: `route_step`, `choose_target`, `clamp_translation`, `inert_reason`, `escape_step`. |
| tools/action.py | the key table, the engine `Action` metadata envelope (`retarget` strips `granularity`/`repeat`/`json_data` before rebuilding a turn — a stale `repeat: 3` fires THREE 90-degree turns) and the bearing algebra (0..3 = N,E,S,W, so `turn_right` is `+1 mod 4`). |
| tools/vlm.py | the perceive → measure → state → ask → govern step. Still calls `evaluator.baseline_step` every step; owns `_inject_memory`, which APPENDS to a stashed pristine `conversation_history[0]` (truncate-then-append deletes the JSON response contract from every later call and breaks the parser). |
| tools/frame.py | frame capture via `evaluator.capture_game_screenshot`. Unchanged. |
| AGENT.md / rules.md / skills/navigation.md / memory.md | the prose priors, rewritten to this build's measured perception (see §change 3). |
| system_prompt.txt / system_prompt.md | **NOT levers** — stale inert copies; `compose_system_prompt()` reads neither, and eval_runner's `initial_prompt` is overridden by the composed `current_prompt`. |
| NOTES.md | this file — proposer/diagnostician facing, never shown to the agent. |

## evolvable levers

- **prose**: AGENT.md / rules.md / skills/*.md / memory.md — what the agent is told.
- **code**: agent.py / tools/*.py — the mechanism (decode, chart, route, govern, inject).
- **tuning** (class attributes on `plan.Pilot`; set them **after** `build_agent`):

| lever | shipped | measured |
|---|---|---|
| `LADDER` | 6 (`Sight.LADDER`) | **0 → 132/150 at 49.3 decisions** (the ancestor); 6 → 144/150 at 32.9. This is the axis; the 6th band is 7 px wide and the gate usually stops the ladder before it. |
| `OBEY_MODE` | `route` | inherited from the sibling that priced it: `route` 132/150, `coverage` 76/150, `off` 34/150 over six tapes. Do not re-sweep without re-pricing the whole panel. |
| `FLOOR_GATE` | `exhaust` | the proof ("no frontier left") is the one with evidence behind it; `stale` is implemented and unswept here. Climbing at the first opportunity is refuted five times over in this lineage — it costs the start-floor exits. |
| `STRIDE_MAX` | 3 | sibling-priced at 1 → 20/25. Unchanged here. |
| `TWIN_MATCHES` / `TWIN_WINDOW` | 2 / 6 | sibling-priced: 1 retires a REAL floor change (19/25). Load-bearing, do not lower. |
| `MAX_CLIMBS` | 8 | an inert safety bound, not a knob. |
| `STALL_LIMIT` | 4 | only reachable once the pose is dropped, which happened in 0 of 150 episodes. |

## anti-cheat

Decide from the observation only. The pilot's single input is `frame_data.image`, the frame the agent
is shown; every field in the injected block derives from those pixels plus the agent's own action
history. A `pre_eval_gate` replica over all seven .py files finds **no privileged attribute or
subscript** (`tower_data`, `all_exits`, `distance_to_exit`, `current_floor`, `maze_file`,
`get_state`, …) and **no `.game` access at all**, no file under the ground-truth tree is opened, and
the agent-facing priors mention none of it. The RGB constants and the band geometry are properties of
the **renderer**, obtained by censusing frames the agent itself receives — the same thing a sighted
agent sees, not episode ground truth. Live runs with `ANTICHEAT_GUARD=1` report `oracle_access` null
on all five episodes.

## what is still open (ranked, for the next proposer)

1. **L25 is the whole residue and it is a pure clock loss.** It ends at the cap with 223 cells
   charted, **all 223 settled**, 0 conflicts, the pose intact and one floor change made — it explores
   correctly and runs out of decisions on the largest level (46 cells stood on, 59 moves, 41 turns).
   This lineage records it as winnable in ~106 decisions. Do **not** attack it with a turn-weighted
   Dijkstra: that was built, exhaustively validated and measured **win-neutral** elsewhere, and the
   premise (zigzag routes) was refuted — the turns are *objective switches*, not route shape. The
   untried lever is an **itinerary**: commit to an ordered chain of frontiers instead of re-electing
   a single target every decision.
2. **The ray is one cell wide; the renderer casts 800 of them.** Every column is a ray, so one look
   charts a cone rather than a line (a sibling lineage measured 5.37 cells/look against 4.20 and
   −18% planner decisions). It must be done with a lattice snap — read the cell off the face the ray
   stopped on, never off the raw intersection — and claims on the axis must be dropped in favour of
   the whole-run colour vote `Sight._corridor` already does.
3. **A refused climb still costs a re-exploration** when the retirement test finds no comparable
   pose inside its window (it fired on none of the floor changes here). A disambiguation that does
   not need the same pose twice would remove the last optimistic assumption in the mechanism.
4. **`move_backward` is unreadable by construction** — nothing is drawn behind the body — which is
   why it is clamped to one cell and why the prose tells the model to turn instead. A chart-backed
   reversal (step back only into a cell already charted plain) is cheap and untried.

## files

Edited this iteration: `tools/sight.py` (new), `tools/plan.py` (new), `tools/action.py` (bearing
algebra + the Action envelope), `tools/vlm.py` (the measure→state→ask→govern loop), `tools/__init__.py`,
`agent.py` (one pilot per episode), `AGENT.md`, `rules.md`, `skills/navigation.md`, `memory.md`,
`NOTES.md`. Untouched: `tools/frame.py`, `system_prompt.txt`, `system_prompt.md`.
