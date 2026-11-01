# maze3d_pro rich harness — module map & levers

Rich harness for the SpatialWorld multi-floor 3D maze (maze3d_pro), frozen agent
Qwen3-VL-30B-A3B-Instruct. `build_agent(evaluator)` is the entrypoint the extended evaluator
installs as `ai_model_func`; the engine `Action` contract, the legal action set, the JSON response
format and the maze `success` scoring are byte-compatible with the prompt-only baseline. Parent:
`anchor_maze3d_pro` (the staged rich anchor — a pass-through that holds no state and reads no
pixels, 2/25 live).

## axis

**code / spatial-grounding — the renderer's projection, INVERTED ANALYTICALLY.**

The steer asked for the region around `cand_r001_1_action-planning` / `cand_r002_0_spatial-grounding`
and named the productive difference as the *token-level* half of φ, so this iteration lands in that
region's component graph (one new `tools/plan.py`, the existing modules kept and their signatures
unchanged) and differs inside the components, which is where the steer pointed.

Those two exemplars ground themselves on an **area census**: count the pixels a tile paints and
compare the count against an empirically measured ladder, with one fraction threshold
(`_ADJACENT_FRAC = 0.30`) separating "adjacent" from "two cells out". This harness takes the same
pixels and reads them the other way round — it **solves the projection** instead of bucketing its
output, so a distance arrives as an *integer cell count* with nothing to threshold, and the two
columns of the frame nobody in this lineage has read yet turn out to carry the cells to either
side of the agent exactly. The downstream shape follows from that: an integer range makes
displacement predictable, predictable displacement makes dead reckoning exact, exact dead
reckoning makes a per-floor map possible, and a map makes the decision a route rather than a
veto. Measurement first, then mechanism.

## change

`tools/plan.py` (new) reads three single columns of the frame the model is about to be shown.

**The forward ray.** The 800x600 view is drawn as nested rectangles about (400, 300): the cell
`d` cells straight ahead is painted at size (600/d, 400/d), and the side faces span x bands that
never cross x = 400. So the centre column carries front faces, ceiling and floor and nothing
else, and the vertical extent `h` of front-face paint on it gives

                                  D = round(400 / h)

for the nearest non-empty cell straight ahead. **Censused over every (floor, cell, bearing) pose
of all 25 levels — 5524 poses — D and the identity of that face are correct 5524/5524, zero
wrong.** The observed heights land on 401/403, 201/203, 133/135, 101/103, 81/83, 67/69, 57/59,
51/53, 45, 41, 37 (the +2 variants are a stair's or the exit's 3-px outline) and every one rounds
to its own D with a wide margin. There is no constant to tune here, which is the point of doing
it analytically.

**The lateral probes.** The cell *immediately* left is the j = 0 trapezoid spanning x ∈ [-800,
100); immediately right spans (700, 1600]. No front face reaches outside [100, 700] and no other
side band outside [100, 350] / [450, 700], so `x = 50` and `x = 750` are painted by those two
cells and by nothing else, at every pose, with occlusion impossible. **Censused on the same poses
(11048 probes): wall / open / stair exact.** The one systematic residue turned out to be the most
useful single fact in the iteration: the renderer special-cases the exit tile *only on the front
face*, so an exit beside you paints nothing and reads as `open` (102 of 102). `open` is still the
correct walkability verdict, so the probe is 11048/11048 sound as a passability read — it is
merely blind to the exit's identity sideways. See §"the invisible exit" below for what that
forced.

**The colour key** is exact rather than tolerant. The renderer is flat-shaded, each surface one
base colour times a fog factor, and the top 60 rows carry an alpha-180 black panel that scales
whatever is under it by 0.294. Under either scaling the families stay disjoint: `R == 0` is the
stair ((0,100,120) fill, (0,255,255) edge — zero survives any scaling and no other palette entry
has a zero red channel), `R ≥ 200 ∧ 150 ≤ G ≤ 230 ∧ B ≤ 60` is the exit, `R > G > B` is a wall,
and floor (50,60,65) / ceiling (15,15,35) satisfy none of them. A live frame holds 7 distinct
colours.

**What the integer buys.**

1. *The stride stops being a judgement call.* A translation covers `min(granularity, reach)`
   cells, and reach follows from the ray: a wall at D ⇒ reach = D−1 exactly; a stair at D ⇒
   reach ≥ D (the stair is enterable and occludes what is behind it); the exit at D ⇒ reach = D
   and arriving ends the episode. Censused on random walks over all 25 levels, 992 translations:
   the prediction matches the engine's displacement **987/992**, and all five misses are the
   documented lower-bound branch — a 3-cell stride asked for with a stair 1 or 2 ahead, which ran
   past it. The planner clips to D so it lands *on* the stair, and under that discipline the
   prediction is exact. `reach == 0` is therefore a free, exact, **pre-step** refusal test: the
   anchor discovered walls by walking into them 2090 times.
2. *Dead reckoning with no HUD and no oracle.* Turns are never blocked and climbs are gated, so
   pose follows by addition from an arbitrary origin: **2197/2197 poses exact against the
   engine's own coordinates**, floor changes included. Independently, a byte-identical frame and
   zero displacement coincide **992/992** — the retrospective check that is trusted whenever a
   prediction and the pixels disagree.
3. *A map, hence routing.* Each frame writes the ray's cells 1..D and the two lateral cells into
   a per-floor lattice; BFS over it turns "where do I go" into "which neighbour is on the
   shortest path to the nearest cell worth visiting".
4. *The tile under your feet is never drawn*, so standing on a stair cannot be seen — but it was
   mapped on the way in. **TP 31 / FP 0 / FN 0 / TN 2166.** The climb is gated on that latch,
   which is why this harness changes floor at all: the anchor emitted 0 climbs in 2340 decisions
   against a prose precondition ("[Q] UP in the bottom HUD") that this build does not render.

`tools/vlm.py` wraps the pristine `evaluator.baseline_step` with observe → inject → ask → govern.
`baseline_step` is still called on **every** step, so the parse path, the fallback and the
VLM-call accounting are untouched and the frozen model still makes every decision. The governor
keeps the model's proposal unless the pixels prove it cannot accomplish anything (a translation
into a touching wall, a climb with no stair underfoot or with unfinished ground on this floor, a
retreat, a fourth consecutive turn), and three branches settle the decision on positive grounds
instead: the exit is identified within a stride, a stair is underfoot and the floor is spent, or
the floor has taught the lattice nothing for `STALE_LIMIT` decisions — the "productive short
orbit" where every translation *succeeds* so a refusal ledger never engages. The distance of a
forward move is always the harness's (clip to `stride_cap`, extend to the mapped route's straight
run); the direction stays the model's.

The four prose priors are rewritten around what the frame actually carries, and `memory.md`
becomes the live per-turn template appended to `conversation_history[0]`.

## verification

A pricing rig drives the **real** evaluator and the **real** shipped harness with a degenerate
stub model and `asyncio.sleep` patched to 0 (25 levels ≈ 100 s), under `ANTICHEAT_GUARD=1`. A
"tape" is a fixed cyclic sequence of answers: the mechanism has to do all the navigating, and the
parent and candidate see the *same* answers. Three rig traps, all hit: the evaluator resolves its
level file **relative to CWD** (run from the SpatialWorld root or you silently load a degenerate
1-floor maze); a stub `choices[0].message` without a `tool_calls` attribute makes the parser throw
and the episode abort after one step as "API retries exhausted"; and `build_agent` purges
`sys.modules["tools*"]`, so a dial override set before the call is thrown away and the "control"
measures the shipped defaults (the dials are class attributes and the rig sets them *after*).

**Paired adversarial tapes, 25 levels each, identical answers to both harnesses:**

| stub tape | parent (anchor) | this harness |
|---|---|---|
| always `move_forward small` | 0 | **18** |
| always `move_forward large` | 0 | **21** |
| always `turn_right` | 0 | **22** |
| always `climb_up` | 0 | **21** |
| `L,F,F,U` cycle | 4 | **15** |
| `F,F,R,D,F` cycle | 4 | **20** |
| **total** | **8 / 150** | **117 / 150** |

No regressions: on both tapes the parent scores on, the candidate's win set is a strict
**superset** (lffu {1,2,3,4} ⊂ {1..10,17,19,22,23,25}; ffrdf {1,3,6,7} ⊂ a 20-level set).
`oracle_access` is null in all 150 episodes and `vlm_calls == steps_taken` in every one.

**The four measurements that moved the score, in the order they were found** — each one a bug the
rig caught and the trace explained, which is the useful part of this record:

| state | tape `fwd_small` | what was wrong |
|---|---|---|
| veto-only governor | 10/25 | "obey unless provably useless" can never *initiate* a climb: standing on the stair with an open cell ahead, the proposal is legal, so the planner walked off the stair forever |
| + the three forcing branches | 17/25 | — |
| − the frame-based climb verdict | 15/25 | two floors seen from one stair render **byte-identically** (L02's stair reads `wall at 1 / wall left / open right` on both floors), so a held frame is not evidence of refusal; treating it as evidence retired the only stair on the level after two decisions |
| + frontier-aware floor choice | 8/25 | **a tie-break sorted alphabetically**: with both directions unopened, `sorted()` put `climb_down` first, and from the bottom floor every first climb was refused — −7 levels from one line |
| + visit every passable cell | 16/25 | see "the invisible exit" |
| + look behind on a blind side | **18/25** | a climb arrival facing a wall with walls to either side has an empty unvisited set, so the floor read as instantly exhausted and the planner climbed straight back: **83 floor changes in one 100-decision episode** |

**The invisible exit** is the census residue turned into policy. Because a laterally-adjacent exit
paints nothing, a floor can be mapped to completion with the exit sitting in it as an
unremarkable `open` cell, and no further looking will ever name it. So the exploration target is
not the frontier but **every mapped passable cell not yet stood on**: arriving on the exit wins
whether or not the pixels identified it. That is worth +1 level on its own and is the reason the
goal set is defined the way it is.

**The dials, all priced.** `FLOOR_BUDGET` (12/20/25/40/50/55/70/999 → 18 every time) and
`MIN_DWELL` (1/4/10 → 18 every time) are **inert**: they are safety bounds, not knobs, and both
became inert *because* the mechanism they were compensating for got fixed — the budget is dominated
by "no unvisited cell is reachable" and the dwell by the look-behind rule. Do not re-sweep them.
`STALE_HARD` (4/8/16/999 → no change) is likewise dominated. `NEW_FLOOR_BIAS` 0 beats 1 by one
level (15 vs 14 at the time it was measured) — prefer returning to a floor already mapped over
opening a floor id never seen. The one dial that is *not* priced is `STALE_LIMIT`, which gates the
forced route; it was never swept.

**Integrity**, checked on a live 30-decision episode with the system turn captured at each call:
30/30 turns still carry the `RESPONSE FORMAT` contract, the block appears exactly once per turn
(never accumulating), and the turn grows by 780 bytes. The block is absent from turn 1 only —
`conversation_history` does not exist yet when the first call is made — and the governor still acts
on decision 1 regardless. 30 decisions including engine stepping and rendering take 1.69 s total,
so the three-column decode is negligible against a ~2000 ms model call.

**Anti-cheat.** `pre_eval_gate` replica by hand: no privileged vocabulary in any agent-facing
prior; an AST walk of all five modules finds **no `evaluator.game` access at all** (not even the
permitted `.screen`), no privileged attribute, no disk read, and no import outside the stdlib +
numpy + this harness's own `tools`. Every field in the injected block is the inverted projection
of the frame the model is already shown plus this harness's own action history. The geometry
constants and the colour masks are properties of the **renderer**, measured from frames the agent
itself receives — the same thing a sighted agent would see — not episode ground truth. The censuses
in this file compare a frame-derived reading against the engine's grid, and that comparison lives
only in the offline rig in the scratchpad; nothing of it ships.

## module map

| file | role |
|------|------|
| harness.py | ENTRYPOINT. `build_agent(evaluator)` returns the async per-step fn, opens ONE `SightedPlanner` per episode and threads it plus the memory template into `tools.vlm.decide`; `compose_system_prompt()` recomposes the prior from the .md files; `memory_template()` loads `memory.md` and validates its placeholders against the planner's fields, falling back to `plan.DEFAULT_TEMPLATE` on any mismatch. Exposes the planner on the returned fn so an offline rig can read its counters. |
| **tools/plan.py** | **NEW — the mechanism.** `read_sight` (the projection inverted + the colour key + the frame digest), `Sight` (`reach` / `stride_cap`), `Lattice` (per-floor grid, `absorb` / `frontier` / BFS `route`), `SightedPlanner` (`observe` → outcome resolution + dead reckoning, `forced` / `routed` / `plan`, `climb_choice`, `unseen_bearing`, `floor_spent`, `govern`, `record`, `render_memory`). |
| tools/vlm.py | the perceive → inject → ask → govern step. Still calls `evaluator.baseline_step` every step. Owns `_inject_memory`, which appends to a **stashed pristine** `conversation_history[0]` and never splices it — truncate-then-append deletes the JSON RESPONSE FORMAT contract from every later turn. |
| tools/action.py | the action/key algebra, now ACTIVE: `name_for` (the inverse table — the engine dispatches on `action.key`), `read_proposal`, and `retarget`, which strips `granularity`, `repeat` AND `json_data` before rebuilding an Action (a stale `repeat: 3` on a replaced translation fires a turn three times, and `_get_action_granularity` re-reads a stale `json_data`). |
| tools/frame.py | frame access, now ACTIVE: `pixels(frame_data)` is the normalisation gate in front of the decode and returns None rather than raising, which degrades one decision to the baseline instead of costing the episode. |
| AGENT.md | **REWRITTEN.** States that there is no HUD and no text of any kind, replaces the `POS`/`FLOOR`/`DIR` and `[Q] UP` fictions with the measured colours, says panel size *is* distance, and names the two facts that are easy to get wrong (the tile underfoot is never drawn; a refused move leaves the picture unchanged). |
| rules.md | **REWRITTEN.** The legal set and granularity contract unchanged; the climb stated correctly (a stair underfoot is the only precondition, the prompt does not exist) and stride economics stated as economics. |
| skills/navigation.md | **REWRITTEN.** Priority order around the grounded block: gold ends it, never repeat a refused move (and do not retreat — it is a 2-cycle), travel in strides, step *onto* the stair, finish a floor before leaving it, prefer unexplored ground. |
| memory.md | **REWRITTEN** into the live runtime template. Placeholders must stay in step with `SightedPlanner.fields()`; a mismatch falls back silently. |
| NOTES.md | this file — proposer/diagnostician facing, never shown to the agent. |
| system_prompt.txt / .md | stale inert reference copies. **Not levers**: `compose_system_prompt()` reads neither. |

## evolvable levers (axes)

- **prose**: AGENT.md / rules.md / skills/*.md / memory.md — what the agent is told.
- **code**: harness.py / tools/*.py — compose, normalise, decode, reckon, map, route, govern.
- **sub-levers inside tools/plan.py**: the goal priority in `_goal_cells`; the three branches of
  `forced()`; the veto set in `govern`; the stride policy (clip vs extend); the `climb_choice`
  ranking; the route metric (plain BFS — turn-aware routing is listed as refuted in the lineage
  and was not retried).
- **tuning**: `STALE_LIMIT` (the only unswept dial), `NEW_FLOOR_BIAS` (0 > 1 by one level),
  `SPIN_LIMIT` (unswept); `FLOOR_BUDGET` / `MIN_DWELL` / `STALE_HARD` all **measured inert**.
- **not levers**: `system_prompt.txt`, `system_prompt.md`, this file.

## what is still open (ranked, for the next proposer)

1. **The two start-floor-exit levels, L13 and L20, are still lost** — the largest residue, and the
   lineage says why: the harness leaves a floor whose exit it had not yet reached. The gate here
   is "no unvisited passable cell is reachable", which is the strongest version of that gate yet
   and still departs too early on those two. The lineage's own finding is that *commitment* is the
   cure and that a return-as-goal is refuted both ways; the untried version with this instrument
   is a **debt comparison** — leave a floor only when the far side owes less than this one — which
   the per-floor lattices now make directly computable (unvisited-passable count per floor) and
   which nothing in this harness reads yet.
2. **`fwd_small` is the weakest forward tape (18) and `fwd_large` the strongest (21)** on an
   otherwise identical policy, so ≈3 levels are pure throughput. The stride extension added here
   recovers nothing on that tape because the nearest unvisited cell is usually 1–2 cells away;
   the fix is a route that *orders* its targets to produce long straight runs (a serpentine or
   nearest-neighbour tour over the unvisited set) rather than always taking the nearest.
3. **The ray reads cells 1..D and the flanks read j = 0 only.** The side bands for j ≥ 1 sit at
   x ∈ [100, 350] / [450, 700] and are occluded only by a front wall nearer than j, so the cells
   *beside the corridor ahead* are decodable too and would fill the lattice several times faster
   per decision. That is the single cheapest coverage win left and it is pure measurement.
4. **Turn 1 carries no measured state** (`conversation_history` does not exist when the first call
   is made). Injecting it means appending to `evaluator.current_prompt` instead, which puts the
   block *before* the AVAILABLE ACTIONS + RESPONSE FORMAT tail — so the step-2 stash must remove
   it by exact string match, never by splitting at the marker, or the contract is truncated. Worth
   one decision's awareness only; sequenced below the above.
5. **Everything above is measured against stub tapes, not the frozen model.** The tapes are the
   right adversarial control — they isolate the mechanism, and the parent sees identical answers —
   but they say nothing about how the rewritten prose interacts with a model that answers in 15
   tokens. `create_openai_response` sends `temperature=1.0, top_p=0.9` and no seed, so a
   ±1-episode live difference still cannot be attributed. Price the next iteration on the tape
   table first and treat live `success` as the noisy confirmation it is.
6. **The rig is reusable** (`rig.py` in the scratchpad): real evaluator, real harness, any tape,
   per-step trace of cell/bearing/floor/D/face/flanks/stale/mode/override, and class-dial sweeps
   applied after `build_agent`. Mind the CWD, the `tool_calls` attribute and the `sys.modules`
   purge.

## files

`tools/plan.py` (new, the mechanism), `tools/vlm.py` (observe/inject/govern + `_inject_memory`),
`tools/action.py` (`name_for`, `read_proposal`, `retarget`), `tools/frame.py` (`pixels`),
`harness.py` (per-episode planner + `memory_template`), `AGENT.md`, `rules.md`,
`skills/navigation.md`, `memory.md`, `NOTES.md`. Unchanged: `tools/__init__.py`,
`system_prompt.txt`, `system_prompt.md`.
