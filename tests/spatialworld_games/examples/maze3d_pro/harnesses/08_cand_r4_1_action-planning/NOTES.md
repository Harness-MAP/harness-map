# maze3d_pro rich harness — module map & levers

Parent: `anchor_maze3d_pro` (the staged rich anchor, which delegates every decision to the
evaluator's pristine VLM step and keeps no state). Iter 4, slot axis **action-planning**. The frozen
agent (Qwen3-VL-30B-A3B-Instruct) and the SpatialWorld engine are untouched; only this harness
changed, and `build_agent` / `compose_system_prompt` / `tools.vlm.decide` keep their signatures.

## axis
**code / action-planning — `tools/plan.py` (new): an EXACT scene decoder feeding a dead-reckoned
occupancy map that the harness routes on.** The steer named the productive difference in the target
region as its *token-level implementation*, so the component graph is deliberately the region's own
(`agent.py` + `tools/{__init__,action,frame,plan,vlm}.py` + the four prose priors — the same five
modules both exemplars ship, nothing added or split), and everything new is token-level content
*inside* those components: `tools/plan.py` inverts the renderer's fog/rect geometry with
module-level look-up tables built by comprehension over the engine's own `fog(d)` and `rects[d]`
formulas, and `tools/action.py` gains the bearing algebra (integer headings, `STEP`, `turn_toward`)
beside the key table. Where the region's exemplars measure *area fractions against thresholds*, this
measures *exact packed RGB against a depth-indexed LUT* — the same perception problem, a different
token-level answer, and it has no tolerance in it anywhere.

## change
`tools/plan.py` reads the maze off the frame instead of guessing at it. `draw_3d_view` is a
flat-shaded painter with a closed-form geometry, and the new decoder inverts it:

* `rects[d] = (600/d, 400/d)` centred on (400, 300), painted **far-to-near**, so the nearest
  non-empty cell on the forward ray covers every farther one. A 21×21 patch at the frame centre sits
  inside `rects[12]`, the smallest front face drawn, so that patch is *always* filled by the nearest
  front face when one exists within 12 cells.
* every face is `darken(base, fog)` with `fog(d) = max(0.4, 1 - 0.08·d)` and integer truncation, so
  the exact RGB of that patch names the material **and** its distance: a front wall is (73,36,27) at
  one cell, (67,33,25) at two, … (32,16,12) from eight on; a stair is `(0,100,120)·fog` behind a
  `(0,255,255)·max(fog,0.8)` border; the exit is a flat (255,200,0) with a white outline. Fog
  saturates from d=8, and there the half-width of the painted run recovers d instead.
* the `d=1` side quads span x ∈ [-800, 100] and [700, 1600], so columns 0–89 and 710–799 are painted
  by **nothing except the cell immediately left / right of the agent** — a one-cell readout with no
  occlusion and nothing to disentangle.

That yields three facts per frame — `(kind, distance)` of the first non-empty cell ahead, and the
passability of both lateral neighbours — and they are measurements, not estimates: scored against the
engine's own `tower_data` over all 25 levels, both floors, every reachable cell and all four
headings, the forward ray is **5624/5624 exact** and the lateral cells **11248/11248 exact**. Two
consequences carry the whole mechanism. First, the number of cells a translation will travel is known
*before* it is spent (`walkable = dist` for a stair or the exit, `dist - 1` for a wall), so a refused
move is refused *pre-emptively* rather than learned by wasting a decision on it. Second, because the
engine never refuses a turn and a stride's travel is predictable, the pose is exactly dead-reckonable,
and each observation writes a whole ray plus two lateral cells into a per-floor occupancy map in one
shared coordinate frame.

BFS over that map's known-passable cells then answers the three questions the anchor's orbits were
failing to answer, in priority order: route onto a **mapped exit** cell; else route to the nearest
**frontier** (a reachable cell with something unsettled next to it); else, only once the floor is
finished, route to a **stair** and change floor. The settling rule is where the exit-detection proof
lives: a ray stops at the first cell that is not plain floor — a wall, a stair *and the exit* all
terminate it — so every cell a ray passes **through** is proven ordinary floor, while a lateral band
proves nothing of the kind (the renderer paints the exit only as a front face, so an exit cell beside
the agent reads exactly like open floor). Those two kinds of "open" are kept apart in `cells` and in
`clear`, and therefore *no reachable frontier left and no exit found* **proves** the exit is not on
this floor. That proof is what makes the floor change free, which is the half of this the lineage
measured as load-bearing: climbing at the first opportunity costs the levels whose exit is on the
start floor, and all nine of those levels are won here.

A floor change is **not observable**: measured directly on this benchmark, **46 of 240 possible
climbs (19.2%) leave the frame byte-identical even though they succeeded**, so frame identity cannot
decide whether a climb happened. Every issued climb is therefore committed optimistically into a
fresh map *epoch* — sound either way, because a refused climb just re-maps the same floor rather than
corrupting the old map — and the ambiguity is settled afterwards for free: per-pose frame digests
from the previous epoch are compared against the new epoch's as they arrive, one mismatch confirms
the transition and `TWIN_MATCHES = 2` matches retire the new epoch back into the finished one. A
stair link is recorded in both directions, so returning to a floor reuses its map instead of
re-exploring it.

The frozen model is asked on **every** step and keeps three channels. It is obeyed whenever its
proposal agrees with the route (agreement is free, and that is 54% of decisions on the rig); it is
obeyed entirely whenever the harness has no route of its own (the map exhausted, or dead reckoning
dropped); and when several frontiers tie at the same BFS distance **its own proposal casts the
deciding vote**, bounded to the turn-minimal subset so the vote can never cost a decision — the
unbounded version let a degenerate always-`turn_right` proposer swing the target every step and spin
in place (11 of 25 levels against 22). What is displaced is only what the measurement refutes:
a translation with `walkable == 0`, a climb with no mapped stair underfoot, or a proposal that
contradicts the mapped route. `render_memory()` states every measured fact back to the model in plain
text — what is ahead and how many cells away, what is immediately left and right, what is under its
feet, whether the last action moved it — appended to a stashed pristine `conversation_history[0]`,
never spliced, because truncating that turn deletes the JSON RESPONSE FORMAT contract from every
later call. All four prose priors were rewritten around the same measured facts, which also deletes
the `[Q] UP` / `[E] DOWN` HUD gate: that text is behind `if False:` on this build's `_draw_ui` and its
pose line is `info = ""`, so the anchor's climb precondition was literally unsatisfiable (0 climbs in
2340 decisions on 25 two-floor levels).

## why (the gradient)
`diagnosis.md` for this parent names two harness bugs and ranks the fixes. **(i) No feedback channel**
— `tools/vlm.py:decide` is a pass-through, so nothing can observe that an action did nothing, and
80.3% of all decisions sit inside terminal periodic orbits (13 episodes press one refused translation
to the cap; 4 run a 4-to-16-action limit cycle; L3 pressed `move_forward small` into a wall 94 times
from one cell away from the exit). **(ii) Prose describing a HUD that is not drawn** — 0 floor
transitions on 25 two-floor levels, 12 of which put the exit on floor 2, i.e. ~48% of the benchmark
unwinnable by construction. §5.1–5.3 are marked confirmed-landed and are reproduced here. The new
work is §5.4 (*"a transition ledger that re-aims repeats"*, sized at 31% of decisions re-proposing a
`(view, action)` already executed twice) and §5.6 (**structural, untried**: *"read the corridor, not
just the tiles … that yields the legal stride length for free and a frontier preference, the one
lever that would put a stair into view on L8/L10/L16/L18/L19, where none was ever seen"*). Those two
are one mechanism once the corridor is read exactly: the stride comes out of `walkable`, the frontier
preference out of BFS on the map, and a *cell*-keyed map strictly dominates a view-keyed ledger here —
the frame digest aliases badly (257 distinct digests over 5624 distinct poses), so keying facts on the
view was never going to be sound. The hypothesis was narrow: the frozen model's syntax is already
perfect (2340/2340 legal, 0 parse fallbacks) and its perception is the ceiling, so the recoverable
loss is entirely in facts the pixels state exactly and nothing was reading.

## verification
A **pricing rig** drives the real evaluator and the real shipped harness dir with deterministic stub
tapes (eval_runner's injection reproduced exactly: `baseline_step` stash → `build_agent` →
`current_prompt` capture), `asyncio.sleep` patched to 0 and video off, so a 25-level sweep costs
about a minute. Its two traps, both hit: the evaluator resolves its level file **relative to CWD**,
so the rig must run from the SpatialWorld root; and `build_agent` purges `sys.modules["tools*"]`, so
a lever override set before the call is thrown away and the "control" silently measures the shipped
config — every lever here is a **class attribute set after `build_agent`**.

**Decoder, against the engine's own `tower_data`** (25 levels × both floors × every reachable cell ×
4 headings): forward ray `(kind, distance)` **5624/5624 exact**, lateral cells **11248/11248 exact**,
`walkable` **0 mismatches**. The one failure found and fixed in the process: the fog plateau from
d=8 collapses the front-wall shade, which read a wall at 9/10/11 cells as 8 (48/5624) until the
half-width fallback was added.

**Six adversarial stub tapes × 25 levels — the mechanism has to do everything:**

| stub tape | anchor | this harness |
|---|---|---|
| always `move_forward small` | 0 | **22** |
| always `move_forward large` | 0 | **22** |
| always `turn_right` | 0 | **22** |
| always `climb_up` | 0 | **22** |
| `L,F,F,U` cycle | 4 | **22** |
| `F,F,R,D,F` cycle | 4 | **22** |
| total | 8/150 | **132/150** |

The won set is **identical on all six tapes** — {1–18, 20–23}, the three losses 19/24/25 — so the
mechanism is invariant to what the proposer says, which is the property that matters against an
unseeded `temperature=1.0, top_p=0.9` sampler the diagnosis blames for unattributable ±1-episode
moves. **Zero regressions: there is no level the anchor won on any tape that this loses.** The three
losses are pure clock, not routing: traced, they run to the cap still exploring correctly (113–180
cells mapped, `lost=False`, `desync=0`) on the largest two-floor mazes.

**Soundness and cost, instrumented over all 25 levels (1232 decisions):** episodes where dead
reckoning was dropped **0**; map conflicts (`desync`) **0**; floor changes 18; overrides 564/1232
(45.8%), i.e. the model's proposal was executed as asked on 54% of decisions; planner cost **5.1 ms
per decision** against a ~2000 ms model call.

**LIVE, real Qwen3-VL-30B-A3B-Instruct, full episodes through `eval_runner.py` with
`ANTICHEAT_GUARD=1`** — `oracle_access` null everywhere, `valid` true, `used_rich_harness` true,
exactly one VLM call per decision (no re-asks, no extra calls). Ten levels: the three the diagnosis
says should convert, the two start-floor traps the lineage lost, the parent's own win, and the rig's
two hardest losses.

| level | anchor live | this harness live |
|---|---|---|
| L01 | loss | **WIN, 3 steps** |
| L03 | loss, sat 1 cell from the exit for 94 decisions | **WIN, 6 steps** |
| L17 | WIN, 33 steps | **WIN, 14 steps** |
| L13 | loss (start-floor exit; the lineage climbed off it) | **WIN, 27 steps** |
| L20 | loss, distance 12 | **WIN, 33 steps** |
| L02 | loss, distance 3, floor 1 | **WIN, 18 steps, floor 2** |
| L04 | loss, distance 5, floor 1 | **WIN, 20 steps, floor 2** |
| L11 | loss, floor 1 | **WIN, 52 steps, floor 2** |
| L19 | loss | loss, 100 steps (clock) |
| L24 | loss | loss, 100 steps, floor 2 (clock) |

**8/10 live against the anchor's 1/10 on the same levels** (its only two wins in the whole benchmark
are L6 and L17), and L02/L04/L11 end **on floor 2** — the floor transitions the anchor never made.

## module map
| file | role |
|------|------|
| agent.py | ENTRYPOINT. `build_agent(evaluator)` returns the async per-step fn and opens ONE `tools.plan.ActionPlanner` per episode, threading it into `tools.vlm.decide`; `compose_system_prompt()` recomposes the prior from the .md files. Note `evaluator.game` does **not** exist yet at this point — `run_evaluation` builds it after the call — so nothing may touch engine state here, and nothing does. |
| **tools/plan.py** | **NEW — the mechanism.** The renderer-derived constants and depth LUTs, `read_scene` (ray + both lateral cells + the frame digest), the dead-reckoned pose, the per-floor occupancy map with `clear` provenance, BFS routing (exit → frontier → stair), the epoch/stair-link bookkeeping with the climb-retirement test, the governor, and `render_memory`. |
| tools/action.py | the engine `Action` contract **and** the bearing algebra. `name_for(key)` (the engine dispatches on the key, not on `metadata["function_called"]`), `turn_toward` / `turns_to` / `bearing_to` / `STEP` (integer headings 0..3 = N,E,S,W, matching the adapter's own turn tables so `turn_right` is `+1 mod 4`), and `retarget`, which strips `granularity`, `repeat` **and** `json_data` before rebuilding a turn — the input source reads all three, and a stale `repeat: 3` on a re-aimed turn fires 270 degrees. |
| tools/vlm.py | the perceive → measure → inject → ask → govern step. Still calls `evaluator.baseline_step` every step (parsing, fallback and the legal set untouched); owns `_inject_memory`, which appends to a stashed pristine `conversation_history[0]`. |
| tools/frame.py | frame capture via `evaluator.capture_game_screenshot`. Unchanged. |
| AGENT.md | **rewritten.** No HUD, no text, no coordinates anywhere; the nested-rectangle corridor; the three materials; the exit is drawn only straight ahead; the tile under your feet is never drawn; an unchanged screen means nothing happened. |
| rules.md | **rewritten.** Legal set + granularity contract unchanged; the climb gate corrected to its real precondition (a stair underfoot — the `[Q] UP` / `[E] DOWN` prompt does not exist on this build); stride economics stated as economics. |
| skills/navigation.md | **rewritten.** Priority order over the measured block: gold ahead, a stair underfoot, a stair adjacent, the open corridor, turn-don't-shove, prefer uncovered ground, finish a floor. |
| memory.md | **rewritten** into the live runtime template. Placeholders must match `plan.ActionPlanner.fields()`; a mismatch falls back silently to `plan.DEFAULT_TEMPLATE`, so keep the two in step. |
| NOTES.md | this file — proposer/diagnostician facing, never shown to the agent. |

## evolvable levers (axes)
- **prose**: AGENT.md / rules.md / skills/*.md / memory.md — what the agent is told.
- **code**: agent.py / tools/*.py — the mechanism (decode, map, route, govern, inject).
- **tuning, all priced here** (class attributes on `ActionPlanner`; set them **after** `build_agent`):

| lever | shipped | measured alternatives |
|---|---|---|
| `OBEY_MODE` | `route` | `route` **132**/150, `coverage` 76/150, `off` 34/150 over the six tapes. `coverage` (obey while the proposal reaches new ground, insist only on a *proven* route) is the agency-preserving setting and costs 56 episodes; it is also wildly tape-dependent (6–21 of 25), which is exactly the instability an unseeded sampler would inherit. |
| `FLOOR_GATE` | `exhaust` | `stale` also 22/25 but on a **different** set (loses L21, keeps L24): equal count, and `exhaust` is the one with the proof behind it. |
| `FRONTIER_RULE` | `nearest` | `forward` (prefer a frontier straight ahead so the stride grows) **refuted**: 21/25 and slower (44.8 vs 42.4 mean steps). |
| `TWIN_MATCHES` | 2 | 1 → 19/25 (loses L4, L16, L23): a single matching digest retires a *real* floor change. Load-bearing, do not lower. |
| `TWIN_WINDOW` | 6 | 16 → byte-identical outcome. **Inert**, do not re-sweep. |
| `MAX_CLIMBS` | 8 | 2 → 22/25. **Inert** safety bound, not a knob. |
| `STRIDE_MAX` | 3 | 1 → 20/25. The stride is worth two levels. |
| `OBSERVE_EVERY_CELL` | False | True (never cross a cell without standing on it) → 21/25; the `clear` provenance makes it unnecessary. |
| `STALL_LIMIT` | 4 | unswept (only reachable once dead reckoning is dropped, which never happened in 1232 decisions). |

- **NOT a lever**: `system_prompt.txt` / `system_prompt.md` are stale inert copies —
  `compose_system_prompt()` reads neither, only the .md priors.

## anti-cheat
Decide from the observation only. The planner's single input is `frame_data.image`, the frame the
agent is shown; every field in the injected block is derived from those pixels plus the agent's own
action history. The `pre_eval_gate` replica was re-run by hand over all six .py files: all parse, and
there is **no privileged attribute or subscript** (`gt.json`, `maze_file`, `distance_to_exit`,
`current_floor`, `target_blocks`, …) anywhere in the code, and none in any agent-facing prior either.
**`evaluator.game` is not accessed at all** — an AST walk finds zero `.game` attributes — and no file
under the ground-truth data tree is opened. The RGB constants and the rect geometry are properties of
the **renderer**, derived once from the drawing code and verified against frames the agent itself
receives; they are the same thing a sighted agent would see, not episode ground truth. Live runs with
`ANTICHEAT_GUARD=1` report `oracle_access: null` and `valid: true` on all ten episodes.

## what is still open (ranked, for the next proposer)
1. **Clock, not routing, is now the binding constraint.** The three remaining rig losses (19, 24, 25)
   and both live losses run to 100 decisions still exploring correctly. The cost is one decision per
   corridor cell, because a lateral band only reads the cell the agent is *in*: a corridor's side
   cells cannot be settled from a distance, so completeness costs a visit. Anything that settles a
   cell's neighbours without standing on it buys levels directly. Price it on decisions-per-settled-
   cell, not on `success`.
2. **Turn churn is the second cost.** A turn costs a whole decision and the route spends 1–2 of them
   at every junction. The depth tie-break already prefers the bearing being faced; a route cost that
   counted turns rather than cells was refuted elsewhere in this lineage as win-neutral, so attack the
   *ordering of looks* at a junction instead (one sweep in a fixed hand, not nearest-first).
3. **A refused climb still costs a re-exploration** when the retirement test finds no comparable pose
   inside its window (it fired on none of the 18 live/rig floor changes). A disambiguation that does
   not depend on having stood at the same pose before would remove the last optimistic assumption in
   the mechanism.
4. **`move_backward` is unmapped by construction** — the backward ray is never rendered — so the
   governor can only reason about it through the map. The model proposed it 0 times in 2340 decisions,
   so this is latent, but a reversal is the one action whose dead reckoning rests on the map alone.
5. **Nothing is seeded.** `create_openai_response` sends `temperature=1.0, top_p=0.9` with no seed.
   The rig makes a mechanism measurable without it, but a ±1-episode live move still cannot be
   attributed. The invariance of this harness's won-set across all six tapes is the workaround, not a
   fix.

## files
- `tools/plan.py` — NEW: the decoder (renderer-derived LUTs, `read_scene`), the dead-reckoned pose and
  occupancy map with `clear` provenance, BFS routing, the epoch/stair-link bookkeeping and climb
  retirement, the governor, `render_memory`.
- `tools/action.py` — rewritten: the key table plus the bearing algebra (`STEP`, `turn_toward`,
  `turns_to`, `bearing_to`, `steps_granularity`, `climb_dz`) and `retarget`.
- `tools/vlm.py` — `decide()` now observes → injects → delegates to `baseline_step` → governs; added
  `_inject_memory` (stash-and-append on `conversation_history[0]`).
- `agent.py` — `build_agent` opens the per-episode planner and threads it into `vlm.decide`; added
  `_positive_int`.
- `AGENT.md`, `rules.md`, `skills/navigation.md`, `memory.md` — rewritten around the measured facts.
- unchanged: `tools/frame.py`, `tools/__init__.py`, `system_prompt.txt`, `system_prompt.md`.
