# maze3d_pro rich harness — module map & levers

Rich harness for the SpatialWorld multi-floor 3D maze (maze3d_pro). The frozen agent
(Qwen3-VL-30B-A3B-Instruct) and the SpatialWorld engine are untouched; only this harness is
optimized. `build_agent(evaluator)` is the entrypoint the extended evaluator installs as
`ai_model_func`; the engine `Action` contract and the maze `success` scoring stay byte-compatible
with the prompt-only baseline.

## axis
**Token-level implementation of spatial grounding, inside the EXISTING component graph.** The steer
asked for the `cand_r001_0_spatial-grounding` region and named the productive difference as the
*token-level* half of φ, so the component graph is deliberately left alone: no module is added,
removed, split or re-wired, `run_episode`/`build_agent`/`compose_system_prompt`/`tools.vlm.decide`
keep their signatures and callers, and every change is *inside* a component —
`tools/plan.py:ActionPlanner` gains a pixel census and three new decision branches (eight new
numeric constants where the parent had three), and the four prose priors are rewritten around the
facts that census measures. The parent's axis was "score the outcome of the last action"; this one
is "measure the two tiles that decide the maze, and the one fact the agent can never see".

## change
`tools/plan.py` keeps its frame-differencing outcome ledger verbatim and adds an **exact-colour
census of the whole frame**. This renderer is flat-shaded: a sweep of 433 live frames over seven
levels contains exactly **30 distinct RGB values**, and in that entire palette the stair family is
the *only* one with `R == 0` ((0,234,234)/(0,214,214)/(0,204,204) borders, (0,92,110)…(0,52,62)
faces) while the exit is the single value (255,200,0) — the walls are all `R > G > B` browns topping
out at (147,78,55), the floor is (50,60,65), the ceiling (15,15,35)/(4,4,10). So `R == 0 and G >= 1`
**is** the stair mask and `R >= 200 and 150 <= G <= 230 and B <= 60` **is** the exit mask: exact
masks, not colour tolerances. The painted **area** is then a distance ladder, and it is quantised —
measured straight down a corridor, a stair paints 1753 px five cells out, 15755 four out, 27401
three, 61505 two, and **243005 (50.6% of the 800x600 frame) when it is the adjacent cell dead
ahead**, then **0** the instant you step onto it, because the tile under your own feet is never
drawn. The exit ladder has the same shape (237009 px = 49.4% adjacent). The gap between adjacent
(0.506) and two-cells (0.128) is a factor of four, so one threshold, `_ADJACENT_FRAC = 0.30`,
separates them with no tuning. That gives the harness the fact the pixels otherwise withhold:
**a saturated stair tile that vanishes under a step that really happened means the stair is now
underfoot.** `govern` gained three branches on top of the parent's three vetoes — step into an
adjacent gold exit; take the floor transition when a stair is measured underfoot *and the floor is
exhausted*; step onto an adjacent stair (always at `small`, never striding past it) when the floor
is exhausted — plus a bound on the climb that cannot work: an ungrounded climb is obeyed
`CLIMB_PROBES = 2` times per floor and re-aimed after that. A floor counts as exhausted after
`STALE_LIMIT = 4` consecutive decisions with no view this floor had not already shown, which is also
the anti-ping-pong lock (a floor change resets it, so the far side has to go stale before the return
is allowed) and is why the climb does not cost the levels whose exit is on the start floor.
`_last_line` no longer reports a refused climb as "executed as you asked", `fields()` stops
advertising `climb_*` as "never tried" where no stair is underfoot, and `memory.md` /`AGENT.md` /
`rules.md` / `skills/navigation.md` are rewritten around the measured perception facts.

## why (the gradient)
`diagnosis.md` for the parent names one binding constraint: **0 floor transitions in 25 episodes,
114 of 114 `climb_up` executions refused, and 12 of the 25 levels put the exit on floor 2** — 48% of
the benchmark unwinnable by construction. Its cause was the three prose priors this parent froze,
which gate both climbs on a `[Q] UP` / `[E] DOWN` prompt in a bottom HUD. **Re-measured here on the
frames the agent is actually shown: there is no text anywhere on this render** (the 30-colour palette
above contains no near-white text pixel outside the exit's outline), so the precondition is
unsatisfiable and the only floor-changing action was disabled. The diagnosis also records that
correcting that prose *alone* is not enough — the sibling did exactly that and reached 179 climb
proposals with still 0 transitions — because an on-stair frame carries **0** cyan, identical to
"no stair anywhere". That is the gap this change closes: §5.1's frame-measured on-stair detector and
§5.2's "make a refused climb visible to the loop", both built *inside* `tools/plan.py` rather than as
a new module, which is what the token-level steer asked for.

## verification
A **pricing rig** drives the REAL evaluator and the REAL shipped harness with a degenerate stub
model, patches `asyncio.sleep` to 0 (a 25-level sweep costs ~60 s) and logs an **offline-only**
oracle pose beside the pixel census so a frame-derived predicate can be scored against ground truth.
Its two traps, both hit and fixed: the evaluator resolves its level file **relative to CWD**, so the
rig must run from the SpatialWorld repo root or it silently loads a degenerate 1-floor maze
(`total_floors=1`, `distance_to_exit=-1`, a frame that never changes); and `build_agent` purges
`sys.modules["tools*"]`, so a tuning override set on the class *before* the call is thrown away and
the "control" measures the shipped config.

**Degenerate stub (always `move_forward small`), all 25 levels — the mechanism has to do everything:**

| | parent | this harness |
|---|---|---|
| wins | 8/25 {1,3,6,7,13,15,17,20} | **13/25** {1,2,3,4,6,7,9,12,13,15,17,19,20} |
| regressions | — | **none** — a strict superset, and the step count on all 8 inherited wins is **identical** (10, 9, 11, 16, 33, 27, 22, 37), i.e. the new branches never fire on an episode that was already being won |
| episodes ending on floor 2 | 0/25 | 12/25 |

`STALE_LIMIT` has an interior optimum and the curve is monotone either side of it:
0 → 8 wins, 2 → 13 (but loses L20, a start-floor exit), **4 → 13 with zero regressions**, 6 → 12,
10 → 11, 16 → 10, 24 → 10. Climbing at the *first* opportunity instead (the prototype that ignores
staleness) scores 11/25 and **loses six levels it already won** — the staleness gate is the half of
this change that makes the floor transition free. `CLIMB_BUDGET` (2/4/8/20) and `CLIMB_PROBES`
(0/2/4) move the score by **0** on every arm: they are safety bounds, not tuned knobs.

**Paired adversarial stubs, 25 levels each — candidate never below the parent, better on 5 of 6:**

| stub model | parent | this harness |
|---|---|---|
| always `move_forward small` | 8 | **13** |
| always `move_forward large` | 8 | **10** |
| always `turn_right` | 6 | 6 |
| always `climb_up` | 0 | **2** |
| `L,F,F,U` cycle | 6 | **7** |
| `F,F,R,D,F` cycle | 3 | **6** |
| total | 31/150 | **44/150** |

**LIVE, real Qwen3-VL-30B-A3B-Instruct, full episodes through `eval_runner.py` with
`ANTICHEAT_GUARD=1`** (`cheat_voided` null everywhere, `used_rich_harness` true, exactly 1 VLM call
per decision — no extra calls, no re-asks). Six levels: the three the diagnosis says should convert,
the two absorption episodes, and two the parent won live:

| level | parent live | this harness live |
|---|---|---|
| L02 | loss, dist 3, floor 1 | **WIN, 46 steps, floor 2** |
| L04 | loss, dist 5, floor 1 | **WIN, 61 steps, floor 2** |
| L11 | loss, 32 consecutive refused climbs, floor 1 | **WIN, 61 steps, floor 2** |
| L17 | WIN | **WIN, 17 steps** (unchanged) |
| L13 | loss, 82 consecutive refused climbs | loss, dist 11 — the absorption is gone (longest same-action run 6) |
| L20 | WIN, 29 steps | **loss, dist 6** — a regression; attributed below |

4 live wins against the parent's 1 on the same six levels, and the **first floor transitions this
arm has ever achieved live** (0 in 75 prior episodes across the parent and both r001 siblings).

The attribution is measured, not inferred. Replaying each live episode's **own recorded proposal
tape — action AND granularity** — through the rig reproduces all six live outcomes exactly
(L02 win/46, L04 win/61, L11 win/61, L17 win/17, L13 loss d11, L20 loss d6), so the same tape can be
fed to the parent harness as a clean counterfactual:

| level | tape → candidate | tape → parent | overrides (cand / parent) | forced climbs |
|---|---|---|---|---|
| L02 | **WIN** | loss, dist 3 | 18/46 / 19/46 | 1 (`turn_right`→`climb_up` @42) |
| L04 | **WIN** | loss, dist 2 | 4/61 / 3/61 | 2 (@51, @57) |
| L11 | **WIN** | loss, dist 4 | 11/61 / 10/61 | 1 (@43) |
| L17 | WIN | WIN | 1/17 / 1/17 | none |
| L13 | loss, dist 11 | loss, dist 11 | 33/100 / 33/100 | none |
| L20 | loss, dist 6 | loss, dist 6 | 29/100 / 29/100 | none |

So each of the three conversions is caused by one or two forced climbs and nothing else — override
volume is otherwise unchanged — and on the other three the governor is **literally a no-op**:
identical override counts, zero forced climbs, identical outcomes. **That includes L20: the
mechanism did not cost it.** Its regression lives entirely in the model's own answers, i.e. the
rewritten prose and/or the unseeded `temperature=1.0, top_p=0.9` sampler — the same noise channel
that made 7/25 vs 8/25 a tie last iter. L13's loss is likewise the model's: its exit is on the start
floor, so declining to climb there is correct, and what remains is the orbit of open item (1).

And the decisive negative control: **the model proposed zero climbs in all six live episodes
(0 of 345 decisions)**, exactly as the parent did, so the corrected prose on its own still does not
make this model change floor — every transition came from the grounded latch, which is the diagnosis
§6 finding ("correcting the missing-HUD prose ALONE buys nothing") reconfirmed on a third harness.

**Anti-cheat**, `pre_eval_gate` replica re-run by hand: PASS. No privileged vocabulary
(`gt.json`, `maze_file`, `distance_to_exit`, `current_floor`, …) in any agent-facing prior
(`system_prompt.txt`, `AGENT.md`, `rules.md`, `memory.md`, `skills/*.md`); all six .py files
AST-parse with zero privileged attribute or subscript access; **no `.game.*` access anywhere**.
The oracle pose exists only in the offline rig, which lives in the scratchpad and ships nothing.
Cost: the full-resolution census is ~2 ms/step against a ~2000 ms model call.

## module map
| file | role |
|------|------|
| harness.py | ENTRYPOINT. `build_agent(evaluator)` returns the async per-step fn and opens ONE `ActionPlanner` for the episode; `compose_system_prompt()` recomposes the prompt from the .md priors. Unchanged this iter except its docstring. |
| tools/plan.py | **CHANGED.** The planner: the frame-difference outcome ledger (unchanged), plus the exact-colour census, the distance ladder, the on-stair latch, floor staleness, the climb spend/budget ledgers, the three grounded `govern` branches, and the rendering of memory.md. |
| tools/vlm.py | The perceive → inject memory → ask → govern step. Appends the block to a stashed pristine `conversation_history[0]`, never splices it. **Untouched.** |
| tools/action.py | the maze3d_pro action/key algebra + the engine `Action` contract (w/s carry granularity 1/2/3; a/d turn; q/e climb); `retarget` strips `granularity`/`repeat`/`json_data` first. Only two stale comments corrected (they asserted the `[Q] UP` prompt). |
| tools/frame.py | frame capture via `evaluator.capture_game_screenshot`. **Untouched.** |
| AGENT.md | **CHANGED.** The measured palette, the area-as-distance ladder, "the tile under your feet is never drawn", and the fact that there is no HUD and no text of any kind. |
| rules.md | **CHANGED.** The legal action set + granularity contract + the climb rule stated correctly: a stair underfoot is the only precondition, the prompt does not exist, and an unchanged screen after a climb is not proof of failure. |
| skills/navigation.md | **CHANGED.** Priority order re-ordered around the grounded block: adjacent exit, then the gated floor change, then the stair step, then open-path movement; plus the two-floor policy and the come-back rule. |
| memory.md | **CHANGED.** New placeholders `{sight_line}` / `{floor_line}` replace `{climb_line}`; must stay in step with `plan.DEFAULT_TEMPLATE` (a mismatch silently falls back to the built-in). |
| NOTES.md | this file — proposer/diagnostician facing, never shown to the agent. |

## evolvable levers (axes)
- **prose**: AGENT.md / rules.md / skills/*.md / memory.md — what the agent is told.
- **code**: harness.py / tools/*.py — the mechanism (compose, capture, measure, decide, govern).
- **tuning, all now priced**: `STALE_LIMIT` (interior optimum at 4, curve measured),
  `CLIMB_BUDGET` / `CLIMB_PROBES` (both inert — do not re-sweep), `_ADJACENT_FRAC` (a factor-4 gap,
  nothing to tune), `stall_limit` / `turn_sense` / `_NULL_FRAC` (inherited, unswept).
- **NOT a lever**: `system_prompt.txt` / `system_prompt.md` are stale inert copies —
  `compose_system_prompt()` never reads either.

## what is still open (ranked, for the next proposer)
1. **The productive short orbit — the largest remaining residue.** `diagnosis.md` §2d: nine losses
   whose tapes are a period-3-to-16 cycle repeated to the cap, where every translation *succeeds*
   so the refusal ledger and the sweep never engage. The rig's 12 remaining losses are the same
   shape. `tried[view]` and `views[view]` exist and `govern` still reads neither; a
   `transitions[(view, action)]` ledger that re-aims a transition already taken twice is §5.3 and is
   untouched by this iter. Price it on coverage and revisit rate, not on `success`.
2. **The middle rungs of the distance ladder are uncalibrated.** Only the adjacent rung (0.506 vs
   0.128, a factor of four) is used by the mechanism and stated as a cell count to the agent; the
   rest is reported as a band on purpose, because an off-axis tile shrinks the paint the same way
   distance does and the two were not separately measured. Separating them would let the harness
   *route* to a stair rather than only recognise one when it is already adjacent.
3. **The exit census is only used as a one-cell closer.** Gold appeared in 1 of 433 explored frames,
   so a line-of-sight commit lever has little opportunity in the loss set (diagnosis §6 measured 0
   LoS frames in all 18 losses) — rank it below (1).
4. **Nothing is seeded.** `create_openai_response` sends `temperature=1.0, top_p=0.9` and no seed,
   so a ±1-episode change still cannot be attributed (diagnosis §5.5). The rig makes a mechanism
   measurable without it, but the live gradient stays noisy.
5. **The offline rig is reusable.** `rig5.py` (scratchpad) drives the real evaluator + the real
   harness with any stub tape, patches `asyncio.sleep`, logs an offline-only oracle pose, and sweeps
   class-level constants *after* `build_agent`. Mind the CWD and the `sys.modules` purge.

## anti-cheat
Decide from the observation only. Every field in the working-memory block is derived from the pixels
the model is already shown plus the agent's own action history. Nothing reads `evaluator.game`
(`.screen` is the only permitted attribute and even that is unused), the level layout, the exit
position, the floor count or the true distance. The exact RGB values and the area rungs in the code
and in the priors are **properties of the renderer**, measured from frames the agent itself receives
— the same thing a sighted agent would see — not episode ground truth.

## files
tools/plan.py (the mechanism), AGENT.md, rules.md, skills/navigation.md, memory.md,
tools/action.py (two comments), harness.py (docstring), NOTES.md
