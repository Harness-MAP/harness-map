# maze3d_pro rich harness — module map & levers

Rich harness for the SpatialWorld multi-floor 3D maze (maze3d_pro). The frozen agent
(Qwen3-VL-30B-A3B-Instruct) and the SpatialWorld engine are untouched; only this harness is
optimized. `build_agent(evaluator)` is the entrypoint the extended evaluator installs as
`ai_model_func`; its signature, the engine `Action` contract and the maze `success` scoring are
byte-compatible with the prompt-only baseline, and `evaluator.baseline_step` still does every VLM
call, parse and fallback (verified: `vlm_calls == steps_taken` in every live episode below).

## axis
**Frame COLOUR census → grounded floor transitions** (code lever: a new `tools/sight.py` threaded
through `tools/plan.py`, with the three prose priors corrected to the perception facts this build
actually has). The parent read `frame_data.image` as ONE BIT — an md5 fingerprint, equal or not
equal. This iter reads the same frame a second way, by exact colour, which is the only reading that
can answer the question the fingerprint cannot: *am I standing on a stair?*

Axis-distinct from the parent (which added the fingerprint ledger + escape sweep) and from the
sibling `cand_r001_0_spatial-grounding` (which dead-reckons a pose/heading/local wall map and
re-asks the model on a proven-null answer). Nothing here reckons a position, and the model is never
re-asked; the colour census is used for exactly two facts — *where the stair and the exit are* and
*whether a stair is underfoot* — and the second one is spent on a climb.

## change
### `tools/sight.py` (NEW — the whole mechanism)
An exact-colour census of `frame_data.image`, **re-measured today on live frames of this build**
(800x600, via the real evaluator driven by a scripted stub — the frames the agent is shown):

* the renderer is flat-shaded and its entire palette is **30 RGB values**, so both special tiles
  decode with an EXACT mask, no tolerance and nothing to tune:
  `stair ⇔ red == 0 ∧ green,blue ≥ 1` (the teal family (0,234,234)…(0,52,62)) and
  `exit ⇔ blue == 0 ∧ red,green ≥ 1` (the single gold (255,200,0)). No other colour in the palette
  has a zero channel — walls are browns with R>G>B down to (21,10,8), ground (50,60,65), ceiling
  (15,15,35)/(4,4,10), and the exit's white outline (255,255,255) has none — so the two masks cannot
  be confused with each other or with the scene. The `≥ 1` floor is what keeps black out of both.
* the painted **area is a quantized distance ladder** and the rungs are byte-identical on every
  level (the view is a pure function of local geometry). As a fraction of the frame:
  stair `0.0148 / 0.0213 / 0.0328 / 0.0571 / 0.1281 / 0.5063` at 6,5,4,3,2,**1** cells and
  exit `0.0297 / 0.0529 / 0.1219 / 0.4938` at 4,3,2,**1**. 2 cells → 1 cell is a factor of **four**,
  so one cut at `ADJACENT = 0.30` means "the very next cell dead ahead" with no tuning. All ten
  measured rungs round-trip exactly through `cells_away`'s cut table.
* **standing on the tile paints zero of it** — the tile underfoot is never drawn. That is the trick,
  and it turns the fact the fingerprint could never supply into an exact frame fact:
  *previous frame painted a stair at ≥ 0.30 ∧ the executed action was a translation ∧ the frame
  CHANGED ⇒ the stair is underfoot.* **Validated against the engine's own `cell_under_foot` over
  1840 decisions / 25 levels (offline pricing only): 67 true positives, 0 false positives,
  0 missed saturated arrivals.**
* `StairSense` is the latch plus its budget. It refuses to over-conclude: a *successful* climb can
  leave the frame byte-identical, so an unchanged frame is never read as a failed climb — the climb
  is spent from `CLIMB_BUDGET` and re-armed only by arriving on a stair again. "Am I on a stair" is
  also never filed against a view fingerprint (two identical-looking poses can disagree about it),
  unlike "is the way ahead blocked", which legitimately is.

### `tools/plan.py` (wiring + three honesty fixes the diagnosis named)
* `observe` now also takes the colour census and feeds `StairSense`. `govern` gains a FIRST branch
  that displaces a proposal on **positive** evidence: a stair underfoot, the floor's price paid, and
  **no exit in sight** (an exit anywhere in view is reachable on this floor and ends the episode, so
  no floor change can improve on it) ⇒ take the climb. Agreement still costs nothing.
* **the price of leaving a floor**, the only two tuned numbers in the harness (swept below):
  `FLOOR_PATIENCE = 24` decisions this floor must have been given at all, and `STALE_FLOOR = 1`
  consecutive decision showing no view this floor had not already produced. A floor change clears
  the floor's view set and its age — **that is the anti-ping-pong lock**, not a direction ban.
* a refused climb is now visible to the loop. `_score` counts an inert climb on a rolling
  `null_climbs` (not in the view-keyed `refused` ledger, which would be unsound); `refusal_reason`
  re-aims any climb after one has been proven inert and widens the stall branch to climbs;
  `last_line` reports "the screen did NOT change at all" instead of "was executed as you asked";
  `climb_line` never asserts a floor change it cannot see; and `untried_here` stops advertising
  `climb_up`/`climb_down` as "never tried" unless a stair really is underfoot — the invitation that
  fired the first climb in both absorbed episodes.
* `_trim`: a `medium`/`large` forward stride is clipped to `small` when the frame shows the target
  tile is the very next cell, so a long stride cannot walk over the stair it was aimed at. The
  action NAME is never changed, so this is obedience-neutral.
* §5.4 correctness: `_reaim` is now PURE and `sweep` is exactly "the loop has stalled", decided in
  `observe` alone. Previously any veto — one refused translation, a climb lock — latched ESCAPING
  mode until a translation succeeded (649 of 1943 live decisions). The sealed-cell guard is `>= 4`
  rotations, not `> 4` (it was spending 450°).

### prose (`AGENT.md`, `rules.md`, `skills/navigation.md`, `memory.md`)
These describe a HUD and a `[Q] UP`/`[E] DOWN` stair prompt **that this build does not render** —
there is no text anywhere in the frame — and `rules.md` made both climbs *conditional* on that
prompt, i.e. on an unsatisfiable precondition. All four now state the measured facts: no text at
all; the brown/gold/cyan palette; a tile is drawn only while it is ahead, so no cyan on screen does
NOT mean no stair underfoot; the size of a coloured tile is its distance; and a climb works only
while a stair is underfoot, with the UNDER YOUR FEET line as the only statement about it. Correcting
the prose **alone** buys nothing — the sibling did exactly that and reached 179 climb proposals with
still 0 floor transitions — which is why it ships with the detector, not instead of it.

## why (the gradient)
`diagnosis.md` for the parent: primary **7/25**, all 18 losses at exactly 100/100 steps, and the
binding constraint is that the harness **never completes a single floor transition** — 0 in 25
episodes, 114 of 114 `climb_up` executions refused — while **12 of the 25 levels put the exit on
floor 2**, so 48% of the benchmark was unwinnable by construction. The agent stood on a stair tile
for ≈121 decisions, including on 7 of those 12 levels, and proposed `climb_up` on none of them,
because the frame shows it nothing and the priors told it to wait for a prompt that does not exist.
Two whole episodes (L11, L13) were absorbed into 114 consecutive refused climbs. This change
attacks exactly that, and the §5.2/§5.4 fixes ride along in the same module.

## verification
**Two independent offline rigs** (real evaluator + real harness, scripted model, ~60 s per 25-level
sweep). Oracle fields were read by the RIGS only, for scoring and for validating the latch; nothing
derived from them is in the shipped harness.

**(a) Replay rig — the model's OWN recorded answers** from the parent's live run fed back in order
with their granularity. It is CALIBRATED: run against the parent harness it reproduces the parent's
live result exactly — **7/25 and the identical win set {1,3,6,7,15,17,20}**.

**(b) Stub rig — a degenerate model** that answers `move_forward small` to every question, so the
mechanism alone has to do the work. Parent: 8/25, 0 floors climbed.

| arm | replay rig | floor 2 | stub rig | floor 2 |
|---|---|---|---|---|
| **parent harness** | **7/25** | 0/25 | **8/25** | 0/25 |
| **this harness (shipped: patience 24, stale 1)** | **10/25** | 12/25 | **17/25** (+9 / **−0**) | 13/25 |
| blind the colour census (`look` → empty) | 6/25 | 1/25 | 8/25, win set *identical to the parent's* | 0/25 |
| patience 0 / 8 / 16 / 24 / 32 | 9 / 9 / 9 / **10** / – | 13/13/13/12 | 19 / 19 / 19 / **17** / 17 | 19/19/15/13/13 |
| stale 0 / 1 / 2 / 4 / 6 / 10 | 9 / 9 / 8 / 7 / 6 / 8 | 14/13/13/10/9/7 | 18 / 19 / 15 / 13 / – / – | 20/19/13/11 |

* **Blinding the census collapses this harness onto the parent, win-set for win-set** on the stub
  rig (8/25, {1,3,6,7,13,15,17,20}) and to 6/25 on the replay rig. The colour reading IS the
  mechanism; everything else in the diff is carried by it.
* **`patience = 24` is the joint pick.** It is the best replay arm (10/25) and the only arm that is
  **zero-regression** on the stub rig — at patience 0–16 the stub rig is nominally higher (19/25)
  but gives back L7 and L20, both start-floor-exit levels, which is precisely the failure the gate
  exists to prevent. 24 of a 100-decision budget = the floor you start on gets the first quarter of
  the clock. The curve is shallow and both ends are measured, so the next iter can move it with data.
* The granularity trim is **score-neutral on the rigs**: with a `large`-only stub it fires 11 times
  across 25 levels and the paired no-trim arm scores identically (12/25 both). It is kept as a
  correctness guard — that same arm reaches floor 2 only 4/25 against 13/25 at `small`, which is the
  measurement that long strides walk over stairs — not as a scoring lever. Do not price it again.

**Live, through `eval_runner.py` with `ANTICHEAT_GUARD=1`, exactly the path the outer loop uses**
(6 episodes on the shipped config; `oracle_access` null and `vlm_calls == steps_taken` in all):

| level | exit floor | parent live | this harness live |
|---|---|---|---|
| **L2** | 2 | loss, 100 steps, dist 3, floor 1 | **WIN, 44 steps, finishes on FLOOR 2** |
| **L4** | 2 | loss, 100 steps, dist 5, floor 1 | **WIN, 37 steps, finishes on FLOOR 2** |
| **L11** | 2 | loss, 100 steps, dist 5 — **32 refused `climb_up`** | **WIN, 50 steps, FLOOR 2**, 2 climbs executed, longest run 6 |
| L13 | 1 | loss — **82 consecutive refused `climb_up`**, longest same-action run 82 | loss, but **1 climb proposed, 83 forward, longest run 9** — the absorbing state is gone |
| L20 | 1 | **win, 29 steps** | **loss ×2** (both 100 steps, dist 6), and in **neither run did a floor change occur** |

**3 live conversions, all three by reaching floor 2** — the first live floor transitions anywhere in
this lineage — against 1 regression. L11 and L13 are the two episodes the parent lost entirely to
the refused-climb absorbing state: one is now a win on the other floor, the other spends 1 decision
on a climb instead of 82.

**L20 is a real live regression and the climb gate is not its cause** — both samples stayed on
floor 1 for all 100 decisions, so nothing the stair latch does was in play. What changed on L20 is
the model's own exploration under the rewritten priors and the new memory lines (parent: 29
decisions, 27 of them `move_forward`; this harness: 100 decisions, 73 forward / 25 turns / 2 back).
Both rigs say patience 24 recovers L20 and the live sample says it does not, twice. That is the one
thing to watch in the next diagnosis, and the cheapest read on it is the prose, not the gate.

**Contract + anti-cheat.** `pre_eval_gate` replica run by hand: **PASS** (no privileged mention in
any agent-facing prior, no privileged attribute or subscript access in `harness.py` or any
`tools/*.py`). No module touches `evaluator.game` at all. End-to-end check with a fake evaluator:
`build_agent(evaluator)` signature unchanged, the prompt composes from all three priors (6337
chars), a saturated-stair frame trims a `large` proposal to `small`, the next frame sets the latch
and the injected block reads "you ARE STANDING ON A STAIR right now", the base system turn is
appended to and never spliced, and `baseline_step` is called once per decision.

## module map
| file | role |
|------|------|
| harness.py | ENTRYPOINT. `build_agent(evaluator)` returns the async per-step fn and opens ONE `ActionPlanner` per episode; `compose_system_prompt()` recomposes the prompt from the .md priors. |
| tools/sight.py | **NEW.** The exact-colour census: the two zero-channel masks, the central-band readout, the measured area→cells ladder, `stepped_onto_stair` (the exact on-stair transition) and `StairSense` (the latch, its budget, the believed floor, and the plain-English line the agent is shown). |
| tools/plan.py | The governor. Whole-frame `fingerprint`/`screen_changed` + the view-keyed refusal ledger + the right-hand escape sweep (unchanged and load-bearing), now plus the grounded-climb branch, the floor-age/staleness price, the inert-climb counter, the granularity trim, and the working-memory block. |
| tools/vlm.py | The perceive → inject → ask → govern step. Appends the block to a stashed `conversation_history[0]`, delegates every call to `evaluator.baseline_step`. **Unchanged this iter.** |
| tools/action.py | the maze3d_pro action/key algebra + `retarget` (strips `granularity`/`repeat`/`json_data`). **Unchanged.** |
| tools/frame.py | frame capture via `evaluator.capture_game_screenshot`. **Unchanged.** |
| AGENT.md | role framing + the perception facts measured on this build: no text anywhere, the palette, a tile is drawn only while it is ahead, size = distance. |
| rules.md | the legal action set + granularity contract + the corrected climb rule (no prompt exists; a climb needs a stair underfoot). |
| skills/navigation.md | the priority order, re-ordered around the grounded block: exit in sight first, then stairs, then the stair underfoot once the floor is paid for. |
| memory.md | the per-step scratchpad TEMPLATE; placeholders must match `ActionPlanner.fields()` or `plan.DEFAULT_TEMPLATE` is rendered instead (verified: memory.md is accepted). |

## evolvable levers (axes)
- prose: AGENT.md / rules.md / skills/*.md / memory.md — what the agent is told.
- code: harness.py / tools/*.py — the mechanism (compose, fingerprint, census, govern, inject).
- knobs inside the new mechanism, both swept above and both shallow: `FLOOR_PATIENCE`,
  `STALE_FLOOR` (tools/sight.py). `ADJACENT`, `CLIMB_BUDGET` and the ladder cuts are MEASUREMENTS
  of the render, not knobs — the adjacent/2-cell gap is a factor of 4, so 0.30 has no gradient.
- UNEXPLORED and ranked next by the diagnosis (§5.3): `tried[view]` and `views[view]` already exist
  and `govern` reads NEITHER. Nine of the parent's losses are a literal period-3-to-16 cycle
  repeated to the cap, with the governor firing 0–19 times; a per-(view, action) transition ledger
  that re-aims a transition already taken twice is the obvious next axis. Price it on coverage
  (cells visited, median 13% in the parent) and revisit rate, not on `success` alone.
- Also unexplored: a per-request `seed` on `evaluator.client` (§5.5) — without it a ±1-episode
  change cannot be attributed to a mechanism at temp 1.0.

## what is still open
- **L20** (above): two live losses against a parent win, with no floor change in either.
- A climb is spent from a budget because a successful climb can be frame-invisible; the harness
  therefore never learns whether a given climb worked, only that it was grounded. `level` (the
  believed floor) is bookkeeping over our own presses and can be wrong after an ungrounded one.
- The on-stair latch only arms by ARRIVING on a tile. A body that starts an episode on a stair, or
  steps onto one backwards (invisible, behind the camera), is not detected.
- Exit line-of-sight is read and shown to the agent but is only a climb veto. The diagnosis measured
  **0** LoS frames in all 18 losses, so an exit-commit lever is a closer with no opportunity in the
  loss set — do not make it the next axis.

## anti-cheat
Decide from the observation only. Every number in the shipped harness is either a constant measured
off the render or a count of the agent's own actions. Nothing reads `evaluator.game` (only `.screen`
is even permitted), the level layout, the exit's position, the floor count or the true distance. The
offline rigs DO read `get_state()`/`tower_data` — for scoring arms and for validating the latch —
and they live in the session scratchpad, not in this directory.

## files
tools/sight.py (new), tools/plan.py, AGENT.md, rules.md, skills/navigation.md, memory.md,
harness.py (docstring), NOTES.md
