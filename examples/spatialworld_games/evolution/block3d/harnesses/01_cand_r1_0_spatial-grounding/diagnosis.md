# Diagnosis — block3d iter 0 / `anchor_block3d` (`staged_anchor`, lane `k32_claude_pv`)

**Headline:** primary (`won`) = **1/20 = 0.050** · mean final completion **14.15%** · placement
precision **17/116 = 14.7%** · target recall **17/116 = 14.7%** · **19/20** episodes burn the full
**50**-decision wallet · **0/951** parse failures · **0** default actions · **0** illegal actions ·
**7** distinct reply strings · **8** completion tokens and `reasoning=None` on **951/951**
completions. The single win is **L01, whose only target cell is the cursor's spawn cell `(2,2,0)`**
(`block3d_1_7eo7zcne`, f0, one VLM call, `place_block`) — a win that requires no spatial reasoning
at all.

**Binding constraint: VLM-CEILING on *cursor localisation and routing* — in front of a harness whose
decision path is two lines, so the ceiling is never approached.** Two measurements in this pass
sharpen that from the previous lanes' blanket "ceiling on everything", and they change what the
next candidate should build:

> **1. The cursor-movement decision is worse than chance; the toggle decision is not.**
> `P(place_block | the cursor stands on an unfilled TARGET cell) = 17/34 = 0.500` versus
> `P(place_block | it stands on an empty NON-target cell) = 141/635 = 0.222` — **ratio 2.25**
> (2.46 excluding frame 0). The commit decision carries real signal. Movement carries *negative*
> signal: **all six directions are emitted more often exactly when that direction is blocked**
> (`move_right` 0.173 at-wall vs 0.042 free = **4.13×**; `move_up` 2.83×; `move_left` 2.75×;
> `move_down` 1.64×; `move_forward` 1.47×; `move_backward` 1.44×; a frame-independent coin gives
> 1.00). The harness clamps **42.5% of its moves (319/751)** against a uniform-random policy's
> **15.8%** on the same wallet and levels (20,000 simulated episodes), and reaches **14.0** of 216
> cells per episode against random's **24.4**. Conditioning movement on the image makes it *strictly
> worse than not looking*. ⇒ **own the route, keep the model on the commit.**
>
> **2. The wallet is not the constraint — it has 1.5×–50× headroom.** With the true targets, the
> optimal build tape (exact TSP for |T| ≤ 9, 2-opt above) costs **1, 5, 3, 3, 6, 8, 9, 8, 9, 8, 8,
> 13, 10, 15, 19, 18, 21, 22, 33, 32** callbacks on L01…L20 — **max 33, mean 12.6, every level
> inside the 50-decision wallet**. 251 callbacks would have won all 20; the anchor spent **951** and
> banked **17 correct blocks**, one per **56 decisions**. Nothing about this run is budget-bound.

Every number below is a *measurement*, not an inference: cursor dead-reckoning from the harness's
own emitted actions (spawn `(2,2,0)`, clamp to `[0,5]³`, `left/right = ∓X`, `up/down = ∓Y`,
`forward/backward = ±Z`) reproduces the engine's `matched_blocks` **exactly on 20/20 episodes**.

---

## 1. What this harness does (mechanism)

`orchestrator.py` is a **pass-through**. Nothing in it decides anything.

- `compose_system_prompt()` (orchestrator.py:21) concatenates `AGENT.md` + `skills/*.md` + `rules.md` and
  installs the result on `evaluator.current_prompt`; `eval_runner.py:293` captures that as
  `initial_prompt`. The composition reproduces `system_prompt.txt` in order: role + grid facts →
  "study the three views / you can remove misplaced blocks" → the 7-action control map + colour
  legend. It supplies **no coordinates, no clamp model, no toggle-parity model, no plan**.
- `build_agent(evaluator)` (orchestrator.py:34) puts `HERE` on `sys.path`, purges `sys.modules["tools*"]`
  so a sibling candidate's `tools/` is never inherited (**keep this line**), and returns
  `agent(frame_data) -> await tools.vlm.decide(evaluator, frame_data)`.
- `tools/vlm.py:decide()` is **two lines**: `return await evaluator.baseline_step(frame_data)`. That
  pristine SpatialWorld step captures the screenshot, appends it to a 29-turn image+text history,
  asks for one JSON action, parses it, and returns `Action(type="key_press", key=...)`.
- `tools/frame.py:capture`, `tools/action.py:key_for` / `KEY_MAPPING` / `CELLS_PER_MOVE`, and the
  whole of `memory.md` are **defined and never called**. The rich surface exists as dead code.

Contract facts that bound any fix (read off `game_config/eval_runner.py` and this run's results):

| fact | value | where |
|---|---|---|
| wallet | **50 decisions**; `max_steps=100` counts *engine* steps (2 per decision — L01 won at `steps_taken=2`, `vlm_calls=1`) | `eval_result.json` |
| anti-cheat | **ON** — `vlm_calls` is non-zero in every `eval_result.json`, and the counter only increments when `ANTICHEAT_GUARD` is truthy | eval_runner.py:276–283 |
| `vlm_calls == 0` | score **voided to 0.0**, `valid:false` — the frozen model must stay on the decision path | eval_runner.py:341 |
| `evaluator.game.<attr>` | only `.screen` passes; every other read is recorded and **raises**, and the episode's score is voided | `_GameGuard._ALLOWED`, eval_runner.py:132 |
| the data tree | `_DiskGuard` blocks harness reads of `SpatialWorld/data` (layouts/targets) via `os.open` | eval_runner.py:306 |
| history window | **29** (`HISTORY_WINDOW` unset; prompt grows +678/turn from 1394 and saturates at **21,093** by turn 30) | eval_runner.py:258 |
| wall | **52.7–77.8 s** of episode time per subprocess against the **1800 s** `subprocess.run(timeout=1800)` wall — **~23× headroom for extra VLM calls per decision** | game_adapter.py:283, `total_time` |
| cost | 1246.7 s of episode time, **14,418,689 prompt tokens to buy 7,608 output tokens (1895:1)** | per-call `usage` |

## 2. What actually happened (causal chain, with evidence)

### The absorbing state — `block3d_2_816ect9z` (target = `{(4,4,0)}`, matched 0, min tape **5**)

| f | action | cursor | outcome |
|---|---|---|---|
| 00 | `move_down` | (2,3,0) | legal |
| **01–49** | `move_backward` **×49** | (2,3,0) | **`z` is already 0 — every one of the 49 presses is a clamped null.** 98% of the wallet, 2 cells visited, 0 toggles. |

A clamped move returns a **byte-identical frame**, so the identical question is re-asked of a
near-deterministic 8-token reflex, which answers identically. The target was `right, right, down,
down, place` — **5 callbacks away**. This is not a parse, key-map or fallback failure: every reply
was legal JSON and was applied. `block3d_6_8kdgovfq` is the same shape (`move_backward` ×44 from
f07, 43 dead decisions, 86% of the wallet).

**The loop is measurable, and the harness builds it:**
`P(repeat the same action | the previous move was a clamped null) = 152/307 = 0.495` vs
`P(repeat | the previous move actually moved) = 96/427 = 0.225` — **2.2×**, a positive feedback loop
with an absorbing state. **33.5% of all observations (319/951) are byte-identical to the
immediately preceding one**, and the trailing 29-observation window holds on average only **15.1
distinct `(cursor, placed)` states (52%)** — the harness fills the model's own context with
duplicates.

### Reach-without-commit — `block3d_8_wcifkwn3` (target = the column `{(2,2,0),(2,2,1),(2,2,2),(2,3,0)}`, matched 0, min tape **8**)

f00–f02 walk `move_forward` from the spawn through **(2,2,1)** and **(2,2,2)** — i.e. **three of the
four target cells, one of them the spawn itself** — and place nothing. The cursor then tours to the
`(0,0,0)` corner, makes **its only toggle of the episode** at f31 on the wrong cell, and spends
**f32–f47 (16 decisions) pressing `move_backward` ×9 then `move_left` ×7 into two walls at that
corner**. 27 of 49 moves clamped, 21 cells visited, 1 toggle, 0 correct.

### Parity churn — `block3d_7_p3d9tgh0` (target = `{(4,4,0),(4,5,0),(5,5,0)}`, matched 0, min tape **9**)

f14–f49 is a **period-10 two-voxel parity orbit** in the opposite corner of the box from the target:
`place (0,3,0)` / `up` / `forward` / `place (0,2,1)` / `backward` / `left`(null) / `down` … The same
two cells are placed and removed in alternating laps — **15 toggles, net +5 blocks, 0 correct** —
while the target sits at maximum L1 distance from the orbit.

### Both at once — `block3d_17_6yshn82p` (target 10 cells incl. the spawn, matched 0, min tape **21**)

Spawns **on** a target cell and leaves at f00 without placing; never returns. Thereafter it
alternates between the `(5,5,0)` and `(5,5,2)` corners, and **24 of its 42 moves are clamped**
(f13–f23 is eleven consecutive nulls broken only by one toggle). 8 toggles, 3 of them removals of
its own blocks, 2 blocks standing at the end, 0 correct.

### Why the model cannot escape: the harness gives its context no state

The prompt-token sequence is **turn-for-turn identical across all 19 full-length episodes**
(758,805 tokens each, zero differing positions). All seven legal replies tokenize to exactly 8
tokens, so this does not prove the action echo is absent — but it *does* prove that **no
level-dependent and no state-dependent text ever enters the context**: any rendered coordinate,
count or scratchpad would have made the sequences differ. The only information distinguishing one
episode from another, inside the model's context, is **pixels** — and 33.5% of those pixels are
duplicates of the previous turn.

### The opening move is anti-correlated with the one free block on offer

The spawn cell `(2,2,0)` is a target cell on **9/20 levels** (L01, 04, 08, 11, 13, 14, 15, 17, 19).
The anchor opened with `place_block` on **13/20** levels overall, but on only **4/9** of the gift
levels — `P(open with place | spawn IS a target) = 0.444` vs `P(… | spawn is NOT a target) =
0.818`. It walked off a free correct block on **L04, L08, L11, L15, L17**.

## 3. Failure breakdown

951 decisions, 20 episodes, all figures dead-reckoned (validated 20/20 against the engine).

| category | rate | evidence / control |
|---|---|---|
| **CLAMPED NO-OP move** (cursor cannot change; frame byte-identical) | **319/751 moves = 42.5%** (33.5% of all decisions) | uniform-random control on the same levels/wallet: **15.8%** (2.7× better) |
| └ concentrated in two presses | `move_backward` at `z=0` (146 nulls, **59.1%** of its 247 presses) and `move_down` at `y=5` (76, **47.2%** of 161) — **70%** of all nulls | `move_backward` is the most-emitted action of the run (247, 26.0%) |
| **DEAD BUDGET** (decisions after the last new `(cursor,placed)` state) | **306/951 = 32.2%**; **10/20** episodes end inside a detected periodic orbit (periods 1,1,3,3,4,4,4,6,8,8,8,12) | L02 49, L06 43, L20 32, L16 31, L04 29, L07 26, L05 23 |
| **WRONG-CELL PLACEMENT** | **99/116 standing blocks = 85.3%** on non-target cells; final precision **17/116 = 14.7%** | `matched_blocks` vs the reckoned final set |
| **CANCELLED WORK** (toggle removes a standing block) | **42/200 toggles = 21.0%**, 11 of them undoing a block placed ≤4 decisions earlier | pure churn: zero of the 42 removals cleared a wrong block the agent then replaced correctly |
| **TOGGLE PARITY CHURN** (toggle hits the same cell as the previous toggle) | **29/200 = 14.5%** | L07 f14–f49, L17 f12–f48 |
| **NEVER REACHED** | the cursor occupies **14.0 of 216 cells per episode (6.5%)**; only **29/116 target cells (25.0%)** are ever visited at all | a target cell the cursor never occupies is unplaceable regardless of perception |
| **REACH-WITHOUT-COMMIT** | of **34** decision points standing on an unfilled target cell, **17 (50.0%)** took the toggle — informative (2.25× its off-target base rate) but there were only **34 of 951 (3.6%)** such points | this is the one decision the model gets *right*; the harness never puts it in that position |
| PARSE / DEFAULT / ILLEGAL | **0 / 0 / 0** of 951 | `evaluation_info.successful_actions=50, default_actions=0` on every full episode |

Per-episode: `won` only on L01. Completion is **exactly 0% on 12/20 levels**; the non-zero ones
(L09 50%, L18 30%, L19 26.7%, L14 25%, L13 20%, L20 18.75%, L15 12.5%) are each the accident of a
toggle that happened to land on a target cell, never the end of a plan.

**Control summary (uniform random over the same 7 actions, 50-decision wallet, 20 levels × 1000 reps):**

| metric | anchor | uniform random | verdict |
|---|---|---|---|
| clamped moves | **42.5%** | 15.8% | **2.7× worse** |
| cells visited / ep | **14.0** | 24.4 | **1.7× less coverage** |
| toggles landing on a target cell | **8.5%** | 7.9% | indistinguishable |
| L1 distance toggled-cell → nearest target | **3.51** | 3.26 | **worse** (it toggles at walls/corners, the cells farthest from centrally-clustered targets) |
| mean matched blocks / ep | **0.85** | 0.36 | better — but it also places more (5.8 vs 4.5 net) |
| wins over a 20-level sweep | **1.00** | 0.246 (L01 19.95%, L02 0.65%, L03 2.20%, L04 1.80%, **0 on L05–L20**) | the anchor's entire measurable edge is that it reliably takes L01's spawn gift |

## 4. Binding constraint — classify it

**VLM-CEILING on cursor localisation + routing, amplified by a HARNESS-BUG-class feedback loop, and
entirely fixable inside the harness.** Attribution, with the alternatives ruled out:

- **Not HARNESS-BUG in the parse/fallback sense.** 0 parse failures, 0 default actions, 0 illegal
  actions across 951 completions; L01 proves the key mapping and the toggle semantics are right.
  There is nothing to fix in `tools/action.py:KEY_MAPPING` or in the parser.
- **Not TASK-GAP.** The task is fully determined by the observation (three orthographic panels are a
  complete tomographic specification; the engine re-tests set equality after every executed action),
  and §Headline-2 shows every level is winnable inside the wallet. Nor is it a commitment gap: the
  model commits at **2.46×** its base rate when it is standing in the right place.
- **It *is* a ceiling, and a narrow one.** The frozen `Qwen3-VL-30B-A3B-**Instruct**` is handed one
  isometric render, a 29-turn *image* history with no textual record of its own position, and a
  reply schema that admits no reasoning (8 tokens, `reasoning=None`, 951/951). **It cannot localise
  a highlighted voxel in an isometric projection, and it has nowhere to put the state that would let
  it count its own moves.** That is why *movement* comes out below chance while *committing* does not.
- **The harness's own contribution is the amplifier.** It re-asks an identical question after a
  provably no-op action (0.495 repeat probability), and it packs the context with 33.5% duplicate
  frames. Re-wording prose cannot touch this — an 8-token reflex has nowhere to hold state.

⇒ **The fix must put a decoded, self-verified cursor and a precomputed build tape on the decision
path and arbitrate the model's proposal against them — while leaving the model the decision it is
actually good at (what to commit / which candidate cell), which also keeps `vlm_calls` honest.**

## 5. Concrete changes to try (the gradient — ranked by leverage)

All of it is derived from the rendered observation only. **Keep the frozen model a real decision
participant** (eval_runner.py:341 voids the episode at `vlm_calls == 0`). `evaluator.baseline_step`
is pure with respect to the game — calling it costs no engine step — so one proposal per decision is
free; arbitrate against it, never silently replace it.

### 5.1 `tools/perception.py:decode_panels(frame_data.image)` — read the target exactly (THE lever)

The left strip carries three 90×90 panels, 6×6 cells at 15-px pitch; a filled cell is a 14×14 square
of exactly `RGB(80,80,80)` on exactly `RGB(20,20,20)`, so the read is all-or-nothing and can **refuse
itself**:

```
PANEL_X0 = 40 ; PITCH = 15
patch  = img[top+15*r+3 : top+15*r+11, 40+15*c+3 : 40+15*c+11]
filled = |median(patch) - (80,80,80)|_1 <= 12
TOP   top y=30    (col,row) -> (x,y) = (row,   5-col)
FRONT top y=180   (col,row) -> (y,z) = (5-col, 5-row)
SIDE  top y=330   (col,row) -> (x,z) = (col,   5-row)
hull  = { (x,y,z) : (x,y) in TOP and (y,z) in FRONT and (x,z) in SIDE }
```

Each lattice axis is drawn on **two** panels (`TOP.cols == FRONT.cols`, `TOP.rows == SIDE.cols`,
`FRONT.rows == SIDE.rows`), so cross-check the marginals and **stand the whole wrapper down to
`baseline_step` on disagreement** rather than guessing. Measured in this chain: all three masks
reproduce the true projections on **20/20** levels and `hull == target` on **18/20**. **Do not
assert the lattice constants — fit them.** The empty grid is drawn in `rgb(70,70,75)` and is
byte-identical on all 20 levels, so the three congruent square outlines give colour/pitch/origin as
*measurements*; a hard-coded constant table reads a wrong mask the moment the renderer moves and an
empty one if it rescales. Add a per-frame census + stability assertion so a silent mis-decode
becomes a refusal.
**Predicted effect:** target-cell visitation **25.0% → ~100%**; the 85.3% wrong-placement rate
collapses; recall 14.7% → ≥90%. This is the precondition for everything else.

### 5.2 `tools/state.py:read_cursor(frame_data.image)` — read the cursor by FOOTPRINT CONTAINMENT, not by the blob centroid

Mask `|px − (100,200,250)|₁ ≤ 10` **restricted to `x ≥ 300`** (the sidebar legend uses the same
colour — this is the one trap). The lattice is affine: corner `(0,0,0)` projects to `(239,305)` with
`+X=(20,40)`, `+Y=(17,−34)`, `+Z=(−40,0)`.

**Inverting the *centroid* to the nearest lattice point is only 122/150 correct** — placed cubes
occlude part of the highlight and drag the centroid off the lattice. Testing the mask against a
frame-measured cube **footprint** and demanding a **unique** covering cell is **150/150**; when the
cover is not unique, refuse the read and fall back to the ledger. **Re-read every callback**, so
dead-reckoning drift is structurally impossible and a clamped move is *observed* as "the cursor did
not move" instead of being integrated as motion. Track the placed set from your own toggles (it is
empty at frame 0 — the frame carries no blue cubes at spawn); **do not count blue pixels**, the 3-D
view self-occludes (13 of 96 real placements read *negative* blue ink). A *held* frame digest is
**not** proof of a null.
**Predicted effect:** removes the 42.5% clamp rate and the 21.0% cancellation rate outright, and
with them the 32.2% dead budget. On its own this is worth more than any prose change in the tree.

### 5.3 `tools/program.py:compile_order(hull, cursor)` — the build order that wins 20/20

The projection-consistent family is **upward-closed inside the hull**, and the engine re-tests set
equality after every executed action, so **a build order whose prefixes are plausible candidates is a
free sweep of hypotheses** — the engine ends the episode on the right one. Build a *minimal* member
first, then grow:

```
forced  = hull cells that are the UNIQUE hull coverer of some lit projection square
tier(c) = (c not in forced, c.z, round(|c - centroid(hull)|^2))
order   = concat over sorted tiers of a nearest-first walk inside the tier (from the live cursor)
```

Measured live in this chain (20 episodes, pixel-derived tapes, `max_steps=100`): **`won=true` on
20/20**, all wins at ≤45 callbacks, with L17 winning at callback 33 and L19 at 43 **as prefixes**
(their hulls, 12 and 19, exceed their targets, 10 and 15 — forced-first + layer + rim ordering puts
the truth in front of the spurious cells). Ablations: dropping the rim-distance key loses L19;
dropping nearest-first routing costs L20 the wallet; flat (no-`z`) tiering loses L17 **and** L19;
**reversing the growth direction loses 2**. This run's oracle-optimal tapes (max 33, mean 12.6)
confirm the wallet can absorb the hull's over-approximation on every level.
**Predicted effect: 1/20 → ~20/20. This is the whole gradient.**

### 5.4 `tools/vlm.py:decide()` — arbitration that keeps the VLM load-bearing *where it is strong*

This pass's §Headline-1 says exactly how to split the decision. Per callback: decode (5.1/5.2) →
`prop = await evaluator.baseline_step(frame_data)` → map `prop.key` back through the inverse of
`tools/action.py:KEY_MAPPING` (this finally puts `key_for` on the live path) → then

- **Movement: the tape wins.** Accept the proposal only when it is route-equivalent (same next tape
  cell, or any legal move that strictly reduces L1 distance to it); otherwise substitute the tape
  action. Hard-veto any move whose *decoded* cursor cannot change — that single veto is aimed at
  319/951 decisions and at the 0.495 repeat loop.
- **Commitment: the model keeps a real vote.** `P(place | on target)/P(place | off target) = 2.46`,
  so when the cursor is on a tape cell, *ask* and honour a `place_block` proposal; veto a toggle only
  when the ledger says the cell is already correct (the 14.5% churn + 21.0% cancellation). Where the
  hull is genuinely ambiguous, hand the model a short **symbolic shortlist** of candidate
  sub-structures to rank — a decision the ceiling does not block.
- **Budget rules the measurements force:** the hardest level needs 33 callbacks of 50 (and up to 45
  with a hull tape), so never spend a callback you cannot justify and never re-route in a way that
  lengthens the tape; prefer **not asking** over asking — an ask-iff-the-answer-can-change-the-key
  policy cut a comparable loop from 304 to 58 calls (82 MB → 3.6 MB of trace) with no loss of wins.
  There is ~23× wall headroom per episode subprocess (53–78 s against 1800 s), so a *verification*
  second call at a genuinely ambiguous decision is affordable; a second call at every decision is not
  worth its trace size.
**Predicted effect:** preserves 5.3's win rate while satisfying `vlm_calls >= 1` honestly.

### 5.5 Cheap, independently-testable sub-lever: take the spawn gift

`(2,2,0)` is a target cell on **9/20** levels and the anchor took it on **4/9**. An unconditional
`place_block` as callback 0 (before any decode has run) banks a free correct block on those 9 levels
and costs one callback + one later removal on the other 11. **Expected value: +5 matched blocks, +0
wins** (L01 is already won, and no other gift level has `|T| = 1`) — so ship it as a one-line tape
prefix inside 5.3, **not** as a standalone candidate, and only if it does not perturb L01.

### 5.6 Inject the ledger (`memory.md`) — but obey the two injection rules that have cost wins

Render each turn: decoded cursor, decoded target list, next tape cell, standing placed set, **which
of the 6 moves are legal-vs-clamped at this cursor**, and callbacks remaining. All self-derived; no
oracle. Two hard constraints:

1. **Never speak on turn 1.** A mechanism that injects on message 0 of the first turn costs the L01
   gift win (3/3 → 2/3 in paired arms). Let the first decision be the pristine one.
2. **Rebuild message 0 from a stashed base; never splice.** A `_merge` that truncates at a marker
   and appends **deletes SpatialWorld's JSON response contract** from every call after the first
   (measured: 1/29 calls kept the contract vs 15/15 when rebuilt, 3.11% parse failures — and the
   parse fallback then masquerades as a vote). Message 0 is the only per-turn channel.

**Predicted effect:** higher proposal-acceptance (fewer overrides to justify) and a readable next
diagnosis; **it is not the win lever by itself** — prose cannot carry state for an 8-token reflex.

### 5.7 Make the applied behaviour auditable

Stamp `Action.metadata` with `frame_number`, decoded cursor, `|hull|`, `|forced|`, tape index,
proposal action, applied action, override reason, clamp verdict, callbacks remaining.
Reconstructing all 20 trajectories offline was the entire cost of this diagnosis; one metadata dict
removes it. Note `metadata["repeat"]` is honoured but **buys latency, not budget** — a clamped
repeat still pays full fare. Never log oracle comparisons.
**Predicted effect:** no score change; the next gradient becomes direct instead of inferred.

### 5.8 Rig hygiene (so the next candidate's offline numbers are real)

- Stub at **`evaluator.client`**, not at `create_openai_response`, or the seed/plan is not actually
  priced; a stub client **must carry the real model name** or the call routes to `client.responses`
  and silently measures nothing.
- Apply every override **after** `build_agent` — it purges `sys.modules["tools*"]`, so a "control"
  configured before it silently measures the shipped config.
- Price a closed loop by **breaking its premises** (a blinded decode should collapse the candidate
  onto the parent answer-for-answer); on a clean split the loop reads as a no-op.
- Don't alternate two harnesses in one process: it leaks `sys.path` and has inverted a verdict.
- `HISTORY_WINDOW` is unset here (29). If a rig sets it, it is not measuring this configuration.

## 6. Exhausted — do NOT repeat

- **The pass-through itself.** `build_agent → tools.vlm.decide → baseline_step` unchanged is **1/20**
  across independent batches and lanes, at 50 callbacks *and* at 60. More callbacks do not help —
  this run proves the wallet already has 1.5×–50× headroom (max optimal tape 33 of 50).
- **Parser / retry / key-map / fallback work.** 0 parse failures, 0 default actions, 0 illegal
  actions in 951 completions; 7 distinct reply strings, all legal; L01 proves the mapping.
- **Re-wording the priors.** "Study the three views" and "you can remove misplaced blocks" are
  already in `skills/building.md`; `rules.md` already states the 6 directions, the toggle, and that
  every move is exactly 1 cell. They supply no coordinates, no parity, no clamp model — and the model
  is an 8-token `Instruct` completion with `reasoning=None` on 951/951, so prose cannot hold state
  for it. The prompt-token sequence being identical across all 19 full episodes is the proof that no
  state-bearing text reaches the context.
- **`memory.md` as prose.** A scratchpad not filled from *decoded* state and not consumed by an
  arbiter cannot stop a period-1 orbit (`block3d_2_816ect9z` f01–f49).
- **Reading the cursor from a HUD caption.** There is none — the `Cursor: (x,y,z)` caption was
  deleted from both engine copies on 2026-10-02. Use the `RGB(100,200,250)` highlight (5.2).
- **Counting blue pixels to confirm a placement.** The 3-D view self-occludes; 13 of 96 real
  placements read *negative* blue ink. Track parity from your own toggles.
- **`evaluator.game.<attr>`.** Only `.screen` is allowed, and in headless mode it is unreliable —
  use `frame_data.image` (already the array, so even `capture_game_screenshot` is optional).
- **Any no-VLM solver.** `vlm_calls == 0` → `score 0.0`, `valid:false` (eval_runner.py:341).
- **Hull-first descent / "build the hull then prune".** The wallet cannot afford removals, and the
  descent is dead on L19's 15,842-member family. Grow upward from a minimal member (5.3).
- **A plain boustrophedon / memoryless sweep as the movement cure.** It folds into a 12-cell orbit
  with centre-seeking 2-cycles; the memoryless cure must be a Hamiltonian face sweep. (This run's
  own evidence: the anchor's unguided wandering reaches 14.0 cells where *random* reaches 24.4.)
- **Turn/route micro-optimisation as the win lever.** Build *order* is near-worthless on its own;
  the leverage is forced-first + upward growth (5.3), not shaving presses.
- **Optimising `cost`.** Dead in this lane's selection — `pick_best` is `(primary, -idx)`.
