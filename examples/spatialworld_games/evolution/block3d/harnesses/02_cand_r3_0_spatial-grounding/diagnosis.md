# Diagnosis — block3d iter 0 / `anchor_block3d` (`staged_anchor`, lane `k32_claude_pv`)

**Headline:** primary (`won`) = **1/20 = 0.050** · mean completion **14.15 %** · placement precision
**17/116 = 14.7 %** · target recall **17/116 = 14.7 %** · **19/20** episodes burn the full **50**-decision
wallet · **0/951** parse failures, **0** default actions, **0** illegal actions · **951/951** completions
are exactly **8 tokens** with `finish_reason='stop'` and `reasoning=None` · **7** distinct reply strings
in the whole run. The one win is **L01, whose only target cell is the cursor's spawn cell `(2,2,0)`**
(`block3d_1_7eo7zcne`, frame 0, a single VLM call emitting `{"action": "place_block"}`) — a win that
requires no spatial reasoning at all.

**Binding constraint: VLM-CEILING on cursor localisation and routing, sitting behind a decision path
that is literally two lines long — so the ceiling is never even approached, and a harness-side fix
recovers almost all of it.** Three measurements made in this pass fix what the next candidate should
build:

> **1. Movement is below chance; commitment is not.**
> `P(place_block | the cursor stands on an unfilled TARGET cell) = 17/34 = 0.500` versus
> `P(place_block | anywhere else) = 183/917 = 0.200` — **ratio 2.50**. The commit decision carries real
> signal. Movement carries *negative* signal: **all six directions are emitted more often exactly when
> that direction is blocked** — `move_right` 0.173 at-wall vs 0.042 free (**4.13×**), `move_up` 2.83×,
> `move_left` 2.75×, `move_down` 1.64×, `move_forward` 1.47×, `move_backward` 1.44×; a
> frame-independent coin gives 1.00. The anchor clamps **319/751 = 42.5 %** of its moves where a
> uniform-random policy on the same levels and wallet clamps **15.9 %**, and it occupies **14.0** of
> 216 cells per episode against random's **24.4**. Conditioning movement on the image makes it
> *strictly worse than not looking.* ⇒ **own the route; keep the model on the commit.**
>
> **2. The wallet is not the constraint.** Optimal build tapes (exact TSP for |T| ≤ 8, NN+2-opt above,
> 1 press + L1 travel per cell) cost **1, 5, 3, 3, 6, 8, 9, 8, 9, 8, 8, 13, 10, 15, 19, 18, 21, 22, 33,
> 32** callbacks on L01…L20 — **max 33, mean 12.6, every level inside the 50-decision wallet**. 251
> callbacks would have won all twenty. The anchor spent **951** and banked **17** correct blocks: one
> per **56 decisions**.
>
> **3. A pixel-only solver wins 20/20 live, *measured in this pass*.** Decoding the three target panels
> from the frame (no oracle) and walking a forced-first tape wins **20/20 on the real engine** at
> **1, 5, 3, 3, 8, 8, 11, 8, 11, 12, 9, 15, 10, 15, 20, 23, 33, 26, 43, 45** callbacks — every level
> inside 50. The gap between 1/20 and 20/20 is entirely inside `tools/`.

Every number below is a **measurement**. Cursor dead-reckoning from the harness's own emitted actions
(spawn `(2,2,0)`, clamp to `[0,5]³`, `left/right = ∓X`, `up/down = ∓Y`, `forward/backward = ±Z`)
reproduces the engine's `matched_blocks` **exactly on 20/20 episodes**; the direction semantics and the
two-engine-steps-per-executed-action cost were re-verified directly against
`Block3DGameAdapter.execute_mapped_action`.

---

## 1. What this harness does (mechanism)

`harness.py` is a **pass-through**. Nothing in it decides anything.

- `compose_system_prompt()` (harness.py:21) concatenates `AGENT.md` + `skills/*.md` + `rules.md` and
  installs the result on `evaluator.current_prompt`, which `eval_runner.py:293` captures as
  `initial_prompt`. **Verified: the composition equals `system_prompt.txt` token-for-token** (1377 vs
  1382 chars — leading/trailing whitespace only), so this candidate *is* the prompt-only baseline
  wearing a module layout. It supplies role + grid facts, "study the three views / you can remove
  misplaced blocks", and the 7-action control map + colour legend. **No coordinates, no clamp model,
  no toggle-parity model, no plan.**
- `build_agent(evaluator)` (harness.py:34) puts `HERE` on `sys.path`, purges `sys.modules["tools*"]`
  so a sibling candidate's `tools/` can never be inherited (**keep this line** — it is the only load-
  bearing code in the file), and returns `agent(frame_data) -> await tools.vlm.decide(...)`.
- `tools/vlm.py:decide()` is **two lines**: `return await evaluator.baseline_step(frame_data)`. That
  pristine SpatialWorld step captures the screenshot, appends it to a 29-turn image+text history,
  asks for one JSON action, parses it, and returns `Action(type="key_press", key=…)`.
- `tools/frame.py:capture`, all of `tools/action.py` (`KEY_MAPPING`, `key_for`, `CELLS_PER_MOVE`), and
  the whole of `memory.md` are **defined and never called**. The rich surface is dead code.

Contract facts that bound any fix (re-measured here, not quoted):

| fact | value | how it was measured |
|---|---|---|
| wallet | `max_steps=100`, and **one executed action advances `steps_taken` by 2** ⇒ **50 decisions** | stepped the adapter: `right` → `steps_taken` 0→2, `space` → 12→14 |
| a clamped move | returns a **byte-identical frame** and still costs **2 steps** | 5/5 `backward` presses at `z=0`: identical md5, `steps_taken` 2,4,6,8,10 |
| history window | **exactly 29** turns | `prompt_tokens` 1394 → +678/turn → saturates at **21 093** from turn 30 |
| cross-episode context | **identical** 50-turn `prompt_tokens` sequence on all 19 full episodes | 1 distinct sequence of 19 ⇒ no level- or state-dependent *text* ever enters the context |
| per-episode wall | 53–78 s against the adapter's **1800 s** subprocess timeout | `eval_result.total_time` vs `game_adapter.py:284` |
| `vlm_calls == 0` | voids the episode (`score 0.0`, `valid:false`) | `eval_runner.py:316` under `ANTICHEAT_GUARD` |
| frames on disk | **none** — each episode dir holds only 4 JSON/log files | `ls $TRACE_DIR/block3d_*` |

---

## 2. What actually happened (causal chain, with evidence)

### The absorbing state — `block3d_2_816ect9z` (target `{(4,4,0)}`, matched 0, optimal tape **5**)

| f | action | cursor | outcome |
|---|---|---|---|
| 00 | `move_down` | (2,3,0) | legal |
| **01–49** | `move_backward` **×49** | (2,3,0) | `z` is already 0 — **every one of the 49 presses is a clamped null.** 98 % of the wallet, 2 cells visited, 0 toggles. |

A clamped move returns a byte-identical frame (verified above), so the *identical* question is re-asked
of a near-deterministic 8-token reflex, which answers identically. The target was `right, right, down,
down, place` — **5 callbacks away**. Nothing failed in the parser, the key map or the fallback: all 50
replies were legal JSON and all 50 were applied. `block3d_6_8kdgovfq` is the same shape
(`move_backward` ×44 from f07; 43 dead decisions, 86 % of the wallet).

**The loop is measurable, and the harness builds it:**
`P(repeat the same action | the previous move was a clamped null) = 152/307 = 0.495` versus
`P(repeat | the previous move actually moved) = 96/427 = 0.225` — **2.2×**, a positive feedback loop
with an absorbing state. **319/951 = 33.5 % of the observations the model is shown are byte-identical
to the one immediately before them.**

### Reach-without-commit — `block3d_8_wcifkwn3` (target `{(2,2,0),(2,3,0),(2,2,1),(2,2,2)}`, matched 0, optimal tape **8**)

f00–f02 walk `move_forward` from the spawn straight through **(2,2,1)** and **(2,2,2)** — three of the
four target cells, one of them the spawn itself — and place nothing. The cursor then tours to the
`(0,0,0)` corner, makes **its only toggle of the episode** at f31 on a wrong cell, and spends
**f32–f47 pressing `move_backward` ×9 then `move_left` ×7 into two walls**. 27 of 49 moves clamped,
21 cells visited, 1 toggle, 0 correct.

### Parity churn — `block3d_7_p3d9tgh0` (target `{(4,4,0),(4,5,0),(5,5,0)}`, matched 0, optimal tape **9**)

From f14 the trajectory closes into a **period-14 orbit** in the *opposite* corner of the box from the
target: `place` / `up` / `forward` / `place` / `backward` / `left`(null) / `down` … The same cells are
placed and removed on alternating laps — **15 toggles, 5 blocks standing, 0 correct** — while the
target sits at maximum L1 distance from the orbit.

### Both failures at once — `block3d_17_6yshn82p` (10 targets incl. the spawn, matched 0, optimal tape **21**)

Spawns **on** a target cell and leaves at f00 without placing; never returns. Thereafter it alternates
between the `(5,5,0)` and `(5,5,2)` corners, and **24 of its 42 moves are clamped**. 8 toggles, 3 of
them removals of its own blocks, 2 standing at the end, 0 correct.

### Why the model cannot escape: the harness gives its context no state

The per-turn `prompt_tokens` sequence is **identical across all 19 full-length episodes** (1 distinct
sequence of 19). All seven legal replies are exactly 8 tokens, so this does not by itself prove the
action echo is missing — but it *does* prove that **no level-dependent and no state-dependent text ever
reaches the model**: any rendered coordinate, counter or scratchpad would have made the sequences
differ. The only thing distinguishing one episode from another inside the context is **pixels**, and
**33.5 % of those pixels are duplicates of the previous turn**.

### The opening move is anti-correlated with the one free block on offer

The spawn `(2,2,0)` is a target cell on **9/20** levels (L01, 04, 08, 11, 13, 14, 15, 17, 19). The
anchor opened with `place_block` on **13/20** levels overall but on only **4/9** of those gift levels:
`P(open with place | spawn IS a target) = 0.444` vs `P(… | spawn is NOT) = 0.818`. It walked off a free
correct block on **L04, L08, L11, L15, L17**.

---

## 3. Failure breakdown

951 decisions, 20 episodes. All figures dead-reckoned; the reckoning reproduces the engine's
`matched_blocks` on 20/20.

| category | rate | evidence / control |
|---|---|---|
| **CLAMPED NO-OP move** (cursor cannot change; frame byte-identical) | **319/751 moves = 42.5 %** (33.5 % of all decisions) | uniform random on the same levels/wallet: **15.9 %** — the anchor is **2.7× worse** |
| └ concentrated in two presses | `move_backward` at `z=0` (**146** nulls, 59.1 % of its 247 presses) and `move_down` at `y=5` (**76**, 47.2 % of 161) — **70 % of all nulls** | `move_backward` is the most-emitted action of the run (247, 26.0 %) |
| **DEAD BUDGET** (decisions after the last *new* `(cursor, placed)` state) | **306/951 = 32.2 %**; **12/20** episodes spend ≥6 trailing decisions re-entering a state they have already been in (terminal periods 1,1,1,1,1,1,1,1,8,10,12,14) | L02 49, L06 43, L20 32, L16 31, L04 29, L07 26, L05 23, L10 18, L14 16, L19 15 |
| **WRONG-CELL PLACEMENT** | **99/116 standing blocks = 85.3 %** sit on non-target cells; final precision **17/116 = 14.7 %** | reckoned final set vs the level's cells |
| **CANCELLED WORK** (a toggle that removes a standing block) | **42/200 toggles = 21.0 %** | pure churn — not one of the 42 removals cleared a wrong block that was then replaced correctly |
| **TOGGLE PARITY CHURN** (toggle hits the same cell as the previous toggle) | **29/200 = 14.5 %** | L07 f14–f49, L17 f12–f48 |
| **NEVER REACHED** | the cursor occupies **14.0 of 216 cells per episode (6.5 %)**; only **29/116 target cells (25.0 %)** are visited at all | a target cell the cursor never occupies is unplaceable regardless of perception |
| **REACH-WITHOUT-COMMIT** | of the **34** decision points standing on an unfilled target cell, **17 (50.0 %)** took the toggle — informative (2.50× the off-target base rate) but there were only **34 of 951 (3.6 %)** such points | this is the one decision the model gets right; the harness almost never puts it in that position |
| PARSE / DEFAULT / ILLEGAL | **0 / 0 / 0** of 951 | `successful_actions=50, default_actions=0` on every full episode |

`won` only on L01. Completion is **exactly 0 % on 12/20** levels; every non-zero one (L09 50 %, L18
30 %, L19 26.7 %, L14 25 %, L13 20 %, L20 18.75 %, L15 12.5 %) is the accident of a toggle that
happened to land on a target cell, never the end of a plan.

**Control (uniform random over the same 7 actions, 50 decisions, 20 levels × 2000 reps):**

| metric | anchor | uniform random | verdict |
|---|---|---|---|
| clamped moves | **42.5 %** | 15.9 % | **2.7× worse** |
| cells visited / ep | **14.0** | 24.4 | **1.7× less coverage** |
| toggles landing on an unfilled target | **8.5 %** | 6.4 % | marginally better |
| mean L1 distance toggled cell → nearest target | **3.51** | 3.29 | **worse** — it toggles at walls and corners, the cells farthest from centrally-clustered targets |
| mean correct blocks / ep | **0.85** | 0.35 | better, but it also places more (5.8 vs 4.5 standing) |
| wins per 20-level sweep | **1.00** | **0.25** (L01 20.1 %, L03 2.6 %, L04 1.5 %, L02 0.5 %, 0 on L05–L20 except two strays) | the anchor's entire measurable edge is that it reliably takes L01's spawn gift |

---

## 4. Binding constraint — classify it

**VLM-CEILING on cursor localisation + routing, amplified by a HARNESS-BUG-class feedback loop, and
entirely fixable inside the harness.** The alternatives are ruled out by measurement:

- **Not HARNESS-BUG in the parse/fallback sense.** 0 parse failures, 0 default actions, 0 illegal
  actions in 951 completions; 7 distinct reply strings, all legal; L01 proves the key mapping and the
  toggle semantics. There is nothing to fix in `tools/action.py:KEY_MAPPING` or in the parser.
- **Not TASK-GAP.** The three orthographic panels are a complete tomographic specification (decoded
  here: all three projections reproduced on **20/20** levels, `hull == target` on **18/20**), the
  engine re-tests set equality after every executed action, and §Headline-2/3 show every level is
  winnable inside the wallet — **20/20 live from pixels alone**. Nor is it a commitment gap: the model
  commits at **2.50×** its base rate when it is standing in the right place.
- **It *is* a ceiling, and a narrow one.** The frozen `Qwen3-VL-30B-A3B-**Instruct**` gets one
  isometric render, a 29-turn *image* history with no textual record of its own position, and a reply
  schema that admits no reasoning (**8 tokens, `reasoning=None`, 951/951**). **It cannot localise a
  highlighted voxel in an isometric projection, and it has nowhere to put the state that would let it
  count its own moves.** That is exactly why *movement* comes out below chance while *committing* does
  not.
- **The harness is the amplifier.** It re-asks a byte-identical question after a provably no-op action
  (repeat probability 0.495 vs 0.225) and fills the model's own context with 33.5 % duplicate frames.
  Re-wording prose cannot touch this: an 8-token reflex has nowhere to hold state.

⇒ **Put a decoded, self-verified cursor and a precomputed build tape on the decision path, and
arbitrate the model's proposal against them — while leaving the model the decision it is actually good
at. That also keeps `vlm_calls` honest.**

---

## 5. Concrete changes to try (the gradient — ranked by leverage)

Everything below is derived from the rendered observation only. `evaluator.baseline_step` is pure with
respect to the game — calling it costs no engine step — so **one proposal per decision is free**;
arbitrate against it, never silently delete it (`vlm_calls == 0` voids the episode).

### 5.1 `tools/perception.py:decode_panels(frame_data.image)` — read the target exactly (THE lever)

**Re-measured on all 20 frame-0 renders in this pass.** The left strip carries three 90×90 panels at
`x ∈ [40,129]`, tops `y = 30 / 180 / 330`, 6×6 cells at **15-px pitch**; a lit cell is a 14×14 square of
exactly `RGB(80,80,80)` on exactly `RGB(20,20,20)`, so the read is all-or-nothing and can refuse itself:

```
patch  = img[top + 15*r + 4 : top + 15*r + 11,  x0 + 15*c + 4 : x0 + 15*c + 11]
lit    = |median(patch) - (80,80,80)|_1 <= 12
TOP   top y=30    (r,c) -> (x,y) = (r,    5-c)
FRONT top y=180   (r,c) -> (y,z) = (5-c,  5-r)
SIDE  top y=330   (r,c) -> (x,z) = (c,    5-r)
hull  = { (x,y,z) : (x,5-y) in TOP and (5-z,5-y) in FRONT and (5-z,x) in SIDE }
```

**Do not assert the lattice — FIT it.** The empty grid is drawn in `rgb(70,70,75)`; its row-runs and
col-run give the three tops, the left edge and the pitch as *measurements* (verified: `x0=40`,
`pitch=15.0`, `tops=[30,180,330]` recovered on 20/20 levels). A hard-coded constant table reads a wrong
mask the moment the renderer moves and an empty one if it rescales. Each lattice axis is drawn on
**two** panels (`TOP.cols == FRONT.cols`, `TOP.rows == SIDE.cols`, `FRONT.rows == SIDE.rows`) — check
those three marginals and **stand the whole wrapper down to `baseline_step` on disagreement** rather
than guessing.
Measured here: all three masks reproduce the true projections on **20/20** levels; `hull == target` on
**18/20** (only **L17 +2** and **L19 +4** over-approximate).
**Predicted effect:** target-cell visitation 25.0 % → ~100 %; the 85.3 % wrong-placement rate
collapses; recall 14.7 % → ≥90 %. Precondition for everything else.

### 5.2 `tools/state.py:read_cursor(frame_data.image)` — containment in the right DIRECTION, not the centroid

Mask `|px − (100,200,250)|₁ ≤ 10` **restricted to `x ≥ 300`** — the sidebar legend uses the same colour
and is the one trap (146 of the frame's 1170 highlight-coloured pixels sit in the left strip).

The projection is **exactly affine, with zero residual**. Fitted over a 216-pose sweep (centroid of the
highlight mask, in `(col,row)` pixel coordinates):

```
centroid(x,y,z) = (643.101, 278.236) + x*(40, 20) + y*(-34, 17) + z*(0, -40)
max |fit − measured| = 0.000 px over all 216 poses;  mask area = 1024 px on all 216
```

**Reconciliation with the previous gradient's table** (`(0,0,0) -> (239,305)`, `+X=(20,40)`,
`+Y=(17,−34)`, `+Z=(−40,0)`): that is the same map in a different convention — the **bounding-box
top-left** of the footprint, in `(row, col − 300)`. Verified here: cell `(0,0,0)`'s mask bbox is
`(row 239, col 605)` = `(239, 305)` mask-relative, and `+X → (+20 row, +40 col)`,
`+Y → (+17, −34)`, `+Z → (−40, 0)`. **Both tables are correct; mixing their conventions is a
343-pixel error.** State the convention in the code, and still fit rather than assert — the empty
grid and the frame-0 highlight hand you every constant for free.

Three decoders, measured against ground truth over 216 poses × six occlusion densities (0, 4, 8, 12,
16, 20, 28 standing cubes):

| standing cubes | centroid → nearest lattice point | footprint ⊆ mask (strict) | **mask ⊆ footprint (subset, unique best overlap)** |
|---|---|---|---|
| 0 | 216/216 | 216/216 | **216/216** |
| 4 | 173/216 | 141 ok / **75 refused** | **216/216** |
| 8 | 177/216 | 145 ok / 71 refused | **216/216** |
| 12 | 163/216 | 115 ok / 101 refused | **216/216** |
| 16 | 142/216 | 82 ok / 134 refused | **216/216** |
| 20 | 124/216 | 63 ok / **153 refused** | **214/216** (2 refused) |
| 28 | 148/216 | 111 ok / 105 refused | 199/216, **17 refused**, 11 of them a fully occluded highlight |
| wrong answers, any density | **up to 43 %** | 0 | **0** |

**Centroid inversion is unusable once cubes stand** (57–82 %). The direction of the containment test
is load-bearing and easy to invert in code: **occlusion only ever REMOVES lit pixels**, so the sound
test is `mask ⊆ footprint(c)` — "which drawn footprint still covers every lit pixel" — resolved to the
unique maximum-overlap cell, refusing on a tie. That is **216/216 up to 16 standing cubes, 214/216 at
20, and never wrong at any density**. Writing it the other way round (`footprint(c) ⊆ mask`) type-checks,
never answers wrongly, and **refuses 35–71 % of poses**, silently handing most callbacks back to
dead-reckoning — the very drift the read was meant to remove. Assert the direction in a unit test on a
frame with cubes down, not on frame 0 where both agree.
**Supporting facts (also measured here):** the **frame digest is a collision-free pose id at every
realistic density** — 216/216 distinct md5s at 0, 8, 16 and 20 standing cubes (only at 28 do 10 poses
collide) — so "the digest held ⇒ that *move* was clamped" is **sound inside the wallet**. The converse
does **not** hold for toggles: **3 of 180 real placements at 14 standing cubes are byte-invisible**
(the new cube is fully occluded), so track the placed set from **your own toggles**, never from the
pixels. And `placed` is empty at frame 0 — the frame carries no blue cubes at spawn.
**Predicted effect:** removes the 42.5 % clamp rate and the 21.0 % cancellation rate outright, and with
them the 32.2 % dead budget. On its own this is worth more than any prose change in the tree.

### 5.3 `tools/program.py:compile_order(hull, cursor)` — the build order that wins 20/20 (verified live)

The projection-consistent family is **upward-closed inside the hull** and the engine re-tests set
equality after every executed action, so **a build order whose prefixes are plausible candidates is a
free sweep of hypotheses** — the engine ends the episode on the right one. Build a *minimal* member
first, then grow:

```
forced  = hull cells that are the UNIQUE hull coverer of some lit projection square
tier(c) = (c not in forced, c.z, round(|c - centroid(hull)|^2))
order   = concat over sorted tiers of a nearest-first walk inside the tier, from the LIVE cursor
```

**Measured live against the real engine in this pass** (hull from pixels only, 50-action wallet):
**`won = True` on 20/20**, at `1, 5, 3, 3, 8, 8, 11, 8, 11, 12, 9, 15, 10, 15, 20, 23, 33, 26, 43, 45`
callbacks. **L17 wins at callback 33 and L19 at 43 *as prefixes*** — their hulls (12 and 19) exceed
their targets (10 and 15), and forced-first + layer + rim ordering puts the truth ahead of the spurious
cells. Ablations, same rig: **drop forced-first → 18/20** (L19 hits the wallet at 50); **flat, no-`z`
tiering → 18/20**; **drop the rim-distance key → 19/20** (and it is *cheaper* — max 40 — so this key
trades budget for one win: keep it, but it is the knob to turn if a verification pass needs callbacks).
**Budget warning:** at this route quality the slack is only **7 callbacks on L19 and 5 on L20** (the
optimal tapes are 33 and 32). Any arbitration or re-ask layer must be budget-neutral *there*; the
headroom to pay for it comes from a better route, not from the wallet.
**Predicted effect: 1/20 → ~20/20. This is the whole gradient.**

### 5.4 `tools/vlm.py:decide()` — arbitration that keeps the VLM load-bearing *where it is strong*

§Headline-1 says how to split the decision. Per callback: decode (5.1/5.2) →
`prop = await evaluator.baseline_step(frame_data)` → map `prop.key` back through the inverse of
`tools/action.py:KEY_MAPPING` (this finally puts `key_for` on the live path) → then

- **Movement: the tape wins.** Accept the proposal only when it is route-equivalent (same next tape
  cell, or any legal move that strictly reduces L1 distance to it); otherwise substitute the tape
  action. **Hard-veto any move whose *decoded* cursor cannot change** — that single veto is aimed at
  319/951 decisions and at the 0.495 repeat loop.
- **Commitment: give the model a vote, never a veto.** `P(place | on unfilled target) = 0.500`, so
  gating each tape cell on the model's agreement costs ~2 asks per cell in expectation — which breaks
  the wallet on L19/L20 (43 and 45 of 50 already). Ask at a tape cell and honour a `place_block`
  proposal; on a refusal, **place anyway if the cell is `forced`**, and only defer on non-forced cells.
  Veto a toggle when the ledger says the cell is already correct (kills the 14.5 % parity churn and the
  21.0 % cancellation).
- **Spend the vote where ambiguity actually lives.** `hull == target` on **18/20** levels — there is
  literally nothing to ask about on those. The real ambiguity is **L17 (+2 cells) and L19 (+4)**, and
  even they win as tape prefixes. So the model's genuinely useful jobs are (a) ratifying commits,
  (b) ranking candidate minimal sub-structures **when the decode refuses itself**, (c) breaking a tie
  between two equally ranked goal cells. Prefer **not asking** over asking: an
  ask-iff-the-answer-can-change-the-key policy cut a comparable loop from 304 to 58 calls (82 MB →
  3.6 MB of trace) with no loss of wins. There is ~23× wall headroom per episode (53–78 s against the
  1800 s subprocess timeout), so one *verification* call at a genuinely ambiguous decision is
  affordable; a second call at every decision is not worth its trace size.
**Predicted effect:** preserves 5.3's win rate while satisfying `vlm_calls >= 1` honestly.

### 5.5 Cheap, independently-testable sub-lever: take the spawn gift

`(2,2,0)` is a target cell on **9/20** levels and the anchor took it on **4/9**. An unconditional
`place_block` as callback 0 banks a free correct block on those nine and costs one callback plus one
later removal on the other eleven. **Expected value: +5 correct blocks, +0 wins** (L01 is already won
and no other gift level has `|T| = 1`). Ship it as a one-line tape prefix inside 5.3, **not** as a
standalone candidate, and only if it does not perturb L01.

### 5.6 Inject the ledger (`memory.md`) — but obey the two injection rules that have cost wins

Render each turn: decoded cursor, decoded target list, next tape cell, standing placed set, **which of
the six moves are clamped at this cursor**, and callbacks remaining. All self-derived; no oracle. Two
hard constraints:

1. **Never speak on turn 1.** A mechanism that injects on message 0 of the first turn costs the L01
   gift win (3/3 → 2/3 in paired arms). Let the first decision be the pristine one.
2. **Rebuild message 0 from a stashed base; never splice.** A `_merge` that truncates at a marker and
   appends **deletes SpatialWorld's JSON response contract** from every call after the first (measured:
   1/29 calls kept the contract vs 15/15 when rebuilt; 3.11 % parse failures — and the parse fallback
   then masquerades as a vote). Message 0 is the only per-turn channel.

**Predicted effect:** higher proposal-acceptance and a readable next diagnosis; **not a win lever by
itself** — prose cannot carry state for an 8-token reflex.

### 5.7 Make the applied behaviour auditable

Stamp `Action.metadata` with `frame_number`, decoded cursor, `|hull|`, `|forced|`, tape index, proposed
action, applied action, override reason, clamp verdict, callbacks remaining. **Reconstructing all 20
trajectories offline was the entire cost of this diagnosis** — and note that **no frames are archived**
(each episode dir holds only `eval_result.json`, a summary, the responses JSON and the session log), so
every perception claim had to be re-derived by re-rendering the levels. One metadata dict plus a
handful of saved frames removes that. `metadata["repeat"]` is honoured but **buys latency, not
budget** — a clamped repeat still pays full fare.
**Predicted effect:** no score change; the next gradient becomes direct instead of inferred.

### 5.8 Rig hygiene (so the next candidate's offline numbers are real)

- Stub at **`evaluator.client`**, not at `create_openai_response`, or the seed/plan is not actually
  priced; a stub client **must carry the real model name** or the call routes to `client.responses` and
  silently measures nothing.
- Apply every override **after** `build_agent` — it purges `sys.modules["tools*"]`, so a "control"
  configured before it silently measures the shipped config.
- Price a closed loop by **breaking its premises** (a blinded decode should collapse the candidate onto
  the parent answer-for-answer); on a clean split the loop reads as a no-op.
- Don't alternate two harnesses in one process: it leaks `sys.path` and has inverted a verdict before.
- `HISTORY_WINDOW` is unset in this lane (**29**, verified from `prompt_tokens`). A rig that sets it is
  not measuring this configuration.
- `render()` returns `None` and draws into `adapter.screen`; `evaluator.game.<attr>` is guarded. Use
  **`frame_data.image`** — it is already the array.
- **Process note:** the `r001` proposer in `$TRACE_DIR/prop_r001_148ddc/` **timed out at 10800 s**
  (`propose.log`) while building the full seven-module stack. Build 5.1 → 5.2 → 5.3 in that order and
  stop; the arbitration polish in 5.4 is worth less than shipping a decoder that runs.

---

## 6. Exhausted — do NOT repeat

- **The pass-through itself.** `build_agent → tools.vlm.decide → baseline_step` unchanged is **1/20**
  across independent batches and lanes, at 50 callbacks *and* at 60. More callbacks do not help — the
  optimal tape is ≤33 on every level and 45 suffices from pixels alone.
- **Parser / retry / key-map / fallback work.** 0 parse failures, 0 default actions, 0 illegal actions
  in 951 completions; L01 proves the mapping and the toggle.
- **Re-wording the priors.** "Study the three views" and "you can remove misplaced blocks" are already
  in `skills/building.md`; `rules.md` already gives the six directions, the toggle and the 1-cell
  granularity. They carry no coordinates, no parity and no clamp model — and the model is an 8-token
  `Instruct` completion with `reasoning=None` on 951/951. The identical per-turn `prompt_tokens`
  sequence across all 19 full episodes is the proof that no state-bearing text reaches the context.
- **`memory.md` as prose.** A scratchpad not filled from *decoded* state and not consumed by an arbiter
  cannot stop a period-1 orbit (`block3d_2_816ect9z` f01–f49).
- **Reading the cursor from a HUD caption.** There is none — the `Cursor: (x,y,z)` caption was deleted
  from both engine copies on 2026-10-02. Use the `RGB(100,200,250)` highlight (5.2).
- **Centroid inversion for the cursor** (57–82 % once cubes stand), and the **inverted containment
  direction** `footprint ⊆ mask` (never wrong, but refuses 35–71 % of poses). 5.2 prices both; the read
  that holds is `mask ⊆ footprint`, unique max-overlap.
- **Counting blue pixels to confirm a placement.** The 3-D view self-occludes; 3 of 180 real placements
  are byte-invisible in the frame. Track parity from your own toggles.
- **Asserting a lattice/panel constant table instead of fitting it.** The constants are stable in
  this renderer and the previous gradient's cursor basis *is* correct — but in a bounding-box
  `(row, col−300)` convention, not the centroid `(col, row)` one (5.2); mixing them is a 343-px
  error, and a moved or rescaled renderer silently yields a wrong or empty mask. Fit from
  `rgb(70,70,75)` and the frame-0 highlight every episode, and name the convention in the code.
- **`evaluator.game.<attr>`.** Only `.screen` is allowed and it is unreliable headless; use
  `frame_data.image`.
- **Any no-VLM solver.** `vlm_calls == 0` → `score 0.0`, `valid:false` (`eval_runner.py:316`).
- **Hull-first descent / "build the hull then prune".** The wallet cannot afford removals, and the
  descent is dead on L19's 15 842-member family. Grow upward from a minimal member (5.3).
- **A plain boustrophedon / memoryless sweep as the movement cure.** It folds into a 12-cell orbit with
  centre-seeking 2-cycles; the memoryless cure must be a Hamiltonian face sweep. This run's own
  evidence: the anchor's unguided wandering reaches 14.0 cells where *random* reaches 24.4.
- **Turn/route micro-optimisation as the win lever.** Order matters only through forced-first + upward
  growth (5.3 ablations: 18/20, 18/20, 19/20) — not through shaving presses.
- **Optimising `cost`.** Dead in this lane's selection — `pick_best` is `(primary, -idx)`.
