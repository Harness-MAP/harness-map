# Diagnosis — block3d / `anchor_block3d` (`staged_anchor`, lane `k32_claude_pv`)

**Headline:** primary (`won`) = **1/20 = 0.050** · mean final completion **14.15 %**, and **exactly
0 % on 12/20** levels · placement precision **17/116 standing = 14.7 %** (**17/158 presses = 10.8 %**) ·
target recall **17/116 = 14.7 %** · **19/20** episodes burn the whole **50**-decision wallet ·
**0/951** parse failures, **0** default actions, **0** illegal actions · **951/951** completions are
exactly **8 tokens** with `reasoning=None`, drawn from **7** distinct reply strings. The one win is
**L01, whose single target cell is the cursor's spawn cell `(2,2,0)`** — `block3d_1_7eo7zcne` frame 0,
one VLM call, `{"action": "place_block"}`, `steps_taken=2`. Strip L01 and this harness is **0/19**.

**Binding constraint: a VLM-CEILING on cursor localisation and routing, sitting behind a decision path
that is two lines long — so the ceiling is never approached — and turned into an absorbing feedback
loop by the harness itself.** The *movement* decision is measurably **worse than not looking at the
frame**; the *commit* decision is not. That asymmetry is the entire gradient.

### What this pass adds over the three prior diagnoses of this same anchor

The headline numbers above reproduce `prop_r001/r002/r003`'s independently (42.5 % clamp rate,
306/951 dead budget, 17/116, 0 parse fails, 29-turn window, 21 093-token saturation — all re-derived
here from the raw responses). Four things are **new**, and they change what the next candidate should
build and how big it should be:

> **A. The harness stops scoring after decision 12, and the decay is monotone.** Oracle-labelled from
> the engine's own `Block placed correctly!` events: correct blocks per 100 decisions, by wallet
> decile — **6.81 → 1.58 → 0.53 → 0.00 → 0.00**. **16 of 17** correct blocks land by decision **12**;
> decisions **30–49 across all twenty levels (380 calls) banked exactly ZERO**. The last 60 % of the
> wallet (570 calls) produced **one** correct block. The wallet is not short — it is *dead*.
>
> **B. All 17 correct blocks are a by-product of a `place, move, place, move` stamping reflex, not of
> perception.** Openings of L10/L14/L16/L18/L19/L20 are literally `place_block, move_forward,
> place_block, move_forward…`. L19 banked 4 correct in its first 8 decisions because that line
> happened to lie inside the target; L16 ran the identical reflex and banked **0**. The "commit signal"
> is real but it is a *stamp*, aimed by nothing.
>
> **C. Fixing the clamp alone is NOT a win lever — measured.** Replaying each episode's own actions but
> substituting a Hamiltonian-sweep step whenever the emitted move would clamp lifts coverage
> **14.0 → 21.6 cells/episode** — still **below** legal-aware random's **27.3**, and random wins
> ≈0.25/20. The clamp veto is a **prerequisite** that frees 33.5 % of the wallet; only the **panel
> decode** converts freed budget into aim. Order the build accordingly.
>
> **D. The per-turn injection channel is now pinned from readable source** (`scripts/game/
> block3d_openai_evaluation.py`, `mllm_base_agent/.../utils/openai_utils.py` — both outside the
> engine-source sandbox). See §1; it closes the "how do I give the model state" question exactly.

Every number below is a **measurement**. Cursor dead-reckoning from the harness's own emitted actions
(spawn `(2,2,0)`, clamp to `[0,5]³`, `left/right = ∓X`, `up/down = ∓Y`, `forward/backward = ±Z`) was
**validated against the oracle**: all 17 of the engine's `Block placed correctly!` events coincide with
a *new* placement in the reckoned set, on **20/20** episodes, and reckoned `|placed| ≥ matched_blocks`
holds on 20/20. No frames are archived (each episode dir holds only `eval_result.json`, the summary,
the responses JSON and the session log), so pixel claims are flagged as inherited and re-verifiable.

---

## 1. What this harness does (mechanism)

`harness.py` is a **pass-through**. Nothing in it decides anything.

- `compose_system_prompt()` (`harness.py:21`) concatenates `AGENT.md` + `skills/building.md` +
  `rules.md` and installs the result on `evaluator.current_prompt`. The composition equals
  `system_prompt.txt` up to leading/trailing whitespace (1377 vs 1382 chars), so **this candidate *is*
  the prompt-only baseline wearing a module layout**. It supplies role + grid facts, "study the three
  views / you can remove misplaced blocks", the 7-action control map and a colour legend. **No
  coordinates, no clamp model, no toggle-parity model, no plan.**
- `build_agent(evaluator)` (`harness.py:34`) puts `HERE` on `sys.path` and purges
  `sys.modules["tools*"]` so a sibling candidate's `tools/` can never be inherited — **the only
  load-bearing line in the file; keep it.** It returns `agent(frame_data) -> await tools.vlm.decide(...)`.
- `tools/vlm.py:decide()` is **two lines**: `return await evaluator.baseline_step(frame_data)`.
- `tools/frame.py:capture`, all of `tools/action.py` (`KEY_MAPPING`, `key_for`, `CELLS_PER_MOVE`) and
  the whole of `memory.md` are **defined and never called**. The rich surface is dead code.

### The contract that bounds any fix (re-read this pass, from non-sandboxed source)

| fact | value | where |
|---|---|---|
| wallet | `max_steps=100`; one executed action advances `steps_taken` by **2** ⇒ **50 decisions** | 19/20 episodes: `vlm_calls=50`, `steps_taken=100`; L01: 1 call, 2 steps |
| a clamped move costs **full fare** | L02 clamped **49 of 50** actions and still reached `steps_taken=100` | `block3d_2_816ect9z/eval_result.json` |
| history window | **29 turns**, and `MODEL_HISTORY_TURNS = 29` is a **ceiling** — `HISTORY_WINDOW` can only *shrink* it (`window_size = min(MODEL_HISTORY_TURNS, …)`) | `openai_utils.py:41,586`; `prompt_tokens` 1394 → +678/turn → saturates **21 093** from turn 30 |
| `current_prompt` is **write-once** | `build_full_prompt(is_first_call=False)` returns `self.current_prompt` **with no tool list and no RESPONSE FORMAT** | `block3d_openai_evaluation.py:157-211` |
| the **only** per-turn channel is `conversation_history[0]` | `build_conversation_history` always re-emits `previous_history[0]` verbatim (the **system** message built on call 0 — which *does* carry the RESPONSE FORMAT) and truncates only the user/assistant tail | `openai_utils.py:600-640` |
| the per-turn user text is hardcoded | `"Current step: N. Continue based on previous conversation history."` + the image — the harness cannot touch it | `openai_utils.py:641-660` |
| `frame_data.image` is the **ndarray** `(H,W,C)` | free to read; no engine access needed | `core/data_classes.py:11` |
| `evaluator.game.<attr>` | only `.screen` passes `_GameGuard`; any other read **voids the episode** | `eval_runner.py:131-150` |
| ground-truth files | `_DiskGuard` blocks `SPATIALWORLD/data/` (layouts/targets) | `eval_runner.py:~175` |
| `vlm_calls == 0` | voids the episode (`score 0.0`, `valid:false`) — the guard **was on** this run (`vlm_calls` is populated) | `eval_runner.py:275-291,317` |
| the engine's own fallback **reads the oracle** (`cursor_x/y/z`, `matched_blocks`) | legal for the pristine engine, **unavailable to a candidate** — do not try to copy it. It never fired: **0/951** | `block3d_openai_evaluation.py:400-460` |
| wall-clock | 1.0 / **64.5** / 77.8 s per episode (min/median/max) vs the **1800 s** subprocess wall ⇒ **23× headroom**; mean inference 0.95 s/call; whole sweep 1247 s | `eval_result.total_time`, session logs |

**Consequence:** prose cannot carry state here. `AGENT.md`/`rules.md`/`skills/*.md` all land in the
*system* message once; the only way to tell the model something *per turn* is to rewrite
`conversation_history[0]`'s text from a **stashed base** (never splice — truncate-at-marker-then-append
deletes the RESPONSE FORMAT block, which is only ever emitted on call 0).

---

## 2. What actually happened (causal chain, with evidence)

### 2.1 The absorbing state — `block3d_2_816ect9z` (1 target, matched 0)

| f | action | cursor | outcome |
|---|---|---|---|
| 00 | `move_down` | (2,3,0) | legal |
| **01–49** | `move_backward` **× 49** | (2,3,0) | `z` is already 0 — **every one of the 49 presses is a clamped null**. 98 % of the wallet, **2** cells visited, **0** toggles. |

A clamped move leaves the frame byte-identical, so the harness re-asks a *byte-identical question* of
an 8-token near-deterministic reflex, which answers identically. **48 of those 49 nulls are repeats of
a `(cursor, action)` pair the episode had already proved to be a no-op.** `block3d_6_8kdgovfq` is the
same shape (`move_backward` × 44 from f07, 86 % of the wallet). Nothing failed in the parser, the key
map or the fallback: all 50 replies were legal JSON and all 50 were applied.

**The loop is the harness's, and it is measurable:**
`P(a_t = a_{t-1} | the previous move CLAMPED) = 152/319 = 0.476` versus
`P(a_t = a_{t-1} | the previous move MOVED) = 96/624 = 0.154` — **3.1×**. Across the run,
**243/319 = 76.2 %** of all nulls re-press a wall the episode had already hit from that exact cell.

### 2.2 Reach-without-commit — `block3d_8_wcifkwn3` (4 targets, matched 0)

f00–f02 walk `move_forward` straight up the `x=2,y=2` column from the spawn, then the cursor tours to
the `(0,0,0)` corner, makes **its only toggle of the episode** at f31, and spends f32–f47 pressing
`move_backward` × 9 then `move_left` × 7 into two walls. **27 of 49 moves clamped, 1 toggle, 0 correct.**
The harness emitted exactly one commit in fifty decisions.

### 2.3 The stamping reflex that produced every point this harness scored

L19 (`block3d_19_w1fx1nm5`) opens `place, forward, place, forward, place, backward, …` and banks
**correct blocks at decisions 0, 2, 4, 7** — four of its fifteen. L18 banks at 2, 4, 11; L20 at 2, 4, 12;
L14 at 0, 2. **L16 runs the identical `place, move, place, move` opening and banks 0 of 9.** The reflex
is level-independent; whether it scores is decided by whether the line it stamps happens to intersect
the target. This is why **12/20 levels end at exactly 0 %** while the non-zero ones cluster at one to
four blocks: there is no aiming step anywhere in the loop.

### 2.4 Why it dies after decision ~12 — the context poisons itself

**319/951 = 33.5 % of the images the model is shown are byte-identical to the one immediately before.**
The window is 29 turns, so from decision ~30 the *entire* visible history can be duplicates of a frozen
frame. The per-turn `prompt_tokens` sequence is **identical across all 19 full-length episodes** (1
distinct sequence of 19) — proof that **no level-dependent and no state-dependent text ever reaches the
model**: any rendered coordinate, counter or scratchpad would have made the sequences differ. Pixels are
the only thing distinguishing one episode from another, and a third of them are duplicates.

Measured consequence (§A): productivity per 100 decisions collapses **6.81 → 1.58 → 0.53 → 0.00 → 0.00**
across the five wallet deciles; terminal orbit period is **1 on 17/20** levels (the cursor and the placed
set simply stop changing).

### 2.5 The opening move is anti-correlated with the free block on offer

The spawn `(2,2,0)` is a target cell on **9/20** levels. The anchor opened with `place_block` on
**13/20** levels overall but on only **4/9** of the gift levels (prior-pass oracle labelling;
`P(open with place | spawn IS a target) = 0.444` vs `0.818` otherwise). It walked off a free correct
block on L04, L08, L11, L15, L17 — all five of which opened `move_forward`/`move_down`.

---

## 3. Failure breakdown

951 decisions, 20 episodes. Dead-reckoned; the reckoning is validated against the engine's own
placement events on 20/20.

| category | rate | evidence / control |
|---|---|---|
| **CLAMPED NO-OP move** (cursor cannot change; frame byte-identical) | **319/751 moves = 42.5 %** (33.5 % of all decisions) | uniform random on the same wallet: **16.0 %** → the anchor is **2.7× worse** |
| └ where the nulls live | `move_backward` at `z=0`: **146** nulls / 247 presses (**59.1 %**); `move_down` at `y=5`: **76**/161 (47.2 %); `move_left`: 54/92 (58.7 %); `move_right`: 28/61 (45.9 %) — vs `move_forward` **3/115 (2.6 %)** | its three most-emitted directions are the three most likely to be no-ops; its one reliable escape direction is the one it under-emits |
| └ repeats of a known wall | **243/319 = 76.2 %** of nulls re-press a `(cursor, action)` pair already proved null in that episode | L02: 48/49 |
| **DEAD BUDGET** (decisions after the last *new* `(cursor, placed)` state) | **306/951 = 32.2 %**; terminal orbit period **1 on 17/20** levels | L02 49, L06 43, L20 32, L16 31, L04 29, L07 26 |
| **ZERO-YIELD TAIL** | decisions **30–49**, all levels: **380 calls → 0 correct blocks**; decisions 20–49: **570 → 1** | engine `Block placed correctly!` events |
| **WRONG-CELL PLACEMENT** | **99/116 standing blocks = 85.3 %** are on non-target cells | precision 17/116 = 14.7 % |
| **CANCELLED WORK** (a toggle that removes a standing block) | **42/200 toggles = 21.0 %** | not one of the 42 removals cleared a wrong block that was then replaced correctly; **no correct block was ever removed** (17 correct events = 17 final matched) |
| **NEVER REACHED** | the cursor occupies **14.0 of 216 cells per episode (6.5 %)**; dwell is pinned to the boundary — top cells `(0,5,1) 63`, `(0,5,2) 61`, `(2,3,0) 61`, `(3,3,0) 52`, `(5,5,0) 46`; **471/951 decisions sit at `z=0`, only 17 at `z=5`** | a target cell the cursor never occupies is unplaceable regardless of perception |
| **REACH-WITHOUT-COMMIT** | of the **34** decisions standing on an unfilled target cell, **17 (50.0 %)** toggled — **2.50×** the off-target base rate, but only **34/951 = 3.6 %** of decisions are such points | *inherited from `prop_r003`'s oracle labelling*; the one decision the model gets right, and the harness almost never sets it up |
| PARSE / DEFAULT / ILLEGAL | **0 / 0 / 0** of 951 | `successful_actions=50, default_actions=0` on every full episode; 7 distinct reply strings |

**Control (uniform random, same 7 actions, same 50-decision wallet, 4000 reps):**

| metric | anchor | uniform random | legal-aware random | verdict |
|---|---|---|---|---|
| clamped moves | **42.5 %** | 16.0 % | 0 % | **2.7× worse than chance** |
| cells visited / ep | **14.0** | 24.1 | 27.3 | **1.7× less coverage than chance** |
| new placements / ep | 7.9 | 5.8 | 6.7 | it toggles *more* and lands *worse* |
| wins / 20-level sweep | **1.00** | ≈0.25 *(inherited)* | ≈0.25 | its only reliable edge is taking L01's spawn gift |

---

## 4. Binding constraint — classify it

**VLM-CEILING on cursor localisation + routing, amplified by a HARNESS-BUG-class feedback loop, and
entirely fixable inside the harness.** The alternatives are ruled out by measurement:

- **Not HARNESS-BUG in the parse/fallback sense.** 0 parse failures, 0 default actions, 0 illegal
  actions in 951 completions; 7 distinct legal reply strings; L01 proves `KEY_MAPPING` and the toggle
  semantics. There is nothing to fix in `tools/action.py` or in the parser.
- **Not TASK-GAP.** Every level is winnable well inside the wallet: the three orthographic panels are a
  complete tomographic specification, and a pixel-only decode + forced-first build tape won **20/20
  live at 1…45 callbacks** (*inherited, `prop_r003` §5.3 — the single most load-bearing inherited
  measurement; re-verify it first*). Nor is it a commitment gap: the model commits at **2.50×** base
  rate when it is standing in the right place.
- **It *is* a ceiling, and a narrow one.** Frozen `Qwen3-VL-30B-A3B-**Instruct**` gets one isometric
  render, a 29-turn *image* history with no textual record of its own position, and a reply schema that
  admits no reasoning (**8 tokens, `reasoning=None`, 951/951**). It cannot localise a highlighted voxel
  in an isometric projection, and it has nowhere to put the state that would let it count its own moves.
  That is exactly why *movement* comes out below chance while *committing* does not.
- **The harness is the amplifier, and §C prices the amplifier separately from the win.** Removing the
  loop (clamp veto) recovers 33.5 % of the wallet and lifts coverage to roughly chance — and chance
  wins ≈0.25/20. **The loop is what makes the score 1/20 instead of ~0.3/20; the decode is what makes it
  20/20.** Ship both, in that order of dependency, and do not expect the veto alone to move `won`.

⇒ **Put a decoded target and a self-verified cursor on the decision path, drive a precomputed build
tape, and keep the model on the commit decision it is actually good at — while keeping `vlm_calls`
honest.**

---

## 5. Concrete changes to try (the gradient — ranked by leverage)

`evaluator.baseline_step` is pure with respect to the game (it captures, asks, parses; it does not
step), so **one proposal per decision is free** — arbitrate against it, never delete it
(`vlm_calls == 0` voids the episode). Everything below reads only `frame_data.image` and the harness's
own bookkeeping.

### 5.0 SCOPE FIRST — this is the change with the highest expected value right now

`prop_r001_148ddc/` and `prop_r003_148ddc/` each already contain a **complete, syntactically valid
9-module stack** (`panels.py` + `lattice.py`/`frame.py` + `reckon.py`/`ledger.py` + `debt.py`/`serpent.py`
+ an arbitrating `vlm.py`; 9/9 modules parse in both), and **all three proposer rounds hit the
10800 s wall** (`propose.log`: `timed out after 10800s` × 3 — r002 came back a byte-identical no-op).
**Zero of those mechanisms has ever been scored.** So:

1. **Port, smoke-test and ship one of the existing stacks** (or build **only** 5.1 → 5.2 → 5.3, three
   modules, and stop) rather than authoring a seventh module set from scratch. A mechanism that exists
   and runs beats a better one that times out.
2. Budget the round as: decode → unit-test the decode on a *rendered frame with cubes down* → tape →
   one live 20-level sweep. The arbitration polish (5.4) is worth less than shipping a decoder that runs.

### 5.1 `tools/perception.py:decode_panels(frame_data.image)` — read the target (THE lever)

*Inherited constants, re-verified on 20/20 frame-0 renders by the previous pass; FIT them, don't assert
them.* Three 90×90 panels in the left strip at `x ∈ [40,129]`, tops `y = 30 / 180 / 330`, 6×6 cells at
**15-px pitch**; a lit cell is a 14×14 square of exactly `RGB(80,80,80)` on `RGB(20,20,20)`, so the read
is all-or-nothing and **can refuse itself**:

```
patch = img[top+15*r+4 : top+15*r+11,  x0+15*c+4 : x0+15*c+11]
lit   = |median(patch) - (80,80,80)|_1 <= 12
TOP   (top y=30)  (r,c) -> (x, 5-c)      FRONT (y=180) (r,c) -> (y,z) = (5-c, 5-r)
SIDE  (y=330)     (r,c) -> (x,z) = (c, 5-r)
hull  = { (x,y,z) : (x,5-y) in TOP and (5-z,5-y) in FRONT and (5-z,x) in SIDE }
```

The empty grid is drawn in `rgb(70,70,75)`; its row-runs and col-runs give the three tops, the left edge
and the pitch as **measurements** (`x0=40`, `pitch=15.0`, `tops=[30,180,330]` recovered on 20/20).
Each lattice axis appears on **two** panels (`TOP.cols == FRONT.cols`, `TOP.rows == SIDE.cols`,
`FRONT.rows == SIDE.rows`) — check those three marginals and **stand the whole wrapper down to
`baseline_step` on disagreement** rather than guessing. `hull == target` on **18/20** (L17 +2, L19 +4).
**Predicted effect:** target-cell visitation 25 % → ~100 %; the 85.3 % wrong-placement rate collapses;
recall 14.7 % → ≥90 %. Precondition for everything else.

### 5.2 `tools/state.py:read_cursor(frame_data.image)` — containment in the right DIRECTION

Mask `|px − (100,200,250)|₁ ≤ 10` **restricted to `x ≥ 300`** (the sidebar legend reuses the colour —
146 of the frame's 1170 highlight-coloured pixels sit in the left strip). The projection is exactly
affine: `centroid(x,y,z) = (643.101, 278.236) + x(40,20) + y(−34,17) + z(0,−40)` in `(col,row)`, mask
area 1024 px, zero residual over 216 poses. *(The older table `(0,0,0) → (239,305)`, `+X=(20,40)`,
`+Y=(17,−34)`, `+Z=(−40,0)` is the same map in the bbox-top-left, `(row, col−300)` convention — mixing
the two is a 343-px error. Name the convention in the code and still fit from the frame-0 highlight.)*

**The direction of the containment test is load-bearing.** Occlusion only ever *removes* lit pixels, so
the sound test is `mask ⊆ footprint(c)`, resolved to the unique maximum-overlap cell, refusing on a tie:
**216/216 up to 16 standing cubes, 214/216 at 20, never wrong at any density.** Written the other way
(`footprint ⊆ mask`) it type-checks, is never wrong, and **refuses 35–71 % of poses**, silently handing
most callbacks back to drift. Centroid-to-nearest-lattice-point is **124–177/216 and up to 43 % wrong**
once cubes stand — unusable. **Unit-test the direction on a frame with cubes down, not on frame 0 where
both agree.** Supporting: the frame digest is a collision-free pose id at ≤20 standing cubes, so "digest
held ⇒ that *move* was clamped" is sound inside the wallet; the converse fails for toggles (3/180 real
placements are byte-invisible behind occlusion), so **track `placed` from your own toggles, never from
pixels**; `placed` is empty at frame 0.

A **dead-reckoning-only** fallback (spawn `(2,2,0)`, clamp to `[0,5]³`, track your own keys) is exact in
this run — validated 20/20 against the engine's placement events — and is ~30 lines. Ship it as the
floor and let 5.2 correct it; **do not ship 5.2 without the pixel check**, because drift is unrecoverable
once it happens.
**Predicted effect:** kills the 42.5 % clamp rate, the 0.476 repeat loop and the 32.2 % dead budget.
Per §C this buys budget and ~chance coverage — **not wins on its own.**

### 5.3 `tools/program.py:compile_order(hull, cursor)` — the build order (the win)

The projection-consistent family is **upward-closed inside the hull** and the engine re-tests set
equality after every executed action, so **a build order whose prefixes are plausible candidates is a
free sweep of hypotheses** — the engine ends the episode on the right one. Build a *minimal* member
first, then grow:

```
forced  = hull cells that are the UNIQUE hull coverer of some lit projection square
tier(c) = (c not in forced, c.z, round(|c - centroid(hull)|^2))
order   = concat over sorted tiers of a nearest-first walk inside the tier, from the LIVE cursor
```

*Inherited, measured live:* **`won` on 20/20** at `1,5,3,3,8,8,11,8,11,12,9,15,10,15,20,23,33,26,43,45`
callbacks; L17 wins at 33 and L19 at 43 **as prefixes**. Ablations: drop forced-first → **18/20**; flat
(no `z`) tiering → **18/20**; drop the rim key → **19/20** but *cheaper* (max 40). **Budget warning:**
slack is only **7 callbacks on L19 and 5 on L20** — any arbitration layer must be budget-neutral there.
**Predicted effect: 1/20 → ~20/20. This is the whole gradient.**

### 5.4 `tools/vlm.py:decide()` — arbitration that keeps the VLM load-bearing where it is strong

Per callback: decode (5.1/5.2) → `prop = await evaluator.baseline_step(frame_data)` → invert
`tools/action.py:KEY_MAPPING` on `prop.key` (this finally puts `key_for` on the live path) → then

- **Movement: the tape wins.** Accept the proposal only when it is route-equivalent (same next tape
  cell, or any legal move that strictly reduces L1 distance to it); else substitute the tape action.
  **Hard-veto any move whose decoded cursor cannot change** — aimed at 319/951 decisions and at the
  0.476 repeat loop.
- **Commitment: a vote, never a veto.** `P(place | on unfilled target) = 0.500`, so *gating* each cell
  on agreement costs ~2 asks/cell and breaks the wallet on L19/L20. Ask at a tape cell, honour a
  `place_block`; on refusal **place anyway if the cell is `forced`**, defer only on non-forced cells.
  Veto a toggle when the ledger says the cell is already correct (kills the 21.0 % cancellation and the
  14.5 % parity churn).
- **Prefer not asking.** `hull == target` on 18/20 — there is nothing to ask about there. Spend the vote
  on (a) ratifying commits, (b) ranking minimal sub-structures **when the decode refuses itself**,
  (c) breaking a tie between equally ranked goal cells. There is 23× wall headroom per episode, so one
  verification call at a genuinely ambiguous decision is affordable; a second call at every decision is
  not worth its trace size.

### 5.5 Shorten the history instead of lengthening the prose

33.5 % of the 29-turn image window is byte-identical duplicates, and productivity is **0.00 per 100
decisions** in the last two wallet deciles (§A, §2.4). The window is a liability once a tape owns the
route: set `HISTORY_WINDOW` small (it can only shrink — `min(MODEL_HISTORY_TURNS=29, …)`) and spend the
reclaimed context on **decoded state text**. Cheap, independently testable, and it also cuts the 0.95 s
mean inference.

### 5.6 Inject the ledger — and obey the two rules that have already cost wins

Render per turn into `conversation_history[0]`: decoded cursor, decoded target list, next tape cell, the
standing placed set, **which of the six moves are clamped at this cursor**, and callbacks remaining.
All self-derived; no oracle. Two hard constraints, both grounded in §1:

1. **Never speak on turn 1** — a mechanism that injects on message 0 of the first turn costs the L01
   gift win (3/3 → 2/3 in paired arms). Let the first decision be the pristine one.
2. **Rebuild message 0 from a stashed base; never splice.** `build_full_prompt` emits the tool list +
   RESPONSE FORMAT **only on call 0**, and `build_conversation_history` then carries that system message
   forward verbatim. A truncate-at-marker-then-append `_merge` deletes the JSON contract from every
   later call (measured: 1/29 calls kept it vs 15/15 when rebuilt; 3.11 % parse failures — and the
   engine's parse fallback then masquerades as a vote).

**Predicted effect:** higher proposal-acceptance and a readable next diagnosis; **not a win lever by
itself** — prose cannot carry state for an 8-token reflex.

### 5.7 Cheap sub-lever: take the spawn gift (ship inside 5.3, not standalone)

`(2,2,0)` is a target on **9/20** levels and the anchor took it on **4/9**. An unconditional
`place_block` as callback 0 banks a free correct block on those nine, costing one callback plus one
later removal on the other eleven. **Expected: +5 correct blocks, +0 wins** (L01 is already won, no
other gift level has `|T| = 1`). Only if it does not perturb L01.

### 5.8 Make the applied behaviour auditable

Stamp `Action.metadata` with `frame_number`, decoded cursor, `|hull|`, `|forced|`, tape index, proposed
action, applied action, override reason, clamp verdict, callbacks remaining. **Reconstructing all 20
trajectories offline was the entire cost of this diagnosis**, and **no frames are archived** — one
metadata dict plus a handful of saved frames removes that. `metadata["repeat"]` is honoured but buys
latency, not budget: L02 proves a clamped press still pays its 2 steps.

### 5.9 Rig hygiene (so the next candidate's offline numbers are real)

- Stub at **`evaluator.client`**, not at `create_openai_response`; a stub client **must carry the real
  model name** or the call routes to `client.responses` and silently measures nothing.
- Apply every override **after** `build_agent` — it purges `sys.modules["tools*"]`, so a "control"
  configured before it silently measures the shipped config.
- Price a closed loop by **breaking its premises** (a blinded decode should collapse the candidate onto
  the parent answer-for-answer); on a clean split the loop reads as a no-op.
- Don't alternate two harnesses in one process — it leaks `sys.path` and has inverted a verdict before.
- `HISTORY_WINDOW` is unset in this lane (**29**, verified from `prompt_tokens`). A rig that sets it is
  not measuring this configuration.
- `render()` returns `None` and draws into the adapter's surface; `evaluator.game.<attr>` is guarded.
  Use **`frame_data.image`** — it is already the `(H,W,C)` array.

---

## 6. Exhausted — do NOT repeat

- **The pass-through itself.** `build_agent → tools.vlm.decide → baseline_step` unchanged is **1/20**
  across independent batches and lanes, at 50 callbacks *and* at 60. More callbacks cannot help: §A
  shows decisions 30–49 bank **zero** correct blocks across all twenty levels.
- **Parser / retry / key-map / fallback work.** 0 parse failures, 0 default actions, 0 illegal actions
  in 951 completions; L01 proves the mapping and the toggle.
- **Re-wording the priors.** "Study the three views" and "you can remove misplaced blocks" are already
  in `skills/building.md`; `rules.md` already gives the six directions, the toggle and the 1-cell
  granularity. None of it carries coordinates, parity or a clamp model, and the model is an 8-token
  `Instruct` completion with `reasoning=None` on 951/951. The **identical per-turn `prompt_tokens`
  sequence on all 19 full episodes** is the proof that no state-bearing text reaches the context.
- **`memory.md` as prose.** A scratchpad not filled from *decoded* state and not consumed by an arbiter
  cannot stop a period-1 orbit (`block3d_2_816ect9z` f01–f49).
- **Writing `evaluator.current_prompt` per turn.** It is read only on call 0
  (`build_full_prompt(is_first_call)`); `conversation_history[0]` is the channel (§1, 5.6).
- **Expecting the clamp fix alone to move `won`.** NEW this pass (§C): veto + sweep lifts coverage
  14.0 → 21.6 cells, still under legal-aware random's 27.3, and random wins ≈0.25/20. It is a
  prerequisite that frees a third of the wallet, not a win lever.
- **Reading the cursor from a HUD caption.** There is none — the `Cursor: (x,y,z)` caption was deleted
  from both engine copies on 2026-10-02. Use the `RGB(100,200,250)` highlight (5.2).
- **Centroid inversion for the cursor** (57–82 % once cubes stand) and the **inverted containment
  direction** `footprint ⊆ mask` (never wrong, refuses 35–71 % of poses). The read that holds is
  `mask ⊆ footprint`, unique max-overlap.
- **Counting blue pixels to confirm a placement.** The 3-D view self-occludes; 3/180 real placements are
  byte-invisible. Track parity from your own toggles.
- **Asserting a lattice/panel constant table instead of fitting it.** The constants are stable in this
  renderer, but a moved or rescaled renderer silently yields a wrong or empty mask, and the two cursor
  conventions differ by 343 px. Fit from `rgb(70,70,75)` + the frame-0 highlight every episode.
- **`evaluator.game.<attr>` beyond `.screen`**, and **reading `SPATIALWORLD/data/`** — both guarded, both
  void the episode. Also: do not replicate the engine's oracle-reading fallback (§1).
- **Any no-VLM solver.** `vlm_calls == 0` → `score 0.0`, `valid:false`.
- **Hull-first descent / "build the hull then prune".** The wallet cannot afford removals and the
  descent is dead on L19's 15 842-member family. Grow upward from a minimal member (5.3).
- **A plain boustrophedon / memoryless sweep as the movement cure.** It folds into a 12-cell orbit with
  centre-seeking 2-cycles; the memoryless cure must be a Hamiltonian face sweep.
- **Turn/route micro-optimisation as the win lever.** Order matters only through forced-first + upward
  growth (ablations 18/20, 18/20, 19/20) — not through shaving presses.
- **Optimising `cost`.** Dead in this lane's selection: `pick_best` is `(primary, -idx)`.
- **Authoring a fresh 7-module stack from scratch.** Three consecutive proposer rounds died at the
  10800 s wall with nothing scored (§5.0). Build three modules, or adopt an existing stack.
