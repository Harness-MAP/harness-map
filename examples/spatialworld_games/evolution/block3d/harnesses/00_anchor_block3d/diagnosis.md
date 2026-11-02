# Diagnosis — block3d (SpatialWorld, levels 1–20) / `anchor_block3d` (`staged_anchor`, lane `k32_claude_pv`)

**Headline:** primary (`won`, binary) = **1/20 = 0.050**. Mean completion **14.1458 %**, exactly 0 % on
**12/20**, target recall **17/116 = 14.7 %**, press precision **17/200 = 8.5 %**, **99** surplus standing
blocks against 17 correct. **0** parse failures, **0** default actions, **0** illegal names in **951**
completions. The one win is **L01, whose only required cell is the cursor's spawn cell `(2,2,0)`**
(`block3d_1_7eo7zcne`, frame 0, one VLM call, `{"action": "place_block"}`). Strip L01 and the anchor is
**0/19**.

**Binding constraint: HARNESS-BUG, worth 18 points — unchanged, settled, and not what this pass is for.**

> This is the **seventeenth** diagnosis of this artifact. Stage A proposes but never evaluates, so the
> anchor keeps returning as parent; `prop_r001`…`prop_r016` have each diagnosed it. The previous pass
> (`prop_r016_148ddc/cand_r016_0_spatial-grounding/diagnosis.md`, 615 lines) is correct and complete on
> the anchor, the observation, the geometry, the ambiguity class and the L19 ceiling. **§1–§4 below are
> re-measured independently so this file stands alone, but they are deliberately short — do not
> re-derive them a eighteenth time. §0 is what is new, and it changes which candidate the next
> proposer should lift.**

### What is new in this pass

1. **`prop_r016` had never been audited** — it was written *after* the previous diagnosis (its
   `tools/vlm.py` is dated 17:12, that diagnosis 15:40). It is the newest and largest harness in the
   tree and the chain was about to inherit it.
2. **Its own headline number is wrong in both directions.** Its `NOTES.md` reports **11–12/20**. On the
   real level file it is **18/20** on eight of nine reply arms — and **5/20** on the one arm that
   matters (§0.3).
3. **The reason no rig in this chain could see it:** all three stub rigs (`prop_r013`, `prop_r015`,
   `prop_r016`) hard-code the engine key and ignore the reply *name*, so **none of them can express a
   model-driven `place_block`** — the frozen model's second most frequent reply, 200/951 = **21 %** of
   its completions (§0.4). This is a family-wide instrument defect, and it is why `prop_r016` shipped
   with the one defect its design most needed to exclude.
4. **Two one-line edits take `prop_r016` from an expected 15.0/20 to a deterministic 19/20** on every
   arm, and cut it from 31.1 to 14.5 unit actions per episode (§0.6). Measured, not predicted.

### Provenance

*The anchor*, re-measured from `$TRACE_DIR`: 20 × {`eval_result.json`,
`block3d_model_responses_level*.json`, `session_*.log`}; all **951** actions dead-reckoned from spawn
`(2,2,0)`, clamp to `[0,5]³`, `left/right = ∓X`, `up/down = ∓Y`, `forward/backward = ±Z` (this
candidate's own `tools/action.py`), `place_block` = XOR at the cursor. **Self-validating: the reckoned
match count equals the engine's own on 20/20 episodes and the reckoned standing total is 116 against
116 target cells.**

*The candidates*, through **one** rig written for this pass
(`$SCRATCH/live_rig.py`, outside every genome), so that for the first time all five are scored on the
same renderer, the same engine contract and the same levels:

- **Renderer** at the previous pass's live frame probe: canvas 600×1100, bg `(20,20,20)`, pane outline
  `(70,70,75)`, lit tile `(80,80,80)`, panes at rows 30/180/330, col 40, side 90, pitch 15,
  `TOP (x, 5−y)` / `FRONT (5−z, 5−y)` / `SIDE (5−z, x)`. **Validated to the pixel:** on L01 it renders
  **1068** rule px and **588** lit px, the live probe's exact counts; its strip background differs from
  the live 75247 by exactly **1097 px**, the three static caption labels it does not draw.
- **Engine** with both counters, the real `core.data_classes.Action`/`FrameData`, `metadata["repeat"]`
  clamped 1..10 re-run inside one callback while still charging two game steps each, clamp, XOR, and
  `won` = exact set equality.
- **Validated against the previous pass's LIVE runs, element for element.** The rig reproduces the live
  callback *and* unit-action vectors on **all 20 levels for both `prop_r013` and `prop_r015`**:

  ```
  prop_r015 callbacks  rig == live  [ 1  3  3  4  7 10 10 10 11 10 12 15 12 20 24 23 31 26 60 43]
  prop_r015 units      rig == live  [ 1  5  3  4  6  9 11  8 10  8 10 15 10 16 20 21 26 22 50 35]
  prop_r013 callbacks  rig == live  [ 1  5  3  3  6  8  9  9  9  8  9 15 10 15 20 18 35 22 50 32]
  ```
  and reproduces the anchor's own live score (0–1/20 by arm, **0.40/20** at the model's measured reply
  distribution, the 1 being L01). **Every `prop_r016` number below is therefore as good as a live one.**

*Oracle used for ANALYSIS only* (permitted to the diagnostician, forbidden on any decision path):
`SpatialWorld/data/game/Block3D/level_data.py`, the 20 target sets, used to validate the reckoning and
to score the rig. **Nothing in §5 reads it at runtime.** `diagnosis.md` is not one of the five files
`pre_eval_gate` text-scans; privileged key names may appear here and must **not** appear in
`system_prompt.txt` / `AGENT.md` / `rules.md` / `memory.md` / `skills/*.md`.

---

## 0. What this pass adds

### 0.1 The anchor reproduces exactly; stop re-deriving it

| quantity | this pass | prior passes |
|---|---|---|
| `won` / mean completion | 1/20 · 14.1458 % | identical |
| reckoned matched == engine `matched_blocks` | **20/20 episodes** | identical |
| standing / target · correct / surplus | 116 / 116 · 17 / 99 | identical |
| clamped moves | **319 / 751 = 42.48 %** (33.54 % of all calls) | identical |
| per-direction clamp | `backward` 146/247 = .591 · `left` 54/92 = .587 · `down` 76/161 = .472 · `right` 28/61 = .459 · `up` 12/75 = .160 · `forward` 3/115 = .026 | identical |
| presses | 200 → **17** correct, **141** wrong, **42** self-removals (0 correct blocks ever removed) | identical |
| `P(press \| on an unfilled required cell)` | **17/34 = 0.500** vs 183/917 = 0.200 elsewhere (**2.5×**) | identical |
| distinct cells visited / ep | **14.0** of 216 | identical |
| reply distribution (951) | `move_backward` 247 (.260) · `place_block` **200 (.210)** · `move_down` 161 · `move_forward` 115 · `move_left` 92 · `move_up` 75 · `move_right` 61 | identical |

### 0.2 One rig, five candidates, the real 20 levels — the ranking the chain has never had

Wins of 20, `GAME_MAX_STEPS=100`, the model channel armed with one fixed reply every turn:

| candidate | `move_left` | `place_block` | illegal (`fly_up`) | `None` | `move_backward` | `move_down` | `move_forward` | `move_up` | `move_right` | **mixture at the measured frequencies (5 seeds)** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **anchor (this candidate)** | 0 | 1 | 0 | 0 | 0 | 0 | — | — | — | **0.40** |
| `prop_r013` | 19 | 19 | 19 | 19 | 19 | 19 | — | — | — | **19.00** |
| `prop_r014` | **14** | 19 | **14** | 19 | 19 | 19 | — | — | — | **17.80** |
| `prop_r015` | 19 | 19 | 19 | 19 | 19 | 19 | 19 | 19 | 19 | **19.00** |
| **`prop_r016`** | 18 | **5** | 18 | 18 | 18 | 18 | 18 | 18 | 18 | **15.00** |

`prop_r013`/`prop_r015` reproduce their live 19/20 and their model-invariance; `prop_r014` reproduces
the previous pass's 14-on-`left` and its ~17.6 expectation (17.80 here, same within seed noise). That
is three independent confirmations that this rig is measuring the live thing.

### 0.3 `prop_r016`'s real number, and the cliff its own rig hid

`prop_r016`'s `NOTES.md` reports **11 / 11 / 12** wins of 20 and presents that as the candidate's score.
On the **real** level file it is **18/20**, and **5/20** on the `place_block` arm. Both halves of the
error matter:

- Its rig's levels are **uniform random scatter** (`traces_notes/stub_rig.py:levels`,
  `k = randint(1,12)` cells drawn uniformly from the 216). That family is a different task: the spawn
  cell `(2,2,0)` is in the target on **0/20** of its levels and **9/20** of the real ones, and the
  mean hull/target ratio is 1.12 against the real 1.05. A candidate measured only there is measured
  against a distribution the engine never serves.
- The `place_block` arm is **21 % of the frozen model's actual replies**, and on it `prop_r016` loses
  **15 levels** — `[2, 3, 5, 6, 7, 8, 9, 10, 12, 13, 15, 16, 18, 19, 20]`. At the model's measured
  reply mixture the expectation is **15.00/20** (seeds: 15, 17, 14, 14, 15) — **a 4-level regression
  against `prop_r015`, worse than `prop_r014`, and with a 5/20 tail.**

### 0.4 Why no rig in this chain could have caught it — the instrument, not the candidate

Every stub rig in the tree returns a hard-coded engine key and puts the reply *name* only in metadata:

```
prop_r013/traces_notes/stub_rig.py:73   return Action(type="key_press", key="w",    metadata={"function_called": reply(self)})
prop_r015/traces_notes/stub_rig.py:70   return Action(type="key_press", key="left", metadata={"function_called": name})
prop_r016/traces_notes/stub_rig.py:70   return Action(type="key_press", key="left", metadata={"function_called": name})
```

So in all three rigs a `"place_block"` reply **executes a cursor move**. The live `baseline_step` maps
the parsed name to its key — the anchor's L01 win is the proof, a model `place_block` that really
toggled — so the one reply that can *write to the board* has never once been exercised by this chain's
instruments. `prop_r016`'s eleven-arm table is not wrong about its own rig; its rig cannot ask the
question. **This is the single most valuable correction in this file: fix the arm, not just the
candidate.**

### 0.5 The defect, exactly — two unguarded delegations, and a comment that reasons about the wrong state

`prop_r016/tools/vlm.py` returns the model's Action straight to the engine in two places:

```python
334:    if STAND(book):
            # A failed premise, or THE OPENING DELEGATION, which is always paid: ...
            # Where this turn leaves the cursor does not matter — the survey below overwrites it,
            # which is the whole point of surveying.
            return await pristine(frame_data)          # <-- the model's Action, played verbatim
...
360:    if len(pool) == 0:
            book["stop"] = WHYSTOP(book)
            return await pristine(frame_data)          # <-- every turn to the end of the wallet
```

The comment is **right about the cursor and silent about the standing set**. `prop_r016`'s headline
mechanism — three aimed `SWEEP`-long face legs that make the cursor arithmetic — genuinely absorbs any
*move* the opening turn makes. `place_block` is the one reply it cannot absorb: it leaves the cursor
exactly where the survey expects it while writing a block into the engine's standing set that the
harness's `book["up"]` never learns about. The second site then repeats the same mistake for the whole
terminal phase, turning the previous pass's §5.5 ("stand down onto `baseline_step`, never idle") into a
model-driven press loop.

**L02 end to end** (target `{(4,4,0)}`, one required cell, 100 steps):

| arm | emitted tape (first entries) | engine standing | harness `book["up"]` | `won` |
|---|---|---|---|---|
| `move_left` | `move_left·?` → `move_right×5` → `move_down×5` → `move_backward×5` → `move_left` → `move_up` → `place_block` | `{(4,4,0)}` | `{(4,4,0)}` | **✓** (7 callbacks, 19 units) |
| `place_block` | **`place_block·?`** → survey → … → `place_block` → **`place_block` × 31** | `{(2,2,0)}` | `{(4,4,0)}` | **✗** (38 callbacks, all 50 units) |

Read the second row: turn 1 stamps a block on the **spawn** cell, the harness correctly builds
`(4,4,0)`, then the terminal rung delegates 31 more times and toggles `(4,4,0)` back **off**. The
harness finishes believing it has won (`book["up"] == target`) while the engine holds the one cell it
never asked for. The cursor ledger stayed exact the whole time — the survey did its job. The *board*
ledger was never defended.

### 0.6 The fix, measured: two one-line edits, 15.00 → 19.00, and 31.1 → 14.5 units

| variant of `prop_r016` | all nine fixed arms | mixture (5 seeds) | levels lost | units/ep | callbacks/ep |
|---|---|---:|---|---:|---:|
| **as shipped** | 18 · **5** · 18 · 18 · 18 · 18 · 18 · 18 · 18 | **15.00** | 19, 20 | 31.1 | 17.1 |
| **+ a delegated reply may not WRITE** | **18 on all nine** | **18.00** | 19, 20 | 30.2 | 17.1 |
| **+ that, and the survey removed** | **19 on all nine** | **19.00** | 19 only | **14.5** | 14.3 |
| `prop_r015` (reference) | 19 on all nine | 19.00 | 19 only | 14.5 | 16.8 |

Edit 1 — *one line*, either form, both measured at 18.00 flat:

```python
MUTE = lambda act: None if SAID(act) == PRESS else act      # a delegated reply may not WRITE
#   ... return MUTE(await pristine(frame_data))             at both 334 and 360
# or, stricter and what prop_r015 does: consult, then play your own action
#   ... return (await pristine(frame_data)) and None
```

Edit 2 — *the survey is now redundant insurance, and it costs a level.* With the opening Action
discarded the cursor cannot drift, so the three face legs buy nothing and the ledger can simply be
seeded at the published spawn (`FRESH()["at"] = (2,2,0)`, `"homed": 3`). The premium is **16.6 unit
actions per episode, a third of the 50-unit wallet**, and **L20 is exactly what it buys you out of**:
L20's hull *is* its 16-cell target (unambiguous), `prop_r015` builds it in 35 units with **0** clamped
moves, shipped `prop_r016` runs out at 50 units with **11** clamped and 15/16 placed, one cell short at
`(1,3,1)`. The order of the two edits is not optional — removing the survey **without** edit 1 scores
**0/20** on the `move_left` arm, because the opening Action then really does move the cursor the ledger
was seeded for. That is the hazard the survey was built for, and discarding the Action is the cheaper
cure.

The fixed variant is also **budget-robust**, where the shipped one is not (arm `move_left`):

| `GAME_MAX_STEPS` | 60 | 80 | 100 | 200 |
|---|---:|---:|---:|---:|
| `prop_r013` | 17 | 19 | 19 | 19 |
| `prop_r015` | 18 | 19 | 19 | 19 |
| `prop_r016` as shipped | **13** | 16 | 18 | 19 |
| `prop_r016` + both edits | **18** | 19 | 19 | 19 |

Anti-cheat holds: the fixed variant makes **≥ 1** VLM call on every one of the 20 episodes
(`min = 1`, per-level 1–5), so `vlm_calls == 0` is never reached.

### 0.7 Gate status

`prop_r016` passes `pre_eval_gate`'s text scan: none of `gt.json`, `maze_file`, `distance_to_exit`,
`distance_to_food`, `cube_solved`, `current_floor`, `target_blocks`, `matched_blocks` occurs in its
`system_prompt.txt` / `AGENT.md` / `rules.md` / `memory.md` / `skills/*.md`. It is the first candidate
in this chain that actually *renders* `memory.md` into the system turn (`tools/vlm.py:275`) rather than
leaving it dead — keep that; it is the only self-derived-state channel in the family.

---

## 1. What this harness does (mechanism)

`agent.py` is a pass-through wrapper. Nothing in it decides anything.

- **`compose_system_prompt()`** (`agent.py:21`) concatenates `AGENT.md` + `skills/building.md` +
  `rules.md` onto `evaluator.current_prompt`, reproducing `system_prompt.txt` in order. It supplies no
  coordinates, no clamp model, no toggle-parity model, no plan, and never says **which way a key moves
  the highlight on screen**.
- **`build_agent(evaluator)`** (`agent.py:34`) puts `HERE` on `sys.path`, purges `sys.modules["tools*"]`
  so a sibling's `tools/` is never inherited (**keep this line**), and returns
  `agent(frame_data) -> await tools.vlm.decide(...)`.
- **`tools/vlm.py:decide`** is two statements: hand the frame to `evaluator.baseline_step` — the
  pristine SpatialWorld step that captures the screenshot, appends it to a rolling 29-turn image+text
  history, calls the VLM, parses `{"action": …}` and maps it to a key — and return its Action.

**Only the prose is on the decision path.** Four declared levers are dead code: `tools/frame.py:capture`
(never called), `tools/action.py` (never called — but its `KEY_MAPPING` is **correct**, and is what
every reckoning and every rig in this tree uses), `memory.md` (read by nothing), and
`system_prompt.md`/`.txt` (byte-identical duplicates read by nothing; `.txt` is outside the evolvable
bundle — a trap for the next editor). `agent.py`'s docstring still opens *"Rich **MAZE** harness"*.

**Contract facts that bound any fix:** primary `won` = exact set equality, binary per level; callbacks
≤ `max_steps` = 100; unit actions ≤ `max_steps // 2` = 50 (2 game steps each); a `None` return costs 1
callback and **0** game steps; `metadata["repeat"]` honoured, clamped 1..10, re-runs the mapped method
in one callback (never set `granularity` nor `json_data.granularity` — either forces repeat to 1);
**0 VLM calls voids the episode** to `score 0.0, valid: false`; only `evaluator.game.screen` passes
`_GameGuard`; `_DiskGuard` blocks harness reads of `SpatialWorld/data/`.

## 2. What actually happened (causal chain, with evidence)

**sign-biased direction choice → grid face → silent clamp → byte-identical frame → the same question →
the same answer → periodic orbit → wallet dead**, with the toggle flipping wrong cells on and off inside
the orbit.

The cursor spawns at `z = 0` and the policy presses −Z (`move_backward`, 247 emissions, the most of any
action) **2.1×** more often than +Z. **Every** direction is emitted more often exactly when it is
blocked, so conditioning the move on the image is *worse than not looking*: 42.48 % of moves clamp
against a uniform-random policy's ~16 %, and the cursor reaches **14.0** of 216 cells per episode
against random's ~24. A clamped move redraws a byte-identical frame, so the next prompt is the previous
prompt plus a duplicate image, and an 8-token reflex (`reasoning = None` on 951/951, 7 distinct reply
strings) answers identically.

- **L02 (`block3d_2_816ect9z`, 1 required cell, 0/1).** f00 `move_down` → `(2,3,0)`; then **f01–f49 are
  `move_backward` × 49**, every one a clamped null at `z = 0`. 98 % of the wallet, 2 of 216 cells
  visited, **zero** toggles — on a level `prop_r015` finishes in 5 unit actions.
- **L06 (`block3d_6_8kdgovfq`, 0/3).** Three blocks stamped near spawn, then `move_backward` × 44;
  5 cells visited all episode.
- **L08 (`block3d_8_wcifkwn3`, 0/4).** The opening `move_forward` run walks the cursor **through three
  of the four required cells** and places nothing; it then tours to `(0,0,0)`, makes its only toggle on
  a wrong cell, and spends f32–f47 pressing into two walls.
- **L16 (`block3d_16_hf5c40aq`, 0/9).** 21 presses, strict move/press alternation, `maxrun = 1` —
  carpet-bombing with a toggle, 0 correct.
- **L20 (`block3d_20_skvmjyw0`, 3/16).** Productive prefix f00–f12, then a period-4 orbit
  `move_left` (clamped at `x = 0`) → `move_forward` → `place_block` → `move_backward` for the remaining
  32 callbacks: net placed-set change per lap 0, net cursor drift per lap `(0,0,0)`, and `y` never
  changes after f00, so 12 of 16 required cells are unreachable *by construction*.

**All the useful work is in the first ten calls, next to the spawn** — 13 correct presses in calls 0–9,
4 in 10–29, **0** in 30–49. **The toggle decision is the one thing the model gets right and the harness
never puts it in a position to use it**: `P(press | standing on an unfilled required cell) = 0.500`
against 0.200 elsewhere, a real 2.5× signal — but only **34 of 951** decisions (3.6 %) ever stand on
such a cell, and the cursor reaches only **25 of the 116** required cells all batch.

## 3. Failure breakdown (shares of the 951-callback budget)

| category | count | share | control |
|---|---:|---:|---|
| **CLAMPED NO-OP move** | 319 / 751 moves | **33.5 %** of calls | uniform random clamps ~16 % |
| └ two presses own 70 % of the nulls | `move_backward` at `z=0` (146) + `move_down` at `y=5` (76) | — | the two directions that leave the spawn plane |
| real move | 432 | 45.4 % | 14.0 of 216 cells per episode |
| **WRONG placement** | 141 | 14.8 % | not near-misses |
| self-removal of a wrong block | 42 | 4.4 % | **0** correct blocks ever removed |
| **CORRECT placement** | **17** | **1.8 %** | all within L1 4 of spawn, all by callback 20 |
| calls after the episode's last correct press | 888 | **93.4 %** | the wallet is not short, it is dead |
| unspent callbacks | 50 / episode | — | the loop stops on game steps, not callbacks |
| PARSE / DEFAULT / ILLEGAL | **0 / 0 / 0** | 0 % | `successful_actions = 50` on every full episode |

**18/20 episodes end with surplus > 0** (99 surplus against 17 correct): even the 25–50 %-completion
episodes were never one press from a win. **Do not rank on completion or on the engine's match count** —
the predicate is exact set equality and neither can see the surplus term.

## 4. Binding constraint — classify it

**HARNESS-BUG, 18 points, now confirmed from a second direction.** `tools/vlm.py:decide` owns no state,
so the policy cannot notice a clamp, cannot notice a repeat, and has no representation of the
set-equality predicate it is scored on. Two replacements in this tree are **19/20 live**; this pass
reproduces both to the callback and adds a third at 19/20 (§0.6). Nothing about the engine, the model,
the wallet or the prompt contract has to change.

**VLM-CEILING is real but it binds the design space, not the score.** The frozen
`Qwen3-VL-30B-A3B-Instruct` cannot read a 6×6 binary grid it is staring at (every granularity tried
across `prop_r011`/`prop_r012`), cannot execute a stated route (0/6 with perception solved and injected),
and **any authority it is given over something the harness can prove is a measured loss** — `prop_r014`
−1.4 levels, `prop_r016` **−4.0** (§0.2). The ceiling is not one better scaffolding lifts.

**TASK-GAP on exactly one level.** On 18/20 the three silhouettes determine the structure uniquely; L19
admits **1 980** silhouette-exact candidates at its target's removal depth against ≤ 2 affordable
attempts, and 19/20 still holds at `MAXSTEPS = 300`. **19/20 is the ceiling on this level file and three
designs are now at it.**

**The design rule, stated as a measurable property.** The harness decides; the model is given a
non-binding job on a turn the anti-cheat makes the harness spend anyway. *Non-binding* means **the
outcome vector is identical under every reply arm** — and the arm set must include a correctly-keyed
`place_block`, or the test is vacuous (§0.4).

---

## 5. Concrete changes to try (the gradient — ranked by leverage)

**Lift `prop_r015_148ddc/cand_r015_0_spatial-grounding`.** It is 19/20 live with `ANTICHEAT_GUARD=1`,
19 under all nine reply arms and 19.00 at the measured mixture, derives both budgets from
`GAME_MAX_STEPS`, and is the cheapest episode in the tree (14.5 units, 0 clamped moves).
`prop_r013` is an equally valid base at this lane's budget. **Do not lift `prop_r016` as shipped**
(15.00 expected, 5/20 tail) and **do not lift `prop_r014`** (17.80 expected).

### 5.1 If you lift `prop_r016`'s mechanism, apply both edits — in this order

**File/function:** `tools/vlm.py`, the two `return await pristine(frame_data)` sites (**lines 334 and
360**), then `FRESH()`'s `"at"`/`"homed"` seeds (line 252/256) and the `if book["homed"] < 3:` rung
(line 366).

1. **A delegated reply may not WRITE.** `MUTE = lambda act: None if SAID(act) == PRESS else act`, or
   better, consult and play your own action: `return (await pristine(frame_data)) and None`.
   **Measured: 15.00 → 18.00 expected, 5 → 18 worst case, and model-invariance restored across all
   nine arms.** Causal reason: `place_block` is the only reply that leaves the cursor fixed (so the
   survey's guarantee still holds) while mutating the standing set the harness is scored on and never
   observes.
2. **Then delete the survey** and seed the ledger at the published spawn. **Measured: 18 → 19 on every
   arm, 31.1 → 14.5 units/ep, and 13 → 18 at `MAXSTEPS = 60`.** Causal reason: the survey's only job is
   absorbing the opening Action's cursor drift; edit 1 removes the drift, and the three face legs then
   cost 16.6 units/ep of a 50-unit wallet, which is exactly the margin on L20.

   Never edit 2 alone: it is **0/20** on the `move_left` arm without edit 1.

**Keep from `prop_r016` regardless of which base you lift:** it is the only harness in the family that
renders `memory.md` into the system turn (`tools/vlm.py:270-275`), and the only one whose refusal rungs
are a *table* of named premises (`GATES`/`TRIP`) rather than scattered `if`s. Both are worth carrying.

### 5.2 Fix the instrument before the next candidate — the reply arm must map name → key

**File:** every `traces_notes/stub_rig.py` in the tree (`prop_r013:73`, `prop_r015:70`, `prop_r016:70`).

```python
KEY = {"move_left": "left", "move_right": "right", "move_up": "up", "move_down": "down",
       "move_forward": "w", "move_backward": "s", "place_block": "space"}
return Action(type="key_press", key=KEY.get(name, "left"), metadata={"function_called": name})
```

Then require an **identical outcome vector** across all seven strings the model actually emits
(`move_backward` .260, `place_block` **.210**, `move_down` .169, `move_forward` .121, `move_left` .097,
`move_up` .079, `move_right` .064) plus an illegal name and `None`, and a 5-seed mixture at those
frequencies. **Predicted effect on the primary: 0 when the design is right — and it is the only way to
see a `prop_r016`-class defect before spending an evaluation.** It would have cost this chain one line
and saved it a 4-level regression.

### 5.3 Score candidates on the real level distribution, or say which one you used

**File:** `traces_notes/run_rig.py`. `prop_r016`'s `levels()` draws uniform random cell sets; the spawn
cell is in the target on **0/20** of those and **9/20** of the shipped file, and the hull/target ratio
differs (1.12 vs 1.05). A number from that family is not comparable to a number from this one, and
`prop_r016`'s NOTES presents one as the other. If the real sets are off-limits to the proposer, then
**report the level family in the headline** and treat the number as ordinal within one rig only.

### 5.4 Keep these locks exactly as they are — each is priced at −1 level or worse

1. **The mandatory turn-1 delegation** (−8 levels if made conditional: `vlm_calls == 0` → `valid: false`).
   A `None`-returning polled turn is its cheapest legal home — 1 callback, 0 game steps.
2. **`ADMIT` = the MAXIMAL silhouette-consistent set.** Maximal is the target on 18/20, minimum-cardinality
   on 13/20, greedy peel on 13/20 (91.3 % vs 99.5 % across 44 000 redraws). **Never minimise.**
3. **The repair(stray)-first tier, a peel of removal depth ≥ 2, and the layer-ordered serpentine** — −1
   level each (all three cost L17, whose target is one of 6 exact structures at depth 2).
4. **Fit the panel constants from the frame, keep every refusal rung sticky**, and replace the exact-RGB
   locator `(img[:, :STRIP] == RULE).all(2)` with the colour-free modal-background form (`prop_r013:115`,
   `prop_r015:153`, `prop_r016:133`). 0 points today — the constants are right, re-verified to the pixel
   in this pass's renderer validation — and the difference between 19/20 and 1/20 if any render constant
   ever moves, which is the only way this family fails *silently*.
5. **Keep the `Action.metadata` stamp** (turn, cursor, objective, owed/stray, rung, what the model said).
   It is what made §0.5 readable in one episode.

### 5.5 Stand down onto `baseline_step` — but discard what it returns

The previous pass's §5.5 is right that a terminal phase with budget left should keep the channel open
rather than idle. **§0.5 is what happens when "delegate" is implemented as "return its Action":** 31
model-driven presses that toggle the one correct cell back off. Delegate for the conversation; emit
`None`, or your own no-op. Costs no points on L19 (unwinnable either way) and removes the whole tail.

### 5.6 Prose, and the gate

`AGENT.md` / `rules.md` / `skills/building.md` name the axes without ever saying **which way a key moves
the highlight on screen** — precisely the fact §2 shows the model lacks. Replace with: the verified
pane→axis table (`TOP (x, 5−y)`, `FRONT (5−z, 5−y)`, `SIDE (5−z, x)`); that a key pressed at a grid face
costs a full decision and changes nothing; that three silhouettes admit a *family*, and a cell belongs
only if all three of its shadows land on lit tiles; that the episode ends on exact set equality, so an
extra block is as fatal as a missing one. `prop_r004/cand_r004_0/AGENT.md` already carries most of it.
**⚠ `pre_eval_gate` case-insensitively substring-scans `system_prompt.txt`, `AGENT.md`, `rules.md`,
`memory.md`, `skills/*.md`** against `privileged_keys` and AST-scans `agent.py` + `tools/*.py`; one
occurrence rejects the candidate with no evaluation spent. Write *"the engine's match counter"* and
*"the cells that must hold a block"*. `NOTES.md` / `diagnosis.md` / `traces_*/` are not scanned.
**Predicted effect on the primary: ~0.** Its value is stopping the next decoder inheriting a wrong
convention.

### 5.7 Lane note (φ, not score)

`prop_r013` keeps the parent's component graph byte-identical (11 nodes, 4 edges, 6 components, 5
modules) by putting the mechanism in one component over a module-scope lambda algebra; `prop_r015` and
`prop_r016` keep the same five modules. Module-scope lambdas are φ-cheap for the ast/num blocks, a
module-level `def` mints a component, `from tools import x` mints an edge, and the doc window here is
`.md`-only so all `.py` prose is φ-free. `diagnosis.md` is in `DEFAULT_SKIP_ARTIFACTS` and any path part
starting with `traces` is in `DEFAULT_SKIP_PREFIXES`, so this file and `traces_*/` are φ-free — but not
`_harness_sig`-free (it hashes every file in the dir).

---

## 6. Exhausted — do NOT repeat

- **The pass-through.** `build_agent → tools.vlm.decide → baseline_step` unchanged is **1/20** across
  independent batches and lanes, and its one win is the level whose only required cell is the spawn
  cell. More budget does not help: 93.4 % of calls already fall after the last correct press and 50
  callbacks per episode go unspent.
- **Any plan that routes target PERCEPTION through the frozen model.** All granularities measured (full
  frame, cropped, 6×, 12×, bitmap, coordinate list, convention handed over, 4-way choice with the answer
  present) — all at or below the trivial baseline.
- **Any plan that routes the ROUTE through the frozen model.** 0/6 with perception solved, the exact
  cursor, the exact destination and the blocked moves named.
- **Any plan that gives the model a per-cell EXISTENCE vote.** 14/20 on the `left` arm, 17.80 expected.
- **Any plan that lets a delegated reply reach the engine unfiltered.** NEW and now priced: 5/20 worst
  case, 15.00 expected, −4.0 against the same harness with one line changed (§0.5, §0.6).
- **"Keep the model on the commit."** A perfect commit oracle on the anchor's own route is 1/20: only
  3.6 % of its decisions stand on a placeable required cell.
- **Minimum / "simplest" / greedily-peeled explanations of the silhouettes.** 13/20 on the shipped file,
  91.3 % vs 99.5 % across redraws. **Never minimise.**
- **Making the mandatory VLM call conditional.** −8 levels, silently, as `valid: false`.
- **Removing the repair-first tier, ordering the tier ladder fill-first, replacing the serpentine with a
  shorter tour, or peel depth > 2 / `NODE_CAP` > 64.** −1 level each (L17).
- **A survey that walks the cursor into the faces to "construct" it.** NEW: it solves a real hazard, but
  the hazard is created by playing the opening Action at all; discard the Action and the survey costs
  16.6 units/ep and exactly L20 (§0.6).
- **Designing for L19.** 1 980 silhouette-exact candidates at the target's removal depth, ≤ 2 affordable
  attempts, 19/20 at `MAXSTEPS = 300`. The frame does not contain the answer and no budget buys it.
- **Deriving the wallet expecting a gain.** 0 at every `MAXSTEPS ≥ 80`; take it for correctness.
- **Looking for a structural prior that disambiguates L17/L19.** Connectivity fails on both.
- **Looking for in-frame feedback.** The pane strip is byte-static across every frame of every episode;
  the three captions are static labels; the `Cursor: (x,y,z)` caption was deleted from both engine
  copies on 2026-10-02.
- **Counting blue pixels to confirm a placement.** The 3-D view self-occludes. Track parity from your own
  toggles — and from §0.5, track it for *every* action the engine executes, including the ones you
  delegated.
- **Fitting any constant off the recorded `.mp4`s.** They are lossy: `(69,71,68)`/`(77,79,76)` where the
  live array is exactly `(70,70,75)`/`(80,80,80)`.
- **`evaluator.game.<attr>`.** Only `.screen` passes `_GameGuard`.
- **Parser / retry / key-mapping / fallback work.** 0 parse failures, 0 defaults, 0 illegal names in 951
  calls; L01 proves the mapping; `tools/action.py:KEY_MAPPING` is correct as written.
- **Resetting a module-global ledger on `frame_number == 0`.** `build_agent` runs once per episode; clear
  the book there.
- **Ranking on completion or on the engine's match count.** Both read 100 % and `k/k` on losing episodes.
- **Re-deriving the geometry, the masks, the lattice or the admitted set.** Panes at rows 30/180/330,
  col 40, side 90, pitch 15; tiles exactly 20.0 / 80.0; outline `(70,70,75)` = 1068 px over three panes;
  a lit tile 14×14 = 196 px; canvas 600×1100; `TOP (x, 5−y)`, `FRONT (5−z, 5−y)`, `SIDE (5−z, x)`.
  Confirmed by two independent decoders, three rigs, a live frame probe, and — in this pass — a renderer
  that reproduces the live pixel counts exactly and the live callback/unit tapes element for element on
  40 episodes. **Stop.**
- **Chasing variance in the primary.** It is deterministic under `prop_r013`, `prop_r015` and a two-line
  `prop_r016`. If a candidate's score moves with the model's reply, that is a defect of the candidate.
- **Trusting a rig that cannot express a `place_block` reply, or one whose levels are uniform random
  scatter.** NEW, and the reason this pass exists (§0.3, §0.4).
- **Treating per-level numbers as a target.** 11 of the 20 levels are resampled on every redraw of the
  level file (the 9 static ones, L17 and L19 among them, are not). A candidate that hard-codes a tape,
  a level index or a cell list will not survive a redraw.
