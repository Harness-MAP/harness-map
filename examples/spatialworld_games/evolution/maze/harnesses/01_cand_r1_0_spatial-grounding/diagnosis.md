# Diagnosis — maze (SpatialWorld, 20 levels) / `anchor_maze` · k32_claude_pv staged anchor

**Headline:** primary = **0.250 (5/20 success: L03, L04, L05, L09, L19)**. All 15 losses are 100-step
timeouts; **0/2000 parse or default failures**; **1017/1214 (83.8%) of forward decisions in the
losing episodes moved zero cells**, and **969/1263 (76.7%) of all forward decisions re-pushed a wall
the harness had already measured as a wall at that exact cell**.
Binding constraint: **HARNESS-BUG — `tools/vlm.py:decide` is an open-loop pass-through that throws
away the one bit the engine hands it free every step (did my last move change my pose), so a single
wall misread becomes an absorbing fixed point; on 7/20 levels the agent never once faces the single
open side of its own start cell.**

Priced offline against the real engine transition (§7): recording that one bit and refusing to
re-probe a measured wall is worth **12–13/20**; adding a frame-measured local wall occupancy is worth
**20/20 at ≤49 of the 100 steps, with any tie-break at all**.

---

## 1. What this harness does (mechanism)

```
config.py::_apply_harness
  └─ orchestrator.py:build_agent(evaluator)
       ├─ evaluator.current_prompt = compose_system_prompt()
       │      AGENT.md  +  skills/navigation.md  +  rules.md      (concatenated, in that order)
       └─ async agent(frame_data) -> tools.vlm.decide(evaluator, frame_data)
                                        └─ await evaluator.baseline_step(frame_data)   # pristine
```

* `tools/vlm.py:decide` is **two lines**: fetch `evaluator.baseline_step` and await it. The baseline
  grabs the pygame surface, appends it to a rolling image+text history, calls Qwen3-VL, parses the
  JSON and returns `Action(type="key_press", key=…)`.
* `tools/frame.py:capture` and all of `tools/action.py` (`KEY_MAPPING`, `granularity_steps`) are
  **dead code** — nothing in the decision path imports them.
* `memory.md` is a template and is **deliberately not injected** (`NOTES.md` says so). The harness
  therefore holds **zero** state across steps: no pose, no heading, no last-action result, no
  blocked-heading set, no visit count.
* Measured prompt shape (from `raw_response.usage`): frame 0 = 1541 prompt tokens, +~518 per turn,
  saturating at **~16,540 from step ≈30** → a ~29-turn sliding image+text window, as documented.
* Measured completion shape: **exactly 19 tokens for a `move_forward`, 11 for a turn**, and
  `reasoning=None` on every one of 2000 calls. The backbone is
  **`Qwen3-VL-30B-A3B-Instruct`** — it emits the bare JSON object and **no deliberation at all**.
  There is no chain of thought in these traces to repair, and nothing in the prompt can make the
  policy "think about it": the output is a reflex.

So: a stateless reflex policy, told in prose to notice walls and notice being stuck, with no code
anywhere that checks whether it did.

---

## 2. What actually happened (causal chain, with evidence)

### 2.0 The start pose is adversarial on 18/20 levels — and the harness walks into it

The engine spawns the agent facing **row − 1** (verified exactly, §7). Every level's `S` sits on
row 1, so **the cell directly ahead at step 0 is a wall on all 20 levels**, and the start cell has
**exactly one open neighbour on all 20 levels**:

| | L01 | L02 | L03 | L04 | L05 | L06 | L07 | L08 | L09 | L10 |
|---|---|---|---|---|---|---|---|---|---|---|
| ahead at step 0 | WALL | WALL | WALL | open | WALL | open | WALL | WALL | WALL | WALL |
| only opening | W | S | E | N | E | N | S | S | E | W |

(L11–L20: WALL on all ten; only opening E,E,S,E,E,S,S,S,E,S.)

The agent's **step-0 action is `move_forward small` on 18 of 20 levels** — i.e. a guaranteed no-op
on 16 of them. Step 0's correct action is *uniquely determined* (rotate to the single opening), and
the policy gets it wrong almost every time. That is the cleanest possible single-frame measurement
of the perception failure: the wall is one cell away, filling the view, and the model says "forward".

### 2.1 The absorbing fixed point: `maze_7_3l9a54wd` (L07)

```
step 0   facing N (wall)   move_forward small   -> 0 cells
step 1                     turn_right           -> facing E (also a wall)
step 2..99                 move_forward small   x 98,  every one 0 cells
```
`S=(1,1)`, the **only** opening is SOUTH. The agent faced **N and E only** — it never once faced S
or W in 100 steps. Final: 1 unique cell visited of 17, **0 cells traversed**, BFS distance to exit
unchanged at 8, timeout.

### 2.2 The orbit is a fixed point of the *frame → action* map, not maze-dependent

`maze_8_sdtjeplh` (L08) and `maze_16_dcn_pvqi` (L16) are **different mazes** (7×7 vs 11×11, exit
distance 16 vs 32) with **different frames** (frame-1 prompt_tokens 2059 vs 2055 — the images are
genuinely not identical), and their **(action, granularity) tapes are byte-identical for all 100
steps**:

```
[ move_forward small , turn_right , move_forward small , move_forward medium ,
  move_forward large , turn_left ]   x 16.67     (period 6, from step 0)
```

The `turn_right`/`turn_left` pair cancels, so the pose is invariant with period 6 → the frame
sequence is periodic → the policy, being a deterministic low-entropy map, reproduces the cycle
forever. Both episodes move **0 cells**, visit **1** cell, and never face their only opening (S).
Note the granularity ladder `small → small → medium → large` walks `rules.md`'s menu in order: the
model is enumerating the *parameter*, having concluded the *action* is right.

Every loss ends in such an orbit. Eventual period of the action tape, and the step it locks in:

| level | 1 | 2 | 6 | 7 | 8 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| period | 8 | 8 | 1 | 1 | 6 | 1 | 6 | 8 | 1 | 1 | 11 | 6 | 1 | 1 | 8 |
| locks in at step | 9 | 50 | 19 | 2 | 0 | 12 | 4 | 48 | 6 | 88 | 49 | 0 | 82 | 25 | 16 |

**10 of 15 losses lock in at or before step 27 — well before the 29-turn window saturates at step
≈30.** Window length is therefore *not* the cause; do not spend an iteration on it (§6).

### 2.3 Progress, then freeze two cells from the exit: `maze_6_74redmv6` (L06)

```
steps 0..12   8 cells traversed; by step 8 the agent is BFS distance 2 from the exit
step 13       turn_left    -> facing E at (5,5);  N and W are both open, E is a wall
steps 14..99  move_forward (large x81, medium x5) x 86, every one 0 cells
```
It froze **two cells from the exit** and burned 86 of its 100 steps pushing one wall, facing exactly
**one** heading for the whole run while two open sides sat beside it. Final distance 4.
`maze_2_ymejypv2` (L02) is the same story at BFS 2 from step 7.
`maze_18_0ro4a8kg` (L18): frozen at `(11,1)` facing S from step 25 — **75** consecutive null
forwards, N and E open the whole time.
`maze_17_2gan2x1e` (L17): 18 null forwards from step 82 at `(7,3)` facing E, N and W open.

### 2.4 The capability exists but is not reliably triggered: `maze_14_w8wb83bn` (L14)

L14 is the healthiest loss — 52 productive forwards, 23/49 cells visited, BFS 32 → 10. And it
contains `turn_right, turn_right, turn_right` at steps 29–31 and `turn_left ×3` at steps 43–45:
**the policy can emit a 3-turn heading scan.** It just doesn't do it when frozen (it still ends on
12 null forwards at `(3,9)` facing E with N and W open). The scan is available to the model and is
not being conditioned on the right signal, because no signal exists.

### 2.5 Healthy controls — everything except the wall judgement works

L03 wins in 7 decisions, L05 in 7 (0 null forwards), L09 in 10 (0 null forwards), L19 in 34.
Across 2000 responses: `successful_actions=2000`, `default_actions=0`, every `response_text` parses
to one of the three legal actions, and replaying the recorded tapes through the engine transition
reproduces **win/loss and final `distance_to_exit` for 20/20 episodes** (§7). Capture, prompt
composition, JSON contract, key mapping, granularity, and exit commitment are all sound.

---

## 3. Failure breakdown

| category | rate | evidence |
|---|---:|---|
| success | **5/20 (25.0%)** | `eval_result.json` × 20 |
| budget timeout | **15/20 (100% of losses)** | every loss `steps_taken=100`; optimal single-cell route is ≤42 actions on every level |
| **null forward** (0 cells moved) | **1017/1214 (83.8%) in losses**; 11/49 (17.2%) in wins | exact transition replay |
| **redundant push** (re-pushed an already-measured wall at the same cell) | **969/1263 (76.7%)** overall; 95–97/97 on L07/L10/L13 | replay + per-`(cell,heading)` ledger |
| never left the start cell | **7/20 (35.0%)** — L01, L07, L08, L10, L13, L16, L20 | 1 unique cell, 0 cells traversed |
| …and never faced the single opening | **7/7** | all seven faced only {N, E}; openings were S or W |
| steps spent after the last productive move | **1041/1500 loss-steps (69.4%)** | last `moved>0` index per episode |
| longest single null-forward run | 98 (L07), 94 (L13), 88 (L10), 86 (L06), 75 (L18) | consecutive-run scan |
| turn-undo while frozen (returned to a heading already faced since the last movement) | 32 (L08/L16), 25 (L20), 24 (L01), 23 (L12) | heading ledger between movements |
| reached within 2 cells of the exit and still lost | 2/15 (L02, L06) | BFS over the layout |
| open cells ever visited, losses | median **11%** (L13/L16 2%, L20 1%, best L14 47%) | replay |
| parse / default / illegal action | **0/2000** | `evaluation_info`, every `response_text` |
| hidden reasoning to repair | **none** | `reasoning=None`, completions exactly 19/11 tokens |

The loss is not malformed output, not engine execution, not budget size, and not exit commitment.
It is the repeated selection of an action the harness had already *measured* to be a no-op.

---

## 4. Binding constraint — classify it

**Primary: HARNESS-BUG (fixable in the harness, large).** `tools/vlm.py:decide` discards the
transition result. The engine gives a perfect, free, observation-side wall oracle on every step —
*a `move_forward` that leaves the pose unchanged proves the cell ahead is a wall* — and the harness
records nothing, so 969 forward decisions re-litigate walls it already knew about and 7 episodes
never test the one heading that works. The orbits in §2.2 are the inevitable consequence of a
stateless policy over a frozen observation, not a model defect.

**Secondary: VLM-CEILING on single-cell wall-ahead perception.** Step 0 is the clean experiment:
the wall is 1 cell ahead on all 20 levels and the model says `move_forward` on 18. A frozen,
zero-reasoning 19-token reflex cannot be prompted into this judgement — `AGENT.md` already tells it
walls are "brick-red / orange-brown blocks" and `skills/navigation.md` already says to turn when
blocked. The fix must **compute** the wall test, not request it.

**Not TASK-GAP.** The 100-step budget is slack on every level: shortest routes cost 6–42 actions at
one cell per action, less with granularity. Entering the exit cell auto-commits (all 5 wins end the
episode on arrival, 0 extra steps). There is no goal-commitment failure to fix.

**Headroom, priced offline (§7, exact engine transition, 100-step budget, no oracle in the policy):**

| policy | wins / 20 | notes |
|---|---:|---|
| this anchor (recorded tapes) | **5** | |
| harness records only *did-I-move*; never re-probe a measured wall; right-hand frontier | **12** | pure mechanism, no VLM |
| same, random tie-break among untested headings | **13.2** mean (min 10, max 17, 200 trials) | tie-break worth ≈ ±4 |
| veto-only over the *recorded tape* (substitute `turn_right`, no plan) | **7** | **a veto alone is weak — see §5.2** |
| **+ local wall occupancy (front/left/right) measured from the frame** | **20** | 6–87 steps; right-hand tie-break |
| …same, with a **random** tie-break | **20.0** (200/200 trials) | the tie-break stops mattering |
| …+ granularity run-lengths on known-open corridors | **20** (4–65 steps) | ~35% cheaper |

Read that table carefully: **once the harness knows which of the four sides of its current cell are
walls, and refuses to re-enter a visited cell, every one of these 20 levels falls inside budget with
an arbitrary tie-break.** This task does not need a planner, does not need exit-seeking, and does
not need the VLM to be smart. It needs local wall occupancy plus a visited set.

---

## 5. Concrete changes to try (the gradient — ranked by leverage)

### 5.1 `tools/pose.py` — a measured pose/transition ledger. **Prerequisite for everything else.**

New module `tools/pose.py`; instantiate **once** in `orchestrator.py:build_agent` (closure state, so it
lives for the episode) and thread it through `tools/vlm.py:decide`.

```
decide(evaluator, frame_data, state):
    obs   = state.observe(frame_data)          # pose key, measured from the frame
    state.settle(obs)                          # attribute the PREVIOUS action's outcome
    action = await evaluator.baseline_step(frame_data)      # unchanged, 1 VLM call
    action = state.arbitrate(action)            # see 5.2
    state.commit(action)
    return action
```

`observe` must produce a **pose key from the agent's own observation.** Three candidate routes,
in order of preference — **calibrate, do not assert:**

1. **The HUD crop.** `AGENT.md` claims a top-left cyan `POS (X, Y)` and top-right `DIR`. Crop those
   two rectangles, threshold to exact-colour masks, and hash. A hash is enough — you never need to
   OCR the digits, only to tell "same cell" from "different cell". **Verify the HUD exists before
   building on it**: capture one frame via `tools/frame.py:capture`, write the PNG to the candidate
   dir, and census the crop across ≥200 frames. Prose is not evidence that pixels exist.
2. **Whole-frame digest.** Cheap and needs no HUD. **Caution:** `AGENT.md` says the exit panel
   *pulses*. If the renderer animates, the frame is not a pure function of pose and a changed digest
   does **not** prove a move. Before relying on it, capture two frames at the same pose and confirm
   byte-equality on ≥50 poses; if it fails, fall back to a digest of a crop that excludes the exit
   glow, or to route 1. (Related trap seen elsewhere in this repo: *a held digest is not proof of a
   null* — an unchanged region can hide a real change.)
3. **Dead reckoning** from the ledger itself: assume each non-null forward advanced by its
   granularity. Self-consistent and free, but it drifts the moment a *partial* move happens
   (granularity `large` into a wall 2 cells out moves 1 cell, not 3). **Mitigation: use `small`
   for every exploratory forward**, so "moved" is an exact one-bit answer, and reserve `medium`/
   `large` for corridors the ledger has already mapped as open.

State to keep (all self-derived; nothing privileged):
`cell` (dead-reckoned integer pair), `heading` (0–3, updated by the turns *you* returned),
`blocked: set[(cell, heading)]`, `open_edges: set[(cell, heading)]`, `visited: set[cell]`,
`came_from: dict[cell, heading]`.

`settle` is the whole point: *if the last returned action was a forward and the pose key did not
change, add `(cell, heading)` to `blocked`* — one line, and it is an exact measurement, not a guess.

**Predicted effect:** on its own this is instrumentation, worth 0. It is the substrate for 5.2–5.4
and the thing whose absence causes 969 redundant pushes.

### 5.2 `tools/plan.py:arbitrate` — never re-push a measured wall, and **supply the substitute**

```
arbitrate(proposal):
    if proposal is forward and (cell, heading) in blocked:
        return <the harness's own next frontier action>     # NOT just "turn_right"
    if proposal is forward and (cell, heading) in open_edges and target in visited and
       an untested heading exists at this cell:
        return <rotate toward the untested heading>
    return proposal
```

**The substitute must come from a plan, not from a reflex.** I measured this: replaying the recorded
tapes with a pure veto that substitutes `turn_right` scores **7/20** — it rescues L01 and L02 but
*loses* L19, because the tape runs out while the agent spins. The same ledger driving a frontier
choice scores **12–13/20**. Veto without plan is the trap here.

The frontier rule, in full, is short:
* prefer an **untested** heading at the current cell, ordered right → straight → left → back;
* else BFS over `open_edges` to the nearest cell that still has an untested heading, and take the
  first edge of that path;
* rotate **monotonically** to the target heading (shortest rotation, but never reverse a rotation
  while the pose has not changed — that is the L08/L16 `R…L` cancellation, 32 wasted turns each);
* probe with `small`.

**Predicted effect:** this is the 5 → **12–13/20** step. It eliminates all 7 never-moved episodes by
construction (the frontier rule faces all four sides of the start cell within 3 turns), ends every
orbit in §2.2 (an orbit requires re-pushing a blocked edge), and converts L06/L18's 86- and 75-step
dead tails into exploration. Expected residue: the 11×11 and 13×13 levels (L12–L17, L20) where blind
probing exceeds 100 steps — that is what 5.3 is for.

### 5.3 `tools/perception.py:wall_mask(frame_data) -> (front, left, right)` — **the 20/20 lever**

This is the single highest-leverage change in the whole gradient. Measure, from pixels, whether the
cell ahead is a wall, and whether the left/right sides of the current cell are open. With that,
**20/20 at ≤49 steps with a right-hand tie-break and 20/20 on 200/200 trials with a purely random
tie-break** — the probe cost vanishes (in the blind version, 50% of the mechanism's budget is spent
on turns and another 335 forwards are deliberate wall probes), and no exit-seeking is needed at all.

How to build it, from the frame only:
* A wall **one cell ahead** fills the centre of a first-person view; an open corridor shows receding
  floor/ceiling converging to a vanishing point. The separable statistic is the **vertical extent of
  wall-coloured paint down the centre column** (or the ink census of a centred band): wall-at-1-cell
  saturates it, wall-at-2 covers less, open corridor less still. A sibling game in this repo
  (`maze3d_pro`) has exactly this result — the forward ray decodes from the vertical extent of the
  paint down one centre column, exact on 5624/5624 poses, and side faces on 23688/23688.
* Side openings appear as a break in the lateral wall band at a fixed screen x-range.
* **Fit the constants, never hardcode them.** Dump frames, histogram the colours (`AGENT.md`'s
  claimed palette — brick-red/orange-brown walls, gray floor, near-black-blue ceiling — is a
  starting hypothesis, not a measurement), and then **validate against the free ground truth you
  already have**: the did-I-move bit from 5.1. A `wall_mask` that predicts `front=WALL` must be
  confirmed by a null forward. Log agreement; ship only at ≥99%, and keep 5.1's ledger as the
  authority when they disagree. That makes the decode a *measurement* with a census behind it, not
  an assertion that silently reads the wrong pixels when the viewport changes.
* **Degrade safely:** if the decode's agreement is low, `wall_mask` returns `UNKNOWN` and 5.2 falls
  back to probing. The floor of the combined system is then 5.2's 12–13/20, never worse.

**Predicted effect:** 12–13 → up to **20/20**. Even a *partially* reliable front-wall test is worth
most of it: it removes the deliberate probe cost and the rotation waste.

### 5.4 Granularity run-lengths on mapped corridors — `tools/action.py` (currently dead code)

Wire up `tools/action.py:granularity_steps`. On a heading whose next 2–3 cells are already in
`open_edges`, emit `medium`/`large`; on any unmapped heading, emit `small`. Measured: 20/20 at
**4–65** steps instead of 6–87 — a ~35% budget saving, which is the margin on L20 (65 of 100).
Keep exploratory forwards at `small` so the did-I-move bit stays a clean one-bit answer (§5.1.3).

### 5.5 Keep the VLM, and give it the one job it is actually good for

Do **not** cut the model out. The mechanism's own weak point is the tie-break among untested
frontier headings (worth ±4 levels: random 13.2, min 10, max 17). That is exactly where vision can
help and the harness cannot: *the exit is a glowing gold panel*, visible down a corridor from
further than one cell. So:

* Let `evaluator.baseline_step` run **once per step, unchanged** (keep `vlm_calls` at 1/step — the
  run already costs 200–640 s per episode at 2.0–6.4 s/step, so a second call per step roughly
  doubles a 640 s episode; do the arithmetic in code, not in tokens).
* Use its proposal **only as the tie-break** among the headings 5.2 has not ruled out. Where it
  proposes something measured-impossible, override it.
* **Rewrite the history after an override.** `tools/vlm.py:decide` should replace the newest
  assistant JSON in `evaluator.conversation_history` with the action actually executed. Otherwise
  the 29-turn window teaches the model that its vetoed wall-push succeeded — which is precisely the
  self-reinforcing pattern that produced the identical L08/L16 tapes.

### 5.6 Prose riders — small, and only alongside the above (`skills/navigation.md`, `AGENT.md`)

Worth one paragraph, not an iteration: "a forward that leaves POS unchanged **proves** that side is
a wall — never retry it"; "a dead-end needs two or three turns **in the same direction**; never
reverse a turn before you have moved"; "you begin facing a wall — your first action is a turn."
Also drop the `granularity` menu's ordering salience from `rules.md`: the L08/L16 orbit walks
`small → medium → large` in menu order, enumerating the parameter instead of questioning the action.
**Prose alone cannot fix this** — the current prompt already instructs POS comparison and the model
emits 19 tokens with no reasoning.

---

## 6. Exhausted / do NOT repeat

From the prior `game_config/harnesses/maze/diagnosis.md` (iter 0, 6/20 — same anchor, same
mechanism, independently reproduced here at 5/20):

* **Do not add "notice the wall / don't get stuck" prose.** It is already in `AGENT.md` and
  `skills/navigation.md` and the model does not act on it.
* **Do not tune granularity as the primary lever.** Every granularity is null when the first cell is
  a wall; L06 pushed `large` 81 times for 0 cells. (It *is* a worthwhile ~35% budget saving *after*
  5.1–5.3 — see 5.4.)
* **Do not spend an iteration on JSON format, examples, retries or fallback.** 0/2000 parse
  failures, 0 defaults, 0 illegal actions, across both runs.
* **Do not activate `tools/frame.py` without wiring its output into state.** Capture is healthy;
  capture that nothing reads is still 5/20.

New from this run:

* **Do not shrink or lengthen the 29-turn history window as the fix.** The window saturates at step
  ≈30; **10 of 15 orbits lock in at or before step 27**, so saturation is not causal. L08 and L16
  produce identical 100-step tapes from *different* images — the policy is already insensitive to
  the visual content once the pose stops changing.
* **Do not ship a veto without a plan.** Measured: substituting `turn_right` for a known-null
  forward over the recorded tape scores 7/20 and *loses* L19. The substitute must be the frontier
  action from the ledger (12–13/20).
* **Do not treat budget as the constraint.** Optimal routes are 6–42 actions against 100; every loss
  is a decision-quality loss.
* **Do not try to make the model reason.** `reasoning=None`, completions exactly 19/11 tokens on all
  2000 calls — this backbone is `…-Instruct`, not `…-Thinking`. There is no deliberation to lengthen.
* **Do not wire a second VLM call per step** without checking wall-clock: episodes already run
  200–640 s (2.0–6.4 s/step).
* **Do not add an exit-seeking planner before 5.1–5.3.** With local wall occupancy, a *random*
  tie-break wins 20/20 on 200/200 trials. The planner is not where the points are.

---

## 7. For the next proposer: you can price your mechanism offline, exactly

I fitted the engine's transition from the recorded tapes and it is **exact**: replaying the 20
recorded action sequences reproduces **`success` and final `distance_to_exit` for 20/20 episodes**.

```
cells            : data/maze3d/Level_NN.txt, '#' = wall, 'S' start, 'E' exit, row/col indexed
initial heading  : row - 1  (so the cell ahead at step 0 is a wall on all 20 levels)
turn_right       : clockwise  [row-1] -> [col+1] -> [row+1] -> [col-1]
move_forward     : small=1 / medium=2 / large=3 single-cell steps, STOPS EARLY at a wall
                   (a partial move is possible: `large` with a wall 2 cells out moves 1)
terminal         : entering the exit cell ends the episode immediately, success=True
distance_to_exit : Manhattan in cells from the final cell
budget           : 100 actions (turns count)
```

A ~60-line rig over that transition evaluates a candidate mechanism on all 20 levels in well under a
second, and reproduces the live result for this anchor exactly. Use it to kill bad arms before
spending a ~90-minute live eval, and to pick thresholds on *margin* rather than on a single run.

**Anti-cheat, explicitly:** that rig is a **development tool only**. The shipped harness must never
read `data/maze3d/*.txt`, the exit's coordinates, the true distance, or any `evaluator`/`game`
attribute carrying them — the pre-eval gate AST-scans the code and text-scans the priors, and it
will catch it. Everything in §5 is observation-side: the frame's pixels, and the pose delta the
frame itself reveals. The agent's **own** pose is fair game (the HUD displays it to the agent); the
**layout and the exit** are not.
