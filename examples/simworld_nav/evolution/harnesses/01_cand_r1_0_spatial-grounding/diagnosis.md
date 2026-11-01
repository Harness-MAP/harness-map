# Diagnosis — SimWorld-MMNav (`hard`, search_hard_50) / candidate `anchor_nav`, arm `hard5010_adaptive_k4c1`

**Headline:** SR **0/50** · SubtaskSR **0.36** · per-leg pass **[1.00, 0.24, 0.11, 0.00]**
(n = 50 / 50 / 45 / 45; 5 tasks have only 2 legs). Cost 80.7k tok · 799 s per episode, not over budget.

**Binding constraint (one sentence):** the harness can *walk* (36/50 episodes physically pass through
the leg-1 pass window) but it cannot *stop* — the single binary landmark recogniser that owns every
`-1` has **20 % recall inside the window and 53 % precision overall**, and two independent,
fully-deterministic harness bugs throw away the legs the walker does reach: the TURN burst commits
**0.2–2.0 forward steps short** of the position box on 7/7 failures, and the FINISH commit never emits
the ±90° facing rotation the env requires on **50/50** tasks, so leg 3 is **arithmetically
unreachable** regardless of the recogniser.

> **Confidence note.** Everything below is a per-frame / per-commit measurement over all 50 episodes
> (348 recogniser calls, 918 planner calls, 2 812 forward steps), cross-checked against `gt.json`
> positions **for analysis only**. `read_audit.json`: 0 forbidden, 0 escalated, 0 undeclared reads on
> every episode. No recommendation below routes privileged state into the policy.

---

## 1. What this harness does (mechanism)

`orchestrator.py:run_episode` — one iteration per *decision*, not per env step.

| # | call site | authority over the outcome |
|---|---|---|
| 1 | `active.should_emit` (`orchestrator.py:301`) | **short-circuits perceive+plan entirely** when it returns a list. Owns (a) the ORIENTATION cardinal burst `[rot…, -1]`, (a″) the **relative-TURN burst `[rot, -1]`** (`active_perception.py:431-440`), (a′) the red-light idle, (b) anti-spin, (b′) wall recovery, (d) the cap-2 forward-momentum cadence → **PP → m → m**, i.e. 3 forwards = 1 500 cm per VLM pair. |
| 2 | `perceiver.perceive` | 1 VLM call, **4 images** (RGB, robot_view hint, object_mask, JET depth) + rolling summary → free-text `Description`. |
| 3 | `planner.plan` | 1 **text-only** call over that description → `{Summary, Match, Actions}`. |
| 4 | `completion.verify` (`:389`) | honours a planner `-1` iff `planner_match == 1 and forward_streak >= 1`. |
| 5 | `cross_check_planner_commit` (`:413`) | **2-image** binary REACH call, gates a planner `-1`; capped at 2 fires/subtask. |
| 6 | `proactive_commit_check` (`:430`) | same **2-image** call, fires at `forward_streak ≥ 4` every 5 forwards, ≤12/subtask; `Match==1` ∧ `_named_side_has_structure` ⇒ inject `[-1]`. |
| 7 | `request_side_scan` / `scan_reach` (`:459`) | depth-gated in-place yaw toward the named side, re-ask REACH, yaw back on miss. |

Stages 5 and 6 are **the same function** — `_binary_reach_vlm_call` (`completion_checker.py:530`) —
and it is the only thing in the stack with real authority over `-1`. It is also the **weakest-evidence
call in the harness**: 2 images, no mask, no depth, no summary, no orientation, and a system prompt
that simultaneously licenses "clearly visible and prominent" and demands "Be conservative"
(`:560-569`).

**Env semantics that the harness does not model** (`traffic_world.py:634-690`, `:389-460`):
- a `-1` whose criterion **fails ends the episode immediately** (`terminated = True`) — every false
  commit is fatal, there is no retry;
- each leg gets `subtask_max_steps` = `int(L1_remaining/100)` for ROAD/FINISH (≈ **5× slack**, since a
  forward covers 500 cm), **10** for ORIENTATION, **50** for TURNING — exceeding it terminates;
- ROAD/FINISH are graded on **L1** distance < 2 000; TURNING on orientation **and** a position box;
  FINISH on distance **and** orientation.

---

## 2. What actually happened (causal chain, with evidence)

### 2.1 The walker is fine. The stopper is not. (leg 1 = 0.24)

Per-episode, measured from `trajectory.json` against leg-1's own criterion:

| | episodes |
|---|---|
| physically entered the leg-1 pass window (min L1 < 2 000) | **36 / 50** |
| …and committed there (leg 1 = pass) | **12** |
| …and walked straight through it, never committing | **24** |
| never got near the window at all | 14 |
| committed **outside** the window (fatal) | 9 |

The 24 walk-throughs are not near-misses — the harness has **no notion of having passed anything**:

```
ep_20_16_task_dist_21_0_1   closest approach   65 cm   →  ended 53 935 cm past it (135 forwards)
ep_20_17_task_dist_21_0_1                      42 cm   →        43 542 cm past
ep_20_7_task_dist_21_0_1                        9 cm   →        23 509 cm past
ep_20_18_task_dist_11_0_1 (FINISH leg)         16 cm   →        36 484 cm past
ep_20_2_task_dist_11_0_1                       50 cm   →        47 950 cm past
```
Eight of them burned their *entire* per-leg substep budget (`used == budget`: 12_11 83/83, 15_11 72/72,
16_21 135/135, 2_11 120/120, 20_11 97/97, 22_11 48/48, 23_11 58/58, 30_21 27/27); the rest died on the
env's out-of-bounds / repeated-position terminators.

### 2.2 The recogniser, measured: recall 0.20 in-window, precision 0.53

348 `_binary_reach_vlm_call` firings across the run (proactive + cross-check pooled — identical
prompt), bucketed by the **true** L1 gap to the then-current subtask's criterion:

| L1 gap | firings | `Match=1` | rate |
|---|---|---|---|
| **< 2 000 (would PASS)** | 44 | **9** | **0.20** |
| 2 000 – 5 000 | 55 | 0 | 0.00 |
| 5 000 – 15 000 | 111 | 2 | 0.02 |
| > 15 000 | 138 | 6 | 0.04 |

So: it misses **4 out of 5** true arrivals, and **8 of its 17 yeses** are 5 000–58 789 cm away — each one
a fatal commit. The prompt is the direct cause of both halves, and the model says so in its own
`Reason` field:

```
FALSE positives (all cite VISIBILITY):
  ep_20_10_21 gap 58 789 : "The reference building with striped awnings is visible AHEAD in the current view."
  ep_20_3_21  gap 35 733 : "The building with large glass panels and brown frames is visible in both views."
  ep_20_5_11  gap 32 586 : "The reference building ... is clearly VISIBLE in the current view."
  ep_20_16_11 gap 26 273 : "The building with arched glass windows ... is visible AHEAD in both views."
TRUE positives (cite BEING THERE):
  ep_20_16_11 gap   337  : "Same building facade and INTERSECTION LAYOUT visible"
  ep_20_9_11  gap   342  : "same INTERSECTION with prominent building and canopy"
```
`completion_checker.py:565-566` literally authorises the false class — *"the same building,
intersection, or scene is **clearly visible and prominent** in the current view"* — while `:567`
("Be conservative") suppresses the true class. One prompt, two opposing criteria; the false one wins
on a 700-frame far-field tail and the conservative one wins at the waypoint.

### 2.3 This is **not** a VLM ceiling — the same model gets it right in the richer channel

At the exact frames where the 2-image recogniser refuses, the **4-image perceiver + planner** on the
same step assert arrival correctly:

```
ep_20_19_task_dist_21_0_1, step 29, true L1 gap = 1 022 cm  (INSIDE the window)
  perceive : "...a large building on the left featuring a golden facade..., consistent with the
              expected landmark ... indicating that the robot has reached the correct location
              at the intersection."
  plan     : Match=1, Actions=[-1]
  REACH    : {"Match": 0, "Reason": "Different building designs and colors"}     <-- vetoes it

ep_20_18_task_dist_21_0_1, step 15, gap = 388 cm            (INSIDE)
  perceive : "...matching the expected landmark ... confirming that the agent has reached the
              correct intersection. The subtask is complete."
  REACH    : {"Match": 0, "Reason": "Different building colors and lighting conditions"} x2

ep_20_17_task_dist_11_0_1, step 11, gap = 574 cm            (INSIDE)
  plan     : "The agent has reached the intersection and is now positioned such that the high
              office building ... " -> Actions=[-1]
  REACH    : {"Match": 0, "Reason": "Different lighting and building details"}
```
Same frozen model, same two images, same step — **the difference is the evidence and the framing the
harness hands it.** The harness gives veto authority to the impoverished call.

Channel discrimination over all 918 plan calls, same bucketing:

| L1 gap | plan calls | planner emits `-1` | recogniser `Match=1` |
|---|---|---|---|
| < 2 000 | 90 | **0.11** | **0.20** |
| 2 000–5 000 | 134 | 0.08 | 0.00 |
| 5 000–15 000 | 297 | 0.01 | 0.02 |
| > 15 000 | 397 | 0.01 | 0.04 |

The planner is the better **far-field rejector** (0.01 vs 0.04); the recogniser has the sharper
**boundary** (0.20 → 0.00 across 2 000 cm). Neither alone is sufficient; the current AND-composition
is right in shape and broken in its cap semantics (§2.4).

### 2.4 `cross_check_planner_commit` is a delay line, not a filter — 5 fatal episodes

`completion_checker.py:748-749`: when the per-subtask cap is spent it returns **`True`** ("defer"),
so the **third** planner `-1` in a subtask commits *unverified*, at whatever position the agent has
drifted to. And `orchestrator.py:495 + 519`: when the cross-check *does* block a bare `[-1]`, the affordance
filter strips `-1`, `chosen_actions` goes empty, and `EMPTY_FALLBACK` injects a **Move_Forward** —
the veto physically carries the agent out of the window it was standing in, 1 500 cm per cycle
(momentum cadence).

The planner emitted a bare `[-1]` 29 times in 8 episodes, always in consecutive runs of 3:

```
ep_20_17_11:  -1 @ gap  574 BLOCKED -> fwd -> -1 @ 2 074 BLOCKED -> fwd -> -1 @ 3 074 COMMITS (cap spent)  FAIL
ep_20_18_21:  -1 @ gap  388 BLOCKED -> fwd -> -1 @ 1 888 BLOCKED -> fwd -> -1 @ 2 888 COMMITS              FAIL
ep_20_19_21:  -1 @ gap  478 BLOCKED -> fwd -> -1 @ 1 022 BLOCKED -> fwd -> -1 @ 2 022 COMMITS (by 22 cm)   FAIL
ep_20_24_11:  -1 @ 7 537 / 6 037 BLOCKED (correctly) -> -1 @ 3 537 COMMITS (cap spent)                     FAIL
ep_20_13_21:  -1 @ 18 634 / 20 134 BLOCKED (correctly) -> -1 @ 21 134 COMMITS (cap spent)                  FAIL
```
In three episodes the gate vetoed the **in-window** claims and then rubber-stamped the out-of-window
one. In two more, cap expiry alone fired the fatal commit. **5/50 episodes die to `return True` on
line 749 plus the forward on line 519.**

### 2.5 The TURN burst commits 0.2–2.0 forward steps short — 7/7 of leg-2's failures

`active_perception.py:431-440` matches "Make a left turn at the intersection." and returns
`[rotate, -1]` **on the first decision of the TURNING leg, with zero translation**. Every one of the
12 turn commits in this run landed **exactly 2 env steps after the ROAD commit**. But the TURNING
position box is centred on the *intersection*, 2 458–8 306 cm further along the street than the ROAD
waypoint the agent just committed at. Distance by which each failure missed the box, measured along
the agent's own travel axis:

```
ep_20_2_21     78 cm  (0.2 forwards)      ep_20_8_21    506 cm  (1.0)
ep_20_13_11   356 cm  (0.7)               ep_20_11_21   627 cm  (1.3)
ep_20_0_11    452 cm  (0.9)               ep_20_1_11    692 cm  (1.4)
                                          ep_20_23_21   980 cm  (2.0)
```
**All 7, outward along the direction of travel.** TURNING is granted **50 substeps** irrespective of
distance, so the approach is free; the burst simply never takes it.

### 2.6 Leg 3 is arithmetically impossible — FINISH needs a ±90° facing the harness never emits

`FINISH` is graded on distance **and** `(correct_robot_orientation − yaw) % 360 < 45`. Across all
**50/50** tasks, the required FINISH yaw is **exactly ±90° off the final travel heading** — never 0:

```
(n_subtasks, (FINISH_yaw − travel_yaw) % 360) -> {(4, 90): 24, (4, 270): 21, (2, 270): 4, (2, 90): 1}
```
and it is **perfectly predicted by the instruction text alone**, 50/50:

```
instruction "... on your right, stop and face that building."  ->  delta +90   (25/25)
instruction "... on your left,  stop and face that building."  ->  delta -90   (25/25)
```
Every commit path that fires on a FINISH leg (`proactive_commit_check` → `planner_actions = [-1]`;
`verify` → bare `-1`) emits `-1` with **no rotation**, leaving the agent facing along the street,
90° or 270° off. `per_position_SR[3] = 0.0` over 45 episodes is exactly what this predicts. The one
mechanism that *would* produce the right pose — `request_side_scan` → `scan_reach` →
`SIDE_SCAN_COMMIT` — is gated behind `_side_band_near` (depth) and fired **2 times in 50 episodes**,
committing **0**.

### 2.7 Two dead safety nets

- **`SpatialMemory.log_step` hard-codes `coll = 0`** (`spatial_memory.py:66,77`). `static_collision_delta()`
  therefore always returns 0, so ActivePerception rule (b′) *wall-stuck recovery* (`:520-533`) and the
  iter_008 momentum collision veto **can never fire**. `read_audit.json` confirms no collision key was
  ever read on any episode. Cost: ep_20_20_21 (19/20 forwards moved < 1 cm), ep_20_22_21 (19/22),
  ep_20_8_11 (7/11) mashed a wall until the env's repeated-position terminator killed them — 3
  episodes, 0 legs.
- **The STUCK detector is a no-op** (`orchestrator.py:280`): `forward_count = sum(1 for a in chosen_actions
  if a == 0)` counts the *current burst* (length 1–3), not history, so `forward_count > 3` essentially
  never holds. `SpatialMemory.forward_count_since_boundary()` is the intended signal and is never
  called. `STUCK` appears in 3/50 notes while 118 forwards across the run moved < 1 cm.
- **`Match` missing on 111/918 (12 %) plan replies** — the shared `reasoning_template` Output Format
  omits it, so `verify()`'s `planner_match == 1` silently fails. Latent, since the planner's `-1` rate
  is only 3 %.

---

## 3. Failure breakdown (50 episodes, leg at which the episode ended)

| category | eps | evidence |
|---|---|---|
| **Walked through the leg-1 window and never committed** | **24** | §2.1 — min L1 9–1 107 cm, final 3 122–65 291 cm past |
| **Fatal false commit, out of window** | **9** | gaps 2 022 / 2 888 / 3 074 / 3 537 / 6 932 / 21 134 (ROAD) · 22 988 / 32 586 / 58 789 (FINISH) |
| **Turn committed short of the position box** | **7** | §2.5 — all 7 miss by ≤ 2.0 forwards |
| Never approached the window (wall-mash / bad heading) | 5 | 20_21, 22_21, 8_11 (wedged, §2.7); 12_21, 24_21 |
| FINISH committed far short after a good turn | 5 | 7 778 / 15 733 / 26 273 / 35 733 cm |
| Reached FINISH position with the right facing | **0** | §2.6 — structurally impossible |

Of the 24 walk-throughs, at least **3 (17_11, 18_21, 19_21)** had an in-window commit actively vetoed
and replaced by a forward (§2.4); the other 21 simply never produced a `Match=1` (§2.2).
Note tokens across the run: `EMPTY_FALLBACK` 21 · `XCHECK_BLOCK` 18 · `REACH_COMMIT` 17 ·
`NO_ACTION_RETURNED` 8 · `STUCK` 3 · `SIDE_SCAN_MISS` 2 · `SIDE_SCAN_COMMIT` **0**.

---

## 4. Binding constraint — classified

**Primary: HARNESS-BUG (three of them), not VLM-CEILING.** The previous gradient classified the
arrival decision as a model ceiling. On 50 episodes that is refuted for the *decisive* cases: §2.3
shows the same frozen model, on the same frame, asserts arrival correctly in the 4-image
perceive+plan channel and denies it in the 2-image REACH channel that holds the veto. The ceiling is
in **what the harness shows the recogniser and what criterion it asks for**, not in the weights.
Concretely fixable in code:
1. `active_perception.py:431-440` — TURN burst commits without approaching (§2.5, 7/7, deterministic).
2. `completion_checker.py:749` + `orchestrator.py:495,519` — cross-check cap defers to `True` and the veto
   *moves the agent* (§2.4, 5 episodes).
3. `spatial_memory.py:66` + `orchestrator.py:280` — two recovery mechanisms wired to constants (§2.7).

**Secondary: TASK-GAP.** §2.6 — the harness has no representation of the *terminal pose* the FINISH
leg is graded on, even though the instruction states it and `_named_side()` already parses it. Leg 3
cannot score until a rotation is emitted before the final `-1`. Separately, the harness has no
"I may have passed it" state at all (§2.1), so a missed window is unrecoverable by construction.

**Residual VLM-CEILING, honestly bounded.** Even after the above, the in-window recall of a *single*
binary call is 0.20 and each episode gets only ~1.2 in-window firings (44 firings / 36 reaching
episodes). That product — not motion, not coverage — is what caps leg 1 at 0.24. The fix must
**scaffold** the judgement (richer evidence, a comparative rather than absolute question, a
position-preserving second look), not re-word it.

---

## 5. Concrete changes to try — ranked by leverage

### A — Approach the intersection before committing the TURN. `modules/active_perception.py:431-440`
Replace the unconditional `return [code, _SUBTASK_DONE]` with a **two-phase latch**: on first entry to
a relative-turn instruction return `[_FORWARD]` (optionally twice); only then return
`[code, _SUBTASK_DONE]`. Track phase in the existing `_turn_burst_done_for` field (make it a small
dict/counter keyed by instruction so the subtask-change reset at `:400-403` still clears it).

- **Why it works:** §2.5 — TURNING is graded on a position box centred on the intersection, the ROAD
  waypoint sits short of it, and all 7 failures missed by 78–980 cm *outward along travel*.
- **Priced on this run** (margin to the far edge of the box measured for the 5 current passes):
  **1 forward** converts 4/7 failures (452, 356, 78, 506 cm) and breaks **0/5** passes (tightest margin
  is ep_20_3_11 at 520 cm). **2 forwards** converts **7/7** and breaks **1/5** (ep_20_3_11) → **net +6**.
  Remaining margins on the other four passes are 3 458–5 237 cm, i.e. 7–10 forwards of headroom.
- **Predicted:** leg-2 `0.111 → 0.24` (2-forward variant), SubtaskSR **+0.03**, and 6 more episodes
  survive to attempt leg 3. Cost: 2 of TURNING's 50 free substeps.
- **Cheapest high-confidence item in this report.** Ship it alone and it is measurable.

### B — Emit the facing rotation before the FINISH `-1`. `modules/completion_checker.py` (new helper) + `orchestrator.py:430-444`
`_named_side(instruction)` already exists (`completion_checker.py:248`). On a FINISH-style leg
("… on your left/right, **stop and face that building**") prepend one rotation to the commit burst:
`planner_actions = [1, -1]` for *left*, `[2, -1]` for *right* (`1 = Rotate_Left = −90°`,
`2 = Rotate_Right = +90°`, per `rules.md`).

- **Why it works:** §2.6 — measured **50/50**: `right → +90`, `left → −90` relative to the travel
  heading, and the env grades FINISH on that orientation within 45°. Derived **entirely from the
  instruction string**, the same honest inference the ORIENTATION cardinal burst already makes.
- **Distinguish the leg:** ROAD instructions read "Continue straight and **stop at the intersection**
  …"; FINISH reads "Proceed forward until reaching … **stop and face that building**." A `face that
  building` / `until reaching` predicate separates them without touching `_is_orientation_instruction`
  (which only matches a named cardinal and must stay as-is).
- **Predicted:** **0 points on its own** — no episode currently reaches the FINISH window — but leg 3
  is **0.00 and will stay 0.00 without it**, whatever else improves. Ship it together with C/D so the
  first correctly-positioned FINISH can actually score.
- **Hazard for whoever also ships the side-scan:** an in-place yaw that commits a **ROAD** leg leaves
  the heading 90° off, and the following relative-turn burst rotates from that wrong base. Yaw **back**
  before `-1` on ROAD (position is identical, ROAD is position-only); commit **from** the turned pose
  on FINISH.

### C — Stop the cross-check from delaying and displacing the commit. `modules/completion_checker.py:748-749`, `orchestrator.py:495-519`
Two one-line semantics changes:
1. `:749` — on `fires >= MAX_XCHECK_FIRES_PER_SUBTASK` return **`False`**, not `True`. A spent budget
   means "cannot verify", which must not mean "commit anyway". Better still: **cache the verdict per
   (subtask, forward_streak-bucket)** so a re-asked identical claim costs nothing and the cap stops
   being a countdown to a free pass.
2. `orchestrator.py:519` — when a bare `[-1]` is vetoed, do **not** inject `Move_Forward`. Emit a
   **position-preserving** probe instead (yaw to the named side → `scan_reach` → yaw back, the
   existing `:459-476` machinery) or, failing that, re-perceive after a net-zero `[2, 1]` pair.
- **Why:** §2.4 — the veto costs 1 500 cm per cycle and the third claim always commits. 3 episodes
  (17_11 574→3 074, 18_21 388→2 888, 19_21 478→2 022) had a **passing** claim vetoed and a **failing**
  one rubber-stamped; 2 more (24_11, 13_21) died purely on cap expiry.
- **Predicted:** leg-1 **+3 episodes** (0.24 → 0.30) from the vetoed-in-window class, plus 2 episodes
  no longer terminated by a far-field rubber-stamp. Confidence high — the mechanism is deterministic
  and the five instances are individually traced.
- **Risk:** returning `False` on cap exhaustion makes the gate strictly tighter; if the recogniser's
  recall is not also raised (D), those 3 episodes convert from "fatal commit" to "walk past" — same
  leg-1 score, fewer fatal terminations. **C is worth most when shipped with D.**

### D — Give the arrival call the evidence the perceiver already gets, and ask a *position* question. `modules/completion_checker.py:530-604`
Two independent edits to `_binary_reach_vlm_call`:
1. **Evidence.** It currently sends RGB + hint and nothing else. Attach `obs['object_mask']` and
   `obs['depth']` with the same captions `prompt_assembler.py:130-137` already uses, and pass the
   current perceiver `Description` (thread it from `orchestrator.py:314`) as text context. §2.3 is the
   direct evidence: the 4-image channel is right on the exact frames the 2-image channel refuses.
2. **Criterion.** Delete the "*clearly visible and prominent in the current view*" clause (`:565-566`)
   — all 8 far-field false positives quote it back verbatim ("visible **ahead**", "visible in both
   views") — and replace it with a **proximity** test: *"1 only if you are standing AT the landmark —
   its facade fills the near field / extends off the side of the frame and the cross-street opening is
   beside you; 0 if you can merely see it further down the street."* Keep the `Reason` field: it is
   what made this diagnosable.
- **Status vs the exhausted list:** the prior gradient exhausted **illumination-invariance
  re-wordings** of the *same* question on the *same* two images. This is different on both axes —
  it changes the **evidence set** (never tried; the REACH call has always been 2 images) and it
  **removes a clause that licenses the observed failure**, rather than adding a hedge. Ship (1) and
  (2) as **separate candidates** so the attribution is clean; (1) is the structural one.
- **Predicted:** in-window recall 0.20 → 0.35-0.45 and far-field `Match=1` 0.03 → < 0.01 would take
  leg 1 from 0.24 to ≈ 0.40 (36 reaching episodes × ~1.2 in-window firings). Largest ceiling in the
  report, lowest confidence.
- **Cost:** +~1 300 input tokens per REACH call (perceive is 2 606 with 4 images, REACH is ~1 030 with
  2). At ≤12 fires/subtask that is well inside the 80.7k/episode budget.

### E — Give the harness an overshoot signal (it has none). `modules/completion_checker.py` + `modules/spatial_memory.py`
Keep the last K RGB frames sampled every N forwards (SpatialMemory already logs per-step actions and
`moving_distance`; add a small frame ring buffer). Ask the recogniser an **ordinal** question instead
of an absolute one: *"Here are three views taken N forwards apart, oldest→newest, plus the reference.
Which is closest to the reference viewpoint: NEWEST / MIDDLE / OLDEST / NONE?"* → `NEWEST` ⇒ commit,
`MIDDLE`/`OLDEST` ⇒ **PASSED**, `NONE` ⇒ keep walking.
- **Why:** §2.1 — 24 episodes walked 3 122–65 291 cm past the window with no mechanism able to notice.
  A relative "which of these is closest" comparison is a strictly easier judgement for a VLM than an
  absolute threshold, and it is the only formulation that can emit a *passed* verdict at all.
- **The recovery is affordable:** ROAD/FINISH legs get `L1/100` substeps and need `L1/500` forwards —
  **5× slack**, verified (`budget` vs `used`: 39/6, 33/6, 30/8, 131/27 on the passing episodes). A
  bounded back-up (`[1,1]` → forwards → `[2,2]`, expressed in action counts, no cm literals) fits.
- **Predicted:** the biggest single population in the failure table (24 episodes) but the most
  speculative lever. Prototype it as a *logging-only* verdict first (emit the ordinal answer into
  `notes` without acting on it) so the next iter can price its accuracy before wiring it to `-1`.

### F — Resurrect the two dead safety nets. `modules/spatial_memory.py:66`, `modules/active_perception.py:520-533`, `orchestrator.py:280`
`info['agent_collision']` is **not** in the sanctioned surface (`rules.md`) — do **not** wire it.
Drive both mechanisms off `moving_distance`, which is sanctioned and already computed:
- wall recovery (b′): fire on `last action == 0 and moving_distance_delta(window=1) < 1.0`, dropping
  the `static_collision_delta > 0` conjunct entirely;
- the stuck detector (`orchestrator.py:280`): call `memory.forward_count_since_boundary()` instead of
  counting the current burst.
- **Predicted:** recovers the 3 wedged episodes (20_21 19/20 stalled forwards, 22_21 19/22, 8_11 7/11)
  from 0 legs to at least a chance at leg 1. Small (+0.02 SubtaskSR at most) but the code is already
  written and simply never executes.

### G — Free-est fix: let the planner emit `Match`. `modules/planner.py:120-133`
The shared `reasoning_template` system block's Output Format omits `Match`, so 111/918 (12 %) replies
drop it and `verify()`'s `planner_match == 1` silently fails. Append `Match` to the system-prompt
Output Format, or treat `match is None` as *unknown* rather than folding it into `== 1`.
Zero risk; unblocks a gate the chain believes is live. (Flagged in the prior gradient's §5-D; still live.)

---

## 6. Exhausted — do NOT repeat

Carried forward from the prior `diagnosis.md`, **re-confirmed on 50 episodes** where noted:

- **Do not touch the ORIENTATION path.** 50/50 here. `_parse_target_cardinal` + `_rotation_burst` is
  the only thing that works end to end. Leave it exactly as it is.
- **Do not re-word the arrival question for illumination invariance.** Six reformulations plus graded
  confidence were priced offline and all lost. Re-confirmed: 3 of the decisive vetoes here cite
  "lighting" verbatim — that rationalisation is a symptom, not a lever. §5-D(2) is *criterion
  replacement* (removing a clause that licenses the far-field FP), not a lighting hedge; keep them
  distinguished when reporting.
- **Do not tune `K_REACH_FORWARD_STREAK` / `K_REACH_INTERVAL` / `MAX_REACH_FIRES_PER_SUBTASK`.** Run to
  death across iters 006-010 in both directions. Re-confirmed: 49/50 episodes got REACH firings and
  44 of them landed inside the pass window — **coverage is not the binding term, the answer is.**
- **Do not re-run the `N_REACH_CONFIRMATIONS_REQUIRED = 2` experiment** (iter 8): it regressed
  SubtaskSR 0.30 → 0.25 and leg-1 to 0.00. With in-window recall at 0.20, a 2-of-2 gate has an
  expected recall of 0.04.
- **Do not use `yel_band` / `pur_sides` / the raw lateral-yellow fraction as an arrival trigger.**
  Measured flat (0.51–0.66) at every position.
- **Do not ship a bare depth-polarity flip**, and do not re-enable `_strongly_far` / `_looming_state`
  as a gate — per-frame-normalised JET has no absolute scale, and the `far`-block was already shown to
  suppress the REACH check at exactly the off-axis waypoints where it is needed.
- **Do not read** `agent_location`, `target_location`, `destination`, `correct_robot_location`,
  `ideal_path`, `gt.json`, `current_instruction`, `env.unwrapped._agent_rotation`, or
  `env.subtask_max_steps` (it is `int(L1_to_target/100)` — laundered `target_location`).
  **`info['agent_collision']` is also outside `rules.md`'s sanctioned surface** — see §5-F.
- **Avoid numeric literals ≥ 2000** in candidate code (the lint WARNs them as map-tuned). Express every
  distance as a count of forward actions.
- **`bootstrap/*/diagnosis.md` in this tree is one file copied 7×** and documents VOID rounds with zero
  traces. Do not mine those `0.0`s as refutations.

**New to this pass — do not re-derive:**
- The `-1` commit that *fails* its criterion **terminates the episode** (`traffic_world.py:634-651`).
  Every false commit is fatal; there is no second attempt. Any lever that trades precision for recall
  must be priced against that.
- **`request_side_scan` as currently gated is dead** (`_side_band_near`, depth): 2 firings in 50
  episodes, 0 commits. The *mechanism* (in-place yaw preserves position, so a location-graded leg stays
  valid) is sound and is the right vehicle for §5-B and §5-C(2) — the **trigger** is what is wrong.
  Re-trigger it on channel *disagreement* (planner says reached ∧ forward REACH says 0), not on depth.
- **`SpatialMemory.static_collision_delta()` always returns 0** and always has
  (`spatial_memory.py:66`), so iters 008 and 010's levers have never executed. Do not read their
  historical score deltas as evidence about collisions.

---

## 7. Provenance

- Candidate: `$OUT_DIR` = `simworld_gym/simworld_config/harnesses/nav` (`orchestrator.py` 613 L + 7 modules,
  2 705 L total). Score: `runs_adaptive/nav_hard__hard5010_adaptive_k4c1/arm_hard5010_adaptive_k4c1/
  anchor_nav_r000/score.json` — SR 0.0, SubtaskSR 0.36, per_position_SR [1.0, 0.24, 0.1111, 0.0],
  distance_progress_mean 0.145, moving_distance_mean 26 856 cm, 80 746 tok / 799 s per episode.
- Traces: `…/anchor_nav/traces/` — 50 episodes, `trajectory.json` + `vlm_trajectory.jsonl` +
  `verdict.json` + `gt.json` each. 348 REACH calls, 918 plan calls, 2 812 forward steps analysed.
- Env semantics quoted from `SimWorldGym/simworld_gym/envs/traffic_world.py:384` (`_get_terminated`),
  `:389-460` (`_get_evaluation_of_subtask`), `:634-651` (`-1` handling + `subtask_max_steps`).
- Instruction↔orientation mapping (§2.6) computed over all 50 `task_config.json` files under
  `envs/setting/single_agent_world/hard/`.
- `read_audit.json`: 15 unique paths, **0 forbidden / 0 escalated / 0 undeclared** on every episode.
  `gt.json` and `trajectory.json` positions were used **only for this analysis**; no recommendation
  above routes them into a policy path.
