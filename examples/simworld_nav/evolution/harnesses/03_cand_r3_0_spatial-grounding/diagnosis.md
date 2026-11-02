# Diagnosis — SimWorld-MMNav (hard, `search_hard_50`) / `anchor_nav`, arm `hard5010_adaptive_k4c1`

**Headline:** SR **0/50** · SubtaskSR **0.360** · per-leg **[50/50, 12/50, 5/45, 0/45]** ·
cost 80 746 tok/ep · `moving_distance_mean` 26 856 cm · `distance_progress_mean` 0.145 ·
`over_budget: false`.

**Binding constraint:** the harness **navigates to the waypoint and then fails to say `-1` there.**
In **26 of the 45** episodes that have a ROAD leg the robot physically drove *through* the ROAD pass
zone — median closest approach **139 cm** on a **2000 cm** threshold, as tight as **9 cm** — and never
committed; and in the **8** episodes that reached a FINISH leg, **7** committed `-1` after only 6–21
steps at **15 733–58 789 cm** from the destination with **188–627** substeps of budget unspent.
The single mechanism that decides every `-1` is one binary landmark-recognition VLM call, and it is
**miscalibrated in opposite directions on the two leg types**: 21 % recall at a true ROAD waypoint,
and "yes" on the first frame of a FINISH leg because the destination tower is *visible* from 30 km
away. Everything else in the loop is downstream of that one call.

> **Scalar hygiene.** 50 episodes, temp 1.0. Every claim below is a per-frame / per-step measurement
> pooled over all 50 episodes against `env_step_log.jsonl` + `vlm_trajectory.jsonl` + `gt.json`, not
> a reading of the SubtaskSR scalar. Oracle coordinates were used **for measurement only** (§Anti-cheat).
> `per_position_SR[1]` = 12/50 pools two different leg types: 45 episodes are
> ORIENTATION→ROAD→TURNING→FINISH and 5 are ORIENTATION→FINISH, so index 1 is ROAD for 45 and FINISH
> for 5. Per-type rates: **ROAD 12/45**, **TURNING 5/45**, **FINISH 0/50**.

---

## 1. What this harness does (mechanism)

`agent.py::run_episode`, one outer iteration per *decision* (not per env step). Seven modules:

| order | call site | what it owns |
|---|---|---|
| 1 | `active.should_emit` (`agent.py:301`) | **short-circuits perceive+plan entirely** when it returns a list. Owns (a) the ORIENTATION cardinal burst `[rot…, -1]`, (a″) the **relative-turn burst** `[rot, -1]` (`active_perception.py:431-440`), (a′) the red-signal idle, (b/b′) anti-spin + wall recovery, (d) **forward momentum, cap 2** → 3 forwards per perceive+plan pair. |
| 2 | `perceiver.perceive` | 1 VLM call, 4 images (RGB, hint, `object_mask`, JET depth) → free-text `Description`. |
| 3 | `planner.plan` | 1 text-only VLM call → `{Summary, Match, Actions}`. |
| 4 | `completion.verify` (`:391`) | honours a planner `-1` only if `planner_match == 1 and forward_streak >= 1`. |
| 5 | `cross_check_planner_commit` (`:413`) | 2nd binary VLM call gating a planner `-1`; **capped at 2 fires/subtask**, then *defers* (returns True). |
| 6 | `proactive_commit_check` (`:430`) | the dominant commit path. At `forward_streak ≥ 4`, every `K_REACH_INTERVAL = 5` forwards, ≤ 12/subtask: one binary REACH call `(live RGB, hint) → {Match}`; `Match==1` ⇒ inject `[-1]`. |
| 7 | `request_side_scan` / `scan_reach` (`:459`) | depth-gated in-place yaw toward the instruction's named side, re-ask REACH, yaw back on miss. |

Stages 5, 6 and 7 all funnel into **one function**, `completion_checker._binary_reach_vlm_call`
(`:530-605`). Its system prompt is the whole arrival policy:

> "…judge whether the navigator has **REACHED** the landmark … 1 if the current view shows the same
> landmark / place as the reference (the same building, intersection, or scene is clearly **visible
> and prominent** in the current view) … **Be conservative — output 1 only when the match is
> unambiguous.**"

Two facts about the environment the harness is not built against (`envs/traffic_world.py:626-651`,
`:384`) and which set the whole cost structure:

* **A failed `-1` terminates the episode outright** — `if self._evaluation_of_subtask: … else:
  terminated = True`. There is no second attempt. A premature commit does not cost a step; it costs
  the entire remaining episode.
* **Each leg has its own step budget** — `subtask_max_steps` = 10 (ORIENTATION), **50** (TURNING),
  `int(L1_dist/100)` (ROAD/FINISH), and `substeps_count >= subtask_max_steps` also terminates. The
  harness's own `max_steps` is 672 and was **never** the binding limit (0 `MAX_STEPS` notes in 50 eps).

Grading (same file, `:388-460`): ORIENTATION = yaw within 45°; **ROAD = L1 distance < 2000 cm** to
`correct_robot_location`; **TURNING = yaw within 45° AND inside `correct_robot_position_area`** (a box,
not a radius); FINISH = L1 < 2000 cm **and** yaw within 45°.

---

## 2. What actually happened (causal chain, with evidence)

### 2.1 ORIENTATION — 50/50, zero VLM calls. Do not touch this.
`_parse_target_cardinal` + `_rotation_burst` close the ORIENTATION leg deterministically in 0–2
rotations. `per_position_SR[0] = 1.0`. This is the only leg that works and it works because it is
**not** decided by a VLM.

### 2.2 ROAD — the harness arrives and does not commit (26/45 episodes)

The ROAD instruction is always of the form *"Continue straight and **stop at the intersection** with
the \<long building description\> **on your left**."* The waypoint is 3 000–7 000 cm ahead; a forward
step is exactly 500 cm.

**Of the 45 episodes with a ROAD leg: 12 pass, 26 enter the 2000 cm pass zone and never commit, 7
never reach it.** Closest approach for the 26, with where they ended up:

| episode | min L1 to ROAD target | L1 where the leg ended | forwards spent |
|---|---|---|---|
| `ep_20_7_task_dist_21_0_1` | **9 cm** | 23 509 | 65 |
| `ep_20_2_task_dist_11_0_1` | **50 cm** | 47 950 | 120 |
| `ep_20_16_task_dist_21_0_1` | **65 cm** | 53 935 | 135 |
| `ep_20_17_task_dist_21_0_1` | 42 cm | 43 542 | 110 |
| `ep_20_19_task_dist_11_0_1` | 212 cm | 65 291 | 177 |
| … 21 more, median min L1 **139 cm** | | | |

These are not navigation failures. The agent walks the correct street, crosses the correct point, and
keeps going until the env kills the leg (9/45 on `substeps_count >= subtask_max_steps`; the rest by
driving off the road polygon into `_get_terminated`).

**Why it does not commit — the recognizer, measured.** Every binary REACH call made during a ROAD
leg (n = 305), bucketed by the robot's true L1 distance to the ROAD target at that step:

| L1 to ROAD target | Match=1 | Match=0 | P(say yes) |
|---|---:|---:|---:|
| **< 2000 cm (PASS zone)** | **9** | **33** | **0.21** |
| 2 000 – 5 000 | 0 | 51 | 0.00 |
| 5 000 – 15 000 | 1 | 102 | 0.01 |
| > 15 000 | 0 | 109 | 0.00 |

Specificity is **99.6 %** (1 false positive in 263 out-of-zone calls). Recall at the true waypoint is
**21 %**. Every defensive mechanism the chain has accreted — `_named_side_has_structure`,
`N_REACH_CONFIRMATIONS_REQUIRED`, the xcheck cap, `_strongly_far` (already disabled) — defends against
a false-positive threat that **does not exist on this leg**, while the actual loss is 79 % missed
detections.

Sample rejections *inside* the zone — the reason string is nearly always the same:

```
ep_20_17_task_dist_11  step 11  L1=  74 cm  {"Match": 0, "Reason": "Different lighting and building details"}
ep_20_18_task_dist_21  step 15  L1= 112 cm  {"Match": 0, "Reason": "Different building colors and lighting conditions"}
ep_20_13_task_dist_21  step 17  L1= 134 cm  {"Match": 0, "Reason": "Different building colors and lighting conditions"}
ep_20_12_task_dist_11  step 17  L1= 361 cm  {"Match": 0, "Reason": "Different lighting and building details"}
```
21/21 in-zone (<1000 cm) rejections cite **global appearance** — lighting/colour — not identity. The
reference `robot_view` hint was captured facing the building under different lighting; the live frame
is a forward view down a dim street with the target building **off-axis on the named side**. The
prompt asks for *"visible and prominent"* and tells the model to *"be conservative"*, and it obeys.

**Why only ~1 shot lands in the window.** Momentum cap 2 ⇒ 1500 cm of travel per decision;
`K_REACH_INTERVAL = 5` ⇒ a REACH look every 2500 cm. The pass window is 4000 cm wide. So a traversal
buys ≈1.6 looks, and the measurement confirms it: 42 of 305 calls (14 %) land in the zone, median 1
per episode. **0.21 recall × ~1 shot ≈ 0.21 → observed ROAD 12/45 = 0.27.** The arithmetic closes.

### 2.3 The cross-check cap turns a veto into a countdown (3 episodes lost outright)

`MAX_XCHECK_FIRES_PER_SUBTASK = 2` (`completion_checker.py:139`), and on cap exhaustion
`cross_check_planner_commit` **defers** (returns True). So the harness vetoes the first two planner
`-1`s of a leg and rubber-stamps the third, **regardless of where the robot is standing**:

| episode | planner `-1` #1 | #2 | #3 (auto-approved) | outcome |
|---|---|---|---|---|
| `ep_20_17_task_dist_11` | **74 cm** ✗blocked | **1 574 cm** ✗blocked | 3 074 cm ✓passed | commit failed → **episode terminated, step 18** |
| `ep_20_18_task_dist_21` | **112 cm** ✗blocked | **1 388 cm** ✗blocked | 2 888 cm ✓passed | failed → terminated, step 22 |
| `ep_20_19_task_dist_21` | **978 cm** ✗blocked | **522 cm** ✗blocked | 2 022 cm ✓passed | failed (22 cm outside!) → terminated, step 33 |
| `ep_20_2_task_dist_21` | 3 578 cm ✗ | 2 078 cm ✗ | **578 cm** ✓ | **passed** |
| `ep_20_3_task_dist_21` | 3 317 cm ✗ | 1 817 cm ✗ | **317 cm** ✓ | **passed** |
| `ep_20_8_task_dist_21` | 5 506 cm ✗ | 4 006 cm ✗ | **1 006 cm** ✓ | **passed** |

ROAD on the planner channel is decided by **whether the third proposal happens to land in the zone**.
The cap is not a safety mechanism; it is a lottery, and it loses 3 episodes and wins 3.

### 2.4 The blocked-commit fallback walks the agent *out* of the zone

`agent.py:495-519`. When the xcheck blocks, the planner's surviving actions are `[a for a in
planner_actions if a in allowed]` — but the planner emitted a bare `[-1]`, so the list is empty and
`EMPTY_FALLBACK` injects **`[0]` (Move_Forward)**. With momentum that is 1500 cm per rejection. Trace
the distances in the table above: 74 → 1 574 → 3 074; 112 → 1 388 → 2 888; 978 → 522 → 2 022. **Every
rejection pushes the robot 1 500 cm further from the place it just claimed to be.** By the third
attempt it is out of the zone, which is exactly when the cap lets the commit through.

### 2.5 TURNING — committed 1–2 forward steps short of the box, every time (7/12)

`active_perception.py:431-440`, rule (a″), matches `turn (left|right)` in the instruction and returns
`[rot, -1]` **on the first decision of the leg, with zero translation and zero VLM calls**. The leg
has a 50-substep budget; the burst spends 2. Since ROAD systematically commits just *before* the
intersection, the rotate-and-commit lands just outside the TURNING box:

| episode | pos at TURNING commit | box | outside by | forwards needed |
|---|---|---|---|---|
| `ep_20_2_task_dist_21` | [-1700, 17222] | [[-2700,17300],[7700,19300]] | **78 cm** | **1** |
| `ep_20_13_task_dist_11` | [-16944,-58300] | [[-22700,-59300],[-17300,-51900]] | 356 cm | 1 |
| `ep_20_0_task_dist_11` | [3152,-38300] | [[-2700,-39300],[2700,-31900]] | 452 cm | 1 |
| `ep_20_8_task_dist_21` | [16794,-18300] | [[17300,-28100],[22700,-17300]] | 506 cm | 2 |
| `ep_20_11_task_dist_21` | [18300, 3327] | [[11900,-2700],[19300,2700]] | 627 cm | 2 |
| `ep_20_1_task_dist_11` | [18300, 3392] | [[11900,-2700],[19300,2700]] | 692 cm | 2 |
| `ep_20_23_task_dist_21` | [-56320, 1700] | [[-59300,-8100],[-57300,2700]] | 980 cm | 2 |

**All 7 misses are 78–980 cm — one or two forward steps.** And because a failed `-1` terminates,
each one also destroys the FINISH leg. `ep_20_0_task_dist_11` ends at step 12 of a 672-step budget.

### 2.6 FINISH — 0/50, and the recognizer fails the *other* way

Of the 8 episodes reaching a FINISH leg, 7 committed in 6–21 steps at 15 733–58 789 cm, with
188–627 substeps unspent. Same call, opposite bias — every accepted frame says **"visible"**:

```
ep_20_10_task_dist_21  L1=58 789  {"Match":1,"Reason":"The reference building with striped awnings is visible ahead in the current view."}
ep_20_5_task_dist_11   L1=32 586  {"Match":1,"Reason":"The reference building with rectangular windows and a large awning is clearly visible in the current view."}
ep_20_3_task_dist_21   L1=35 733  {"Match":1,"Reason":"The building with large glass panels and brown frames is visible in both views."}
ep_20_16_task_dist_11  L1=26 273  {"Match":1,"Reason":"The building with arched glass windows and stone facade is visible ahead in both views."}
ep_20_9_task_dist_11   L1= 7 778  {"Match":1,"Reason":"The building with grid-like glass windows and colorful billboards is visible in both views."}
```

FINISH-leg calls by distance: P(say yes) = 0.00 (<2000, n=1) · 0.11 (2–10k) · 0.15 (10–30k) · **0.23
(>30k)**. The acceptance rate *rises with distance* — because a FINISH target is a tall building
straight down an open street, visible from the moment the leg starts, whereas a ROAD target is a
corner building off-axis to the side.

And the mirror-image miss: the two 2-leg episodes that never committed — `ep_20_18_task_dist_11` and
`ep_20_4_task_dist_21` — passed within **16 cm** and **27 cm** of the destination and kept walking
(147 and 143 steps on the leg).

**One prompt produces both failure modes.** "Visible and prominent" is a *recognition* predicate; the
grader applies a *ranging* predicate (L1 < 2000 cm). The loop contains **no ranging signal at all** —
`_depth_gray`'s own docstring records that the env normalizes depth per frame, so there is no absolute
scale, and `_strongly_far` was disabled for exactly that reason.

---

## 3. Failure breakdown (50 episodes)

| category | eps | share | evidence |
|---|---:|---:|---|
| **ROAD reach-but-no-commit** — drove through the 2000 cm zone, never `-1` | **26** | 52 % | §2.2; median min-L1 139 cm |
| **TURNING commit 1–2 forwards short of the box** (fatal) | 7 | 14 % | §2.5; 78–980 cm outside |
| **FINISH commit 15–59 km short** (fatal) | 7 | 14 % | §2.6; "visible ahead" |
| **xcheck cap-lottery: correct commits vetoed, wrong one approved** (fatal) | 3 | 6 % | §2.3 |
| **Never reached the ROAD zone** (wrong street / wedged / off-road early) | 7 | 14 % | min-L1 3 537–42 770 cm; 2 `STUCK`, 1 wall-mash (`ep_20_8_11`, 28 732 cm) |
| Parse failure (`NO_ACTION_RETURNED` ×8) | 1 | 2 % | `ep_20_12_task_dist_21`; loop tolerated it, not a leg loss |

Side-channel signals (not scored, but real): `red_light_violation_mean = 8.0` — the iter_009 red-signal
idle is not preventing violations; `static_collision_mean = 1.78`, with `ep_20_0_task_dist_21` logging
20 building collisions in 133 forwards. Neither enters the objective; ignore them until the commit
channel is fixed.

**Dead code, verified:** `SIDE_SCAN_COMMIT` fired **0 times in 50 episodes** (`SIDE_SCAN_MISS` twice).
`request_side_scan`'s trigger `_side_band_near` is gated on the per-frame-normalized depth and
effectively never fires. The whole iter_005 turn-to-face mechanism is inert.

---

## 4. Binding constraint — classified

**Primary: VLM-CEILING on `_binary_reach_vlm_call` as currently posed**, wrapped in **HARNESS-BUGs
that amplify it**.

* **VLM-CEILING** — the frozen Qwen3-VL-30B cannot answer "have I arrived?" from (live RGB, reference
  photo) alone. Measured: 21 % recall at a true ROAD waypoint (rejecting on lighting/colour), and
  23 % *acceptance* at >30 km on a FINISH leg (accepting on visibility). Re-wording the prompt alone
  will not fix this — it must be **scaffolded**: give the decision a ranging signal it does not
  currently have, or make the *sampling schedule* compensate for the low recall, or both.
  Note `cand_r2` already tested "give the same call more channels (seg + depth + description)" and
  **ROAD regressed 0.24 → 0.12** — richer evidence into the same question does not help.
* **HARNESS-BUG (fixable, high confidence)** — (i) the xcheck cap-exhaustion lottery (§2.3),
  (ii) the blocked-commit fallback that walks the agent out of the zone at 1500 cm per rejection
  (§2.4), (iii) the relative-turn burst committing with zero translation (§2.5), (iv) a REACH
  schedule that spends 86 % of its calls where the answer is known to be "no" (§2.2).
* **TASK-GAP (secondary)** — the agent has no representation of *how far along the leg it is*. It has
  honest odometry (`info['moving_distance']`, +500 cm/forward, exact) and never uses it to reason
  about leg progress; nothing in the loop counts intersections or blocks.

**This is fixable in the harness.** The robot is already standing on the right spot in 26/45 ROAD
legs. Nothing about perception-for-locomotion is broken.

---

## 5. Concrete changes to try (ranked by measured leverage)

**Ground rule from the chain:** the anchor's **ROAD 12/45 is the best ROAD number in the lineage**
(`cand_r1` 1/50, `cand_r2` 6/50). Both regressions came from bundles that touched the ROAD commit
criterion. **Change the ROAD commit path one lever at a time**, and keep §5-A separable from the rest.

---

### A. Stop walking away from a blocked commit — replace `EMPTY_FALLBACK` with a zero-displacement hold
`agent.py:495-519`. When `commit_done` was revoked (XCHECK_BLOCK) or the planner's surviving action
list is empty, **do not inject `[0]`**. Emit a net-zero re-look instead: either nothing (fall through
to the next perceive+plan on a fresh frame) or the in-place pair `[2, 1]` already used by the traffic
idle. Also reset `active._consecutive_momentum` so the next decision is not a 1500 cm momentum burst.
*Predicted:* the 3 lottery-loss episodes (§2.3) keep re-proposing from 74 / 112 / 978 cm instead of
being pushed to 3 074 / 2 888 / 2 022 cm. **+2–3 ROAD legs** and 3 fewer fatal terminations.
*Why:* measured — each rejection currently costs 1 500 cm of ground truth error (§2.4). This is the
one edit that strictly adds opportunity and cannot cause a false commit.
*Caveat:* `cand_r2` shipped something like this **bundled** with the evidence change and ROAD fell to
0.12. Ship it **alone** so it is attributable.

### B. Delete the xcheck cap, or make exhaustion DENY instead of DEFER
`completion_checker.py:139` (`MAX_XCHECK_FIRES_PER_SUBTASK = 2`) and `:748`. Today the third planner
`-1` of a leg is auto-approved wherever the robot stands. Either (i) remove the cap (the call is
~1 026 input tokens; at ≤4 planner `-1`s per leg this is < 5 k tok/ep against an 80 k/ep budget and
`over_budget: false`), or (ii) on exhaustion return **False**.
*Predicted:* removes the 3 fatal auto-approved commits. Note the honest trade: with the cap gone the
21 %-recall recognizer becomes authoritative on the planner channel too, which would also have blocked
the 3 lottery *wins* — so **A+B must ship together with D**, which is what restores the recall.
*Why:* a cost cap is currently making the correctness decision.

### C. Put 1–3 forwards in front of the relative-turn commit
`active_perception.py:431-440`. Replace the unconditional `return [code, _SUBTASK_DONE]` with a staged
latch: return `[0]` for the first N decisions of the leg, then `[code, -1]`. Price the approach off
odometry — if a forward moves `moving_distance` < 1 cm the agent is against a facade, so stop
advancing and commit.
*Predicted:* **+4 to +7 TURNING legs** (all 7 misses are 78–980 cm = 1–2 forwards; §2.5), i.e.
per-leg TURNING 5/45 → ~10/45, SubtaskSR **+0.03 to +0.04**, and 7 episodes survive to attempt FINISH.
*Why:* directly measured geometry, and **independently corroborated by the chain** — `cand_r1` and
`cand_r2` both carried a `_TURN_APPROACH_FORWARDS = 2` latch and the *conditional* rate
P(TURNING | ROAD) rose **5/12 = 42 % → 5/6 = 83 %** (`cand_r2`) and 1/1 (`cand_r1`). Their SubtaskSR
regressions came from the ROAD edits bundled alongside, not from this latch. **Port this one verbatim.**
Budget is not a constraint: TURNING gets 50 substeps and the burst currently spends 2.

### D. Re-aim the REACH schedule: fewer calls far away, many calls in the plausible window
`completion_checker.py:83,96,114` + `proactive_commit_check`. Today 86 % of REACH calls fire where
P(yes) ≈ 0.005 and ~1 lands where it matters. Two composable sub-levers:

* **D1 (cheap, mechanical).** Set `K_REACH_INTERVAL = 1` once a *candidate-arrival* condition holds,
  keeping `MAX_REACH_FIRES_PER_SUBTASK` as the cost bound, and raise `K_REACH_FORWARD_STREAK` so the
  early-leg calls (P(yes) ≈ 0) are not wasted. With `p = 0.21` per call, going from ~1 to 4 in-zone
  shots gives `1 − 0.79⁴ = 0.61` detection. *Predicted:* **ROAD 12/45 → ~20–25/45**, SubtaskSR
  +0.04–0.07 before downstream effects. Cost +~10 REACH calls/ep ≈ **+10 k tok/ep (+13 %)**; headroom
  exists. Specificity is 99.6 %, so extra shots carry very little false-commit risk **on the ROAD leg**.
* **D2 (the trigger).** The candidate-arrival condition should be the **intersection cue the
  instruction names**, not a blind forward count. Measured on this run's own seg masks (building class
  purple `(128,0,128)`, rows 0.30–0.62, outer 25 % of the named side): the named-side building fraction
  is **0.65–0.69 median along the block** and drops to **0.42 median (p25 0.31) inside the ROAD pass
  zone** — the cross-street gap. A running-median drop test (`frac < 0.6 × median-so-far`, 3-frame
  warm-up) fires in 28/45 episodes with **median first-fire at 1 880 cm** — right at the zone edge —
  and enriches in-zone call density from 14 % to 25 %. It is **too noisy to commit on** (32 in-zone vs
  95 out-of-zone fires) but it is a good *gate for spending REACH calls*. Validate before shipping.

### E. Split the arrival question into RANGE and IDENTITY
`completion_checker.py:552-570`. The one prompt asks a recognition question and is graded on a range
predicate, which is why it says "yes, visible ahead" at 58 km (§2.6) and "no, different lighting" at
74 cm (§2.2). Replace the single `Match` with two fields the model can actually answer from an
egocentric frame, e.g. `{"Identity": 0|1, "Range": "adjacent"|"near"|"mid"|"far"}` where *adjacent* is
defined concretely ("the building fills more than half the frame height / you could touch the wall in
a few steps"). Commit only on `Identity==1 AND Range=="adjacent"`. Drop the sentence
**"Be conservative — output 1 only when the match is unambiguous"** from the identity half: with 99.6 %
specificity on ROAD it is buying nothing and costing 79 % of the recall.
*Predicted:* removes the 7 fatal FINISH commits (**FINISH 0/45 becomes attemptable**; those episodes
have 188–627 substeps unspent) and raises ROAD in-zone recall.
*Why:* the failure text is unambiguous — the model answers exactly the question it was asked.
*Risk:* this is the criterion lever that `cand_r1` moved and ROAD collapsed to 1/50. Ship it **after**
A/B/C/D land and are measured, and change **only the criterion wording**, not the evidence channels.

### F. Give the loop a leg-progress representation (structural, larger scope)
Nothing in the harness knows how far along a leg it is, though `info['moving_distance']` is exact
(+500 cm/forward) and sanctioned. Ask the perceiver **once at the start of each leg** for a coarse
count ("how many intersections ahead is the target intersection?"), then count them down with the D2
side-band-drop cue and odometry, and open the dense REACH window on the last one. This is the honest
version of "know when to look" and it also bounds the walk-forever cluster (§2.2), which currently
runs 65 000 cm past the waypoint. Larger change; propose only if A–E have been absorbed.

### G. Housekeeping (free, do alongside anything)
* `_strongly_far` is a disabled no-op (`:382`), `request_side_scan`/`scan_reach` fired 0 successful
  commits in 50 episodes, and `N_REACH_CONFIRMATIONS_REQUIRED = 1` makes the consecutive-confirm
  counter a no-op. Remove them — they are prompt-space and reader-attention cost, and they hide the
  real control flow from the next proposer.
* `_named_side_has_structure` is used as a `> 0.02` veto on **both** commit paths. Measured: the
  named-side building fraction is **> 0.35 on 68–92 % of frames at every distance** (§5-D2), so it
  almost never vetoes, and where it does correlate it does so **backwards** (lowest in the pass zone).
  It is not a far-field sanity check; delete or invert it deliberately, but do not trust its docstring.

---

## 6. Exhausted — do NOT repeat

* **Feeding `_binary_reach_vlm_call` more channels** (segmentation + JET depth + the perceiver's
  description alongside the two RGBs). `cand_r2_0_spatial-grounding_r002` shipped exactly this:
  **ROAD 12/50 → 6/50, SubtaskSR 0.36 → 0.33.** The bottleneck is the *question*, not the evidence.
* **A "commit is a pose" rewrite that tightens the ROAD criterion** (`cand_r1_0_spatial-grounding_r001`,
  four bundled edits): **ROAD 12/50 → 1/50, SubtaskSR 0.36 → 0.285.** The ROAD gate has no slack to give.
* **Depth as an absolute range signal.** `_depth_gray`'s docstring and `_strongly_far`'s disablement
  record it: the env normalizes depth **per frame**, so the nearest object always maps to ~0 and there
  is no scale to distinguish "at the facade" from "30 km down an open street". Any predicate of the
  form "commit when depth reads near" is unimplementable on this channel. `_side_band_near`
  (`request_side_scan`'s trigger) is built on it and fired 0 successful commits in 50 episodes.
* **Re-tuning the red-signal detector.** Two iterations of ROI/threshold retuning already happened
  (iter_004 → iter_009) and `red_light_violation_mean` is still 8.0. It does not enter the objective.
* **Raising `MAX_REACH_FIRES_PER_SUBTASK` alone** (already 12) without fixing `K_REACH_INTERVAL = 5`.
  The cap is not what binds — the 2 500 cm spacing across a 4 000 cm window is (§2.2). Measured
  in-zone calls: median 1 per episode, with 7 fires still unspent at leg end in most episodes.
* **Touching the ORIENTATION cardinal burst.** 50/50 at zero VLM cost. Leave it alone.

---

## Anti-cheat note
Oracle fields (`gt.json`, `agent_location`, `target_location`, `correct_robot_*`) were read **by this
diagnosis only**, to measure where the robot actually was when each decision was made. Every fix above
keeps the frozen agent deciding from the sanctioned surface — `obs['rgb'] / obs['depth'] /
obs['object_mask']`, `info['moving_distance']`, `info['agent']['agent_rotation']`,
`task_spec['subtasks']`, `env.unwrapped.current_subtask`. In particular §5-D2 uses only
`obs['object_mask']` with a **self-normalizing running median** (no absolute coordinate, no map-tuned
constant), §5-C prices its approach off odometry deltas, and §5-F counts intersections from perception
plus odometry. No recommendation introduces a coordinate, a distance-to-goal, or a route read.
