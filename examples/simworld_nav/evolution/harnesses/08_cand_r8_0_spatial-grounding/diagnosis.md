# Diagnosis — SimWorld-MMNav (hard) / `anchor_nav`, arm `hard5010_adaptive_k4c1`, n=50

**Headline:** SR **0.00** · SubtaskSR **0.360** · per-leg pass **[1.000, 0.240, 0.111, 0.000]**
(ORIENTATION 50/50 · ROAD 12/50 · TURNING 5/45 · FINISH 0/50) · distance_progress 0.145 ·
audit clean (0 forbidden / 0 undeclared reads over 21 458 reads).

**Binding constraint (one sentence):** the harness has no working *stopping rule* — in **28/50
episodes the robot physically stood inside the live waypoint's acceptance window** (median closest
approach **179 cm** against a 2000 cm L1 threshold) **and never committed**, then walked a median
**~23 500 cm past it** until the env killed the episode; the arrival predicate cannot fire because
it compares a **forward-facing** frame against a reference photo captured **facing the landmark
~90° off the travel axis**, giving in-window recall **9/42 (21 %)** on ROAD and **0/43** on FINISH.

> **Method note.** Every number below is a full replay of all 50 `env_step_log.jsonl` traces against
> `traffic_world.py:389-462` (the subtask grader) with yaw reconstructed from `gt.spawning_point`
> + the action stream. The replay reproduces **50/50** of the evaluator's `subtask_success` masks
> exactly, so the per-step "was the acceptance window satisfied?" labels are exact, not estimated.
> GT positions are used **for this analysis only**; no recommendation below reads a privileged field.

---

## 1. What this harness does (mechanism)

`harness.py:run_episode` runs one *decision* per loop iteration, but a decision can emit a multi-step
*burst*, so decisions and env steps are different clocks (**3014 env steps / 918 decisions = 3.28**).

| order | site | role in this run |
|---|---|---|
| 1 | `active.should_emit` (`harness.py:301`) | **short-circuits perceive+plan entirely** when it returns a list (`:310`). Owns (a) the ORIENTATION cardinal burst `[rot…, -1]`, (a″) the relative-turn burst `[rot, -1]` (`active_perception.py:434-440`), (a′) the red-signal idle, (b) anti-spin, (b′) wall-recovery, (d) cap-2 forward momentum (`:256`). |
| 2 | `perceiver.perceive` | 1 VLM call, 4 images (RGB, hint, seg mask, JET depth) → free-text `Description`. |
| 3 | `planner.plan` | 1 text-only VLM call → `{Summary, Match, Actions}`. |
| 4 | `completion.verify` (`harness.py:391`) | honours a planner `-1` iff `planner_match == 1 and forward_streak >= 1`. |
| 5 | `cross_check_planner_commit` (`:413`) | 2nd binary REACH VLM call; vetoes a planner `-1`. Fired 18× (note `XCHECK_BLOCK`). |
| 6 | `proactive_commit_check` (`:430`) | **the commit path that actually decides this run.** At `forward_streak ≥ 4`, every 5 forwards, ≤12/leg: one binary REACH call `(live RGB, hint) → {Match}`; `Match==1` ⇒ inject `[-1]`. Fired 17× (`REACH_COMMIT`). |
| 7 | `request_side_scan` → `scan_reach` (`:459-470`) | the *only* component that can acquire an off-axis viewpoint. Fired **2× in 50 episodes** (`SIDE_SCAN_MISS` ×2, `SIDE_SCAN_COMMIT` ×0). |

**Env facts the harness does not model** (`traffic_world.py:615-650`, `:302-386`):
a *rejected* `-1` sets `terminated = True` — **one wrong commit ends the episode**; each leg carries
`subtask_max_steps` (ORIENTATION 10, TURNING 50, ROAD/FINISH `int(L1_to_waypoint/100)`); and
`_get_terminated` also kills on leaving the road-block polygon or on 20 identical positions.

**Grading, per leg** (`traffic_world.py:396-462`): ROAD = **L1 position only**;
TURNING = **yaw ±45° AND inside a junction box**; FINISH = **L1 position AND yaw ±45°**.
One predicate ("does the forward frame look like the hint?") is used for all three.

---

## 2. What actually happened (causal chain, with evidence)

### 2.1 The whole story in one episode — `ep_20_18_task_dist_11_0_1` (2 legs: ORIENT → FINISH)

FINISH target `(-38300, 4523)`, required yaw **0**, threshold 2000 L1. The agent faces yaw 90
(+y, along the street) and walks. Replayed L1-to-target:

```
step 63  L1 = 2016   \
step 64  L1 = 1516    |
step 65  L1 = 1016    |  NINE consecutive steps inside the position window
step 66  L1 =  516    |  (a single Rotate_Left, yaw 90->0, would have satisfied
step 67  L1 =   16    |   the yaw criterion -> FINISH pass -> episode SR = 1.0)
step 68  L1 =  484    |
...      ...          /
step 71  L1 = 1984   /
step 148 the robot leaves the world polygon -> env terminates. mask = [True, False].
```

The three VLM channels at those exact steps (`vlm_trajectory.jsonl`):

* **perceiver (step 68)** — *"the high gray and beige building featuring Chinese characters
  prominently visible **on the left**… occupies a significant portion of the left side of the frame"*.
  **Correct.**
* **planner (steps 59, 62, 65, 68, 71, 74 — byte-identical Summary all six times)** —
  *"the agent is now very close to the high gray and beige building with Chinese characters on the
  left… **but it is still not directly in front**… must continue moving forward"* → `Match: 0,
  Actions: [0]`.
* **REACH recognizer (steps 62, 68, 74)** — `{"Match": 0, "Reason": "The current view does not show
  a high gray and beige building with Chinese characters on the left."}` — flatly contradicting the
  perceiver on the same frame.

**Named failure mode: `unreachable-arrival-predicate`.** The agent's own success test is *"the
landmark fills the frame / is directly in front"* (inherited from the shared baseline
`perception_template`, plus *"ensure proximity… look for close-up"*). The landmark is **beside** the
agent and the reference photo is frontal. **No amount of forward motion can ever satisfy that test** —
forward motion moves the agent *past* the landmark, never in front of it. The phrase "on your left"
is read as a descriptor, never as an instruction to yaw. The policy is a fixed point that walks
forever. `ep_20_4_task_dist_21_0_1` is the identical story (closest approach **27 cm**, then 22 more
steps and out of the world).

### 2.2 The ROAD leg: the recognizer is a near-constant `0`

All 348 REACH-family calls, cross-tabbed against the exact replayed acceptance window:

| leg | calls **inside** window | of those `Match=1` | calls **outside** | of those `Match=1` |
|---|---|---|---|---|
| ROAD | **42** | **9  (recall 21 %)** | 263 | 1 (FPR **0.4 %**) |
| FINISH | **0** *(window never satisfiable — yaw always wrong)* | – | 43 | **7 (all 7 fatal)** |

Overall `Match=1` on **17/348 = 4.9 %** of calls. Combined with coverage — the ROAD window is
~4 forward steps deep, the momentum cap steals 2 of every 3 steps from the decision clock, and
`K_REACH_INTERVAL = 5` spaces fires 2500 cm apart — the harness gets **0.93 questions per
window-pass**, so `P(commit | entered window) ≈ 1 − (1−0.21)^0.93 = 0.20`, versus the observed
ROAD pass rate **0.24**. The closed form reproduces the score.

### 2.3 The TURNING leg: a blind zero-motion commit

All **12/12** TURNING commits fired at `substeps_in_leg == 1` — i.e. `active_perception.py:434-440`
returns `[rotate, -1]` on the **first** decision of the leg, with **zero translation** from the ROAD
waypoint it had just committed at. The rotation is *always* correct:

| episode | yaw err | position | miss vs the junction box |
|---|---|---|---|
| ep_20_2_21 | **0** | (-1700, 17222) | 78 cm short in y |
| ep_20_13_11 | **0** | (-16944, -58300) | 356 cm |
| ep_20_0_11 | **0** | (3152, -38300) | 452 cm |
| ep_20_8_21 | **0** | (16794, -18300) | 506 cm |
| ep_20_11_21 | **0** | (18300, 3327) | 627 cm |
| ep_20_1_11 | **0** | (18300, 3392) | 692 cm |
| ep_20_23_21 | **0** | (-56320, 1700) | 980 cm |

**7/7 failures are yaw-correct and 78–980 cm (1–2 forward steps of 500 cm) short of the box.**
Counterfactual, replayed exactly: inserting **K** forwards along the **pre-turn** heading before the
rotate converts **5/12 → 9/12 (K=1) → 11/12 (K=2) → 10/12 (K=3)**, and the already-earned ROAD
window still holds at K≤2 in 12/12 (in-place rotation preserves position, so nothing is risked).

### 2.4 The FINISH leg: the only commits it makes are fatal

10 episodes reached FINISH. All 7 FINISH commits were rejected: **6/7 failed on distance *and* yaw
(7 778 – 58 789 cm out, yaw err 90–270°)**; 1 on distance alone. Every one was injected by
`proactive_commit_check` on a `Match=1`. Down a straight city street the destination facade *is*
"clearly visible and prominent" from tens of km away, so the question has **no stopping power** on
this leg: its precision here is **0/7**. Meanwhile the two episodes that actually arrived (16 cm,
27 cm) got `Match=0` and walked out of the map.

### 2.5 Everything the episode dies of, and why

| terminal cause | n | mechanism |
|---|---|---|
| **fatal rejected `-1`** | **20** (ROAD 6 · TURNING 7 · FINISH 7) | `traffic_world.py:650` — a failed grade sets `terminated=True` |
| **left the world polygon** | **14** | verified with the actual `build_road_polygon` union: outside on **exactly the final step** in 14/14. **This is the "walk past forever" tail, not an independent road-keeping failure** — median closest approach to the live waypoint 179 cm, median 47 steps (~23 500 cm) walked after it |
| **`subtask_max_steps` deadline** | **9** (all on ROAD) | `substeps_count >= int(L1/100)`; the harness cannot see this clock |
| **20 identical positions** | **4** | wedged against a facade for 20 steps |
| harness self-break (`STUCK`) | 3 | `harness.py` forward-stall detector |

---

## 3. Failure breakdown (rates, from the replay)

| category | rate | note |
|---|---|---|
| **reach-but-no-commit on the live leg** | **28/50 (56 %)** | entered the window (min L1 < 1200 cm), never validly committed |
| ROAD: entered window / committed / never reached | **26 / 12 / 7** of 45 | recognizer recall is the binding term |
| TURNING: blind zero-motion commits | **12/12** | 7 fail by 78–980 cm, yaw always right |
| FINISH: valid commits | **0/50** | every one of 7 attempts fatal |
| fatal commits (episode-ending) | 20/50 (40 %) | |
| walked out of the map after overshooting | 14/50 (28 %) | |
| planner action mix | **Move_Forward 1289/1356 = 95 %** | 37 rotates, 29 `-1` in 918 calls |
| planner `Match=1` rate | 32/918 (3.5 %) | `verify()` therefore almost never gates |
| REACH `Match=1` rate | 17/348 (4.9 %) | near-constant-`0` classifier |
| off-axis look (`request_side_scan`) fires | **2 / 50 episodes** | the one mechanism that could fix §2.1 is switched off |
| parse failures / truncated VLM calls | `NO_ACTION_RETURNED` 8 · truncated 8/2184 | **not** a contributor |
| red-light violations | 8.0 / episode; idle burst fired ≤5× in 3014 steps | secondary metric, detector effectively dead |

---

## 4. Binding constraint — classified

**Primary: HARNESS-BUG (viewpoint acquisition), with a VLM-CEILING component on the frozen
recognizer.** Not a threshold problem and not a prompt-wording problem:

* The frozen model is **already right**. On the frame where the robot stands 16 cm from the
  destination, the perceiver says *"very close… prominently visible on the left"* and the planner
  repeats it verbatim. The information needed to commit is in the harness's own text channels and
  the harness never reads it (`plan_out["summary"]`/`["reason"]` are logged and thrown away).
* The recognizer's 21 %/0 % recall is a **framing** failure, not a calibration failure: it is shown
  a forward frame and a reference photo taken facing 90° away. Its FPR is 0.4 %, so there is a large
  precision budget to spend — but spending it via wording has already been refuted five times in
  this chain, because the geometry, not the criterion, is wrong.
* The mechanism that fixes the geometry **already exists** (`request_side_scan` → in-place yaw →
  `scan_reach` → yaw back on a miss; in-place turns preserve position, so a position-graded leg is
  never harmed and a miss can never be a fatal commit). It is gated behind `_side_band_near`
  (`completion_checker.py:406`), a predicate over `obs['depth']` — a **per-frame min-max normalised
  JET rendering with no absolute scale**, whose own module comment records `near_frac == 0.000` on
  24/24 real frames. **2 firings in 50 episodes.** That is a wiring defect, recoverable in code.

**Secondary: TASK-GAP on TURNING.** The agent operates legally and rotates correctly but no
component's job is to put the body inside the junction box before committing (§2.3).

**Confirmed dead code in the anchor** (not the binding constraint, but it makes two safety rules
unreachable): `spatial_memory.py:66` hard-codes `coll = 0`, so `static_collision_delta()` is
identically `0`, so **both** `active_perception`'s wall-recovery rule (b′) (`:520-534`) and the
momentum collision veto (`:579-585`) can never fire. Evidence they were needed: 4 episodes died to
the env's 20-identical-position check and 3 more to `STUCK`. Note `info["agent_collision"]` is a
**forbidden** field, so the honest repair is to delete the accessor and key those rules on the
odometry delta alone. Also unused: `SpatialMemory.summary()` is never called from `harness.py`.

---

## 5. Concrete changes to try (the gradient — ranked by leverage)

### A — Commit from the *faced* pose: make the off-axis look unconditional, and make it the FINISH commit route. `modules/completion_checker.py:768-800` (`request_side_scan`) + `harness.py:455-478`

Replace the `_side_band_near` depth trigger with the **arrival-look cadence itself** (fire whenever
`proactive_commit_check` would have been eligible: `forward_streak ≥ 4`, spacing 4, cap ~4/leg), and
demote segmentation to a *veto only* (`named-side building fraction < 0.02` ⇒ nothing to look at,
don't spend the cap). Keep the existing miss handling (yaw back, walk on). On a **FINISH** leg the
commit must happen **from the faced pose** (that *is* the yaw criterion); on a **ROAD** leg restore
the travel heading **before** the `-1` so the next leg's turn burst rotates from the right heading.

* **Why it should work:** §2.1 — two episodes stood 16 cm and 27 cm from the destination with the
  perceiver naming the target on the correct side, and a single in-place quarter-turn would have
  scored FINISH (and SR). The predicate the recognizer is asked becomes answerable for the first
  time: the hint was captured from exactly this pose.
* **Predicted:** FINISH 0.000 → first non-zero in the lineage; ROAD +0.05–0.10 via the same
  mechanism; **SR > 0 becomes reachable** (it is currently unreachable in principle, because FINISH
  needs a yaw nothing ever sets). Cost: 2 env steps + 1 VLM call per look; the ROAD/FINISH leg budget
  is `int(L1/100)` = **5× the forwards actually needed**, so there is headroom.
* **Risk:** bounded by construction — a wrong look costs 2 net-zero steps, never a fatal commit.

### B — Stop `proactive_commit_check` from injecting `-1` on a FINISH leg. `modules/completion_checker.py:606-688` + the `_is_finish_instruction` predicate

* **Why:** measured precision on FINISH legs is **0/7** — every accept it has ever made on that leg
  is one of the 7 fatal commits (7 778–58 789 cm out, yaw err 90–270°). A call whose every accept is
  a false positive is not a detector, it is a cost. FINISH stays reachable via A and via the
  planner-`-1` + cross-check path.
* **Predicted:** removes 7/50 episode-ending commits outright; converts them into episodes that
  survive to try A. Pure variance reduction, no coverage loss.

### C — Walk into the junction before the TURNING commit. `modules/active_perception.py:420-440`

Intercept the `[rotate, -1]` burst; emit single `[0]`s along the **pre-turn** heading until the
lateral segmentation band on the turn side opens (facade gives way to the cross street), with a
**2-forward floor** and a **3-forward cap**, then release the held `[rotate, -1]`.

* **Why:** §2.3 — 7/7 failures are yaw-correct and 1–2 forward steps short.
* **Predicted, from the exact replay:** TURNING **5/12 → 11/12** of episodes that reach the leg
  (per-leg SR 0.111 → ~0.24), and TURNING is the gate to FINISH, so it multiplies A and B.
* **⚠ This lever is NOT refuted — it has never been tested cleanly.** `cand_r1_…_r001` shipped it
  bundled with three other changes including an arrival-prompt rewrite; ROAD collapsed 0.24 → **0.02**
  (1/50), so only **1 episode ever reached TURNING** and the turn change was unobserved.
  `cand_r6_…_r006` shipped it inside a larger rewrite and got TURNING **0.20 (9/45) with 9/10
  conversion from ROAD** — the mechanism works. **Ship C alone, leaving every ROAD commit path
  byte-identical to this anchor.** That is the highest-confidence, lowest-variance move available.
* Express the floor/cap as **action counts**, never cm (lint WARNs literals ≥ 2000).

### D — Read the text the harness already has: an arrival vote from the perceiver/planner prose. `harness.py:352-370` (`planner_response` is built and discarded) + a new predicate in `completion_checker.py`

The planner's own `Summary`/`reason` says *"very close to the … building … on the left"* at 4.5 % of
decisions and 78 % of decisions contain the *"not directly in front / continue forward"* clause.
Cheap, no extra VLM call: when the planner text asserts the named landmark is **close and on the
named side**, treat that as a **trigger** for A's look (not as a commit). Pair it with a
**turn-down / peak detector** over a *graded* 0–3 proximity score across consecutive looks — commit
when the score stops rising — instead of a threshold on a single frame.

* **Why:** the window is only ~4 forward steps deep, so "am I past the maximum?" is a far better
  shaped question than "am I there?", and it is robust to the model's absolute calibration.
* **Distinguish from exhausted work:** the chain has refuted *re-wording* the question and
  *thresholding* a graded score at one frame. A **monotone-change detector across frames** is a
  different estimator and has not been tried.
* Also: break the planner's frozen loop. The rolling `Summary` repeats **byte-identically for 6+
  consecutive decisions** (§2.1) because the harness feeds it back verbatim; splice in the unused
  `SpatialMemory.summary()` (`step / steps_in_leg / cum_path`) so the state channel carries
  information the model cannot copy.

### E — Reframe the evidence geometrically, at zero env-step cost. `modules/completion_checker.py:530-603`

Pass `_binary_reach_vlm_call` a **crop of the named-side quarter of the current RGB, upscaled**,
alongside the full frame, and ask about the crop. This puts the off-axis building near frame-centre
using pixels the harness already has — no turning, no position risk, available on **every** step
rather than on a cadence.

* **Why:** it attacks the same *framing* cause as A but with zero step cost, so it can run at full
  rate; the two compose (crop screens, look confirms).
* **Note the precision budget:** out-of-window FPR is **0.4 %** (1/263), so trading precision for
  recall is strongly favourable *on ROAD* (graded on position only). Do **not** loosen on FINISH
  without A's pose fix — there the same loosening produced all 7 fatal commits.
* Also delete the `"Be conservative — output 1 only when the match is unambiguous"` clause
  (`completion_checker.py:567`) **on ROAD only**; it is directly responsible for the 21 % recall and
  it contradicts the rest of the same prompt.

### F — Make `_named_side_has_structure` earn its place, or delete it. `modules/completion_checker.py:442-470`

Measured on **743 real ROAD frames** from this run (not the 32 frames it was fitted on):
it passes **96.2 %** of in-window frames and **93.5 %** of out-of-window frames — it discriminates
nothing. Worse, the in-window named-side building fraction is **lower** (mean 0.434, p50 0.412) than
out-of-window (mean 0.583, p50 0.664), because at an intersection the near block face ends. So any
*tightening* of this veto preferentially kills true arrivals.
**The inverted form is the real signal:** "the named-side building fraction has just fallen from its
running maximum" is an intersection cue (and is exactly what C needs for its release condition).
Either re-derive it that way or delete the veto — do not leave a fitted no-op in a commit path.

---

## 6. Exhausted — do NOT repeat

* **Do not touch the ORIENTATION path.** 50/50 here. The closed-form cardinal burst is the only
  thing in this harness that works.
* **Do not re-word or re-shape the arrival question, and do not threshold a graded confidence at a
  single frame.** Six reformulations were priced offline in the bootstrap gradient and lost;
  `cand_r002` (four channels into the recognizer) → 0.330, `cand_r001` (criterion rewritten in frame
  geometry) → **ROAD 0.02**. The cause is the viewpoint (§4), not the wording. An
  illumination-invariance clause is a known dead end.
* **Do not tune `K_REACH_FORWARD_STREAK` / `K_REACH_INTERVAL` / `MAX_REACH_FIRES_PER_SUBTASK`.**
  Run to death across iters 006–010; `cand_r007` replaced the quota with a refill bucket → 0.345
  (ROAD **0.14**); `cand_r005` attacked the decision-vs-step clock → 0.345 (ROAD 0.20). Coverage is
  a real multiplier but it is **not** the binding term while recall is 21 %.
* **Do not re-ship the `_named_side_has_structure` veto as-is or tighten it** — §F, measured no-op
  on 743 frames, and tightening is actively harmful.
* **Do not ship a bare depth-polarity flip, and do not build a new gate on `obs['depth']` magnitude.**
  Per-frame min-max JET has no absolute scale; `_strongly_far` is already disabled for this reason
  and `_side_band_near` is the same mistake still wired in.
* **Do not try to wire `static_collision_delta`.** `info["agent_collision"]` is a forbidden field;
  delete the accessor (`spatial_memory.py:66`) and its two unreachable consumers instead.
* **Do not read** `agent_location`, `target_location`, `destination`, `correct_robot_location`,
  `ideal_path`, `gt.json`, `current_instruction`, the analog `_agent_rotation` yaw, or
  `env.subtask_max_steps` (it is `int(L1_to_target/100)` — a disguised distance oracle).
* **Avoid numeric literals ≥ 2000** in candidate code; express distances as action counts
  (1 forward = 500 cm of odometry).
* **Do not bundle a TURNING change with a ROAD-commit change.** `cand_r001` did exactly that and the
  ROAD regression made the turn fix unobservable — one leg per candidate, or the evaluation is void.

---

## 7. Provenance

* Candidate: `$OUT_DIR` = `simworld_gym/simworld_config/harnesses/nav` (`harness.py` 613 lines +
  7 modules, 2705 lines total). Score: `arm_hard5010_adaptive_k4c1/anchor_nav_r000/score.json`.
* Traces: `arm_hard5010_adaptive_k4c1/anchor_nav/traces/` — 50 episodes, 3014 env steps,
  2184 VLM calls (918 perceive · 918 plan · 348 REACH-family), 8 truncated.
* Env semantics quoted from `SimWorldGym/simworld_gym/envs/traffic_world.py:302-386` (`_get_terminated`,
  `_get_evaluation_of_subtask`) and `:615-650` (`step(-1)`, `subtask_max_steps`).
* Replay validated: reconstructed `subtask_success` matches the evaluator's on **50/50** episodes.
  Out-of-bounds deaths verified against the real `build_road_polygon(progen_world.json)` union
  (outside on exactly the final step, 14/14).
* Pixel measurements: 743 ROAD-leg segmentation frames + their JET depth frames decoded from
  `traces/*/frames/` via a nearest-neighbour `cv2.COLORMAP_JET` inverse. Mask palette census
  (25 frames): yellow `(255,255,0)` ground/road 0.447 · purple `(128,0,128)` buildings 0.275 ·
  bright green `(0,255,0)` sky 0.160.
* `read_audit.json`: **0 forbidden, 0 escalated, 0 undeclared** over 21 458 reads on 15 paths.
