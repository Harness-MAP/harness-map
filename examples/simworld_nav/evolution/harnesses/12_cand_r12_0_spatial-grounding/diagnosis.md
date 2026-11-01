# Diagnosis — SimWorld-MMNav (`hard`, `search_hard_50`) / `anchor_nav` (the honest H_0 anchor)

**Headline:** SR **0.000** (0/50) · SubtaskSR **0.360** · per-leg pass
**[1.000, 0.240, 0.111, 0.000]** · 50/50 episodes complete · 80.7k tok/ep · 799 s/ep ·
distance_progress 0.145 · moving_distance 26.9k cm/ep · static_collision 1.78 · red_light 8.0.

**Binding constraint (one sentence):** the agent *physically stands inside* the active leg's
graded success window in **28/50 episodes and commits in 12**, and the last leg is
**structurally unwinnable** — a `FINISH` leg is graded on location **and** a 45°-tolerance
facing, and this harness's FINISH commit is a bare `[-1]` that never performs the turn, so
`per_position_SR[3]` is 0.000 by construction, not by bad luck.

Two of the 50 episodes stood **16 cm** and **27 cm** from their FINISH target and would have
been **full SR=1.0 successes** if the commit burst had contained one in-place rotation toward
the side the instruction names (§2.4). That is the largest single uncollected point in the run.

---

## 1. What this harness does (mechanism)

`orchestrator.py:159 run_episode` is one `while not terminated and len(actions_log) < max_steps`
loop wiring 7 modules in fixed order:

| order | call site | role |
|---|---|---|
| 1 | `orchestrator.py:280` stuck detector | `forward_count > 3 ∧ Δmoving_distance < 0.5 → break` |
| 2 | `orchestrator.py:301` `ActivePerception.should_emit` | deterministic burst that **short-circuits the whole VLM pipeline**: (a) cardinal burst `[rot…, -1]` on "face <cardinal>"; (a'') relative-turn burst `[rot, -1]` on "turn left/right" (`active_perception.py:432-440`, once-per-instruction latch); (a') red-signal idle `[2,1]`; (b) anti-spin; (d) forward momentum |
| 3 | `Perceiver.perceive` | 1 VLM call over 4 images (RGB, leg hint, `object_mask`, JET depth) → free-text `Description` |
| 4 | `Planner.plan` | 1 **text-only** VLM call over that description → `{Summary, Match, Actions}` |
| 5 | `AffordanceEstimator.estimate` | full-motion-set stub |
| 6 | `orchestrator.py:391` `CompletionChecker.verify` | honour planner `-1` iff `Match==1 ∧ forward_streak ≥ 1` |
| 7 | `orchestrator.py:413` `cross_check_planner_commit` → `:430` `proactive_commit_check` → `:459` `request_side_scan`/`scan_reach` | the four commit gates |

All four commit gates terminate in **one function**, `completion_checker.py:530
_binary_reach_vlm_call`, which shows the frozen VLM the current RGB + the leg's reference
photo and asks `{"Match": 0|1}`.

So the leg split is: **ORIENTATION and TURNING are decided deterministically from instruction
text with zero VLM calls; ROAD and FINISH are decided entirely by photo-to-frame matching.**
Everything in §2–§4 turns on that asymmetry.

### 1.1 Three env terminators the harness does not model
`SimWorldGym/simworld_gym/envs/simple_world.py:318-326,546-567` ends the **whole episode** on:
leaving the union-of-roads polygon; 20 consecutive identical positions; `steps_count ≥
max_steps (672)`; **or `substeps_count ≥ subtask_max_steps`** — a *per-leg* budget reset at
each successful `-1` to 10 (ORIENTATION) / 50 (TURNING) / `int(L1_to_next_waypoint / 100)`
(ROAD, FINISH). And `step(-1)` on a leg the env grades FAIL sets `terminated = True`
immediately — **a wrong commit is instantly fatal.** None of this reaches the harness as
anything but a bare flag, so 21/50 verdicts have empty `notes` and look like clean exits.

---

## 2. What actually happened (causal chain, with evidence)

I replayed `_get_evaluation_of_subtask` step-by-step over each episode's `env_step_log.jsonl`
against its `gt.json`. The replay reproduces all 50 verdicts **exactly** (agreement
`{(1,1):38, (2,2):7, (3,3):5}`), so the attribution below is mechanical, not inferred.

### 2.1 Episode end causes (n=50)
| n | cause |
|---|---|
| 20 | env-side termination after the harness kept walking (left the road polygon / position-repeat); `notes` empty |
| 13 | **fatal `-1`** on a leg the env graded FAIL — 7 TURNING, 6 ROAD *(plus 7 FINISH ones that cost nothing, see §2.4)* |
| 9 | per-leg `subtask_max_steps` exhausted on leg 1 |
| 5 | harness `STUCK` / `SIDE_SCAN_MISS` / `NO_ACTION_RETURNED` / `EMPTY_FALLBACK` exits |
| 3 | other |

Action mix across all 50 episodes: **2 812 Move_Forward (93 %)**, 114 turns (4 %), 1 strafe,
87 commits (3 %). The planner emits a bare `[0]` on **750 / 918** calls. This agent walks in a
straight line and almost never decides anything.

### 2.2 The dominant mode: reach the window, walk out the far side
For each episode I took the leg it died on and asked whether its trajectory ever entered that
leg's graded window:

> **28/50 episodes (56 %) stood inside the active leg's success window and never committed;
> 26 of them walked straight out the far side.**

Minimum L1 reached, then final L1 at death:
`ep_20_7_21` **9 cm → 23 509 cm`; `ep_20_19_21` 22 cm → 2 022; `ep_20_17_21` 42 cm → 43 542;
`ep_20_2_11` 50 cm → 47 950; `ep_20_16_21` 65 cm → 53 935; `ep_20_17_11` 74 cm → 3 074;
`ep_20_5_21` 80 cm → 5 286; `ep_20_18_21` 112 cm → 2 888; `ep_20_23_11` 122 cm → 3 122;
`ep_20_13_21` 134 cm → 21 134; `ep_20_12_11` 139 cm → 32 639; `ep_20_19_11` 212 cm → 65 291.

The window is ±2000 L1 and a forward step is 500 cm, so the agent is **inside the window for
exactly 8 consecutive steps** and spends all 8 of them walking.

### 2.3 Why it does not commit: the arrival recognizer, priced
348 two-image arrival calls were made (they log as `kind:"perceive"` because
`evaluator.py:462` keys `kind` on image-presence; identify them by the string
`"You compare two egocentric outdoor images"` in `prompt`). Bucketing each by the **true** L1
to that leg's waypoint at the moment of the call:

| leg type | TP | FN | FP | TN | recall | precision | FPR |
|---|---|---|---|---|---|---|---|
| **ROAD** | 9 | 33 | 1 | 262 | **0.21** | **0.90** | **0.004** |
| **FINISH** | 0 | 1 | 7 | 35 | **0.00** | **0.00** | **0.167** |

Two different pathologies in one function:

**ROAD — highly precise, badly under-triggered.** It refuses at point-blank range and tells
you why. `ep_20_17_11` step 11, L1 = **74 cm**, `frames/196c646aafb9.png` vs hint
`frames/87198badef01.png` → `{"Match": 0, "Reason": "Different lighting and building
details"}`. `ep_20_18_21` step 15, L1 = **112 cm** → `"Different building colors and lighting
conditions"`. `ep_20_13_21` step 17, L1 = 134 cm → same clause. The reference photos are
rendered under different lighting than the live frame and the `sys_prompt`
(`completion_checker.py:560-571`) ends with *"Be conservative — output 1 only when the match
is unambiguous."*

**FINISH — anti-correlated.** All **7** FINISH `Match=1` events in the run fired at
7 778 / 15 733 / 22 988 / 26 273 / 32 586 / 33 733 / 58 789 cm, and each one ended its
episode. The reasons give it away: `ep_20_5_11` step 10 (L1 = 32 586 cm,
`frames/820c0513e366.png`) → `"The reference building … is clearly visible in the current
view."`; `ep_20_21_11` step 20 (L1 = 22 988 cm) → `"… is visible in both views."`
**The prompt asks for visibility; the grader asks for arrival.** Its own wording licenses
this: *"the same building … is clearly visible and prominent in the current view"*. Down a
straight city street the destination facade is visible from 300 m.

**Firing more does not help — the answers are correlated within a leg.** Of 32 ROAD legs that
got ≥1 in-window call: legs with exactly 1 call said yes **9/26**; legs with **≥3** in-window
calls said yes **0/6** (`ep_20_17_11` fired 4 times inside the window and refused 4 times).
This independently reproduces the chain's "coverage is not the term" result — *I tested the
opposite hypothesis (fire every step) and the data refutes it.*

### 2.4 The last leg is structurally unwinnable — the biggest finding
`simple_world.py:384-397`: a `FINISH` leg passes only if `L1 < dist_threshold` **and**
`(target_yaw − agent_yaw) % 360 < 45`. The instruction always says
*"…on your left/right, **stop and face that building**."* The harness's FINISH commit is a
bare `[-1]` — there is no rotation anywhere in the commit path (`orchestrator.py:529-548`), and the
one mechanism that would turn (`request_side_scan`, `:459`) is gated on `_side_band_near`,
which is measured noise (§3).

Over the 10 episodes that entered a FINISH leg:

| | count |
|---|---|
| ever held the required facing anywhere on the leg | **1 / 10** |
| ever stood inside the FINISH location window | 2 / 10 |
| ever held the required facing **while** in the window | **0 / 10** |

And the two that stood in the window were one rotation away from a perfect episode:

| ep | n_legs | best L1 | heading there | required yaw | instruction side | `Rotate_<side>` gives |
|---|---|---|---|---|---|---|
| `ep_20_18_11` | 2 | **16 cm** | 90 (N) | 0 | "**on your left**" | 90 − 90 = **0 ✓** |
| `ep_20_4_21`  | 2 | **27 cm** | 90 (N) | 180 | "**on your right**" | 90 + 90 = **180 ✓** |

Both scored 0.5. Both would have scored **SR = 1.0**. The rule "rotate toward the side the
instruction names, then `-1`" is correct in **2/2** of the cases where it could be tested, and
it is derivable from `task_spec` text alone.

### 2.5 The TURNING leg loses by less than one forward step
`active_perception.py:432-440` fires `[rot, -1]` on the **first** tick of a TURNING leg — zero
translation. But the ROAD leg before it may commit anywhere within L1 < 2000 of its waypoint,
while the TURNING leg additionally requires the agent inside `correct_robot_position_area`.
12 episodes reached a TURNING leg; 7 failed and **every one of them failed by 78–980 cm**:

```
ep_20_2_21    78 cm short   ep_20_13_11  356 cm   ep_20_0_11   452 cm   ep_20_8_21   506 cm
ep_20_11_21  627 cm         ep_20_1_11   692 cm   ep_20_23_21  980 cm
```

Counterfactual, replayed along each episode's own measured heading (K extra 500-cm forwards
inserted before the rotate):

```
K=0 → 5/12 inside    K=1 → 8/12    K=2 → 11/12    K=3 → 11/12    K=5 → 10/12
```

### 2.6 `cross_check_planner_commit`'s cost cap inverts the gate
`completion_checker.py:748`: `if fires >= MAX_XCHECK_FIRES_PER_SUBTASK (2): return True`
— "I am out of budget" is treated as "approve". The planner proposes `-1` in **runs** (it
re-proposes every ~3 steps), so the cap blocks the first two proposals and **rubber-stamps the
third**, by which point the agent has walked out of the window:

| ep | proposal 1 | proposal 2 | proposal 3 (honoured) |
|---|---|---|---|
| `ep_20_17_11` | L1 **74 cm** → BLOCKED | 1 574 cm → BLOCKED | **3 074 cm → FATAL** |
| `ep_20_18_21` | L1 **112 cm** → BLOCKED | 1 388 cm → BLOCKED | **2 888 cm → FATAL** |
| `ep_20_19_21` | L1 **978 cm** → BLOCKED | 522 cm → BLOCKED | **2 022 cm → FATAL** (22 cm over threshold) |

Two of those first proposals were inside the window. Note also the blocked proposal is
`[-1]` *only*, so after `XCHECK_BLOCK` the affordance filter yields `[]` and
`orchestrator.py:519 EMPTY_FALLBACK` injects a forward — the veto literally pushes the agent
further past the waypoint. `EMPTY_FALLBACK` (21) and `XCHECK_BLOCK` (18) always co-occur.

### 2.7 The stuck detector is dead code on the dominant path
`orchestrator.py:280`: `forward_count = sum(1 for a in chosen_actions if a == 0)` counts over the
**current burst only**, and 750/918 bursts are `[0]`, so `forward_count > 3` is unreachable
except on the 75 `[0,0,0,0]` bursts. Measured consequence: 119/2 812 forwards (4.2 %) produced
< 50 cm of displacement, and three episodes were wedged against geometry for most of their
run with no `STUCK` note — `ep_20_20_21` 19/20 forwards blocked (95 %), `ep_20_22_21` 19/22
(86 %), `ep_20_8_11` 8/11 (73 %). All three scored 0.25.

---

## 3. Failure breakdown

| # | failure mode | rate | evidence |
|---|---|---|---|
| F1 | **Reach-and-walk-through** — stood in the graded window, never committed | **28/50 eps (56 %)**; 26 exited the far side | §2.2 |
| F2 | **FINISH facing never performed** — last leg unwinnable | **10/10 FINISH legs**; `per_position_SR[3]=0.000` | §2.4 |
| F3 | **Arrival recognizer answers "visible", not "arrived"** on FINISH | FPR 0.167, precision 0.00, 7/7 FINISH commits fatal | §2.3 |
| F4 | **TURNING committed with zero approach** | 7/12 TURNING legs, all by 78–980 cm | §2.5 |
| F5 | **xcheck cost cap flips veto→approve at the worst moment** | 3 fatal commits; 2 in-window claims destroyed | §2.6 |
| F6 | **Wall-wedge invisible to the stuck detector** | 3 eps wedged ≥73 %; 4.2 % of all forwards blocked | §2.7 |
| F7 | planner returns `[]` → `NO_ACTION_RETURNED` with no env step | 8 occurrences, 1 ep | verdicts |
| F8 | duplicate arrival call on the identical frame+step (xcheck and proactive paths ask the same model the same question) | 9 | trace scan |

Cross-check vs the numeric pre-pass: `per_position_SR` [1.00, 0.24, 0.111, 0.00] is consistent
— leg 0 is the zero-VLM cardinal burst (50/50), leg 1 is F1+F5, leg 2 is F4, leg 3 is F2+F3.

---

## 4. Binding constraint — classified

**Primarily HARNESS-BUG, with a hard VLM-CEILING underneath it.**

* **HARNESS-BUG (recoverable by code, ~all of the near-term headroom):** F2 (no facing action
  in the FINISH commit burst — this alone caps leg 3 at exactly 0.000 no matter how good
  perception gets), F4 (turn burst fires with zero approach), F5 (cap inverts the gate), F6
  (stuck detector scoped to the wrong list), F3's prompt asking the wrong predicate on FINISH.
* **VLM-CEILING (must be *scaffolded*, not re-prompted):** the ROAD arrival decision itself.
  The frozen model, shown a live frame and a differently-lit reference photo, answers *yes* on
  21 % of in-window frames, and the answers are **correlated within a leg** (0/6 legs with ≥3
  in-window calls ever said yes). More calls, different schedules and re-wordings are all
  closed by the chain (§6). The fix must **replace the channel that decides ROAD arrival** with
  a geometric read of `obs['object_mask']`, not improve the question.
* **Not TASK-GAP:** the agent does reach the waypoints (28/50) and has ample budget (0/50
  episodes hit `max_steps = 672`; the largest run was 180 steps). Stopping, not travelling, is
  the problem.

---

## 5. Concrete changes to try (the gradient — ranked by measured leverage)

### A. `modules/active_perception.py` + the FINISH commit path — **emit the facing rotation before `-1` on a FINISH leg** *(highest leverage; strictly necessary)*
Add a `_named_side(instruction)` parse (the same `on your left|right` regex already in
`completion_checker.py:250-260`) and make every FINISH commit a **two-action burst**
`[Rotate_Left if side=="left" else Rotate_Right, -1]` instead of `[-1]`. Apply it at all three
places that can set `planner_actions = [-1]` (`orchestrator.py:437`, `:472`, and the `commit_done`
trim at `:523`), gated on "the instruction contains *stop and face that building*".
*Why it works:* the rotation is **in place** — `simple_world.py` only updates
`_agent_location` on translation actions — so it cannot break the location half of the
grade; it can only add the orientation half that is currently always missing. Verified in
2/2 of the episodes that reached a FINISH window (§2.4).
*Predicted:* `per_position_SR[3]` 0.000 → non-zero for the first time in the chain; and
SR 0.00 → ≥0.04 (the 2 near-miss episodes are 2-leg episodes already holding leg 0).
*Risk:* one wasted substep per FINISH commit; a mis-parsed side yields a 180°-wrong heading,
which fails a commit that fails today anyway.

### B. `modules/active_perception.py:432-440` — **spend 2 forwards before the relative-turn burst**
Replace `return [code, _SUBTASK_DONE]` with a two-phase latch: on the first tick of a
turn-verb leg return `[0, 0]`, on the next return `[code, _SUBTASK_DONE]` (splitting it across
ticks keeps the red-signal and collision checks live between the forwards). Constant expressed
as a forward-action count, not a distance.
*Predicted:* `per_position_SR[2]` 0.111 → ≈0.244 (5/12 → 11/12 of the legs reached, §2.5);
SubtaskSR +0.03. Independently corroborated by the chain (r9: TURNING 5/45 → 9/45).
*Risk:* `ep_20_3_11` shows K=2 can overshoot a short block (1/12); K=2 is still the measured
optimum.

### C. `modules/completion_checker.py` — **give the intersection-stop leg a geometric decider instead of the photo match**
Add `_roadway_band_fraction(obs)`: the fraction of `obs['object_mask']` that is the **roadway
class** over the centre-forward band (rows 0.45–0.70, cols 0.30–0.70). The class is
`(0,127,255)` — match **both channel orders** `(0,127,255)` and `(255,127,0)`, it is not
palindromic. Return `None` (not `0.0`) on a missing/odd mask. Route legs whose instruction
contains *intersection* (43/54 of them) through a three-valued `road_surface_arrival`:
`None` = not my leg → fall through unchanged; `True` = commit; `False` = **not yet, and the
VLM path stays suppressed**. Latch one geometric claim per leg (a wrong `-1` is fatal) and arm
it only after the leg's first few forwards.
*Measured on this candidate's own 918 decoded mask frames:* P(band fraction > 0.05) = **0.59**
in-window vs **0.11** at 5–15 km; but **0.36** at >15 km, because episodes often spawn at or
near a junction — hence the "arm after N forwards" guard is load-bearing, not decoration.
This is also the chain's single largest proven move (`cand_r10`: leg 1 0.24 → 0.60,
SubtaskSR 0.360 → 0.535) — **re-probe the palette on your own traces before shipping.**
*Predicted:* `per_position_SR[1]` 0.24 → 0.45–0.60; SubtaskSR +0.06 … +0.10.

### D. `modules/prompt_assembler.py:77-78,132` — **fix the segmentation legend the harness tells the model**
It currently says *"purple = buildings, yellow = ground/road, green = SKY, blue = other. There
is no 'trees' and no 'driveway' class."* Probed on this run's own masks the palette is
`(255,255,0)` **sidewalk/crosswalk** (48.8 % of a typical frame), `(0,127,255)` **roadway**
(1.2 %), `(128,0,128)` buildings (35.2 %), `(0,255,0)` (14.8 %), `(0,128,0)` trees. The legend
collapses the two ground classes the task is built out of and throws the roadway into "other".
Add the actionable sentence: *yellow ahead turning into blue ahead means you are entering an
intersection.* **Do not touch the Match criterion wording** (§6).
*Predicted:* small on its own; it is the precondition for the perceiver ever describing the
junction correctly, and it costs nothing.

### E. `modules/completion_checker.py:748` — **stop letting the cost cap approve**
`if fires >= MAX_XCHECK_FIRES_PER_SUBTASK: return True` → return `False` (budget spent means
"no more claims this leg"), or better: allow exactly one refusal per leg and then stop
re-asking. Also fix the companion bug at `orchestrator.py:507-521`: a blocked `[-1]` falls through
to `EMPTY_FALLBACK` and injects a forward, pushing the agent *out* of the window — on a block,
hold position (emit nothing / a net-zero `[2,1]`) instead.
*Predicted:* removes 3 fatal commits (§2.6) and stops the veto from manufacturing overshoot.
*Note:* the chain found the cross-check itself is net-positive (20/22 vetoes were genuine
saves) — fix its arbitration, do not delete it.

### F. `orchestrator.py:280` — **scope the stuck detector to the leg, not the burst**
Count forwards from `SpatialMemory`'s per-leg history (or a `nonlocal` counter) instead of
`chosen_actions`, and test Δ`moving_distance` over the last N forwards.
*Predicted:* recovers the 3 fully-wedged episodes (§2.7) from a silent 95 %-blocked walk into
a recorded `STUCK`, freeing their remaining budget for a different tactic.

### G. `modules/completion_checker.py:560-571` — **FINISH-specific predicate** *(do this only as part of C/A, not as a re-wording)*
The FINISH arm asks about *visibility* and gets visibility (§2.3). Do **not** re-word the ROAD
criterion (closed, §6). For FINISH specifically the honest options left are: suppress the
forward photo-match commit entirely on FINISH legs and let only the geometric/approach path
commit, or ask a **separate second call** about framing ("does the reference facade fill the
frame edge-to-edge?") — the chain showed a *single* call asserting identity **and** position
scores 0/14 recall, so it must be two calls.
*Predicted:* removes 7 far-field fatal commits. Those 7 cost little score directly (FINISH is
the terminal leg) but they consume the fire budget and end episodes that still had leg credit
to earn.

---

## 6. Exhausted — do NOT repeat

**Refuted in *this* diagnosis (I tested it; the data says no):**
- **Firing the arrival check more often / every step.** Per-call in-window yes rate is 0.21 but
  the answers are **correlated within a leg**: 9/26 legs with exactly one in-window call said
  yes, **0/6** legs with ≥3 in-window calls ever did. Coverage is not the term.
- **Any `object_mask` *building*-class band statistic as an arrival cue.** Named-side building
  fraction: 0.366 in-window vs 0.343 at >15 km; lowest-building-row 0.504 vs 0.506 — flat
  across every distance band over 918 frames. `_named_side_has_structure`
  (`completion_checker.py:442`) vetoes 5.8 % of frames, uncorrelated with arrival — it is
  inert scaffolding, not a safeguard.
- **Any depth-based proximity gate.** With the JET inversion applied correctly, the forward
  cone's relative median is 0.80–0.88 in *every* distance bucket and
  P(side band reads NEAR) is 0.22 in-window vs 0.20 at >15 km. `_side_band_near`
  (`:406`) and `_looming_state` (`:322`) carry no distance information — `obs['depth']` is
  per-frame min–max normalised and has no absolute scale.

**Carried forward from the chain (r9/r10/r11), still true:**
- **Do not re-word, re-shape or "compromise" the arrival criterion** at
  `completion_checker.py:560`. Seven reformulations priced across the chain; flat ROC.
- **Do not add an illumination-invariance clause** to any prompt — the refusals just migrate.
- **Do not tune `K_REACH_FORWARD_STREAK` / `K_REACH_INTERVAL` / `MAX_REACH_FIRES_PER_SUBTASK`**
  (`:83,:96,:114`). Run to death in prior iters and re-confirmed above.
- **Do not propose "turn to face the building, *then ask*".** Faced-view in-window recall is
  **0/20**. (§5-A is different: it turns as part of the *commit burst*, for the **grader**, not
  to re-query the recognizer.)
- **Do not block a commit *path* without fixing the commit *criterion*** — a prior iter took
  forward FINISH commits 7 → 0 and `scan_reach` immediately produced 9 fatal ones.
- **Do not put identity and in-junction position in one VLM call** (0/14 recall). Two calls.
- **Do not touch the ORIENTATION cardinal burst** (`active_perception.py:177`): 50/50, 0 VLM
  calls, four candidates in a row.
- **Do not chase `max_steps`.** 0/50 episodes reached 672; the largest was 180. Every
  budget-shaped loss is the env's *per-leg* `subtask_max_steps`, i.e. a **stopping** problem.
- **Do not read** `agent_location`, `target_location`, `destination`, `correct_robot_location`,
  `correct_robot_position_area`, `ideal_path`, `gt.json`, `criteria.dist_threshold`,
  `env.subtask_max_steps` (it is `int(L1_to_waypoint / 100)` — a laundered `target_location`),
  `current_instruction`, or `env.unwrapped._agent_rotation`.
- **Avoid numeric literals ≥ 2000** in candidate code — express distances as forward-action
  counts or frame fractions.

**Anti-cheat note for the proposer:** every distance, window membership, per-leg attribution
and counterfactual in §2 is **my** oracle-side measurement, computed from `gt.json`,
`env_step_log.jsonl` and the env source *to price the levers only*. None of it may be
hard-coded. Every recommendation in §5 decides from `obs['rgb']`, `obs['object_mask']`,
`obs['depth']`, `info['moving_distance']`, `info['agent']['agent_rotation']`, the static
`task_spec['subtasks']` text, and the harness's own bookkeeping.

---

## 7. Provenance
- Candidate: `$OUT_DIR` = `simworld_gym/simworld_config/harnesses/nav` — `orchestrator.py` (613 L)
  + 7 modules (2 092 L), `AGENT.md`, `rules.md`, `NOTES.md`.
- Score: `…/arm_hard5010_adaptive_k4c1/anchor_nav_r000/score.json` — 50/50 episodes,
  `complete: true`, subset `search_hard_50`, model pool `[Qwen3-VL-30B-A3B-Instruct]`.
- Traces: `$TRACE_DIR` (50 episodes) — `verdict.json`, `gt.json`, `env_step_log.jsonl`,
  `llm_calls.jsonl`, `vlm_trajectory.jsonl`, `frames/*.png`.
- Measurements here: 3 014 env steps replayed against `_get_evaluation_of_subtask` (50/50
  verdicts reproduced exactly); 2 184 VLM calls (88.1k tok/ep); **348** two-image arrival calls
  bucketed by true L1; 918 planner calls; **918** frame quadruples decoded for the
  `object_mask` class statistics and the JET-inverted `obs['depth']` statistics of §2.3/§6;
  12 TURNING legs replayed under K ∈ [0,5] extra approach forwards; 10 FINISH legs replayed
  against the facing requirement.
- Env semantics quoted from
  `SimWorldGym/simworld_gym/envs/simple_world.py:318-397, 532-567`.
