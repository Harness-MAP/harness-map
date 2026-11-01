# Diagnosis — SimWorld-MMNav (`hard`, search_hard_50) / `cand_r002_0_spatial-grounding` ("the arrival channel"), iter 2, parent `cand_r1_0_spatial-grounding`

**Headline:** SR **0/50** · SubtaskSR **0.33** (parent 0.285, anchor 0.36) · per-leg pass
**[1.00, 0.12, 0.1111, 0.00]** (parent [1.00, 0.02, 0.022, 0.00]; anchor [1.00, 0.24, 0.1111, 0.00])
· static_collision 5.12 → **2.86** · cost 97.2k → **90.1k** tok/ep (anchor 80.7k).
The reverts worked — leg 1 recovered 1/50 → **6/50** — and the harness is no longer self-destructing
(`XCHECK_BLOCK` 117 → **6**, `EMPTY_FALLBACK` 118 → **2**). It is still **below the anchor**, and the
one genuinely new lever this iter shipped bought nothing.

**Binding constraint (one sentence):** the arrival recogniser now works again on ROAD legs
(in-window accept 0.143, far-field 0.010, LR ≈ 14) but it is **asked exactly once per pass through
the 8-step-wide window** — 35 firings across 266 in-window steps in 36 entering episodes — so leg 1
converts at recall-per-question, not recall-per-window; on FINISH legs the same call is
**0-for-15 inside 5 000 cm and 5-for-73 beyond 15 000 cm**, i.e. it is a pure false-positive
generator there because the FINISH reference view is 90° off the travel axis.

> **Confidence / method.** Every number is a per-call or per-step measurement over all 50 episodes,
> A/B'd against the anchor's own traces (`anchor_nav/traces/`, 348 REACH calls) with the *same*
> script. Candidate: 388 REACH calls / 1 042 perceive / 1 042 plan / 3 123 forwards. `gt.json` was
> read **for this analysis only**; no recommendation routes privileged state into a policy path.
> Evaluator read-audit: **forbidden = 0, undeclared = 0 on 50/50 episodes.**

---

## 0. Verdict on the five changes this iter shipped

| # | change | file | verdict |
|---|---|---|---|
| 1 | arrival call sees mask + JET depth + perceiver description (2 → 4 images) | `completion_checker.py:750-806` | **NO EFFECT, +11.6 % cost — close this axis** (§2.2) |
| 2 | criterion reverted to the anchor's wording byte-for-byte | `completion_checker.py:718-735` | **correct, and it is what recovered leg 1** (§2.1) |
| 3 | arbitration: cap decides nothing, veto bounded by refusals, falls through to motion floor | `completion_checker.py:1015-1017, 1034-1039` | **half right** — killed the veto loop, but the fallback is a *delayed rubber-stamp*: 5 fires → 1 pass, **4 fatal commits** (§2.4) |
| 4 | vetoed `-1` answered with `HOLD_PROBE=(Rotate_L, Rotate_R)` | `orchestrator.py:109, 598-604` | **does what it says** (0 cm displacement vs 1 500 cm) but **cannot refresh the evidence** (§2.4) |
| 5 | REACH path raises a bare `[-1]`; single pose composer | `orchestrator.py:486` | **correct, and completely unexercised** — all 6 FINISH commits were 15–45 km false positives (§2.5) |
| — | turn-approach latch (kept from parent, untouched) | `active_perception.py:504-532` | **confirmed at n=6 — the best thing in this lineage** (§2.3) |

---

## 1. What this harness does (mechanism)

`orchestrator.py:run_episode`, one iteration per *decision* (not per env step). Seven modules, unchanged graph:

| # | call site | authority over `-1` |
|---|---|---|
| 1 | `active.should_emit` (`orchestrator.py:~310`) | short-circuits perceive+plan: ORIENTATION cardinal burst, the **turn-approach latch**, red idle, anti-spin, cap-2 momentum cadence (≈3 forwards per VLM pair) |
| 2 | `perceiver.perceive` | 1 call, 4 images (RGB, hint, mask, JET depth), temp 0.2 |
| 3 | `planner.plan` | 1 text-only call → `{Summary, Match, Actions}`, temp 0.2 |
| 4 | `completion.verify` (`orchestrator.py:420`) | honours a planner `-1` iff `planner_match == 1 ∧ forward_streak` floor |
| 5 | `cross_check_planner_commit` (`:444`) | REACH call on the planner's claim; 1 refusal/leg, cap 2, then **motion floor decides** |
| 6 | `proactive_commit_check` (`:472`) | REACH call on a schedule: `forward_streak ≥ 4`, then every `K_REACH_INTERVAL = 5` forwards, cap `MAX_REACH_FIRES_PER_SUBTASK = 12` |
| 7 | `request_side_scan` / `scan_reach` (`:508`) | depth-gated yaw-to-named-side + re-ask |

Stages 5, 6 and 7 are all the **same function**, `_binary_reach_vlm_call` (`completion_checker.py:685`).
It owns every commit in the harness. This iter changed what it *sees* (change 1) and what happens
when it says no (changes 3/4).

### 1b. Four env facts that govern this task and are in no prose file the agent reads

Established by reading `SimWorldGym/simworld_gym/envs/traffic_world.py` (analysis only — the harness
must **not** read these fields; everything recommended in §5 is derived from the harness's own
action history and odometry):

1. **A failed `-1` ends the episode immediately** (`:646-654`: `if evaluation: current_subtask += 1 … else: terminated = True`). A false commit is not a lost leg, it is a lost episode. **16/50 episodes died this way.**
2. **`subtask_max_steps = int(L1(position_at_leg_start, target) / 100)`** for ROAD/FINISH legs (`:648`); 10 for ORIENTATION, 50 for TURNING. A 500 cm forward covers 5 budget units ⇒ **the agent gets ≈5× the minimum number of forwards, and every non-translating action spends one of them for zero progress.** 10/50 episodes ran the leg-1 budget to exactly 0.
3. **Stepping outside the road polygon terminates the episode** (`:367`).
4. **20 consecutive identical positions terminates the episode** (`:370-382`). **5/50 episodes died here** — wedged against geometry while emitting forwards.

---

## 2. What actually happened (causal chain, with evidence)

### 2.1 The revert worked. Three-way A/B of the same call on the same 50 tasks

`_binary_reach_vlm_call` firings bucketed by the true planar gap to the then-current leg's criterion:

| run (ROAD legs only) | < 2 000 (would PASS) | 2 000–5 000 | 5 000–15 000 | > 15 000 | LR |
|---|---|---|---|---|---|
| **ANCHOR** — anchor wording, 2 images | **9/43 = 0.209** | 0/52 = 0.000 | 1/105 = 0.010 | 0/105 = 0.000 | ∞ |
| **ITER 1** — frame-geometry wording, 2 images | **1/58 = 0.017** | — | — | 6/139 = 0.043 | 0.40 |
| **ITER 2 (this)** — anchor wording, 4 images + description | **5/35 = 0.143** | 4/61 = 0.066 | 1/105 = 0.010 | 1/96 = 0.010 | 14 |

In-window recall 0.017 → **0.143** is the whole of the leg-1 recovery (1/50 → 6/50). §5-A of the
prior gradient was right and is now closed.

### 2.2 Change 1 (the evidence upgrade) is the one new lever, and it did nothing

Same criterion string, same model, same 50 tasks, same schedule — the *only* difference vs the
anchor is mask + JET depth + description attached to the current view (`completion_checker.py:750-806`).

| ROAD legs | anchor (2 img) | this iter (4 img + desc) | |
|---|---|---|---|
| in-window recall | 9/43 = **0.209** | 5/35 = **0.143** | Fisher p = 0.56 — **no gain, if anything worse** |
| 2 000–5 000 cm accepts | 0/52 = **0.000** | 4/61 = **0.066** | p = 0.12 — a hint of *added* near-miss FPs |
| > 15 000 cm accepts | 0/105 | 1/96 | unchanged |
| tokens / call | ≈1 050 | **2 371** (2 330 in + 41 out) | **+126 %** |
| share of the 50-ep budget | ≈8.9 % | **20.4 %** (920 220 tok) | +11.6 pts of episode cost |

The extra channels are not being used: the `Reason` strings are pure RGB identity statements
("*The current view shows a building with a yellowish lower facade … while the reference shows a
different building with green-tinted glass*", ep_20_11_21 s7) and **not one** of the 388 replies
cites depth hue or the segmentation layout. The prior gradient's §5-E — the last "unshipped,
well-motivated" lever on this call — is now **measured and negative**. Both axes on this call
(wording, evidence set) are now exhausted; what remains is *when* it is asked and *from what pose*.

### 2.3 The turn-approach latch is confirmed — and it is the best result in this lineage

TURNING commits, by distance outside the box, anchor vs this iter (the latch spends 2 forwards
closing on the box before rotating; `active_perception.py:504-532, _TURN_APPROACH_FORWARDS = 2`):

```
 ANCHOR  12 commits:  0 0 0 0 0 | 78  356  452  506  627  692  980      -> 5/12 pass (0.42)
 ITER 2   6 commits:  0 0 0 0 0 | 22                                    -> 5/6  pass (0.83)
```

The anchor's 7 failures are all *outward misses of 78–980 cm* — exactly the mode the latch targets.
It is now validated at n=6 instead of n=1. **This is why leg 2 held at 0.1111 on half the leg-1
passes** (anchor needed 12 ROAD passes for 5 TURNING passes; this iter needed 6). Keep it verbatim.
The single miss is **22 cm** outside the box — `_TURN_APPROACH_FORWARDS = 3` is worth one experiment,
but n=1.

### 2.4 Change 3+4: the veto loop is gone, and was replaced by a *delayed* rubber-stamp

The pathology the parent shipped is genuinely fixed: `XCHECK_BLOCK` 117 → **6**, `EMPTY_FALLBACK`
118 → **2**, stalled-forward rate 9.0 % → **5.9 %**, `static_collision_mean` 5.12 → **2.86**,
cost −7.1k/ep. No episode walks itself out of its own window any more.

But `completion_checker.py:1015-1017` (`if refusals >= MAX_XCHECK_REFUSALS_PER_SUBTASK: return walked`)
honours the *next* claim on the motion floor `forward_streak >= K_REACH_FORWARD_STREAK` — a floor that
was **already satisfied at the moment of the refusal in all 5 instances**. Between the refusal and the
re-claim sits `HOLD_PROBE = (Rotate_Left, Rotate_Right)` — net-zero yaw, zero translation — so the next
decision perceives the **identical frame**, and both perceive and plan run at **temperature 0.2**.
The second claim is the same claim, and it is honoured for free:

```
 ep_20_14_21  s9  gap  6 613  planner -1 -> REACH Match=0 -> XCHECK_BLOCK -> HOLD
              s11 gap  6 613  planner -1 -> allowance spent -> walked=True -> COMMIT   FATAL
 ep_20_16_21  s26 gap  2 065  planner -1 -> REACH Match=0 -> XCHECK_BLOCK -> HOLD
              s28 gap  2 065  -> COMMIT   FATAL (65 cm outside the 2 000 threshold)
 ep_20_24_11  s27 gap  4 463  ... same shape ...                          FATAL
 ep_20_4_21   s86 gap 19 527  ... same shape ... (FINISH leg)             FATAL
 ep_20_19_21  s5  gap 11 478  REACH Match=0 (refusal spent)
              s31 gap    522  planner -1 -> free commit                   PASS
```

**5 commits through this path → 1 pass, 4 fatal.** Precision 0.20, and all four killed episodes that
still had leg budget left. The hold preserves the *position* (its stated goal) but that is precisely
why it cannot change the *answer*.

Two related observations on the same function:
- `_named_side_has_structure` (`:997`) still sits **ahead** of the refusal counter and returns `False`
  as a free, uncounted veto. It vetoed **0 of the 16** accepts this run — currently inert, but it is
  the same unbounded-free-veto shape the parent was condemned for.
- The prior gradient's §2.6/§5-F — "the first planner `-1` per leg is a 0.58-precision arrival
  detector" — **does not replicate.** Here: 14 first-claims per leg, gaps
  `351, 1480, 2065, 3183, 4463, 5006, 5715, 6613, 11478, 19527, 21016, 21712, 25965, 44711` →
  **2/14 = 0.14 in-window**, against a 0.073 base rate over all 1 042 plan decisions. The planner's
  `Match` field is likewise near-useless (in-window rate 0.103 for `Match=1` vs 0.073 base) and is
  dropped entirely in **164/1 042 = 15.7 %** of replies. **Do not build a commit path on this channel.**

### 2.5 Change 5 is correct and irrelevant: leg 3 is pinned by *distance*, not by facing

All **6 FINISH commits** happened at gaps `15 396 / 19 527 / 21 016 / 25 965 / 29 586 / 44 711 cm`.
The terminal pose was never the binding term. The reason is structural: the FINISH instruction is
*"Proceed forward until reaching the high beige office building with three large arched glass windows
on the lower level **on your left**, stop and face that building"* — the reference photo was captured
**facing the building**, i.e. 90° off the travel axis. The forward camera at the destination does not
contain the reference view, and the model says so correctly:

```
 ep_20_24_21 — FINISH leg, agent walks THROUGH the window (min gap 719 cm)
  s121 gap  4 292  REACH Match=0 "different building with vertical columns and rectangular windows,
                                  not the beige office building with three large arched glass windows"
  s127 gap  1 412  REACH Match=0  <-- INSIDE the pass window
  s133 gap  1 888  REACH Match=0  <-- INSIDE the pass window; this is fire #12, the cap
  s139 gap  4 810  no allowance left; walks out to 31 638 cm and the leg budget ends the episode
  (perceiver, s124/127/130/133/136: byte-identical summaries, "still approaching the target …
   the current view shows a tall gray commercial building … which does not match")
```

Both channels are *correct* — the target genuinely is not in the forward frame. Measured over all
91 FINISH-leg firings: **0 accepts in 15 fires below 5 000 cm; 5 accepts in 73 fires above 15 000 cm.**
It has never once been right on a FINISH leg, and its 5 accepts are 5 of the 6 fatal commits.

The mechanism that exists for this (`request_side_scan` → yaw → `scan_reach`) fired **once in 50
episodes** (`SIDE_SCAN_MISS 1`, `SIDE_SCAN_COMMIT 0`). Its trigger `_side_band_near` is a percentile
band over the **per-frame-normalised** JET depth — the same absolute-scale problem that forced
`_strongly_far` to be permanently disabled (`completion_checker.py:489-501`). The mechanism is right;
the trigger is unusable.

The cap is the other half of the FINISH story. FINISH legs start **30 000–51 000 cm** out;
`MAX_REACH_FIRES_PER_SUBTASK = 12` at `K_REACH_INTERVAL = 5` forwards covers ≈36 000 cm of uniform
spacing, so the allowance is spent before the window in 4 of the 5 episodes that reached a FINISH leg
after passing legs 0–2:

```
 ep_20_15_21  12 fires from 48 536 down to 15 598 cm, cap spent; then reaches min gap 1 646 cm
              with ZERO arrival checks available, and walks back out to 30 909.
 ep_20_11_21  12 fires; #12 accepts at 15 396 cm -> fatal.
 ep_20_21_21  12 fires while the gap MONOTONICALLY GROWS 3 745 -> 34 295 (the agent turned onto the
              FINISH leg and walked away from the destination for 86 steps).
```

**3 of the 10 episodes that entered a FINISH leg physically reached within 2 000 cm of the
destination (211 / 719 / 1 646 cm) and none committed.** Leg 3 is not out of reach; it has no usable
detector.

### 2.6 The coverage arithmetic — the actual binding term on leg 1

| | anchor | this iter |
|---|---|---|
| episodes that physically entered the leg-1 pass window | 40/50 | **36/50** |
| …and passed leg 1 | 12 (0.30) | **6 (0.167)** |
| in-window REACH firings, total | 44 | **35** (over **266** in-window env-steps) |
| in-window firings per entering episode | `{0:6, 1:26, 2:6, 3:2}` | `{0:6, 1:25, 2:5}` |
| per-fire in-window accept rate | 0.209 | 0.143 |

The window is **≈8 env-steps wide** (4 000 cm at 500 cm/forward). The schedule
(`forward_streak ≥ 4`, then every 5 forwards) asks the question **once** in 25 of the 36 entering
episodes, twice in 5, and **never** in 6. Expected passes = 36 × (1 − 0.857^0.97) ≈ **5.0**; observed
5 in-window REACH accepts (+1 planner-path) = 6. **The model is reproducing the score exactly**, which
means the lever is the exponent, not the base: *fires inside the window*, not another wording.

Meanwhile **88 % of the 297 ROAD-leg firings happen outside the window**, where they can only
generate false positives.

---

## 3. Failure breakdown (50 episodes, by what actually ended the episode)

| category | eps | evidence |
|---|---|---|
| **Walked through the leg-1 window, never committed** | **30** | §2.6; median overshoot past closest approach **15 049 cm** (range 0–52 184) |
| — of which killed by env off-road / leg budget | 22 | `traffic_world.py:367` / `:648`; 10 legs ran the budget to exactly 0 |
| — of which **wedged** (stall, no recovery) | 12 total | 7 on the harness's own `STUCK` break, **5 on the env's 20-identical-positions rule** (ep_20_13_11, ep_20_1_11, ep_20_21_11, …); stalled-forward rate 5.9 % (183/3 123) |
| **Fatal false commit** | **16** | 9 ROAD (2 065–19 801 cm), 1 TURNING (22 cm), 6 FINISH (15 396–44 711 cm) |
| — of which via the new refusal-exhausted fallback | 4 | §2.4 |
| — of which via the FINISH-leg forward REACH | 5 | §2.5 |
| Never approached the leg-1 window | 14 | 10 committed elsewhere, 4 never got close |
| **Reached the FINISH window and failed to commit** | **3** | §2.5 — min gaps 211 / 719 / 1 646 cm |
| Leg 1 → leg 2 conversion | **5/6 = 0.83** | §2.3 — the latch, the one thing working better than the anchor |

Funnel: **50 → 6 (ROAD) → 5 (TURNING) → 0 (FINISH).**

---

## 4. Binding constraint — classified

**Primary: HARNESS-BUG — a firing *schedule* mismatched to the geometry it is sampling.**
The recogniser is a usable detector on ROAD legs (LR ≈ 14 between in-window and >15 km) and it is
asked ~once per 8-step window while 88 % of its budget is spent far from any waypoint; on FINISH
legs a uniform 12-fire allowance is exhausted 15–35 km before the destination. Both are pure
scheduling logic in `proactive_commit_check` (`completion_checker.py:865-880`) and cost nothing but
tokens to change — and the tokens already exist, wasted on change 1 (§2.2).

**Secondary: VLM-CEILING, now sharply localised to the FINISH viewpoint.** Three independent axes
on `_binary_reach_vlm_call` have now been measured on the same 50 tasks: two criterion wordings
(0.209 vs 0.017 in-window at an *invariant* far-field FP rate) and two evidence sets
(2-image 0.209 vs 4-image+description 0.143). Nothing about the *question* moves it. On FINISH legs
the failure is not the model at all — the target is not in the frame (§2.5, ep_20_24_21 s121-139,
where the perceiver and the recogniser independently and *correctly* report a non-match while
standing 719 cm from the destination). The fix must change the **pose from which the question is
asked**, not the question.

**Tertiary: TASK-GAP, unchanged and now priced.** Still no "I have passed it" state: 30 episodes
walked a median 15 049 cm beyond the window, and under env fact 1b-3 that overshoot is what kills
them. And 12/50 episodes end wedged with no recovery maneuver in a harness that has **never once
emitted a strafe** — env-level action histogram for the whole run:
`{Move_Forward: 3 123, Rotate_Right: 72, Rotate_Left: 31, commit: 77, strafe: 0}`.

---

## 5. Concrete changes to try (the gradient — ranked by leverage)

### A — **Re-spend the evidence budget on coverage. Cost-neutral.** `modules/completion_checker.py:750-806` + `:96`
Two coupled edits:
1. Delete the `sys_prompt +=` evidence block and the mask/depth/description attachments in
   `_binary_reach_vlm_call` (`:750-806`); return to the anchor's 2-image call. Keep the criterion string.
2. `K_REACH_INTERVAL = 5` → **2** (non-ORIENTATION legs).
- **Why:** §2.2 prices the evidence at **2 371 tok/call vs ≈1 050** for zero measured gain — 20.4 % of
  the whole episode budget. §2.6 shows the window is 8 forwards wide and gets **0.97 questions**.
  At 1 050 tok/call you can afford ≈875 firings for the same money as today's 388.
- **Predicted:** in-window firings/entering-episode 0.97 → ≈2.5; per-window recall
  1 − 0.857^2.5 = **0.32** vs 0.167 observed → leg-1 passes 6 → **≈11-12** (anchor level, reached by
  coverage rather than by the anchor's higher per-fire rate). With §2.3's 0.83 ROAD→TURNING
  conversion, leg 2 ≈ **9-10/45 = 0.21** (anchor 0.111). **SubtaskSR ≈ 0.33 → 0.39-0.41.**
  (Sanity: the arithmetic above reproduces this run's 0.33 and the anchor's 0.36 to 2 decimals.)
- **Risk:** far-field ROAD firings also ×2.5 at an accept rate of 0.010 → ≈2 extra fatal commits.
  Net strongly positive. Ship A **alone** so the coverage/evidence attribution is clean.
- **Re-opening a closed axis, deliberately.** The prior gradient closed
  `K_REACH_INTERVAL`/`MAX_REACH_FIRES` with "58 in-window firings produced 1 yes." That was measured at
  per-fire recall **0.017**; recall is now **0.143** and passes = coverage × recall. The closure was
  correct for its regime and is void in this one.

### B — **On FINISH legs, stop asking forward and ask sideways.** `modules/completion_checker.py:830-913` + `orchestrator.py:472-535`
1. In `proactive_commit_check`, **suppress the forward REACH when `_is_finish_instruction(instruction)`**
   (the predicate is already there and is 199/199 correct).
2. Replace `request_side_scan`'s trigger: drop `_side_band_near` (1 fire / 50 eps — a per-frame-normalised
   depth band cannot carry "a building is beside me", same defect that disabled `_strongly_far`) and fire
   on the **same forward-count schedule** as the forward REACH. On a FINISH leg whose instruction names a
   side: emit `[Rotate_toward_side]`, re-perceive, run `scan_reach` on the now-frontal frame, commit on
   `Match == 1`, else `[Rotate_back]`.
- **Why:** §2.5. The forward call on FINISH legs is **0/15 below 5 000 cm and 5/73 above 15 000 cm** —
  strictly harmful. The look-turn costs **2 steps of a ≈5× leg budget** (env fact 1b-2) per probe, and
  it is the only way to reproduce the pose the reference photo was taken from.
- **Predicted:** −5 fatal commits immediately (no recall cost — the suppressed call has never been right).
  Opens the 3 episodes that stood 211 / 719 / 1 646 cm from the destination with nothing to ask.
  Leg 3 **0.00 → 0.02-0.07**, and it is the only realistic path to **SR > 0** in this lineage.
- Also raise/remove `MAX_REACH_FIRES_PER_SUBTASK = 12` on FINISH legs, or make the spacing widen-then-tighten
  with `forward_streak` — §2.5 shows 4/5 FINISH legs spent the allowance 15–35 km short. Derived from
  forward counts only.
- Bug to fix while in there: `orchestrator.py:518` calls `scan_reach` **without** `description=`, unlike the
  other two call sites (moot if the 2-image revert in A lands first).

### C — **Delete the delayed rubber-stamp; re-arm on displacement, not on a hold.** `modules/completion_checker.py:1015-1017`
```python
-   if refusals >= MAX_XCHECK_REFUSALS_PER_SUBTASK:
-       return walked
+   # A refusal is only spent once the agent has left the frame it was refused on.
+   if refusals >= MAX_XCHECK_REFUSALS_PER_SUBTASK:
+       if forward_streak - self._streak_at_last_refusal.get(cur_idx, 0) < K_REACH_FORWARD_STREAK:
+           return False            # same place, same frame -> same answer
+       self._xcheck_refusals[cur_idx] = 0   # new evidence available: re-arm the call
+       # fall through and ASK again
```
- **Why:** §2.4 — 5 commits, 1 pass, **4 fatal**, all four at the exact position the recogniser had just
  refused, one decision earlier, with `HOLD_PROBE` guaranteeing an identical frame and temp 0.2
  guaranteeing an identical answer. `forward_streak >= K_REACH_FORWARD_STREAK` is not a gate here: it was
  already true at the refusal in 5/5 cases.
- **Predicted:** −4 fatal commits, −1 leg-1 pass → **+3 episodes that survive with budget left**, which
  A then converts. Low risk, deterministic, 5 traced instances.
- Also move `_named_side_has_structure` (`:997`) **below** the refusal counter or delete it: it is a free,
  uncounted veto that vetoed 0/16 accepts this run — inert today, the parent's failure shape tomorrow.

### D — **Un-wedge instead of dying.** `orchestrator.py:292-304` + `modules/affordance_estimator.py:38`
- **Why:** 12/50 episodes (24 %) end wedged — 7 on the harness's `break`, **5 on the env's
  20-identical-positions rule** (env fact 1b-4), stalled-forward rate 5.9 %. `forward_count =
  sum(1 for a in chosen_actions if a == 0)` counts only the **current burst** (≤3), so the detector
  barely fires, and when it does it ends the episode.
- Track stalls over a rolling odometry history (`info['moving_distance']`, sanctioned). On 2 consecutive
  stalled forwards emit a recovery maneuver instead of `break`: **`[3]`/`[4]` (strafe) were emitted
  0 times in 3 323 motion steps** — a fully legal, completely unused lever — or `[Rotate, Forward, Rotate_back]`.
- **Predicted:** recovers part of the 24 % that currently terminate with legs unattempted. Cheap, no VLM calls.

### E — **Overshoot detection (open axis, lower confidence).** `modules/spatial_memory.py` + `completion_checker.py`
30 episodes, median 15 049 cm past the closest approach. There is a free signal in the traces: the
perceiver emits **byte-identical summaries** for 5+ consecutive decisions when nothing is changing
(ep_20_24_21 s124/127/130/133/136). "Description unchanged for N decisions ∧ still walking" is a
self-derived not-making-progress state; what to *do* with it (re-orient, widen the arrival schedule)
is the open question. Do not ship this together with A — it would confound the coverage measurement.

### F — Free hygiene, **not** a lever
`modules/planner.py:120-133` — the shared `reasoning_template` Output Format omits `Match`, so
**164/1 042 (15.7 %)** of plan replies drop the field. Worth fixing, but §2.4 measures `Match=1` at
in-window rate 0.103 against a 0.073 base: this channel carries ≈no arrival information. Also still
dead: `spatial_memory.py:66` (`coll = 0` hard-coded).

---

## 6. Exhausted — do NOT repeat

**New, established by this run:**
- **Do not enrich the arrival call's evidence set with `object_mask` / JET depth / the perceiver
  description.** Measured against the anchor at the identical criterion on the same 50 tasks: ROAD
  in-window **5/35 = 0.143 vs 9/43 = 0.209**, near-miss (2–5 k) FPs **4/61 vs 0/52**, cost
  **2 371 vs ≈1 050 tok/call (+11.6 pts of the episode budget)**. Not one of 388 `Reason` strings cites
  depth or segmentation. Together with the two wording experiments, **every axis on the *question* is
  now closed** — move the *schedule* (§5-A) or the *pose* (§5-B).
- **The planner channel is not a better arrival detector.** First planner `-1` per leg: **2/14 = 0.14**
  in-window against a 0.073 base rate; `Match=1` at 0.103. This **refutes the prior gradient's §2.6/§5-F
  (7/12 = 0.58)** on fresh data — do not promote planner claims to a commit path.
- **A net-zero hold cannot refresh evidence.** `HOLD_PROBE = (Rotate_Left, Rotate_Right)` returns the
  identical pose; perceive and plan both run at temperature 0.2. Honouring the re-raised claim is
  `return True` with two wasted steps: 5 fires → 1 pass, 4 fatal (§2.4).
- **`_side_band_near` is not a usable trigger** — 1 fire in 50 episodes. Per-frame-normalised depth
  cannot answer "is a building beside me" (the same defect that permanently disabled `_strongly_far`).
  Keep the look-turn *mechanism*; schedule it on forward counts.
- **A uniform fire schedule cannot cover a variable-length leg.** `MAX_REACH_FIRES = 12` ×
  `K_REACH_INTERVAL = 5` ≈ 36 000 cm of coverage; FINISH legs start at 30 000–51 000 cm and 4/5 spent
  the allowance before the window (§2.5).
- **`K_REACH_INTERVAL` / `MAX_REACH_FIRES_PER_SUBTASK` are NO LONGER exhausted.** The prior closure was
  measured at per-fire in-window recall 0.017; it is now 0.143 and coverage is the binding term (§2.6).
  This is the one entry on the old exhausted list that is deliberately re-opened.

**Carried forward, re-confirmed here:**
- **Do not touch the ORIENTATION cardinal burst.** 50/50 again, 0 VLM calls.
- **Keep the turn-approach latch verbatim** (`active_perception.py:504-532`) — now 5/6 in-box vs the
  anchor's 5/12, outward misses 7 → 1 (§2.3). The only candidate tweak is
  `_TURN_APPROACH_FORWARDS 2 → 3` (the single miss was 22 cm), at n=1.
- **Do not re-word the REACH criterion in either direction** (two operating points, flat ROC).
- **Do not use "how much of the frame the facade fills" as an arrival feature** (flat at ≈0.28 city-wide).
- **Do not re-run `N_REACH_CONFIRMATIONS_REQUIRED = 2`.**
- **`return False` on cross-check cap exhaustion is refuted** (parent: 117 free blocks); `return True`
  is refuted (anchor); `return walked` is refuted here (§2.4). **The budget must not decide at all** —
  gate on displacement since the last refusal (§5-C).
- **A blocked `-1` must never be replaced by `Move_Forward`** — fixed this iter, keep it fixed.
- **Do not read** `agent_location`, `target_location`, `destination`, `correct_robot_location`,
  `ideal_path`, `gt.json`, `current_instruction`, `env.unwrapped._agent_rotation`,
  `env.subtask_max_steps`, or `info['agent_collision']`. In particular **§1b's env facts are for
  reasoning, not for reading** — every recommendation above is derived from the harness's own executed
  action history, `info['moving_distance']`, and the static `task_spec['subtasks']` text.
- **Avoid numeric literals ≥ 2000** in candidate code; express distances as forward-action counts.
- **Verify a commit-composition change by driving the REACH path**, not only the planner path.

---

## 7. Provenance

- Candidate: `…/prop_r002_fe2b86/cand_r002_0_spatial-grounding` (`orchestrator.py` 700 L + 7 modules).
  Score: `…/cand_r2_0_spatial-grounding_r002/score.json` — SR 0.0, SubtaskSR 0.33,
  per_position_SR [1.0, 0.12, 0.1111, 0.0], distance_progress 0.173, static_collision 2.86,
  90 090 tok / 744.1 s per episode, 50/50 episodes complete, read-audit clean.
- Candidate traces: `…/cand_r2_0_spatial-grounding/traces/` — 50 episodes; **388** REACH calls
  (`vlm_trajectory.jsonl`, `kind="perceive"` whose prompt begins *"You compare two egocentric outdoor
  images"* — now 4 images, so image count no longer separates them from perceive proper),
  1 042 perceive, 1 042 plan, 3 123 forwards, 77 commits.
- A/B baseline: `…/anchor_nav_r000/score.json` + `…/anchor_nav/traces/` — SubtaskSR 0.36,
  per_position_SR [1.0, 0.24, 0.1111, 0.0]; 348 REACH calls (2 images), 2 812 forwards.
- Parent: `…/cand_r1_0_spatial-grounding_r001/score.json` — SubtaskSR 0.285,
  per_position_SR [1.0, 0.02, 0.0222, 0.0]. Its REACH calls carry a different system prompt and are
  quoted from its own `diagnosis.md`.
- Env mechanics (§1b): `SimWorldGym/simworld_gym/envs/traffic_world.py:365-385, 646-654`.
