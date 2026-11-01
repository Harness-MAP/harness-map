# Diagnosis — SimWorld-MMNav (`hard`, `search_hard_50`) iter 13 / `cand_r13_0_spatial-grounding` ("geometry supplies the WHERE"), parent `cand_r9_0_spatial-grounding`

**Headline:** primary (**SubtaskSR**) **0.550** — the chain's best, ahead of r010's 0.535 and the parent's
0.400 · **SR 0.00** · per-position SR **[1.00, 0.58, 0.5778, 0.00]** (parent `[1.00, 0.26, 0.2667, 0.00]`).
Read-audit **PASS** (15 paths, 0 forbidden / 0 escalated / 0 undeclared). Cost 145,863 tok/ep,
wall 1,289 s/ep, `over_budget: false`.

**This iteration's bet is confirmed by its own traces, at the operating point it pre-registered.** ROAD
0.26 → **0.58** (predicted 0.45–0.60), TURNING 0.2667 → **0.5778** (predicted ≈ 0.45), and the identity
conjunct measured **live** at 90.9 % in-window / 38.5 % far-field on ROAD against the 93 % / 39 % the
proposer priced offline. F2 (the fail-closed cross-check cap) worked: **zero** FINISH fatal commits where the
parent had four. Nothing about the ROAD/TURNING half of this candidate needs revisiting.

**Binding constraint (one sentence):** SR is pinned at exactly 0.00 by a **single dead conjunct** —
`_FRONTAGE_SYS` was called **29 times and answered `Here=0` all 29 times, including 3 calls made from inside
the success ball (L1 = 484 / 640 / 1144 cm)**, and because it is the *only* position conjunct on a FINISH leg,
**all 31 reached FINISH legs ended without the harness ever emitting a single `-1`** — those 3 frames were the
run's three live chances at SR > 0, each with identity already at `Match=1` and `commit_pose_prefix` able to
supply the exact graded yaw, and one 0/29 veto discarded all three.

---

## 1. What this harness does (mechanism)

`orchestrator.py:run_episode` is a 10-node per-step loop. Each loop iteration: stuck check → read the structural
subtask index → `ActivePerception.gate_burst` → `RoadGeometry.update(obs, instruction, cur_idx)` (new this
iter, inside the perceive branch) → `Perceiver` (1 VLM call, 4 images) → `Planner` (1 VLM call → an action
list) → `ArrivalGate` → affordance filter → **execute the whole action list as a burst** with no
re-perception between steps (`orchestrator.py:631`, median burst = **3** forwards, 1261/1601 runs exactly 3).

The decision that matters is the **`-1` LICENSE**, owned entirely by `modules/arrival_gate.py`:

* `proactive_commit` (harness-injected `-1`, `orchestrator.py:529`) — a **conjunction**:
  `_place_verdict` (POSITION) ∧ `_identity_call` (IDENTITY, two images, no leg text) ∧
  `_named_side_has_structure` (free segmentation veto).
* `allow_planner_commit` (cross-check on a planner-proposed `-1`) — same conjuncts, now **fail-closed** at
  its cap (F2, this iter): it replays `_xcheck_memo` instead of `return True`.
* `_place_verdict` routes the POSITION conjunct by leg type:
  * **ROAD** (`is_intersection_stop_leg`) → `RoadGeometry.place_verdict`, a free rising-edge detector on the
    roadway class `[0,127,255]` in the forward corridor (rows 0.45–0.70, cols 0.30–0.70), tri-valued
    (`None` = not owned, `True` = edge, `False` = **free veto, returned before the schedule**).
  * **FINISH** → `_FRONTAGE_SYS`, a VLM call: *"1 if the robot has arrived at a building's frontage — a
    facade is close enough that it fills a large part of the frame…"*
  * ORIENT / TURNING → untouched (cardinal burst / turn-approach latch).
* Schedule (`completion_checker.py`): `K_REACH_FORWARD_STREAK = 4`, `K_REACH_INTERVAL = 5`,
  **`MAX_REACH_FIRES_PER_SUBTASK = 12`**, `MAX_XCHECK_FIRES_PER_SUBTASK = 2`. On a leg with a live geometric
  edge the floor drops to `GEO_MIN_TRANSLATIONS = 1` and the interval is waived.
* `viewpoint_controller.commit_pose_prefix` prepends one quarter-turn toward the named side at the **one**
  place a `-1` is materialised (`orchestrator.py:594`) — so a FINISH commit is yaw-legal *if it ever happens*.

Prose (`AGENT.md`, `rules.md`) is byte-identical to the parent by design; every change this iter is in the
decision graph.

**Grading (env source, `traffic_world.py:388–460`), for attribution only:**
ROAD = `L1 < 2000`; TURNING = yaw within a one-sided 45° window **and** inside a box;
**FINISH = `L1 < 2000` AND yaw within the one-sided 45° window**. Note `trajectory.json`'s `dist_to_target`
is **Euclidean**; every distance below is recomputed as **L1**, the metric actually graded. The env also
terminates a leg at `substeps_count >= subtask_max_steps = int(L1_at_leg_start / 100)` — this is why
episodes end at 106–225 steps against a `max_steps` of 656–1344, and **no episode in the run hit `MAX_STEPS`**.

---

## 2. What actually happened (causal chain, with evidence)

### 2.0 The leg ledger — where the 50 episodes actually died

| leg | passed | failed by a **rejected `-1`** (fatal) | failed with **no `-1` ever emitted** |
|---|---:|---:|---:|
| ORIENTATION | **50 / 50** | 0 | 0 |
| ROAD | **29 / 45** | 6 | 10 |
| TURNING | **26 / 29** | 3 | 0 |
| **FINISH** | **0 / 31** | **0** | **31** |

114 `-1` actions in the run. **Not one of them was on a FINISH leg.** `REACH_COMMIT` appears 0 times in 50
episodes; every commit the harness made (31) was labelled `GEO_REACH`. The VLM position channel contributed
**29 vetoes and zero licenses**, and `_JUNCTION_SYS` — the wording this chain spent three iterations on — was
called **0 times in 50 episodes**: it is now dead code.

### 2A. The dominant failure: the agent arrives at the destination and the harness cannot say so

31 episodes reached the FINISH leg. **10 of them physically entered the L1 < 2000 success ball**; 24 of 31
got within L1 4000 (half a block). Zero committed. Closest approach, then where they ended up:

| ep | L1 min on the FINISH leg | L1 at episode end | steps in the ball |
|---|---:|---:|---:|
| `ep_20_23_21` | **105** | 11,032 | 7 |
| `ep_20_18_11` | **16** | 36,484 | 8 |
| `ep_20_4_21`  | **27** | 10,973 | 8 |
| `ep_20_9_21`  | **188** | 16,948 | 8 |
| `ep_20_24_21` | **265** | 31,765 | 7 |
| `ep_20_23_11` | **617** | 31,704 | 6 |
| `ep_20_3_11`  | **723** | 53,223 | 6 |
| `ep_20_2_11`  | 1,170 | 29,545 | 4 |
| `ep_20_20_11` | 1,216 | 35,216 | 4 |
| `ep_20_9_11`  | 1,778 | 11,036 | 2 |

Median final L1 over all 31 FINISH legs: **29,545 cm**. The crossing is always the same shape —
`ep_20_23_21` steps 131–138, L1 1538 → 1039 → 541 → **77** → 469 → 966 → 1465 → 1965, action on every one of
those eight steps: `Move_Forward`. It stood 77 cm from the grading point and walked on for another 19 steps.

This is the same mode the iter-9 diagnosis named ("walk in, decline, walk 13 km past, 28/50"), except the
decline is now **free of charge** — in 7 of the 10 ball crossings the harness did not decline, it **never asked**.

### 2B. The proximate cause: `_FRONTAGE_SYS` is a veto that cannot open, and it killed all three live chances

All 29 `_FRONTAGE_SYS` calls in the run, by L1 to the FINISH grading point, answered `Here=0`. Three were
asked from **inside the success ball**:

```
ep_20_18_11 step  68   L1   484   {"Here":0,"Reason":"building facade is distant and not close enough to fill the frame"}
ep_20_23_11 step  82   L1   640   {"Here":0,"Reason":"buildings are distant and not close enough to be at the frontage"}
ep_20_24_21 step 129   L1  1144   {"Here":0,"Reason":"buildings are distant and not close enough to be at the frontage"}
```

On each of those three frames the gate had **already passed every other conjunct**: `_identity_call`
returned `Match=1`, `_named_side_has_structure` passed, `arrival_gate.py:517` then read `here != 1` and
returned `False` (`blocked_by_place`, 28 + 1 = 29 run-wide — i.e. *every* frontage call was a block).

And the commit would have been **legal**. Verified from the perceive prompts at those exact steps:

| ep | FINISH instruction | heading at the ask | named side | yaw after `commit_pose_prefix` | yaw required | L1 |
|---|---|---|---|---|---|---:|
| `ep_20_18_11` | "…Chinese characters **on your left**, stop and face that building" | North (90) | left → `Rotate_Left` | **0 = West** | **0** | 484 ✓ |
| `ep_20_24_21` | "…three large arched glass windows **on your left**…" | West (0) | left → `Rotate_Left` | **−90 = South** | **−90** | 1144 ✓ |
| `ep_20_23_11` | "…white school with gray accents **on your right**…" | West (0) | right → `Rotate_Right` | **90 = North** | **90** | 640 ✓ |

All three satisfy `L1 < 2000` **and** the one-sided 45° yaw window. **Deleting one conjunct that has never
once returned 1 converts 0.00 SR into 2–3 successes.** (Two of the three — `ep_20_18_11` and `ep_20_24_21` —
are recoverable by identity alone, because their *first* FINISH-leg `Match=1` is already the in-ball one.
`ep_20_23_11` needs a position pre-filter that rejects its earlier `Match=1` at L1 6117, or it is foreclosed.)

### 2C. The structural cause: a per-leg call budget that goes silent exactly where the answer lives

The ball crossing is **not** too fast for the loop: across the 10 ball-entering episodes there were 60
in-ball env steps and **20 in-ball perceive decision points** (median burst is 3, not 8). The gate had the
frames. It did not have the budget:

| ep | identity asks on the FINISH leg | hit `MAX_REACH_FIRES_PER_SUBTASK` = 12? | asks made in the ball | L1 min |
|---|---:|---|---:|---:|
| `ep_20_18_11` | 12 | **YES** | 1 | 16 |
| `ep_20_20_11` | 12 | **YES** | 1 | 1216 |
| `ep_20_23_11` | 12 | **YES** | 1 | 617 |
| `ep_20_23_21` | 12 | **YES** | 2 | 105 |
| `ep_20_24_21` | 12 | **YES** | 1 | 265 |
| `ep_20_2_11`  | 13 | **YES** | 1 | 1170 |
| `ep_20_3_11`  | 12 | **YES** | 1 | 723 |
| `ep_20_4_21`  | 12 | **YES** | **0 — budget exhausted before the ball** | **27** |
| `ep_20_9_11`  | 11 | no | 0 | 1778 |
| `ep_20_9_21`  | 12 | **YES** | **0 — budget exhausted before the ball** | **188** |

**9 of 10 hit the cap.** Two of them (`ep_20_4_21` at L1 **27 cm**, `ep_20_9_21` at L1 **188 cm**) stood
essentially *on* the grading point with **no arrival question left to ask**. A FINISH leg is 20–50 km long;
12 asks at `K_REACH_INTERVAL = 5` translations cover roughly the first 30,000 cm and then the gate is mute
for the rest of the leg. This is the **same class of bug as F2** — a budget meant to bound *cost* ends up
determining the *verdict* — only here it fails silent instead of failing open.

Sampling ledger across the 10 ball-entering episodes: **60 in-ball steps → 20 decision points → 8 identity
asks → 3 frontage asks → 0 licenses.**

### 2D. The same identity call is 2.4× worse on FINISH than on ROAD, through identical code

| leg type | L1 < 2000 | 2–4 k | 4–10 k | > 10 k |
|---|---|---|---|---|
| **ROAD** | **30/33 = 90.9 %** | 1/1 | 0/0 | **5/13 = 38.5 %** |
| **FINISH** | **3/8 = 37.5 %** | 0/15 = 0 % | 9/57 = 15.8 % | 17/288 = 5.9 % |

Same `_IDENTITY_SYS`, same model, same two-image format. The difference is the **reference frame**: a FINISH
hint image is the destination facade shot from the *faced* pose, while the robot asks from the *travel* pose
— ~90° off-axis. The iter-9 diagnosis identified this precisely (§2D, F3: *"the pose is the bug, not the
strictness"*) and this iteration **deliberately deferred it** (NOTES risk (d): *"left for the next iter so it
is not confounded with a ROAD-leg measurement"*). That ROAD-leg measurement is now **complete and
successful**. There is nothing left to confound. F3 is no longer deferrable.

### 2E. The ROAD residual is a *disambiguation* failure that scales with leg length

29/45 ROAD legs pass. The 16 failures split three ways:

* **6 fatal premature commits.** One is a near miss (`ep_20_0_11`, L1 2548, threshold 2000). The other five
  are at **L1 17,578 / 17,780 / 20,712 / 20,801 / 20,920** — the agent committed at the *first* intersection
  it crossed on a 21 km leg. These are exactly the far-field identity false positives: 5/13 = 38.5 %, the
  rate the proposer priced at 39 %.
* **7 legs walked over their own waypoint and the gate asked 0–2 times in the whole leg** (L1 min 41 / 113 /
  149 / 162 / 246 / 346 / 1107). Because `geo_v is False` returns **before the schedule** (`arrival_gate.py:470`,
  `geo_block` = 359 run-wide), an owned ROAD leg gets an arrival question *only* on a claimable geometric
  edge. If the edge misses the ball, the leg gets **no arrival check at all**. NOTES risk (a) priced this at
  ≈ 1 leg of expected loss; it cost **7**.
* **3 legs where geometry never raised an edge** (`GEO(below=…)`, `claimable=0`) — risk (a) as priced.

Pass rate by leg length (env budget `int(L1_at_leg_start/100)`, a proxy for how many intersections the leg
crosses):

| env substep budget | n | pass | fatal | no-commit | pass rate |
|---|---:|---:|---:|---:|---:|
| ≤ 8 | 1 | 0 | 0 | 1 | 0.00 |
| 9–40 | 5 | 2 | 1 | 2 | 0.40 |
| **41–120** | 22 | **19** | **0** | 3 | **0.86** |
| **> 120** | 17 | 8 | **5** | 4 | **0.47** |

The detector+identity pair is near-solved on medium legs and leaks badly on long ones, and **all five
far-field fatal commits are in the > 120 bucket.** 43/43 distinct ROAD instructions name a **side**
("…on your left/right"); **0/43 carry an ordinal** ("the second intersection"). So the only disambiguating
cue is the named-side corner building's *appearance* — and the identity call currently compares the **whole
frame**, where a generic street resembles every other generic street.

One pathological instance worth naming: `ep_20_10_11` spawns 154 cm from its own ROAD waypoint, so the env
budget is `int(154/100) = 1` substep. The leg had exactly one decision point, geometry answered `False` on it
(`GATE(geo_block=1)`), zero identity calls were made, and the episode was over at step 3. A leg that begins
inside the ball is unwinnable under a geometry-first free veto.

### 2F. `NO_ACTION_RETURNED` is output truncation on the plan call, not a parse bug

23 occurrences. **All 23 are `finish_reason: "length"` on the plan call at `max_tokens=8192`**
(`modules/planner.py:149`), plus 1 on perceive. The model is `Qwen3-VL-30B-A3B-**Instruct**` — there is no
`<think>` block to pay for; the planner simply runs the whole budget and gets cut mid-JSON. Each one burns a
full perceive+plan cycle (~188 k wasted output tokens run-wide) and `ep_20_11_21` was **killed by
`TOO_MANY_ERRORS` at 11 of them while holding 3 banked legs**.

### 2G. Cost went the opposite way from the prediction

NOTES predicted token cost **down**. Actual **117,979 → 145,863 tok/ep (+24 %)**. The gate did get cheaper
(415 identity + 29 place = **444** calls vs the parent's 595), but episodes now survive ~3× longer (mean
106.5 steps; 158 on banked-3 episodes), and perceive+plan fire every loop iteration: **1,650 + 1,650 = 3,300
of the run's 3,744 VLM calls (88 %)**. The cost lever is the perceive+plan pair, not the gate.

---

## 3. Failure breakdown (50 episodes, by what ended them)

| category | n | share |
|---|---:|---:|
| **Reach-or-approach FINISH, never commit, env substep budget expires** | **26** | 52 % |
|  · of which **physically entered the L1 success ball** | 10 | 20 % |
|  · of which **had identity=1 in-ball + a legal yaw, killed by the frontage veto** | **3** | 6 % |
| Fatal premature `-1` on ROAD (far-field identity FP at a geometric edge) | 6 | 12 % |
| ROAD leg walked over, gate structurally silent (`geo_block` before schedule) | 7 | 14 % |
| Fatal `-1` on TURNING (yaw/box miss) | 3 | 6 % |
| `STUCK` (forward into a wall) | 7 | 14 % |
| ROAD leg, geometry never raised an edge | 3 | 6 % |
| `TOO_MANY_ERRORS` (11 truncated plan calls, 3 legs banked) | 1 | 2 % |

Ledger totals, run-wide: `geo_block` 359 · `ident_0` 350 · `ident_1` 65 · `geo_1` 35 · `place_0` **29** ·
`place_1` **0** · `blocked_by_place` 29 · `xcap_replay` 23 (F2 firing, in 4 eps) · `veto_side` 1 ·
`GEO(edge)` 50, `claimable` 48, `edge_over_cap` 2, `no-readings` 5 (all five are the 2-subtask
ORIENT→FINISH tasks — correct abstention).

---

## 4. Binding constraint — classification

**HARNESS-BUG, on the FINISH leg.** Not VLM-CEILING: the frozen model *already produced the right answer* on
the right frames — identity returned `Match=1` at L1 484 / 640 / 1144 cm, and `commit_pose_prefix` already
computes the exact graded yaw in all three cases. The harness then threw those answers away through
(a) a position conjunct measured at **0/29** that can only ever block, and (b) a per-leg call budget that
runs out before the robot arrives. Both are decision-graph defects a code change recovers, and the recovery
is priced from this run's own trace: **SR 0.00 → 0.04 with one deletion, → 0.06 with a position pre-filter,
ceiling 0.20 (the 10 ball-entering episodes).**

The unifying statement for both halves of the run: **the gate's sampling is not where the answer is.** On
ROAD the budget is too *tight* (event-keyed, ≤ 2 asks per leg, so 7 legs got no check while standing on the
waypoint); on FINISH it is too *uniform* (12 asks spread over 40 km, so 2 legs got no check while standing
on the destination). The 29 % → 56 % win this iter came from moving the asks, not from adding them.

Secondary, genuinely **VLM-CEILING**: the identity call's 38.5 % far-field rate on long ROAD legs. Re-wording
is closed (iters 1–5, plus junction A/B). The fix must **narrow the comparison**, not re-phrase it.

Residual **TASK-GAP**: there is still no arrival *controller* — no path in `orchestrator.py` can slow down, stop,
or back up. Every in-ball frame is a `Move_Forward`. This remains the deepest structural gap, but it is now
*downstream* of the two bugs above: fix the license and the controller question becomes measurable.

---

## 5. Concrete changes to try (the gradient — ranked by leverage)

### F1 — Delete the FINISH position conjunct. `modules/arrival_gate.py:_place_verdict` / `_FRONTAGE_SYS`
Make `_place_verdict` return `None` on `leg_type(instruction) == LEG_FINISH` (or drop `_FRONTAGE_SYS`
outright — it is 29 calls / 0 positives and `_JUNCTION_SYS` is already 0 calls, so the whole `place` channel
and `MAX_PLACE_FIRES_PER_SUBTASK` can go). **Careful:** `None` currently falls through to `_legacy_verdict`
(`arrival_gate.py:511`), which is the parent's identity-with-leg-text call measured at 24 % — do **not** let
it land there on FINISH; license on the text-free identity call.
*Predicted:* `per_position_SR[3]` 0.00 → **0.04–0.06**, SR 0.00 → **0.04–0.06**, primary +0.01–0.015.
*Why it works:* measured directly — 3 frames this run had every other conjunct satisfied and a legal yaw.
*Why it is safe:* FINISH is the **terminal** leg. A premature FINISH `-1` forfeits only a leg that already
scores 0 and has nothing downstream — the recall/precision asymmetry that justifies conservatism on ROAD
**does not exist here**. This chain has never exploited that, and it is the single cheapest change available.

### F2 — Make the arrival budget leg-type-aware; stop going silent on FINISH. `completion_checker.py:114` + `arrival_gate.py:proactive_commit`
`MAX_REACH_FIRES_PER_SUBTASK = 12` is global. On FINISH, drop the cap (or re-arm it on a progress event —
e.g. every N translations of accrued `info['moving_distance']` — rather than spending it once on a clock).
*Predicted:* in-ball identity asks **8 → ~20** (the decision points already exist, §2C), and at the measured
37.5 % per-ask in-window rate that is ≈ 6 in-ball positives instead of 3. With F1, SR → **0.08–0.12**.
*Why it works:* 9/10 ball-entering episodes hit the cap; 2 hit it *before arriving* at L1 27 and 188 cm.
*Cost:* bounded — FINISH legs already run 12 identity calls, so this at most doubles 415 → ~700 gate calls,
against 3,300 perceive+plan calls. Pair with F1 or the extra asks just feed the dead veto.

### F3 — Ask the FINISH identity question from the pose its answer requires. `viewpoint_controller.py` + `arrival_gate.proactive_commit`
When the travel-pose identity returns `Match=1` on a FINISH leg, **first** emit the quarter-turn toward the
named side (the turn `commit_pose_prefix` was going to make anyway), re-grab `obs['rgb']`, and ask identity
again from the faced pose. Commit on a second `1`; emit the reverse quarter-turn and keep walking on a `0`.
Cost: 2 env substeps + 1 VLM call per candidate frame.
*Predicted:* FINISH identity in-window 37.5 % → the ROAD channel's ~90 % (it becomes the same
same-viewpoint comparison), far-field 5.9 % → lower (a side-on facade at 20 km does not fill the frame), and
the commit is **yaw-legal by construction**. This is the lever that reaches the 0.20 ceiling.
*Mandatory caveat for the proposer:* iter 9 deleted `request_face_scan` as **0/128**, and it is still present
and unreferenced in `viewpoint_controller.py`. Re-entering this axis is only honest if the proposal states
what is different — then, a *forward-facing frontage question* with no position pre-filter; now, a
*travel-pose identity `Match=1`* keying it. Say so, and pre-register the 0/128 as the thing being beaten.

### F4 — Narrow the identity comparison to the named side. `arrival_gate.py:_identity_call`
43/43 ROAD and 30/30 FINISH instructions name a side; 0/43 carry an ordinal. Crop **both** images to the
named-side half before base64-encoding them (image preparation only — no wording change, so the 93 %
in-window result is not at risk in the way a re-phrasing would be). Fall back to the full frame when
`named_side()` is `None`.
*Predicted:* far-field identity 38.5 % → lower on long ROAD legs, removing most of the 5 fatal commits in the
`> 120` bucket; `per_position_SR[1]` 0.58 → ~0.68, carrying TURNING at the measured 90 % conditional rate.
Primary +0.03–0.05.
*Offline probe before shipping (recipe):* this run archives all 3,744 call frames. Replay the ROAD-leg
identity calls cropped vs uncropped at temperature 0, scored against L1 to the ROAD waypoint (oracle labels,
proposer-side only). The uncropped baseline to beat is **30/33 in-window, 5/13 > 10 k**.

### F5 — Let the geometric veto spend one ask per leg instead of zero. `arrival_gate.py:proactive_commit`
`geo_v is False → return False` (before the schedule) cost **7 ROAD legs** that walked over their own
waypoint with 0–2 asks, and made `ep_20_10_11` (spawned 154 cm from its waypoint, 1-substep env budget)
unwinnable. Keep the free veto as the *default*, but allow a small reserve the veto cannot consume — e.g.
one identity ask per leg permitted regardless of `geo_v`, or ignore the veto on the first decision point of a
leg (a leg may begin inside the ball).
*Predicted:* recovers 1–3 of the 7; primary +0.01–0.03. Low risk, small surface.
*Note:* NOTES measured and rejected "arming" (requiring a closed reading *before* edge #1). This is the
**converse** and was never measured — it is about the veto suppressing the gate, not about edge timing.

### F6 — Cap the plan call's output. `modules/planner.py:149`
`max_tokens=8192` on a non-thinking model; 23/1,650 plan calls truncate at `finish_reason: "length"`, each
burning a whole cycle, and one killed an episode holding 3 banked legs. Drop to ~1,024–2,048 and/or retry
once on truncation instead of `continue`.
*Predicted:* −23 wasted cycles, ~−188 k output tokens, +1 episode saved from `TOO_MANY_ERRORS`. Partly
reverses the +24 % cost regression (§2G). Cheap; bundle it, don't spend an iter on it.

---

## 6. Exhausted — do NOT repeat

Carried forward from the iter-9 gradient (all still closed): re-wording the leg-text arrival recognizer
(iters 1–5); putting instruction text into the two-image identity call (24 % vs 93 %);
`side_building_fraction` as a junction detector (AUC ≈ 0.72); the depth predicate (per-frame normalised, no
absolute scale, fired 2×); the iter-6 ROAD face-scan (0/128 — **FINISH is a different question, see F3**); a
blanket `allow_forward_commit` FINISH block (relocates the deaths); junction wording A/B; tightening
`_FRONTAGE_SYS` further. Plus from this iter's own NOTES: `_JUNCTION_SYS` wording (CLOSED — and now 0 calls
in 50 episodes, i.e. dead code); band geometry `(0.45–0.70)×(0.30–0.70)` (flat 0.40–0.65 … 0.50–0.75);
`k = 2` confirmations on a `place == 1`; "arming" the geometric edge.

**Newly closed by this run — four things the next proposer must not spend an iter on:**

* **`MAX_CLAIMABLE_CROSSINGS` 2 → 1 is REFUTED.** The NOTES pre-registered it as the lever for risk (b). The
  trace says the opposite: **all 6 fatal ROAD commits were `claimable=1, edge=1` (edge #1)**, and **all 4
  `claimable=2` claims were PASSES**. 2 → 1 removes 4 passes and 0 fatalities. If anything the data points
  at 2 → 3; the `edge_over_cap=1` cases (`ep_20_0_21`, `ep_20_21_21`) were both far-field, so expect little
  either way. **Leave this constant alone.**
* **A segmentation-geometry position channel for FINISH does not exist.** I replayed the **1,241 archived
  FINISH-leg `object_mask` frames** and computed the purple building-class fraction in the named-side
  lateral band across **5 band geometries** (rows 0.30–0.62 / 0.00–1.00 / 0.00–0.40 × outer 0.25 / 0.12 of W):
  in-window median **0.700** vs **0.594** beyond 10 km, firing on **94 %** of far-field frames at every
  threshold from 0.05 to 0.50, and the per-leg **argmax lands in-window on 0–2 of the 9 legs** that had an
  in-window frame. The destination is a point along a continuous built-up facade — there is **no geometric
  signature**. Do not transplant `RoadGeometry` to FINISH. The `where` on FINISH has to come from
  appearance (F3/F4), which is the inverse of the ROAD leg.
* **Burst truncation / F1's pacing half is not the FINISH bottleneck.** The NOTES deferred it (risk (c)) on
  the grounds that the window is 8 forward steps wide and the gate asks ~1.6 times. Measured here: the
  median burst is **3**, not 8, and the 10 ball crossings already contain **20 perceive decision points**.
  The frames exist; the *budget* (§2C, F2) and the *conjunct* (§2B, F1) are what fail. Truncating the burst
  would also perturb the rising-edge detector's sampling cadence for no measured gain.
* **F2 (the fail-closed cross-check cap) is done and worked.** `xcap_replay` fired 23 times in 4 episodes;
  there were **zero** FINISH fatal commits against the parent's four, and zero episodes thrown away by the
  cap. Do not re-open it. The `XCHECK_BLOCK; EMPTY_FALLBACK` pairs still visible in 48/49 episodes are now
  the *correct* behaviour (block, inject one forward, re-decide), not the old hole.

---

*Anti-cheat note: every distance, waypoint, yaw and `subtask_max_steps` figure in this document is an oracle
field (`gt.json`, `trajectory.json`, `traffic_world.py`) read **post-hoc, by the diagnostician, to label
frames and attribute outcomes**. The offline mask replay in §6 scores archived frames from a finished run
against oracle L1 labels — proposer-side probe only. Every fix above decides from `obs['rgb']`,
`obs['object_mask']`, the per-subtask hint image, the static `task_spec['subtasks']` text, the sanctioned
`info['moving_distance']` odometry and cardinal `info['agent']['agent_rotation']`, and the harness's own
bookkeeping. No fix requires `subtask_max_steps`, `correct_robot_location`, `ideal_path`, `agent_location`,
`target_location`, or any coordinate. The candidate's `read_audit` verdict is **PASS** with 15 paths and no
forbidden, escalated or undeclared read, and none of these fixes widens that surface.*
