# Diagnosis — SimWorld-MMNav (`hard`, `search_hard_50`) iter 9 / `cand_r009_0_spatial-grounding` ("the ARRIVAL GATE"), parent `cand_r6_0_spatial-grounding`

**Headline:** SR **0.00** (0/50) · SubtaskSR **0.400** — the chain's best (anchor 0.360, iter-6 parent 0.370) ·
per-position SR **[1.00, 0.26, 0.2667, 0.00]** (parent `[1.00, 0.20, 0.20, 0.00]`). Read-audit clean
(15 paths, no forbidden field). The iteration's central bet **is confirmed by its own traces**: deleting the
`Subtask instruction:` text from the two-image identity call lifted in-window ROAD recall from the parent's
measured **8/34 (24 %)** to **39/42 (93 %)**. The gate's `legacy` degradation path fired **0 times** — the
new channel never failed to parse.

**Binding constraint (one sentence):** the harness now *recognises* arrival but still has **no arrival
controller** — it samples a 29 %-recall binary roughly **1.6 times** while crossing a window that is
**8 forward steps wide**, has no way to slow down, stop, or come back, and so on **28/50 legs the robot walks
through the acceptance window (closest approach median ≈ 210 cm, best 16 cm) and then keeps going a median of
42 more steps / 13 008 cm until the environment kills it.**

---

## 1. What this harness does (mechanism)

`orchestrator.py:run_episode` is the parent's 8-node loop with a 9th node, `ArrivalGate`
(`modules/arrival_gate.py`, 502 lines), which takes the `-1` **license** away from
`CompletionChecker._binary_reach_vlm_call` and makes it a **conjunction of two independent VLM calls**:

```
SpatialMemory → PromptAssembler → Perceiver(VLM) → Planner(VLM) → AffordanceEstimator
   ↘ ActivePerception.should_emit → ViewpointController.gate_burst   (TURN approach, iter 6, kept)
   ↘ CompletionChecker.verify  →  [X] ArrivalGate.allow_planner_commit   (orchestrator.py:462)
   ↘ if not commit_done        →  [P] ArrivalGate.proactive_commit       (orchestrator.py:498)
   ↘ ViewpointController.commit_pose_prefix at the single `-1` materialisation site (orchestrator.py:549)
```

Both commit sites run the same conjunction:

| conjunct | prompt | inputs | when |
|---|---|---|---|
| `_identity_call` (`arrival_gate.py:270`) | `_IDENTITY_SYS` — "do these two photos show the same place? **Judge the IMAGES ONLY**" | current RGB + hint image, **no leg text** | schedule imported verbatim from `completion_checker`: `K_REACH_FORWARD_STREAK=4`, then every `K_REACH_INTERVAL=5` translations, cap `MAX_REACH_FIRES_PER_SUBTASK=12` |
| `_place_call` (`arrival_gate.py:318`) | `_JUNCTION_SYS` on ROAD/TURN, `_FRONTAGE_SYS` on FINISH — one image, no hint, no text | current RGB only | only if identity returned 1 |

Deleted this iter: the iter-6 face-before-look edge (`request_face_scan` → yaw → `scan_reach`) and the
iter-6 `allow_forward_commit` FINISH block — so the FINISH leg is commit-eligible again, guarded only by
`_FRONTAGE_SYS`.

Env facts that price every decision (`simple_world.py:548-571`, `:305-325`): a **rejected `-1` sets
`terminated=True`** — a wrong commit costs the *episode*, not the leg; `subtask_max_steps` = 10 (ORIENT) /
50 (TURN) / `int(L1(leg_start, waypoint)/100)` (ROAD, FINISH); **and leaving the road polygon also
terminates** — which, given a 500 cm forward step, is what actually ends most of this run's episodes.

A detail that governs everything in §2: the gate is consulted **once per loop iteration**, on the frame
captured at the *start* of that iteration, and `ActivePerception` + the planner then execute a **burst of
3–7 forwards (1 500–3 500 cm) blind** before the next consult.

---

## 2. What actually happened (causal chain, with evidence)

I recovered all 595 gate calls from `vlm_trajectory.jsonl` (they are logged as `kind:"perceive"`, so they
had to be matched by system-prompt substring — see fix **F6**). The recovered counts reproduce the
`GATE(...)` ledger in every `verdict.json` exactly: ident 297×0 / 151×1, junction 116×0 / 19×1,
frontage 12×0 / 0×1, `legacy` 0, `ident_none`/`place_none` 0.

### 2A. The identity fix worked. The place conjunct is the recall bottleneck.

Each call scored against the true distance to that leg's own waypoint (oracle used for *analysis only*):

| call | IN (≤2000 cm) | NEAR (2–4 k) | FAR (>4 k) |
|---|---|---|---|
| `identity`, ROAD legs | **39/42 = 93 %** | 34/46 = 74 % | 66/169 = 39 % |
| `junction` (conditional on identity=1) | **11/38 = 29 %** | 5/31 = 16 % | 3/66 = 4.5 % |
| `identity`, FINISH legs | **2/10 = 20 %** | 1/17 | 9/164 = 5 % |
| `frontage` (conditional on identity=1) | **0/2** | 0/1 | 0/9 |

So the conjunction's **precision is excellent** (≈1.8 % far-field fire per ask) and its **recall is the
problem**: ≈26 % per ask in-window. That is the whole ROAD story — `per_position_SR[1] = 0.26`.

### 2B. The dominant failure: walk in, decline, walk 13 km past (28/50)

The window is ±2000 cm → **8 forward steps wide**. The gate's schedule asks once per **5** translations
(2 500 cm), and a burst can carry the body 1 500–3 500 cm between consults. Across the 50 failing legs the
robot spent **291 steps standing inside the acceptance window** and the gate was asked in-window about
**38 times total** — roughly one ask per leg. `ep_20_0_task_dist_11_0_1` is the canonical trace:

```
iter @ step 3   d=3452   plan [0]        → 3 forwards
iter @ step 6   d=1952   plan [0]        → 3 forwards            (already inside the window)
iter @ step 9   d= 452   GATE: ident=1, junction=0 "unbroken facades on both sides"   ← DECLINED AT 4.5 m
                         plan [0,0,0,0,0] → 5 forwards executed BLIND  (452 → 2548 cm)
iter @ step 16  d=2548   GATE: ident=1, junction=1 "crosswalk is at or just in front"
                         → REACH_COMMIT → `-1` REJECTED → episode over
```

`ep_20_17_task_dist_11_0_1` is byte-for-byte the same shape (declined at d=926, committed at d=2074 — 74 cm
outside). On the 28 legs where no `-1` was ever licensed, the robot simply kept walking:

| ep | leg | closest approach | steps walked **after** closest approach | final distance |
|---|---|---|---|---|
| `ep_20_18_11` | FINISH | **16 cm** | 81 | 36 484 |
| `ep_20_23_21` | ROAD | **20 cm** | 7 | 3 520 |
| `ep_20_4_21` | FINISH | **27 cm** | 22 | 10 973 |
| `ep_20_17_21` | ROAD | **42 cm** | 87 | 43 542 |
| `ep_20_4_11` | ROAD | 58 cm | 46 | 2 241 |
| `ep_20_19_21` | FINISH | 169 cm | 235 | 72 953 |
| … 22 more | | median **210 cm** | median **42** | median **13 008** |

20 of the 28 ended **more than 4 000 cm past** a window they had been standing in. Only **4 of 35**
non-fatal failing legs actually exhausted `subtask_max_steps` — the budget is generous (median 65 steps used
of a 100–530 budget). The episodes die because the robot **walks off the road polygon**
(`simple_world.py:307`). There is no overshoot detector anywhere in the harness.

### 2C. `_JUNCTION_SYS` is a crosswalk detector with a fixed half-block offset and no target identity

Every one of the 19 positives, verbatim:

```
d=    9  "a crosswalk is at or just in front of the robot"     d= 1738  "crosswalk is at or just in front…"
d=  639  "crosswalk is at or just in front of the robot"       d= 1808  "crosswalk is at or just in front…"
d=  644  "crosswalk is at or just in front of the robot"       d= 2074  "crosswalk is at or just in front…"  ← FATAL
d=  796  "a crosswalk is at or just in front of the robot"     d= 2548  "crosswalk is at or just in front…"  ← FATAL
d=  950  "crosswalk is at or just in front of the robot"       d= 2785  "a crosswalk is at or just in front…" ← FATAL
d=  978  "a crosswalk is visible in front of the robot"        d= 2873  "a crosswalk is at or just in front…" ← FATAL
d= 1139  "crosswalk is at or just in front of the robot"       d= 2963  "building facade ends at edge of frame" ← FATAL
d= 1494  "crosswalk is at or just in front of the robot"       d=17780  "crosswalk is at or just in front…"  ← FATAL
d= 1580  "crosswalk is at or just in front of the robot"       d=20078  "a crosswalk is visible in front…"    ← FATAL
                                                               d=22842  "crosswalk is at or just in front…"  ← FATAL
```

18/19 cite **one** of the prompt's three disjuncts. Every in-window *negative* is the same sentence —
"unbroken building facades on both sides" (d=37, 42, 68, 163, 285, 373, 452, 480, 658, 683, 692, 712 …). So
the ANY-CUE wording collapsed to a single binary cue, and that cue is **geometric, not semantic**: it fires
when a crosswalk is underfoot, which sits at a roughly fixed offset from the grading point — sometimes
inside the window (11 times), sometimes 74–963 cm outside it (5 times, all fatal). And because the city is a
grid, the question **cannot distinguish the target junction from any other**: three positives at 17 780 /
20 078 / 22 842 cm are different intersections entirely.

Identity cannot cover for this, because identity is a **leg-level**, not a **position-level**, signal. On
`ep_20_19_11` it answered 1 on all 13 asks from 20 212 cm before to 14 288 cm past the waypoint; on
`ep_20_12_21`, 1 on all 10 asks from 25 301 cm. Nothing in the harness estimates *how far* anything is.

### 2D. FINISH is structurally impossible: the frontage question is asked from a pose that guarantees "no"

`_FRONTAGE_SYS` was called **12 times in 50 episodes and answered 0 every single time**, including:

```
d=  484 cm  ep_20_18_11 st68  "building facade is distant and not close enough to fill the frame"
d= 1226 cm  ep_20_5_21  st185 "building facade is distant and not close enough to fill the frame"
```

The cause is a **pose/question mismatch introduced by this iteration**. A FINISH instruction reads
*"Proceed forward until reaching the high gray and beige building with signs in Chinese characters **on your
left**, stop and face that building"*, and its hint image is (the perceiver's own words, `ep_20_18_11`)
*"a close-up of the target building's facade, **directly in front** of the agent"*. The body travels along
the street; the destination facade is **90° off-axis**. `_FRONTAGE_SYS` asks whether a facade "fills a large
part of the frame" — from a heading where, by construction, it cannot. The one component that would have
fixed this, `viewpoint.commit_pose_prefix`, is applied at `orchestrator.py:549` — *after* `commit_done` is
already true, i.e. **after** the question has been asked and answered 0. Same cause explains FINISH identity
recall (2/10 in-window) being 4.7× worse than ROAD's (39/42) through identical code.

Net: `per_position_SR[3] = 0.00` is not noise. **The gate cannot license a FINISH commit at all**, and SR is
therefore pinned at 0 by construction.

### 2E. `allow_planner_commit` fails OPEN at its cap — and that is what produced every FINISH death

`arrival_gate.py:allow_planner_commit`:

```python
fires = int(self._xcheck_fires.get(cur_idx, 0))
if fires >= MAX_XCHECK_FIRES_PER_SUBTASK:      # == 2, imported from completion_checker
    return True                                 # ← DEFERS: a planner `-1` passes with NO gate at all
```

The run exhibits the resulting signature 10 times, exactly as predicted by the code
— `XCHECK_BLOCK; EMPTY_FALLBACK; XCHECK_BLOCK; EMPTY_FALLBACK; …` (21 blocks / 22 empty fallbacks run-wide):
the planner proposes `-1`, the gate blocks it twice, `orchestrator.py:585` injects one forward each time, and on
the **third** proposal the cap is spent and the `-1` executes **ungated**. In 6 of the 15 fatal commits the
rejected `-1` came through this hole, including **all four FINISH deaths**:

| ep | leg | distance at the fatal `-1` | gate state |
|---|---|---|---|
| `ep_20_10_21` | FINISH | **44 789 cm** | `ident_0=7`, no place call ever made |
| `ep_20_22_11` | FINISH | **32 527 cm** | 2 × XCHECK_BLOCK then open; had **banked 3 legs** |
| `ep_20_5_11`  | FINISH | **31 370 cm** | 2 × XCHECK_BLOCK then open |
| `ep_20_7_21`  | FINISH | **24 389 cm** | 2 × XCHECK_BLOCK then open; had **banked 3 legs** |
| `ep_20_12_21` | ROAD   | 5 216 cm | 2 × XCHECK_BLOCK then open |
| `ep_20_13_21` | ROAD   | 2 366 cm | 2 × XCHECK_BLOCK then open |

The conjunction never got to veto these. `_FRONTAGE_SYS`, which answered 0 on all 9 of its far-field calls
and would have blocked every one of them, was never consulted. Two episodes that had already banked three
legs were thrown away by a cap meant only to bound cost.

---

## 3. Failure breakdown (50 episodes, by what ended the episode)

| category | n | share |
|---|---|---|
| **Reach-but-no-commit → walk past → env terminates** (out of road polygon / leg budget) | **28** | 56 % |
| Fatal `-1` (rejected commit ⇒ `terminated`) | **15** | 30 % |
|  · gate-licensed **overshoot** (committed 2 074–2 963 cm out after having been 37–215 cm in) | 5 | |
|  · gate-licensed **far-field** false fire (17 780 / 20 078 / 22 842 cm) | 3 | |
|  · **`allow_planner_commit` cap fail-open** (§2E) — 4 FINISH + 2 ROAD | 6 | |
|  · TURNING yaw/box miss at d = 1 639 (`ep_20_12_11`) | 1 | |
| `STUCK` (wedged, harness detector) | 5 | 10 % |
| Never approached the waypoint at all (failing leg min-distance > 2 000 cm, no commit) | 1 | 2 % |
| `TOO_MANY_ERRORS` | 1 | 2 % |

Failing leg type: ROAD **32**, FINISH **17**, TURNING **1**. ORIENT is 50/50 and untouched by the gate.
**37/50** failing legs entered the 2 000 cm window at some point. Conditional pass rates are informative:
of the 13 episodes that cleared ROAD, **12 (92 %) also cleared TURNING** — the iter-6 turn-approach lever is
healthy and ROAD is the gate; of those 12, **0 cleared FINISH**.

Cost: 595 gate calls / 50 eps ≈ 12 per episode, `cost_per_episode` 117 979 tok, `over_budget: false`. The
gate costs **zero env substeps** (unlike the 256 the deleted face-scan burned), so cost is not the
constraint — there is clear headroom to ask *more*, not less.

---

## 4. Binding constraint — classification

**Primary: TASK-GAP.** The robot gets to the goal legally and often precisely (16 cm, 20 cm, 27 cm, 42 cm
closest approaches) and the harness has no machinery that converts "I am here" into "stop here". The
decision is sampled ~1.6 times across an 8-step window, is one-way, and has no notion of distance, peak, or
overshoot. 33 of 50 episodes (28 reach-no-commit + 5 overshoot fatals) are lost to this single gap.
Perception is **not** the ceiling: the identity conjunct already detects the window at 93 % recall.

**Secondary, both fixable in code this iteration:**
- **HARNESS-BUG** — `arrival_gate.py:allow_planner_commit` defers (returns `True`) once `_xcheck_fires ≥ 2`,
  so the third planner `-1` on a leg bypasses the gate entirely. 6/15 fatal commits, 4 of them FINISH, two
  of them discarding 3 banked legs.
- **HARNESS-BUG (prompt/pose composition)** — `_FRONTAGE_SYS` asks a head-on question from a side-on pose,
  because `commit_pose_prefix` runs *after* the gate instead of before it. 12/12 zeros; FINISH SR is
  structurally 0.

**Not VLM-CEILING.** Every one of the three measured channels behaves correctly *given what it is shown*;
all three failures are about *when*, *how often*, and *from where* the harness asks.

---

## 5. Concrete changes to try (the gradient — ranked by leverage)

### F1 — Local arrival search: when identity fires, slow down and ask every step
*(`orchestrator.py` burst materialisation ≈ `:546-562`; `modules/arrival_gate.py:proactive_commit`)*

Add an **APPROACH latch** to `ArrivalGate`: the first time `_identity_call` returns 1 on a ROAD/FINISH leg,
set `self._approach[cur_idx] = True`. While latched, for up to `N≈10` translations:
1. `orchestrator.py` truncates `chosen_actions` to **one** forward per loop iteration (the forwards still happen —
   what changes is that the loop re-decides between them instead of executing 3–7 blind);
2. the gate's interval drops to `1` for that leg, so `_place_call` runs on **every** 500 cm;
3. commit on the **first** `place == 1`.

*Predicted:* in-window asks per leg **1.6 → ~8**. At the measured 29 % per-ask recall that is
`1 − 0.71^8 ≈ 0.94` P(license | in window), and it fires at the *first* in-window positive rather than the
*second scheduled* ask 2 500 cm later — which is exactly the 5 overshoot fatals. `per_position_SR[1]`
0.26 → plausibly 0.45–0.60, carrying TURNING with it at the measured 92 % conditional rate.
*Cost:* ≈10 extra one-image calls per approached leg, **zero extra env substeps**; budget headroom is
median 65 used of 100–530, and `over_budget` is false with 3× margin.
*Why it must work:* the signal is already there and already accurate; it is being discarded by sampling rate.

### F2 — Make the cross-check cap bound CALLS, not the VERDICT
*(`modules/arrival_gate.py:allow_planner_commit`, the `if fires >= MAX_XCHECK_FIRES_PER_SUBTASK: return True`)*

Cache the verdict per `(cur_idx, forward_streak)` (the `_place_memo` pattern already in the file) and, when
the cap is spent, **return the cached verdict** instead of `True`; only recompute once the body has
translated. Add a hard rule: on a FINISH leg a planner `-1` may **never** pass by deferral — if the gate
cannot answer, block. Also stop the `XCHECK_BLOCK → EMPTY_FALLBACK → re-propose` churn (21/22 occurrences,
2 VLM calls per blocked step) by suppressing the planner's `-1` for k steps after a block rather than
re-asking.
*Predicted:* removes 6 episode-ending commits including all four at 24 389–44 789 cm; `ep_20_22_11` and
`ep_20_7_21` keep their 3 banked legs and continue. SubtaskSR +0.02–0.04 directly, and it is a precondition
for F3 (otherwise a fixed frontage question is still bypassable).

### F3 — Ask the FINISH place question from the pose its answer requires
*(move `viewpoint.commit_pose_prefix` from `orchestrator.py:549` to before `_place_call` in
`arrival_gate.py:_place_verdict`, FINISH only)*

On a FINISH leg: yaw toward the named side **first** (2 substeps), then run `_identity_call` *and*
`_FRONTAGE_SYS` on the faced frame, commit from that pose if both are 1, restore the travel heading if not.
*Predicted:* the only available route to `per_position_SR[3] > 0`, and therefore the only route to SR > 0.
The grading yaw is required at commit anyway, so on FINISH the turn is not a detour.
*Caveat to carry forward:* iter 6 cut this edge because the faced look scored 0/128 — but that was the
*leg-text* recognizer (now independently proven broken: 24 % → 93 % when the text is removed) and it was
applied on **ROAD**, where yaw is not graded and the rotations are pure cost. Re-enable on **FINISH only**;
keep it off ROAD, where this run shows the forward view already works.

### F4 — Replace the junction conjunct with a self-referential "have I passed it?" call
*(`modules/arrival_gate.py:_place_call` / `_JUNCTION_SYS`; frames from `modules/spatial_memory.py`,
baseline from the sanctioned `info['moving_distance']`)*

`_JUNCTION_SYS` has no target identity (3 positives at 17–22 km are other intersections) and a fixed
half-block offset (5 positives at 2 074–2 963 cm). Replace it with a **two-image call over the agent's own
frames**: frame(t) vs frame(t−k), with the odometry baseline in the text — *"between these two views of the
same street, is the landmark getting closer, at its closest now, or already behind you?"*. This supplies the
**position-level** signal the harness has never had; identity cannot supply it (saturated-1 across whole
legs: 13/13 on `ep_20_19_11`, 10/10 on `ep_20_12_21`). It also composes with F1: during the APPROACH latch
the frame buffer gives a dense 500 cm baseline. Pairs naturally with an **overshoot recovery** — ROAD is
graded on **position only** (`simple_world.py:_get_evaluation_of_subtask`), so `[1,1]` + k forwards to back
up is legal and free apart from step cost, and the budget has room.

### F5 — Do NOT add a `k ≥ 2` confirmation rule on the place conjunct
`NOTES.md` §risk-(d) names this as the intended next lever. **The traces refute it:** across the whole run,
19 legs produced *exactly one* `place == 1` and **zero legs produced two**. A `k=2` rule would have taken the
gate's 19 commits to 0 and SubtaskSR to the ORIENT-only floor of 0.25. If confirmation is wanted, confirm
with a **different question at the same step** (F4 next to F3), never the same question at a later step —
by then the body has moved 2 500 cm.

### F6 — Label the gate's calls in the trace
*(wherever the trace wrapper tags `kind`)* — emit `kind:"gate_identity"` / `"gate_place"`. All 595 gate calls
currently land as `kind:"perceive"` and were only recoverable here by substring-matching `_IDENTITY_SYS` /
`_JUNCTION_SYS` / `_FRONTAGE_SYS` out of the prompt. Cheap, and it makes every subsequent diagnosis cheaper.

### Keep, do not touch
The text-free `_IDENTITY_SYS` call. 39/42 = 93 % in-window ROAD recall against the parent's 8/34 = 24 % is
the largest measured perception win in this chain, and it is why ROAD and TURNING both moved this iter.
Never put `Subtask instruction:` back into an image-identity prompt.

---

## 6. Exhausted — do NOT repeat

- **Re-wording the leg-text arrival recognizer** (5 prior candidates, iters 1–5): evidence set, look
  schedule, criterion wording, in-burst sampling, confirmation counts. The problem was the text element
  itself, and it is now removed.
- **Putting the subtask instruction text into a two-image identity call** — measured 14 % vs 100 % offline,
  24 % vs 93 % in-window on this run. Settled.
- **`side_building_fraction` (`viewpoint_controller.py:160`) as a junction detector** — best mask feature
  AUC ≈ 0.72; fires 11.9 % near / 16.1 % far. Not a signal.
- **The depth predicate** (iter 5) — `obs['depth']` is per-frame normalised, so it has no absolute scale;
  fired 2× in 50 episodes.
- **The iter-6 ROAD face-scan** (`request_face_scan` → `scan_reach`) — 0/128 matches, 256 wasted substeps.
  Dead on ROAD. (FINISH is a *different* question — see F3.)
- **A blanket `allow_forward_commit` FINISH block** (iter 6) — it only relocated the premature finishes.
- **Junction wording A/B** — three variants were tried offline this iter and the best shipped; this run shows
  the model collapses all of them onto the crosswalk cue, so the *question type* is what must change (F4),
  not the phrasing.
- **Tightening `_FRONTAGE_SYS` further** — it is already at 0/12 from the travel heading, including at
  484 cm. The pose is the bug (F3), not the strictness.

*(All distances and waypoints in this document are oracle fields read for post-hoc analysis only. Every fix
above decides from `obs['rgb']` / `obs['object_mask']`, the per-subtask hint image, the static
`task_spec['subtasks']` text, the sanctioned `info['moving_distance']` odometry and cardinal
`info['agent']['agent_rotation']`, and the harness's own bookkeeping — no coordinate, no `gt.json`, no
`subtask_max_steps`.)*
