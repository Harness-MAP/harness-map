# Diagnosis — SimWorld-MMNav (`hard`, `search_hard_50`) iter 13 / `cand_r013_0_spatial-grounding` ("the FREE POSITION CHANNEL"), parent `cand_r9_0_spatial-grounding`

**Headline:** SR **0.00** (0/50) · SubtaskSR **0.550** — the chain's best by a wide margin (anchor 0.360,
parent r9 0.400, previous best r10 0.535) · per-position SR **[1.00, 0.58, 0.5778, 0.00]** (parent
`[1.00, 0.26, 0.2667, 0.00]`). Cost 145,863 tok/ep (chain max; parent 117,979), `over_budget: false`.
Read-audit clean — 15 paths, no forbidden field.

**The iteration's bet is confirmed, quantitatively and almost exactly on its own priced prediction.**
`NOTES.md` §why predicted `per_position_SR[1]` 0.26 → "0.45–0.60" and TURNING 0.2667 → "≈0.45"; the run
returned **0.58** and **0.5778**. The mechanism is visible in the traces: the geometric channel moved
**33 of 47 (70 %)** ROAD identity asks inside the 2,000 cm acceptance window, against the parent's measured
29 %, and the text-free identity call converted **30/33 = 91 %** of those. Of the 30 in-window licences,
**29 became ROAD passes** — a 97 % licence→pass conversion. The `_JUNCTION_SYS` VLM call was made **zero**
times in the whole run; the free mask channel answered its question for nothing. `REACH_COMMIT` (the
parent's clock-keyed channel) raised **0** commits; `GEO_REACH` raised **31**.

**Binding constraint:** the task is now gated entirely on the **FINISH leg, which is 0/31 attempts and has
never once emitted a `-1` in this run**, and the single proximate cause is that `_FRONTAGE_SYS` — the one
required position conjunct on a FINISH leg — answered `Here: 0` on **29/29** live asks, including on all
**3** frames where the text-free identity call had correctly recognised the destination from **484 / 640 /
1,144 cm**. It is not a weak question; it is a veto that cannot open, and `NOTES.md` shipped it knowing so.

---

## 1. What this harness does (mechanism)

`run_episode` (agent.py:174-735) is a single loop over 10 modules. Per iteration:

1. **stuck check** on Δ`info['moving_distance']` → `ActivePerception.should_emit` (cardinal burst on
   ORIENT, red-idle, anti-spin) → `ViewpointController.gate_burst` (holds the TURN `[rotate,-1]` burst
   until the lateral segmentation band opens). A non-empty burst **short-circuits perceive+plan**.
2. otherwise: **`RoadGeometry.update(obs, instruction, cur_idx)`** (agent.py:363, new this iter) →
   `Perceiver` (1 VLM call, 4 images: rgb / hint / object_mask / depth) → `Planner` (1 VLM call) →
   `CompletionChecker.verify` → if the planner proposed `-1`, `ArrivalGate.allow_planner_commit`; else
   `ArrivalGate.proactive_commit`.
3. on a commit, `ViewpointController.commit_pose_prefix` prepends one quarter-turn toward the named side
   (FINISH only) and the burst `[…, -1]` executes.

**The `-1` licence (`modules/arrival_gate.py`)** is a conjunction of an *identity* half and a *position*
half:

| leg | position conjunct | identity conjunct | ask schedule |
|---|---|---|---|
| ORIENT | — | — | cardinal burst commits directly, gate not consulted |
| ROAD (intersection-stop) | **`RoadGeometry.place_verdict`** — free, `object_mask` | `_IDENTITY_SYS` (2 images, no leg text) | **event-keyed**: the rising edge waives `K_REACH_FORWARD_STREAK=4` and `K_REACH_INTERVAL=5` |
| TURNING | — | — | `viewpoint_controller` turn-approach latch commits directly (every TURNING leg in this run is **exactly 5 substeps**) |
| FINISH | **`_FRONTAGE_SYS`** (1 image VLM call) | `_IDENTITY_SYS` | **clock-keyed**: fire at streak 4, then every 5 translations, hard cap `MAX_REACH_FIRES_PER_SUBTASK=12` |

`RoadGeometry` (`modules/road_geometry.py`) computes one statistic — the fraction of the forward corridor
(rows 0.45–0.70 H × cols 0.30–0.70 W) of `obs['object_mask']` carrying the roadway class `(0,127,255)` — and
runs a per-leg rising-edge machine above `ARRIVAL_FRAC = 0.03`, claiming at most
`MAX_CLAIMABLE_CROSSINGS = 2` edges past `GEO_MIN_TRANSLATIONS = 1`. It is tri-valued: `None` (not owned →
the VLM question runs), `True` (edge here), `False` (owned, not here → **free veto, no token spent**).

The environment's grading (`simworld_gym/envs/simple_world.py:_get_evaluation_of_subtask`) is L1 position
< `dist_threshold = 2000` for ROAD and FINISH, plus a ≤45° yaw check on ORIENT/TURNING/FINISH. **A rejected
`-1` terminates the episode immediately** (`step(-1)` → `terminated = True`), but it does **not** erase
already-banked legs.

---

## 2. What actually happened (causal chain, with evidence)

### 2A. The ROAD/TURNING win is real and is the geometric channel's (confirmed)

Every identity ask in the run, classified by prompt text and labelled offline with the true L1 distance to
that leg's graded waypoint (415 identity calls, 29 frontage calls, **0** junction calls, 1,650 perceive,
1,650 plan):

```
ROAD   identity by distance band          FINISH identity by distance band
     0-2,000   n= 33   Match=1: 30  (91%)      0-2,000    n=  8   Match=1:  3  (38%)
 2,000-4,000   n=  1   Match=1:  1             2,000-4,000 n= 15   Match=1:  0
16,000-32,000  n= 11   Match=1:  5  (45%)      4,000-8,000 n= 41   Match=1:  8  (20%)
32,000+        n=  2   Match=1:  0             8,000-16,000 n= 94  Match=1:  7  ( 7%)
                                               16,000-32,000 n=119 Match=1:  9  ( 8%)
                                               32,000+     n= 91   Match=1:  2  ( 2%)
```

30 in-window ROAD licences → 29 ROAD passes → 26 TURNING passes (90 % conditional conversion, matching the
parent's measured 92 %). `REACH_COMMIT = 0`, `GEO_REACH = 31` across the 50 verdicts: **the clock-keyed
schedule contributed nothing; the event-keyed one contributed everything.**

### 2B. The FINISH leg never commits — 0 `-1` in 31 attempts, and the gate is the reason

Per-episode commit counts equal the per-episode passed-leg counts in **all 50 episodes**. There is no
premature FINISH commit, no rejected FINISH commit — there is **no FINISH `-1` at all**. 26 episodes
banked `[T,T,T,F]`, 5 two-subtask episodes banked `[T,F]`; all 31 ended with the robot still walking
(last action `Move_Forward` in 26/31).

`_FRONTAGE_SYS` answered `Here: 0` on **29/29** live asks. All 29 Reasons are the same sentence:

```
d=    484  ep_20_18_task_dist_11_0_1  step 68  {"Here": 0, "Reason": "building facade is distant and not close enough to fill the frame"}
d=    640  ep_20_23_task_dist_11_0_1  step 82  {"Here": 0, "Reason": "buildings are distant and not close enough to be at the frontage"}
d=  1,144  ep_20_24_task_dist_21_0_1  step 129 {"Here": 0, "Reason": "buildings are distant and not close enough to be at the frontage"}
…  (26 more, 4,060 → 49,119 cm, identical wording)
```

Because `proactive_commit` requires `here == 1` on a FINISH leg (and `_place_verdict` returns the frontage
answer, not `None`, so the legacy degradation edge never runs), **the FINISH `-1` is structurally
unreachable.** 0/45 at position 3 is not noise at temp 1.0; it is a closed gate.

### 2C. The decisive episode — `ep_20_18_task_dist_11_0_1`

A two-subtask episode (`ORIENTATION → FINISH`). ORIENT passed. The full identity sweep on the FINISH leg,
with true L1 to the destination:

```
step  8  d=29,516  Match 0  "Different buildings and street layouts"
step 14  d=26,516  Match 0      step 44  d=11,516  Match 0
step 20  d=23,516  Match 0      step 50  d= 8,516  Match 0
step 26  d=20,516  Match 0      step 56  d= 5,516  Match 0
step 32  d=17,516  Match 0      step 62  d= 2,516  Match 0  "different buildings and perspectives"
step 38  d=14,516  Match 0
step 68  d=   484  Match 1  "same building wall and tree visible"        <-- CORRECT, IN WINDOW
step 68  d=   484  FRONTAGE Here 0  "building facade is distant…"        <-- VETO
step 74  d= 3,484  Match 0  "Different buildings and surroundings"
```

The perceiver on the same frame (step 68) says *"a prominent high gray and beige building featuring Chinese
characters on its facade located on the left… the agent is walking forward along a yellow-colored
sidewalk"* — i.e. the scene is read correctly. Closest approach was **L1 = 16 cm at env step 67**. The
robot then walked on for 81 more substeps and the episode ended at **L1 = 36,484 cm** by leaving the road
polygon. One `-1` at step 68 would have scored this episode **SR = 1.0**.

This is a textbook demonstration that the identity call is a *sharp, well-localised* arrival detector when
it is asked at the right place (0 at 2,516 → 1 at 484 → 0 at 3,484), and that the position conjunct bolted
in front of it is a constant-`False` function.

### 2D. The deeper recall problem: the harness barely looks while it is in the window

Per FINISH leg, measured: **125.9 substeps**, **40.1 perceive frames** (3.14 substeps/frame), **11.9
identity asks** (10.6 substeps/ask). The acceptance window is ±2,000 cm L1 ≈ 8 forward steps wide. So:

* **20 of 1,244** FINISH perceive frames (1.6 %) are inside the window;
* **8 of 368** FINISH identity asks (2.2 %) are inside the window — ≈ 0.8 asks per leg that enters it;
* the 12-ask cap at interval 5 is spent by translation ≈ 59, i.e. **the back half of a 120–200-substep
  FINISH leg gets zero arrival checks.** `ep_20_18_11`'s asks span steps 8→74 of a 147-substep leg; it got
  its in-window ask only because the destination happened to fall in the first half.

**10 of 31 FINISH legs physically enter the window** (min L1 = 16, 27, 105, 188, 265, 617, 723, 1,170,
1,216, 1,778 cm) and 24/31 come within 4,000 cm. Every one of them walks straight through and keeps going;
end-of-leg distances are 8,302–59,937 cm.

### 2E. How the FINISH legs actually die

```
19/31  left the road polygon        (simple_world._get_terminated: is_robot_inside_complex False)
 6/31  env wedge check              (20 consecutive identical positions)
 4/31  harness STUCK note
 2/31  subtask step budget exhausted
```

Subtask-step-budget utilisation on FINISH legs is **median 0.31** (min 0.17, max 1.11); **25/31 legs used
under half their budget.** The leg does not run out of steps — the robot runs out of *city*, walking
forward until the road network ends. There is therefore ≈ 3× step headroom available to spend on anything
that would improve the decision.

### 2F. The residual ROAD losses, exactly located

45 ROAD legs: **29 PASS · 6 fatal commit · 10 no-commit**.

* **5 of the 6 fatal commits** are the NOTES' own risk (b) realised: edge #1 of a long leg, identity
  answered `Match 1` at **17,578 / 17,780 / 20,712 / 20,801 / 20,920 cm**, and the episode died at substep
  4–23 with only ORIENT banked. In all five, this was the leg's **very first** identity ask, and the
  Reasons hedge — *"same street scene with similar buildings and perspective"*, *"same intersection with
  similar buildings and crosswalk"*. The NOTES priced the identity conjunct as a 39 % far-field filter; it
  measured **45 %** (5/11), which is not a filter. The 6th fatal (`ep_20_0_11`) committed at 2,548 cm, 548
  cm outside.
* **7 of the 10 no-commit legs entered the window** (min L1 = 41, 113, 149, 162, 246, 346, 1,107 cm):
  * 3 in-window identity = 0 (at 459, 1,107, 1,746 cm) — `ep_20_7_11`, `ep_20_11_11`, `ep_20_30_21`;
  * 1 killed by the **free side veto**: `ep_20_14_21`, identity = 1 at **113 cm**, `veto_side=1` in the
    ledger. `_named_side_has_structure` fired exactly **once** in 50 episodes and it was a false negative
    on a correct in-window commit — 1 measured loss, 0 measured saves;
  * 2 killed by `MAX_CLAIMABLE_CROSSINGS = 2`: `ep_20_0_21` (`GEO(above=29,below=25,edge=3,edge_over_cap=1)`,
    asks at 20,338 and 39,838 cm, min approach 162 cm) and `ep_20_21_21` (`edge_over_cap=1`, asks at
    18,851 and 39,851 cm, min approach 149 cm). On both long legs the **third** edge was the in-window one
    and the cap refused it.

### 2G. Confirmed fixed: F2 (the fail-closed cross-check cap)

The parent's `XCHECK_BLOCK → EMPTY_FALLBACK → ungated third `-1`` hole is gone. Fatal commits 15 → **9**,
and **all four FINISH deaths are eliminated** (there are now zero FINISH commits of any kind). The
signature still appears in the notes (48 `XCHECK_BLOCK`, 49 `EMPTY_FALLBACK`, `xcap_replay=1` in
`ep_20_17_11`) but it no longer leaks a commit. Note the interaction: the planner proposed `-1` on FINISH
legs **37** times and **0 of the 37 were in-window** (closest 4,403 cm), so blocking that channel on FINISH
is correct — and the planner's own `Match` field is **0/19** in-window on FINISH and 7/57 on ROAD, i.e. the
planner has no arrival signal at all.

### 2H. Cost

**368 of 415 identity calls (89 %) are spent on FINISH legs at a median distance of 18,693 cm**, on a leg
whose gate cannot open. That is the entire 117,979 → 145,863 tok/ep increase. The ROAD side got *cheaper*
(47 identity asks + 0 junction asks for 50 episodes, exactly as the NOTES predicted).

---

## 3. Failure breakdown

| leg | attempts | pass | dominant failure | rate |
|---|---:|---:|---|---|
| ORIENT | 50 | 50 | — | 1.00 |
| ROAD | 45 | 29 | far-field edge-#1 licence (5) · in-window no-commit (7, of which 3 identity-0, 2 edge-cap, 1 side-veto) | 0.644 (0.58 at position 1, which also carries 5 failed two-subtask FINISH legs) |
| TURNING | 29 reached | 26 | 3 yaw/position rejections at 1,139 / 4,680 / 5,422 cm | 0.897 conditional |
| FINISH | 31 | **0** | **`_FRONTAGE_SYS` = 0/29 (closed veto)** + in-window ask density 2.2 % | **0.00** |

Episode-level termination: 21 off-road-polygon · 9 rejected commit · 9 env wedge · 7 harness STUCK · 3
subtask budget · 1 parse. No socket deaths, no crashes. 23 `NO_ACTION_RETURNED`, 24/3,744 truncated calls
(0.6 %) — parsing is not a problem.

---

## 4. Binding constraint — classify it

**Primary: HARNESS-BUG, on the FINISH position conjunct.** `_FRONTAGE_SYS` is a required conjunct that
returns `False` unconditionally (29/29 live, including 3/3 in-window at 484–1,144 cm). A function that is
constant-`False` in a conjunction is not a weak signal to be improved — it is a bug, and removing it is a
code fix that recovers points immediately. The parent's diagnosis already said *"tightening `_FRONTAGE_SYS`
further — it is already at 0/12 from the travel heading, including at 484 cm. The pose is the bug, not the
strictness"*; this run reproduces that at 0/29 including **the same 484 cm**, and shipped it anyway.

**Secondary: HARNESS-BUG, on ask density.** The FINISH schedule is clock-keyed and capped at 12 fires /
interval 5, so it covers only the first ≈ 59 translations of a 126-substep leg and lands 2.2 % of its asks
in the window. Only 20 of 1,244 FINISH frames are even *observed* inside the window. Any detector's
per-frame accuracy is irrelevant at that sampling rate.

**Not VLM-CEILING on recognition, but VLM-CEILING on the *position* question.** The identity half works:
91 % in-window on ROAD, and on FINISH it produced a clean 0…0→1→0 sweep across `ep_20_18_11`. What the
frozen model cannot do from the travel heading is answer *"am I standing at this building's frontage?"* —
and **there is no free geometric substitute.** I probed four candidates offline on all 1,244 archived
FINISH `object_mask` / `depth` frames, labelled by true L1 distance, and all four are dead (§6). So the fix
must **scaffold the decision with a better observation (pose) and a better schedule**, not re-word the
question.

**Not TASK-GAP on steering.** 10/31 FINISH legs and 7/10 failing ROAD legs *do* reach their graded
waypoint; the controller gets the body to the right place and the gate refuses to stop it. The remaining
7/31 FINISH legs that never come within 4,000 cm are a genuine steering gap, but they are the minority and
not where the next iteration's points are.

---

## 5. Concrete changes to try (the gradient — ranked by leverage)

### F1 — Delete `_FRONTAGE_SYS` from the FINISH licence and replace it with a *free history* conjunct
*(`modules/arrival_gate.py:_place_verdict` — the `LEG_FINISH` branch; `proactive_commit`)*

Two edits that must ship together, because a FINISH episode gets **exactly one `-1`** (a rejection
terminates it), so this is a one-shot stopping problem, not a recall problem:

1. On a FINISH leg, `_place_verdict` returns `None` (no VLM position call) instead of the frontage answer.
2. Replace it with a free per-leg guard inside `ArrivalGate`: **a FINISH `Match=1` may licence only if it is
   preceded by ≥ `m` consecutive `Match=0` answers on the same leg.** Counterfactual over this run's own
   368 FINISH identity answers, "commit on the first qualifying `Match=1`":

   ```
   m=0  in-window 2  far-field 11  no-commit 18
   m=1  in-window 2  far-field 10  no-commit 19
   m=4  in-window 2  far-field  7  no-commit 22
   m=6  in-window 2  far-field  6  no-commit 23      <-- recall is flat in m; precision is not
   ```

   `m ≈ 4–6` is free (it is bookkeeping, not a call) and strictly dominates `m=0`. 4 of the 11 far-field
   first-positives are ask #1 of the leg, which `m ≥ 1` removes outright.

*Predicted:* `per_position_SR[3]` 0.00 → **≈ 0.044 (2/45)**, SR 0.00 → **≈ 0.04 (2/50)** — `ep_20_18_11`
(484 cm) and `ep_20_24_21` (1,144 cm) convert; SubtaskSR 0.550 → ≈ 0.565. Small, but it is the first
non-zero SR this lineage has had, and it is the *floor*, not the estimate, because it is computed by
replaying the actual answer strings.
*Why the downside is ≈ zero, stated precisely:* the harness currently treats FINISH as the most dangerous
place to commit (`xcap_closed` blocks it, the frontage veto never opens, the cap fails closed). **That has
the asymmetry backwards.** A rejected `-1` does not erase banked legs — `subtask_passed[pre_idx]` is
already `True` — and the FINISH leg is worth 0.00 today, so a premature FINISH commit forfeits only a
continuation that in 31/31 observed legs produced nothing. The *only* real cost of a far-field FINISH
commit is the one-shot opportunity, which is exactly what `m` protects. The parent's "9 fatal FINISH
commits discarded episodes that had banked three legs" reasoning double-counted legs that were already
banked.

### F2 — Ask the FINISH arrival question **from the faced pose** (the chain's one untested hypothesis)
*(`modules/viewpoint_controller.py:request_face_scan` / `face_code` / `restore_code`, left unreferenced
since iter 9; call site in `arrival_gate._place_verdict` / `proactive_commit`, FINISH only)*

This was the parent's F3 and was **not taken** (NOTES risk (d): *"left for the next iter so it is not
confounded"*). It is now the highest-leverage item with a measured prior:

* iter 9's own deletion note records the faced pose at **9 matches in 35 FINISH calls (26 %)** against this
  run's **29/368 = 7.9 %** from the travel heading. The faced pose is the only configuration in this
  chain's history that has ever produced FINISH identity recall at rate. iter 9 misread those 9 as "fatal
  commits"; they are *recall at the wrong distances*, which is a gating problem (F1's `m`), not a
  recognition failure.
* The hint image **is** a head-on facade photograph; the robot approaches along the street with the
  building at ~90°. The two images share almost no viewpoint — that is the whole reason FINISH identity is
  38 % in-window where ROAD is 91 %.
* **The turn is on the critical path anyway.** Measured over all 50 `gt.json`: the FINISH graded yaw differs
  from the previous leg's graded yaw by **90° or 270° in 50/50 episodes**. `commit_pose_prefix` already
  prepends that quarter-turn — so asking from the faced pose costs *nothing extra at commit time* and makes
  the committed pose legal by construction instead of by a blind prefix.
* **Cost is affordable and measured:** 2 substeps per look (face + restore). At one look per scheduled ask
  (~12/leg) that is 24 substeps against a budget utilisation of **median 0.31** — 25/31 legs use under half
  their subtask step budget, and 19/31 die by leaving the map, not by running out of steps.

*Shape:* keep the forward-view identity call as the cheap pre-filter; when it returns 1 **and** the `m`
guard is satisfied, face the named side, re-ask identity from that pose, and commit from there if it holds;
restore the heading if not. That makes the expensive turn conditional on a free-ish positive.
*Predicted:* in-window FINISH identity recall 0.38 → toward the ROAD-measured 0.91, and the 10 legs that
physically enter the window plus the 14 that reach 4,000 cm become live: `per_position_SR[3]` → **0.10–0.20**,
SR → **0.08–0.18**. Must be measured first by replaying the archived FINISH frames — but note the archive
has **no faced frames**, so this one is only priceable live; that is an argument for shipping it behind F1
(which is priceable offline) rather than instead of it.

### F3 — Uncap and re-pace the FINISH ask schedule
*(`modules/completion_checker.py:MAX_REACH_FIRES_PER_SUBTASK`; `arrival_gate.proactive_commit`;
`agent.py` burst materialisation ≈ `:600-640`)*

The recall ceiling in F1 is 2 because only 8 in-window asks exist. Two sub-levers, both cheap:

* **(a) Drop the 12-fire cap on FINISH legs** (keep interval 5). Measured: the cap stops asking at
  translation ≈ 59 on legs of 52–199 translations, so the destination half of a long leg is unchecked.
  Uncapping takes FINISH asks ~12 → ~25/leg, in-window asks 8 → ≈ 20, and *roughly triples* the number of
  legs that get any in-window ask. Cost ≈ +400 identity calls / 50 eps ≈ +8k tok/ep against a `false`
  `over_budget` flag at 145,863.
* **(b) Truncate the action burst to ≤ 2 translations per loop iteration on FINISH legs** — the pacing half
  of the parent's F1, still untaken. Measured 3.14 substeps per perceive frame today; halving it doubles
  in-window frames from 20 to ≈ 40 for **zero extra env substeps** (the forwards still happen; the loop
  re-decides between them).

*Predicted:* F1+F3 together ≈ 4–7 FINISH legs, `per_position_SR[3]` → **0.09–0.16**, SR → **0.08–0.14**.
*Interaction warning:* (b) changes `RoadGeometry`'s sampling cadence, and the ROAD operating point
(`ARRIVAL_FRAC = 0.03`, rising-edge) was measured at loop cadence. **Scope (b) to FINISH legs only**, or
re-measure the ROAD edge statistics at the new cadence first — the NOTES' risk (c) is still valid.

### F4 — ROAD: suppress the far-field edge-#1 licence (5 measured episodes)
*(`modules/road_geometry.py:RoadGeometry.update` / `place_verdict`)*

5 of 6 ROAD fatals are identity `Match=1` at 17.5–20.9 km on the leg's first geometric edge, raised at
`GEO_MIN_TRANSLATIONS = 1`. Options, in order of how well the evidence supports them:

* **k-of-n self-consistency on the identity call, ROAD only.** Far-field positives measure 45 % (5/11),
  in-window 91 % (30/33); requiring 2/2 samples predicts far-field 0.45² ≈ 0.20 and in-window 0.91² ≈ 0.83,
  i.e. fatals 5 → ≈ 2 and in-window licences 30 → ≈ 27. Net ≈ +2 legs for +47 calls. Honest caveat: this is
  a coin-flip-independence estimate, and at temp 1.0 the two draws may be correlated — it should be
  measured by re-sampling the archived far-field and in-window frames before shipping.
* **Re-measure "arming"** (require a closed reading before edge #1). The NOTES rejected it on first-edge
  placement (24/45 vs 25/45) — a metric blind to the 5 fatal far-field commits it would remove. Re-score it
  against **leg outcome**, not edge placement.
* **Do NOT** use the "`m` prior zeros" guard from F1 on ROAD. Measured: `m ≥ 1` destroys **26 of 30**
  in-window ROAD licences, because the geometric channel already restricts ROAD to ~1 ask per leg. The
  guard is FINISH-specific.

*Predicted:* `per_position_SR[1]` 0.58 → ≈ 0.64–0.68, carrying TURNING at the measured 90 % conditional
rate → `[2]` 0.578 → ≈ 0.62; SubtaskSR 0.550 → ≈ 0.59.

### F5 — Two one-line ROAD fixes, each with a named episode
*(`modules/road_geometry.py:MAX_CLAIMABLE_CROSSINGS`; `modules/arrival_gate.py:_named_side_has_structure`)*

* **`MAX_CLAIMABLE_CROSSINGS` 2 → 3.** `ep_20_0_21` and `ep_20_21_21` both show `edge_over_cap=1` with
  their two asks at 20,338/39,838 and 18,851/39,851 cm and minimum approaches of **162** and **149** cm —
  the refused third edge was the in-window one. +2 legs. (The NOTES already call 3 "the obvious next
  value"; it interacts with F4, so change one at a time.)
* **Delete or fail-open `_named_side_has_structure`.** It fired exactly once in 50 episodes
  (`ep_20_14_21`, `veto_side=1`) and killed a correct in-window commit at **113 cm**. 1 measured loss, 0
  measured saves.

### F6 — Make the harness notice a wedge, and notice an overshoot
*(`agent.py:316` `forward_count = sum(1 for a in chosen_actions if a == 0)`)*

`chosen_actions` holds only the **current burst** and is reset to `[]` on every commit, so the stuck
detector fires only when the latest burst happened to contain > 3 forwards. Measured consequence: **9
episodes ended on the env's own 20-identical-positions check** and **21 by leaving the road polygon**, with
`building_collision` climbing to 18–20 while `moving_distance` sat frozen (e.g. `ep_20_9_11`, last three
env steps identical at `moving_distance = 40,258.239`) — the harness never noticed. Replace with a
persistent counter of translations-since-the-last-odometry-gain (`info['moving_distance']` is sanctioned),
and on a wedge **change tactic** (strafe / rotate) rather than ending. Free, and it is a precondition for
F3(b) — a denser loop with a broken wedge detector just burns more substeps into the same wall.
Related and cheap: an **overshoot flag** — once a FINISH leg has seen one `Match=1` and a later ask returns
0, the robot has passed the building (visible verbatim in `ep_20_18_11`: `…0 @2,516 → 1 @484 → 0 @3,484`).
ROAD and FINISH are graded on position alone plus yaw, so backing up is legal.

### F7 — Label the gate's calls in the trace (re-raised from the parent's F6, still not done)
All 415 identity and 29 frontage calls are logged as `kind: "perceive"`; this diagnosis recovered them by
substring-matching `_IDENTITY_SYS` / `_FRONTAGE_SYS` out of the prompt for the second iteration running.
Emit `kind: "gate_identity"` / `"gate_place"`. Pure cost reduction on every future diagnosis.

### Keep, do not touch
* **The text-free `_IDENTITY_SYS` call** — 91 % in-window ROAD recall, and the only arrival signal anywhere
  in this harness. Never put leg text back into it.
* **`RoadGeometry` on intersection-stop ROAD legs**, and its routing (`leg_type == LEG_ROAD` ∧ the word
  *intersection*). 70 % of asks moved in-window, 97 % licence→pass. `ARRIVAL_FRAC = 0.03` and the band
  geometry are measured-flat plateaus — do not re-tune them.
* **The ORIENT cardinal burst** (50/50) and the **TURNING approach latch** (90 % conditional, exactly 5
  substeps per leg, zero gate calls).
* **The fail-closed cross-check cap** (F2 from the parent) — it did its job.

---

## 6. Exhausted — do NOT repeat

Carried forward from the parent's §6, all still valid: re-wording the leg-text arrival recognizer (iters
1–5) · putting instruction text into the two-image identity call · `side_building_fraction` as a junction
detector · the absolute-depth predicate (`obs['depth']` is per-frame normalised) · the **ROAD** face-scan
(0/128) · a blanket `allow_forward_commit` FINISH block · `_JUNCTION_SYS` wording A/B · `k ≥ 2`
confirmations of the *same* question at a *later* step.

**Newly closed this iter — four free geometric channels probed offline on all 1,244 archived FINISH
`object_mask`/`depth` frames, labelled by true L1 distance. All four are dead; do not spend an iteration
on any of them:**

| candidate FINISH position cue | <2,000 cm | 2–4 k | 4–8 k | 8–16 k | >16 k | verdict |
|---|---:|---:|---:|---:|---:|---|
| named-side lateral **building-class** fraction (median) | 0.500 | 0.397 | 0.344 | — | 0.425 | **flat** — buildings line the whole block |
| named-side **depth-near** fraction (median) | 0.445 | 0.433 | 0.433 | 0.433 | 0.433 | **flat** |
| named-side building **base row** (median, normalised) | 0.532 | 0.510 | 0.507 | 0.510 | 0.507 | **flat** |
| `RoadGeometry` roadway-corridor detector, fire rate | **30 %** | 65 % | 76 % | 69 % | 73 % | **anti-discriminative** |

The fourth row is the important one: the obvious next move — *"route `RoadGeometry` to FINISH legs too"* —
is measured **backwards**. On a FINISH leg the robot walks *in* the roadway (median band fraction 0.23–0.68)
and near the destination it is on the frontage (median **0.019**, below `ARRIVAL_FRAC`). The *inverted*
falling-edge form was swept too (thresholds 0.02/0.03/0.05/0.10 × min-run 1/2) and raises an in-window edge
on **0–1 of 31** legs, because only 20 of 1,244 FINISH frames are sampled in-window at all — the per-frame
polarity cannot be rescued at that sampling rate (which is F3).

Also closed:
* **The "`m` prior zeros" guard on ROAD** — kills 26/30 in-window licences (keep it FINISH-only).
* **The planner as a FINISH arrival channel** — `Match` is **0/19** in-window on FINISH, and 0/37 planner
  `-1` proposals on FINISH were in-window (closest 4,403 cm). Do not try to recover FINISH through the
  planner or `CompletionChecker.verify`.
* **A `Reason`-text precision filter on identity.** The far-field ROAD positives hedge with *"similar"*
  (*"same street scene with similar buildings and perspective"*) where in-window ones say *"identical"* /
  *"same building"* — but `ep_20_17_11`'s correct 74 cm match also says *"similar buildings and
  perspective"*. Suggestive, not separable; do not build a licence on it.

---

*Anti-cheat note: every distance, waypoint and `gt.json` field in this document was read **offline, after
the rollout**, solely to label archived frames and answers — the standing practice of this chain's
diagnoses. Every fix proposed above decides from `obs['rgb']` / `obs['object_mask']` / `obs['depth']`, the
per-subtask `visual_hint` image, the static `task_spec['subtasks']` text, the sanctioned
`info['moving_distance']` odometry and cardinal `info['agent']['agent_rotation']`, the structural
`env.unwrapped.current_subtask`, and the harness's own per-leg bookkeeping. No coordinate, no
`agent_location` / `target_location`, no `ideal_path`, no `current_instruction`, no analog `_agent_rotation`,
no `subtask_max_steps`. The candidate's own `read_audit.json` is clean at 15 paths.*
