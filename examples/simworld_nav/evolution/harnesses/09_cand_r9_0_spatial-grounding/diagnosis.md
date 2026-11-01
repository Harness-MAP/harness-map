# Diagnosis — SimWorld-MMNav (`hard`, search_hard_50) iter 6 / `cand_r006_0_spatial-grounding` ("the POSE node"), parent `anchor_nav`

**Headline:** SR **0.00** (0/50) · SubtaskSR **0.370** (anchor 0.360 — the chain's best so far, but by
+2 net legs out of 190) · per-position SR **[1.00, 0.20, 0.20, 0.00]** (anchor `[1.00, 0.24, 0.1111, 0.00]`).
Re-split by leg *type*: ORIENT **50/50**, ROAD **10/45**, TURNING **9/45** (anchor 5/45), FINISH **0/50**.
Held-out `val_hard_10` agrees (SR 0.00, SubtaskSR 0.30, `[1.0, 0.1, 0.1, 0.0]`), so the +0.010 is inside noise
and the *per-leg* movement is the only stable signal. Read-audit clean (`forbidden_reads: []`, 15 paths).

**Binding constraint (one sentence):** exactly one of the three pose levers paid — the turn approach, and
only through its *constant* `APPROACH_CAP_FORWARDS = 3` — while the other two re-routed the SAME
visibility-criterion recognizer through a new pose and so inherited its behaviour unchanged: on ROAD legs the
faced look scored **0 matches in 128 calls** (pure cost: 256 substeps), and on FINISH legs it scored 9
matches in 35 calls of which **9/9 were fatal premature commits** at 8 666–58 789 cm. The harness's
*arrival question*, not the *pose it is asked from*, is what is broken — and a probe on this run's own frames
shows the biggest single lever on that question is **a prompt-composition bug in the recognizer**, not the model.

---

## 1. What this harness does (mechanism)

`harness.py:run_episode` is the anchor's 7-node loop plus one new node, `ViewpointController`
(`modules/viewpoint_controller.py`, 436 lines, no VLM call), wired at four sites:

```
SpatialMemory → PromptAssembler → Perceiver(VLM) → Planner(VLM)
      ↘ ActivePerception.should_emit → [A] viewpoint.gate_burst → burst short-circuit
      ↘ CompletionChecker.verify → cross_check_planner_commit
      ↘ [B] viewpoint.allow_forward_commit → CompletionChecker.proactive_commit_check
      ↘ [C] viewpoint.request_face_scan → _scan_step(face) → CompletionChecker.scan_reach → ±commit
      ↘ [D] viewpoint.commit_pose_prefix at the single `-1` materialisation site
```

* **[A] `harness.py:329` → `gate_burst`** — on a `TURN` leg the `[rotate, -1]` burst from
  `active_perception` is *captured*, single forwards are emitted instead, and the burst is released when
  `side_building_fraction(obs, turn_side) ≤ 0.10` past `APPROACH_FLOOR_FORWARDS = 2`, else unconditionally at
  `APPROACH_CAP_FORWARDS = 3`.
* **[B] `harness.py:466` → `allow_forward_commit`** — returns `False` on a `FINISH` leg, so
  `proactive_commit_check` (the forward-facing reach) may no longer inject a `-1` there.
* **[C] `harness.py:518-553` → `request_face_scan`** — on `ROAD`/`FINISH` legs with a named side, at cadence
  (`K_FACE_STREAK=4`, `K_FACE_INTERVAL=4`, `MAX_FACE_SCANS_PER_LEG=4`), yaw 90° toward the named side, run
  `completion.scan_reach` on the new frame, commit on `Match==1`; on `ROAD` restore the heading first, on
  `FINISH` keep the faced pose.
* **[D] `commit_pose_prefix`** — prepend the quarter-turn on a `FINISH` planner `-1`.

Everything that *judges appearance* is byte-for-byte the anchor's: `scan_reach` and
`proactive_commit_check` both call `completion_checker._binary_reach_vlm_call` (`:530`), whose system prompt is
*"1 if the current view shows the same landmark / place as the reference (the same building, intersection, or
scene is **clearly visible and prominent** in the current view)"*, temperature 0.0, `max_tokens=2048`.

Env facts that price every decision (already established in the chain, restated because §2 counts against
them): a **rejected `-1` sets `terminated = True`** — a wrong commit is a lost *episode*, not a lost leg
(`simple_world.py:551-571`); each leg carries `subtask_max_steps` = 10 (ORIENTATION) / 50 (TURNING) /
`int(L1(leg_start, waypoint)/100)` (ROAD, FINISH), charged on **every** action including in-place rotations;
and leaving the road polygon or repeating 20 identical positions also terminates.

---

## 2. What actually happened (causal chain, with evidence)

### 2A. Lever [A] worked — but the adaptive half of it never ran (0/10)

All 10 TURNING legs reached in the run executed **exactly the same sequence**:

```
ep_20_0_11  leg2 FFFL-1 PASS     ep_20_3_11  leg2 FFFR-1 PASS
ep_20_11_21 leg2 FFFL-1 PASS     ep_20_4_11  leg2 FFFR-1 PASS
ep_20_16_11 leg2 FFFR-1 PASS     ep_20_8_21  leg2 FFFL-1 PASS
ep_20_23_11 leg2 FFFR-1 PASS     ep_20_9_11  leg2 FFFL-1 PASS
ep_20_24_21 leg2 FFFL-1 PASS     ep_20_19_21 leg2 FFFR-1 FAIL
```

`FFF` = 3 forwards, i.e. **every release was at `APPROACH_CAP_FORWARDS = 3`; the segmentation cue released
early 0/10 times.** TURNING went 5/45 → **9/45** (9/10 conditional on reaching the leg, vs the anchor's ~42%),
so the lever is real — but the working mechanism is a *constant 3 forwards*, not the adaptive cue.

I measured the cue independently on 329 ROAD-leg `object_mask` frames from this run (recomputing
`side_building_fraction` exactly as `viewpoint_controller.py:160` does): `min(left,right) ≤ 0.10` fires on
**11.9 %** of frames within 2000 cm of the ROAD waypoint and **16.1 %** of frames beyond 12 000 cm — no
signal (best single mask feature over 5 classes × 6 bands peaked at AUC ≈ 0.72; the lateral *road*-class
fraction saturates at 1.00 everywhere because the robot stands on the road). **`side_building_fraction` is not
a junction detector.** Its appearance of success in [A] is entirely the cap.

The one failure is an **overshoot of the constant**: `ep_20_19_21` entered leg 2 at `y = 1678` — already
inside the box `y ∈ [700, 2700]` — and the 3 mandatory forwards walked it to `y = 178`, outside on the cross
axis (`x` was in range). The box is only 2000 wide on that axis; 3 × 500 cm cannot be right when the leg
starts inside it.

### 2B. Lever [C] on ROAD legs: 128 calls, **0 matches**, 256 substeps burned

| reach-call population (from `vlm_trajectory.jsonl`, 395 two-image calls) | n | Match=1 |
|---|---|---|
| forward-facing (`proactive_commit_check` / xcheck), ROAD | 230 | 11 |
| **faced (`scan_reach`), ROAD** | **128** | **0** |
| faced (`scan_reach`), FINISH | 35 | 9 |
| forward-facing, FINISH (xcheck only — [B] blocked the rest) | 2 | 0 |

Restricted to frames **inside the 2000 cm window of the leg's own waypoint**: the forward view fires
**8/34 (24 %)**; the faced view fires **0/20 (0 %)**. The premise of lever [C] — *"the hint is a robot view
captured facing the building, so a match needs the building head-on"* — is therefore **refuted by the data
it was built to fix**: turning to face the side made recognition strictly worse, not better. This is why
ROAD slipped 0.24 → 0.20 while TURNING doubled.

The cost is exact: 45 ROAD legs spent **2 586 substeps**, of which 2 297 forwards, 273 rotations, and
**256 (10 %) were face-scan rotation pairs** — 12–18 % of the leg budget on tight legs. Five of the 13
budget-death legs died at `used == budget` **exactly** (`ep_20_17_11` 44/44, `ep_20_22_11` 48/48,
`ep_20_18_21` 61/61, `ep_20_13_21` 68/68, `ep_20_30_21` 27/27) with 4–8 of those substeps spent on face
scans that returned 0.

### 2C. Lever [B]+[C] on FINISH: the fatal-commit path was moved, not removed

[B] did its job — forward-facing FINISH commits went 7 → **0** (only 2 xcheck calls, both Match=0). But
`scan_reach` replaced them one-for-one and worse. All 9 FINISH matches committed, **all 9 failed, all 9 ended
the episode**:

```
ep         leg  dist to destination at the -1   the model's own Reason
ep_20_10_21  1   58 789   "The building with striped awnings is visible in both views"
ep_20_5_11   1   32 586   "...clearly visible in the current view"
ep_20_24_21  3   28 440   "...is visible in both views."
ep_20_3_11   3   27 243   "...is visible ahead in both views."
ep_20_16_11  3   22 206   "...is visible ahead in the current view"
ep_20_21_11  1   21 488   "same building with large windows and trees in front"
ep_20_4_11   3   20 075   "The building with commercial signage is visible in both views"
ep_20_18_11  1   18 016   "...shows the same gray and beige building..."
ep_20_9_11   3    8 666   "...is visible in both views."
```

Threshold is 2000 cm. **Nine of nine reasons assert visibility; not one asserts arrival.** The criterion
string is literally *"clearly visible and prominent"* — a 90° yaw changes which facade fills the frame, it
does not add a distance predicate. Five of these were episodes that had already banked 3 legs and had spent
**8–18 substeps of a 173–333 substep budget** (`ep_20_3_11` 8/307, `ep_20_4_11` 9/262, `ep_20_16_11` 8/272,
`ep_20_24_21` 8/311, `ep_20_9_11` 18/173) — the harness threw away ~30× the budget it had used.

`completion_checker.scan_reach`'s own docstring (`:807-824`) states *"the harness commits (-1) only on an
explicit 1; on 0/None it restores the travel heading and walks on, so a wrong look **NEVER** produces a false
(fatal) commit."* That was true only while the trigger was dead. Once [C] switched it on, it became the
harness's single largest source of episode death.

### 2D. Every commit in the run fired on the FIRST Match=1 ever seen on that leg

Verdict sequences per leg, for the 20 legs that ever saw a 1 (`0`/`1` = successive reach verdicts):

```
seq=1  seq=001  seq=0001  seq=00001  seq=0000001  seq=0000000001  seq=00000000001  ...
```
**`11` never appears — 0/20.** Not because a second look disagreed, but because there is never a second look:
the first `1` commits. The effective confirmation count on the lethal action is **1 single frame, 1 single
call**, on every leg but ORIENT.

### 2E. Why the 50 episodes actually died

| cause | n | note |
|---|---|---|
| **failed `-1` → instant episode death** | **17** | 9 `SIDE_SCAN_COMMIT` (new), 5 planner `-1` past xcheck, 3 forward `REACH_COMMIT` |
| per-leg `subtask_max_steps` exhausted | 13 | all on ROAD; `used` = budget + 0…5 |
| walked outside the road polygon | 10 | final position **1 977–21 085 cm** from the nearest road centreline vs a 1 700 cm spawn offset (measured against each task's `roads.json`) |
| harness `STUCK` break | 6 | |
| 20 identical positions (wedged) | 4 | |

**0 episodes ran out of global `max_steps`** — the budget was never the ceiling; the harness's own commit
decisions and its overshoot were. And the position evidence says navigation is *not* the bottleneck on ROAD:
the body comes within 2000 cm of the ROAD waypoint on **~35/45 legs (median closest approach 183 cm, min 6 cm)**
and commits on **10**. Same story on FINISH: 4 of 14 FINISH legs entered the destination window
(`ep_20_4_21` reached **27 cm**, `ep_20_23_11` 688 cm, `ep_20_8_21` 925 cm, `ep_20_11_21` 2 036 cm) and none
committed there — `ep_20_4_21` then committed 8 973 cm later.

Two pre-existing bugs that the counts above depend on, both still live:
* `harness.py:~293` `forward_count = sum(1 for a in chosen_actions if a == 0)` counts only the **current
  burst**, so `> 3` is unreachable when plans are 1 action long. `ep_20_20_21` sat wedged against a building
  (`building=1` from step 0, `moving_distance` frozen at 136 for 11 steps, 9 human collisions) and `STUCK`
  never fired; it died on a failed `-1` instead.
* Nothing records that a window was entered and left (`spatial_memory`), which is exactly `ep_20_4_21`.

---

## 3. Failure breakdown

| leg | pass | dominant failure | rate |
|---|---|---|---|
| ORIENT | 50/50 | — | closed-form cardinal burst; do not touch |
| ROAD | **10/45** | reach-but-no-commit (recognizer recall) → overshoot → budget/out-of-bounds death | ~25/45 legs pass through the window uncommitted |
| TURNING | **9/45** | only 10/45 episodes *arrive* at the leg; 9/10 of those pass | 1 failure = constant-3 overshoot of a 2000-wide box axis |
| FINISH | **0/50** | premature commit from the visibility criterion (9) or never committing after entering the window (4) | 9 fatal fires / 35 faced calls |

Cross-check against the numeric pre-pass: `static_collision_mean` 1.78 → **3.42** and
`red_light_violation_mean` 8.0 → **11.74** vs the anchor — the extra distance walked
(`moving_distance_mean` 29 912) is a *symptom* of the missing ROAD stop, not an independent problem.
`distance_progress_mean` 0.145 → 0.161, cost 46.3k tok/ep, `over_budget: false`.

---

## 4. Binding constraint — classify it

**Primary: HARNESS-BUG (prompt composition), and it is recoverable in code.**

I re-ran the recognizer against this run's own saved frames on the live `Qwen3-VL-30B-A3B-Instruct`
(`:8000`), temperature 0.0, 14 ROAD frames inside 2000 cm ("NEAR") and 14 beyond 8000 cm ("FAR"):

| criterion asked of the frozen model | recall (NEAR) | false fire (FAR) |
|---|---|---|
| **A — the live `_binary_reach_vlm_call`, reproduced exactly** (system prompt + `"Subtask instruction: <the real 40-word ROAD text>"` + current view + hint) | **2/14 (14 %)** | 0/14 |
| A with the **instruction-text block removed** (images only) | **14/14 (100 %)** | 6/14 (43 %) |
| C — single image, **no hint**: *"is the robot right now inside a junction — has the facade line ended, is a crosswalk at/under it, are cross lanes open beside it?"* | 6/14 (43 %) | 1/14 (7 %) |
| **A-without-text AND C** (same 28 frames, verdicts intersected) | **6/14 (43 %)** | **0/14** |
| B — proximity/apparent-scale wording ("facade must fill a comparable portion of the frame") | 0/14 | 0/14 |
| D — one call asserting the conjunction (building present AND in junction) | 0/14 | 0/14 |

The dominant effect in the table is **the instruction text**: prefixing the real ROAD instruction collapses
recall 100 % → 14 %, and the rejection reasons switch to *"Different building colors and lighting
conditions"* / *"Different building features"* — the model stops comparing the two images and starts
adjudicating a 40-word verbal description against the frame, which it loses. That is a harness bug at
`completion_checker.py:571` (`instr_text` injected into a call whose job is image identity), not a model
limit. Note also that recall/precision live on **opposite** sides of that one text block, and that asking
the model to hold both criteria in *one* call destroys it (D = 0/14) while the **conjunction of two separate
cheap calls** keeps recall and kills the false fires (0/14 vs the live ~5 % fire rate).

Sample is 28 frames, one seed, temperature 0, drawn from the legs this harness actually visited — treat
"0/14" as "low", not "zero". But the 14 % vs 100 % gap is far larger than that sample can explain away, and
it reproduces exactly across two independent runs of the same two prompts.

**Secondary: TASK-GAP on the commit action.** A `-1` is irreversible and lethal (17/50 deaths) yet is taken
on a single frame from a single call (§2D). No amount of recognizer tuning fixes a 1-sample decision with an
episode-sized downside.

**Not VLM-CEILING on pose.** The pose hypothesis that motivated this candidate is now measured and
falsified: faced 0/20 in-window vs forward 8/34 in-window (§2B). Do not spend another iteration on *where the
camera points* for arrival.

---

## 5. Concrete changes to try (ranked by leverage)

**(1) `modules/completion_checker.py:571` — drop `instr_text` from `_binary_reach_vlm_call`'s user content.**
Delete the `{"type": "text", "text": f"Subtask instruction: {instr_text}"}` element (keep the system prompt
and both images). *Predicted:* in-window ROAD recall 24 % → ~60-100 %, out-of-window fire ~5 % → ~40 %.
**Ship it only together with (2)** — alone it converts reach-but-no-commit into fatal-commit. *Why:* measured,
§4 row 2; the text makes the model adjudicate prose instead of matching images.

**(2) `modules/completion_checker.py` — add a second, independent `_junction_vlm_call(obs)` and require the
conjunction before any ROAD `-1`.** One image (`obs['rgb']`), no hint, no instruction text, the §4 row-3
prompt, `temperature=0.0`, `max_tokens=512`. Gate: `commit ⇔ image_match(no text) AND in_junction`. Wire it
in `harness.py` at the `proactive_commit_check` site and inside `verify`'s planner-`-1` path.
*Predicted:* ROAD 10/45 → ~19/45 at a *lower* false-fire rate than today (measured 6/14 recall, 0/14 fire on
the same frames) ⇒ SubtaskSR +0.045, and ~9 more episodes reach TURNING, where §2A shows 90 % conversion
⇒ a further +0.04. Two extra VLM calls per arrival check; cost is 46.3k tok/ep against no budget pressure.
*Why:* the ROAD leg's grading point **is** the intersection ("stop at the intersection with X on your left");
the harness has never asked about the junction, only about the building.

**(3) `harness.py:518-553` + `modules/viewpoint_controller.py:request_face_scan` — delete the face-scan
commit path.** Keep `gate_burst`; make `request_face_scan` return `None` unconditionally (or remove the
block). *Predicted:* removes 9 of the 17 fatal commits outright and returns 256 substeps (10 % of ROAD
substeps, 12–18 % of budget on tight legs) for **0** forgone matches — it has never once committed a leg
correctly. FINISH goes from "commits early and dies" to "keeps walking with 250–330 substeps in hand", which
is the only state from which (2)/(5) can score. *Why:* §2B, §2C — 0/128 on ROAD, 9/9 fatal on FINISH.

**(4) `modules/completion_checker.py` + `harness.py` — make `-1` require k=2 confirmations at ≥1 forward
apart, on every leg except ORIENT.** Hold a per-leg `pending_commit` when the gate first returns true, emit
one forward, re-ask, and commit only on a second true; drop the pending state on a false.
*Predicted:* would have blocked **all 9** fatal `SIDE_SCAN_COMMIT`s and is untested-by-construction today
(§2D: `11` appears 0/20 times only because the first `1` commits). Cost: one extra forward + one call per
candidate commit; risk: a genuine window is 4000 cm wide and a forward is 500 cm, so the window survives the
extra step. *Why:* the action is irreversible and episode-fatal; a 1-sample decision is the wrong shape.

**(5) `modules/spatial_memory.py` + the FINISH commit path — record "window entered and left".** Latch the
first frame on a leg where the arrival gate returns true; if the gate later returns false for N consecutive
checks while forwards continue, emit the commit anyway rather than walking on. *Predicted:* recovers
`ep_20_4_21` (reached 27 cm, committed 8 973 cm later), `ep_20_23_11` (688 cm), `ep_20_8_21` (925 cm) —
3 of 14 FINISH legs, the first non-zero `per_position_SR[3]` in the chain. *Why:* §2E; nothing in the harness
knows it has passed the goal.

**(6) `modules/viewpoint_controller.py` — make the turn approach conditional instead of constant, and delete
the dead cue.** Replace the `side_building_fraction ≤ 0.10` release (0/10 firings, AUC ≈ 0.5 per §2A) with:
if the previous ROAD leg committed on **arrival evidence** (the gate in (2) returned true), set
`APPROACH_FLOOR/CAP = 0` — the body is already at the junction; only when ROAD was committed without
arrival evidence emit the 3 forwards. *Predicted:* recovers `ep_20_19_21` (entered the box at `y = 1678`,
3 forwards took it to `y = 178`) and protects the 9 current TURN passes, whose ROAD legs were mostly
committed *without* arrival evidence. Honest signal only (which harness path emitted the previous `-1`).

**(7) `harness.py:~293` — scope the stuck detector to history, not the current burst.**
`forward_count` must count forwards since the last odometry change across steps, not `sum(a == 0 for a in
chosen_actions)`; with 1-action plans the `> 3` test is unreachable. *Predicted:* catches the 4
frozen-20-positions deaths and `ep_20_20_21` (11 steps wedged, `moving_distance` frozen at 136, `STUCK`
silent) before the env kills them. Small, independent, low-risk.

---

## 6. Exhausted — do NOT repeat

Carried forward from the anchor's §6 and still true: re-tuning `K_REACH_FORWARD_STREAK` /
`K_REACH_INTERVAL` / `MAX_REACH_FIRES_PER_SUBTASK`; re-**scheduling** *when* the forward reach is asked
(r003, 0.360); re-wording the criterion globally (r001 0.285, r004 0.355); in-burst arrival sampling
(r005 0.345); any **depth**-based proximity gate — `obs['depth']` is per-frame min-max normalised JET and
carries no absolute scale.

Newly closed by **this** iteration:

* **Pose/viewpoint as the arrival lever.** Faced-view in-window recall is **0/20** against the forward
  view's 8/34. Do not propose "look at the building before asking" again in any form.
* **`side_building_fraction` (or any single `object_mask` band statistic) as a junction / arrival cue.**
  11.9 % in-window vs 16.1 % far over 329 frames; a 5-class × 6-band sweep peaked at AUC ≈ 0.72; the lateral
  road-class fraction saturates at 1.00. It released the turn burst early **0/10** times — `APPROACH_CAP = 3`
  is doing all the work.
* **Blocking a commit *path* without fixing the commit *criterion*.** `allow_forward_commit` took forward
  FINISH commits 7 → 0 and `scan_reach` immediately produced 9 fatal ones with the same question. Any future
  candidate that re-routes `_binary_reach_vlm_call` to a new trigger without changing what it asks will
  reproduce this exactly.
* **Single-call conjunctions.** Asking one reach call to assert *both* landmark identity and in-junction
  position scores **0/14** recall (§4 row D). The conjunction must be two separate calls.

Keep, unchanged: the ORIENT cardinal burst (50/50) and `gate_burst`'s pass-through for it; the
`APPROACH_CAP_FORWARDS = 3` turn approach (TURNING 5/45 → 9/45, 9/10 conditional) — this iteration's one
genuine gain.

**Anti-cheat note for the next proposer:** §2A/§2B/§4 distances, `per_position_SR` splits and the
`roads.json` out-of-bounds check are *my* oracle-side measurements (`gt.json`, `env_step_log.jsonl`) and
exist only to price the levers. Every recommendation above decides from `obs['rgb']` /
`obs['object_mask']`, `info['moving_distance']`, `info['agent']['agent_rotation']`, the static
`task_spec['subtasks']` text and the harness's own bookkeeping. In particular **do not read
`env.subtask_max_steps`** — it is `int(L1_to_waypoint / 100)`, a laundered `target_location`.
