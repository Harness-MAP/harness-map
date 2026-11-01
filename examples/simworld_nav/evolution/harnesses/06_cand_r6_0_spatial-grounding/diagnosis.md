# Diagnosis — SimWorld-MMNav (`hard`, search_hard_50) iter 0 / `anchor_nav`

**Headline:** SR **0.00** (0/50) · SubtaskSR **0.36** · per-leg SR = ORIENTATION **1.00** (50/50),
MOVE_ALONG_ROAD **0.24** (12/50), TURN_AT_INTERSECTION **0.11** (5/45), REACH_DESTINATION **0.00** (0/45).
**Binding constraint:** every non-orientation leg is decided by ONE mechanism — a forward-facing
"does this frame match the hint image?" binary VLM call — and that question is the wrong question for
all three legs: at a ROAD waypoint the landmark is 90° off-axis so the recognizer says NO (recall
**21%**, 9/43 in-window fires) and the agent walks 10–23 km past; at FINISH the landmark is on-axis and
visible from 26 km so it says YES far too early (**7/7** fatal FINISH commits); and TURNING is never
asked at all — `ActivePerception` fires `[rotate, -1]` on the first frame of the leg with **zero**
forward motion, landing **78–980 cm** outside the intersection box in 7/7 failures.

> Method note: this file uses `gt.json` / `env_step_log.jsonl` (privileged) for *attribution only*.
> Every fix in §5 is decidable from `obs` + odometry + the cardinal compass + the instruction text.

---

## 1. What this harness does (mechanism)

`run_episode` (harness.py, 613 loc) runs perceive→plan→gate→act per step over 7 modules:

```
 ActivePerception.should_emit  ──(non-None)──► short-circuit: execute burst, skip perceive+plan
        │ None
        ▼
 Perceiver (VLM #1: rgb+hint+seg+depth → description)
        ▼
 Planner   (VLM #2: description → {Actions:[...], Match:0|1, Summary})
        ▼
 CompletionChecker.verify           — honor planner `-1` iff Match==1 AND forward_streak>=1
        ├─ cross_check_planner_commit  (VLM #3, binary REACH) → may REVOKE     [harness.py:413]
        ├─ proactive_commit_check      (VLM #3, binary REACH) → may INJECT -1  [harness.py:430]
        └─ request_side_scan + scan_reach (turn-to-face + binary REACH)        [harness.py:459]
        ▼
 burst executor: env.step per action; `-1` → env grades the leg
```

Three facts about the **environment** dominate everything below (`simple_world.py:532-560, 300-326`):

1. **A wrong `-1` is instantly fatal.** `step(-1)` → `if self._evaluation_of_subtask: current_subtask += 1
   else: terminated = True`. There is no retry. Commit precision is worth more than recall.
2. **Each leg has its own step budget.** After a commit the env sets `subtask_max_steps =
   L1(current, next_waypoint)/100` (ROAD/FINISH), 50 (TURNING), 10 (ORIENTATION); overrunning it
   terminates. Overshooting a waypoint therefore ends the episode too — just silently.
3. **The three legs are graded on three different quantities**: ORIENTATION = yaw only;
   ROAD = **L1 position only**, ±2000 (= 4 forward steps of 500); TURNING = yaw **AND** a
   ~5400×7400 position box; FINISH = position ±2000 **AND** yaw. The harness has exactly one
   notion of "arrived" (`Match==1` against the hint image) and applies it to all three.

`ActivePerception.should_emit` has two closed-form bursts that bypass perceive+plan entirely
(`active_perception.py:409-412` cardinal, `:431-440` relative-turn), both of the form
`[<rotations>, -1]`.

---

## 2. What actually happened (causal chains, with evidence)

### 2.1 ORIENTATION — the one thing that works (50/50)
Instruction is `"Turn to face west. ..."`. `_parse_target_cardinal` matches, branch (a) emits the
exact rotation bridge + `-1`, no VLM call. **Do not touch this path.** It is the whole of the 0.36.

### 2.2 ROAD — "reach-but-never-commit" is the single largest bucket (22/50 episodes)
`ep_20_7_task_dist_21_0_1`, ROAD waypoint L1 distance per step (cols: step, action, L1-to-waypoint,
VLM calls):

```
 14  0   2490.9  perceive,plan,REACH=0
 17  0    990.9  perceive,plan               ← INSIDE the ±2000 window
 19  0      9.1                              ← 9 cm from the waypoint
 20  0    509.1  perceive,plan,REACH=0       ← still inside; recognizer says NO
 23  0   2009.9  perceive,plan               ← out of the window, forever
 ...
 66  0  23509.1                              ← 23 km past; env kills the episode
```
The planner emitted `Move_Forward` on **every** cycle (`Match:0`, 801/918 plans were a single
action) and the binary REACH fired 10× and answered `0` 10×. Same shape in
ep_20_2/11, ep_20_12/11, ep_20_14/11+/21, ep_20_15/11+/21, ep_20_16/21, ep_20_17/21, ep_20_19/11,
ep_20_21/21, ep_20_23/11, ep_20_5/21, ep_20_6/11, ep_20_7/11+/21, ep_20_30/21 … (22 episodes; min
L1 achieved = 9, 41, 42, 50, 65, 80, 113, 122, 139, 139, 149, 162, 210, 213, 215, 215, 220, 220,
238, 246, 346, 1107).

**Why the recognizer says NO.** Every ROAD instruction ends `"...on your left/right"`. The hint is a
robot-view captured *facing* that building; at the waypoint the building is ~90° off the travel axis
and outside the camera FOV. Measured over all 305 in-leg REACH fires:

| leg | fires while GT-inside window | said Match=1 | fires while outside | said Match=1 |
|---|---|---|---|---|
| ROAD | 43 | **9 (recall 21%)** | 262 | 1 (FPR 0.4%) |
| FINISH | 0 | – | 43 | **7 (all 7 fatal)** |

### 2.3 ROAD, second cause — the cross-checker vetoes *correct* planner commits (7 legs lost)
When the planner did propose `-1`, `cross_check_planner_commit` re-asked the *same* forward-facing
question and revoked it:

| planner proposed `-1` | GT says AT waypoint | GT says AWAY |
|---|---|---|
| xcheck returned 0 (**blocked**) | **7** | 11 |
| xcheck did not run | 3 | 7 |

Those 7 are free legs thrown away. Worse, the recovery is wrong: with `commit_done=False` the
planner's `[-1]` filters to an empty list, so `harness.py:519` injects a **forward** (`EMPTY_FALLBACK`,
21 occurrences / 11 episodes), pushing the agent *out* of the ±2000 window it was standing in.

### 2.4 TURNING — the turn burst commits before reaching the intersection (7/7 failures)
Instruction is literally `"Turn right at the intersection."` `active_perception.py:431-440` matches
`_parse_relative_turn` and returns `[Rotate_Right, -1]` on the **first frame of the leg**, with no
perception and no motion. `ep_20_1_task_dist_11_0_1`:

```
 47  0   loc=(18300, 3392)                   ← ROAD leg still running
 48 -1   loc=(18300, 3392)  REACH=1          ← ROAD committed (PASS)
 49  5   yaw 270→180                         ← turn burst, no perceive/plan
 50 -1   loc=(18300, 3392)  yaw-err 0        ← TURNING: box is y∈[-2700,2700]; y=3392 → FAIL, episode over
```
All 7 failures have **yaw error = 0** — the rotation is always right — and are only **452, 627, 356,
692, 980, 78, 506 cm** outside the box. Simulating N extra forwards along the pre-turn heading before
the burst, over all 12 episodes that reached TURNING:

| extra forwards N | 0 | 1 | **2** | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| TURNING legs passing | 5/12 | 8/12 | **11/12** | 11/12 | 11/12 | 10/12 | 9/12 |

### 2.5 FINISH — the recognizer flips to over-eager, and nothing ever sets the final heading (0/45)
`ep_20_16_task_dist_11_0_1`: TURNING commits at step 11; six forwards later, at step 18,
**26,273 cm** from the destination, the binary REACH returns `1` → `REACH_COMMIT` → fatal.
Same in ep_20_10/21 (58,790 cm), ep_20_21/11 (22,988), ep_20_3/21 (35,733), ep_20_4/11 (15,733),
ep_20_5/11 (32,586), ep_20_9/11 (7,778). The prompt asks whether the landmark *"is clearly visible and
prominent"* — down a straight city street that is true from tens of km away.
Independently, 6 of those 7 also had **yaw error 270°**: the FINISH criterion requires facing the
building (`"...on your right, stop and face that building"`), and **no code path ever turns toward it**
before committing. The 2 episodes that did walk onto the destination (ep_20_18/11 min L1 **16 cm**,
ep_20_4/21 min L1 **27 cm**) never committed and timed out.

### 2.6 The mechanism designed for §2.2/§2.5 is dead code
`request_side_scan` (physically yaw to the named side, re-ask, yaw back on a miss) is exactly right —
turns are in-place, so a ROAD position grade is unaffected, and a miss costs 2 steps, never a fatal
commit. It fired **2 times in 50 episodes** (`SIDE_SCAN_MISS` ×2, `SIDE_SCAN_COMMIT` ×0) because
`completion_checker.py:801` gates it on `_side_band_near(obs, side)` — a depth predicate that the
module's own comments already record as unusable ("`near_frac == 0.000` on 24/24 real frames", which
is why `_strongly_far` was disabled). The gate fails **closed**, so the lever is off.

### 2.7 Secondary: the stuck detector cannot fire
`harness.py:280` — `forward_count = sum(1 for a in chosen_actions if a == 0)` counts the *current
burst*, not history; 801/918 plans are 1 action long, so `forward_count > 3` is unreachable in the
common case. Measured: 8 episodes contain a run of ≥4 forwards with <50 cm of progress (longest runs:
19, 19, 19, 19, 12, 9, 8, 8 steps); only 3 were flagged `STUCK`. ep_20_20/21 issued 20 forwards for
**136 cm** of travel (efficiency 0.01) and was never flagged. And when `STUCK` does fire the harness
just `break`s — there is no recovery (strafe-out / re-plan).

---

## 3. Failure breakdown (50 episodes, the leg that ended each one)

| cause | n | share |
|---|---|---|
| ROAD: reached the ±2000 window, never committed, walked out | **22** | 44% |
| FINISH: fatal premature `-1` from the binary REACH (7.8k–58.8k cm out) | 7 | 14% |
| TURNING: fatal `-1` from the zero-motion turn burst (78–980 cm out) | 7 | 14% |
| ROAD: fatal premature `-1` (2.0k–21.1k cm out) | 6 | 12% |
| ROAD: never reached the window (wedged / wrong direction) | 5 | 10% |
| FINISH: reached but never committed / never reached | 3 | 6% |

Cross-cutting: **20/50 episodes die on a wrong `-1`; 30/50 die by never committing again.**
Both halves trace to the same recognizer. Notes-tag counts: `EMPTY_FALLBACK` 21, `XCHECK_BLOCK` 18,
`REACH_COMMIT` 17, `NO_ACTION_RETURNED` 8 (all in ep_20_12/21), `STUCK` 3, `SIDE_SCAN_MISS` 2.
Cost: 799 s and 80.7k tokens per episode, ~44 VLM calls, mean 60 env steps against an env budget of
672–1328 — **the harness is nowhere near the step budget; it is losing on decisions, not on time.**

---

## 4. Binding constraint

**Primary: TASK-GAP (mis-specified commit criterion), with a HARNESS-BUG rider.**
The harness has one arrival predicate ("current frame ≅ hint image, looking forward") and the task has
three, graded on quantities that predicate does not measure. This is *not* mainly a VLM ceiling: the
frozen model is 90% precise when it does answer YES on ROAD (9/10), and the two structural failures —
the zero-motion turn burst (§2.4) and the never-set FINISH heading (§2.5) — involve no perception at
all. What IS a genuine **VLM-CEILING** is asking a forward-facing camera to recognise a landmark the
instruction says is *beside* you: 21% recall is not fixable by re-wording or re-scheduling the same
question. It must be fixed by **changing the viewpoint** (turn to face it) or **changing the evidence**
(segmentation cue for the cross-street), not by asking more often.

Rider **HARNESS-BUG**: (a) `request_side_scan`'s depth gate is permanently false (§2.6);
(b) the stuck detector's counter is scoped to the current burst (§2.7);
(c) a blocked commit recovers by stepping *forward*, which is the one direction that destroys the
position it was trying to claim (§2.3).

---

## 5. Concrete changes to try — ranked by measured leverage

### A. `active_perception.py:431-440` (and `:409-412`) — walk into the intersection before the turn burst
The instruction says *"Turn right **at the intersection**"*; the burst turns wherever it happens to
stand. Replace the immediate `[code, _SUBTASK_DONE]` with an approach phase: emit `[0]` (forward) and
return without `-1` until either (i) an intersection cue is seen, or (ii) a hard floor of 2 forwards
has been executed **inside this leg**, then emit `[code, -1]`. Gate this on the leg being a
*relative-turn* instruction — branch (a) must keep firing bare for the ORIENTATION leg (50/50, §2.1);
the cleanest discriminator is `_is_orientation_instruction()` (already in completion_checker.py:169)
vs `_parse_relative_turn()`.
**Predicted:** TURNING leg SR 5/12 → **11/12** (measured sweep, §2.4) ⇒ per_position_SR[2] 0.11 → 0.24,
SubtaskSR **+0.03**, and 6 more episodes get to attempt FINISH. Honest inputs only (instruction text +
a self-maintained forward counter). *Caveat: the constant 2 is fit on 12 samples. Prefer the adaptive
version (step forward until the lateral segmentation band opens on the turn side, cap 3) with 2 as the
floor; report the floor in NOTES so the next iter can see it was a fit.*

### B. `completion_checker.py:801` — re-trigger the turn-to-face scan on a signal that is not always false
Delete the `_side_band_near` (depth) gate; trigger `request_side_scan` on
`_named_side_has_structure(obs, instruction)` (segmentation, already in this file at :442, already
measured working: side building fraction 0.60–0.72 in-window vs 0.000 on the far-field frame) **AND**
the existing REACH firing schedule, and raise `MAX_SIDE_SCANS_PER_SUBTASK` from 2 to ~4.
This is the only lever that attacks the 21% recall at its cause: the hint was captured *facing* the
building, so the scan shows the recognizer the viewpoint it was trained on. Risk is bounded by
construction — rotations are in-place (position-graded ROAD is unaffected) and `scan_reach` commits
only on an explicit `1`, restoring the heading otherwise.
**Predicted:** attacks the 22-episode bucket (§2.2). Even a 3× recall improvement on those 43 in-window
fires converts ~8–10 ROAD legs ⇒ ROAD SR 0.24 → ~0.40, SubtaskSR **+0.04–0.06**.
**Wiring caveat:** after a ROAD commit via side-scan the agent is yawed 90° off the travel axis —
`harness.py:459-490` must restore the heading on a *commit* as well as on a miss, or fix A's turn
burst will double-rotate on the next leg.

### C. `harness.py:430` + `completion_checker.py:606` — make the FINISH commit a *pose*, not a sighting
FINISH is the only leg graded on yaw as well as position, and nothing sets the yaw: 6/7 fatal FINISH
commits had a 270° heading error. On a FINISH-shaped leg (`_named_side` present and the instruction
contains "stop and face"), forbid `proactive_commit_check` from injecting a bare `-1` from the travel
heading; require the turn-to-face path of (B) — rotate toward the named side, re-ask, and commit only
from the rotated pose. This removes the *only* mechanism that produced the 7 fatal FINISH commits and
simultaneously satisfies the yaw criterion.
**Predicted:** FINISH fatal commits 7 → ~0, so those episodes keep walking (the env budget is ample,
§3). Combined with A+B this is the only route to a non-zero SR, since FINISH is 0/45 today.

### D. `completion_checker.py:689` — stop revoking correct commits with the same weak question
`cross_check_planner_commit` re-asks the identical forward-facing query that `verify` already
implicitly relied on, and blocked 7 true commits vs 11 false ones (§2.3) — roughly a coin flip, with
asymmetric cost since the AT-window ones are unrecoverable. Two options, in order of preference:
(i) make the cross-check the **turn-and-look** of (B) instead of a second forward-facing call;
(ii) if it still returns 0, recover with an in-place look/hold rather than `harness.py:519`'s forced
`EMPTY_FALLBACK` forward, which walks the agent out of the window it was standing in.
**Predicted:** recovers up to 7 ROAD legs directly (SubtaskSR +0.03), net of whatever the 11 blocked
false commits cost back — so pair it with (B) so the replacement evidence is genuinely independent.

### E. `harness.py:280` — fix the stuck counter's scope, and give it a recovery
Count forwards over the per-subtask action history (`SpatialMemory` already logs it) instead of over
`chosen_actions`; and on trigger, strafe (action 3/4) + re-perceive instead of `break`.
**Predicted:** small — touches 4–8 episodes (§2.7), most of which were far from any waypoint. Low
priority, but it is a real bug and cheap.

---

## 6. Exhausted — do NOT repeat

Five candidates (`cand_r1..r5_0_spatial-grounding`) have already forked from this anchor. **All five
re-worked the arrival-recognizer axis and none beat it:**

| candidate | axis (from its NOTES `## axis`) | SubtaskSR |
|---|---|---|
| `anchor_nav` | — | **0.360** |
| r001 | "commit-as-pose": tightened commit criteria inside the existing modules | 0.285 (ROAD 12/50 → 1/50) |
| r002 | commit-gate internals | 0.330 |
| r003 | REACH **look-schedule** — event-driven window instead of a fixed stride | 0.360 |
| r004 | commit-gate internals | 0.355 |
| r005 | `BurstMonitor` — in-burst arrival sampling, 8th module | 0.345 |

Do **not** propose again: re-tuning `K_REACH_FORWARD_STREAK` / `K_REACH_INTERVAL` /
`MAX_REACH_FIRES_PER_SUBTASK` / `N_REACH_CONFIRMATIONS_REQUIRED`; re-scheduling *when* the
forward-facing binary REACH is asked; re-wording its prompt; tightening the commit criterion globally
(r001 shows that costs ROAD legs outright); or reviving a **depth-based** proximity gate —
`obs['depth']` is per-frame min-max normalised, so it carries no absolute scale and cannot separate
"beside a facade" from "in open space" (documented in this file's own `_strongly_far` at
completion_checker.py:382, and confirmed by §2.6's dead side-scan).

Verified untouched by all five prior candidates, i.e. **genuinely open**:
`active_perception.py:431-440` (the zero-motion turn burst — §5A),
`completion_checker.py:801` (the dead `_side_band_near` side-scan gate — §5B),
`harness.py:280` (the burst-scoped stuck counter — §5E).
