# Diagnosis — SimWorld-MMNav (`hard`, `search_hard_50`) / candidate `cand_r002_0_spatial-grounding` ("the arrival channel: evidence + arbitration"), iter 2, parent `cand_r1_0_spatial-grounding`

**Headline:** SR **0/50** · SubtaskSR **0.330** (parent 0.285, anchor **0.360**) · per-leg pass
**[1.00, 0.12, 0.111, 0.00]** (parent [1.00, 0.02, 0.022, 0.00]; anchor [1.00, 0.24, 0.111, 0.00])
· distance_progress 0.173 · 90.1k tok/ep (anchor 80.7k) · 744 s/ep · static_collision 2.86
(parent 5.12) · red_light 8.5.

The iter's two claims both held. The revert + arbitration rewrite **did** stop the veto from
destroying in-window claims — **0** in-window planner claims were vetoed this run against 16 last
run — and the collision/cost regressions the veto caused reverted with it. It recovered leg 1 from
1/50 to 6/50. It did **not** reach the anchor's 12/50, and the reason is that the thing it repaired
was never the term that binds.

**Binding constraint (one sentence):** the agent physically stands inside leg 1's graded success
window in **35/45** episodes and commits in **6**, because every commit in this harness is decided by
asking the frozen VLM whether a live frame "matches" a differently-lit reference photo — a channel
measured here at **4.1 % yes in-window vs 2.6 % yes out-of-window** (likelihood ratio 1.6, i.e. very
nearly no information about position at all) — while the same `obs['object_mask']` the harness
already ships carries a near-binary *geometric* arrival cue (the ground class under the robot
flipping from sidewalk to roadway) that it currently mislabels in its own prompt legend and never
reads in code.

---

## 1. What this harness does (mechanism)

`run_episode` (harness.py:172-700) is a single while-loop over `len(actions_log) < max_steps`, with
seven modules wired in a fixed order:

1. **ActivePerception.should_emit** (`modules/active_perception.py:420`) can short-circuit the whole
   VLM pipeline with a deterministic burst. Three branches matter:
   (a) *cardinal burst* — the instruction names a compass target ("Face north"), so emit the closed-form
   rotation list + `-1`. (a'') *relative-turn burst* — "Turn right at the intersection" → spend
   `_TURN_APPROACH_FORWARDS = 2` forwards, then `[Rotate, -1]`. (b) red-signal idle / anti-spin / momentum.
2. **Perceiver** (`modules/perceiver.py`) — one VLM call over 4 images (RGB, the leg's `robot_view_*`
   hint, `object_mask`, JET depth) → free-text `Description`.
3. **Planner** (`modules/planner.py`) — one **text-only** VLM call over that description (no pixels)
   → `{Summary, Match, Actions}` using the published baseline's `reasoning_template` verbatim.
4. **AffordanceEstimator** — full-motion-set stub.
5-7. **CompletionChecker** (`modules/completion_checker.py`, 1127 L) owns every `-1`, through four
   serial gates: `verify()` (planner `Match == 1` **and** `forward_streak >= K_MIN_FORWARD_BEFORE_PLANNER_COMMIT = 1`)
   → `cross_check_planner_commit()` (an independent binary REACH VLM call, one refusal allowance per leg,
   `HOLD_PROBE = (1, 2)` on a veto) → `proactive_commit_check()` (the same binary call when the planner
   did *not* propose `-1`) → `request_side_scan()` / `scan_reach()` (off-axis framing recovery).
   All four gates terminate in **one function**, `_binary_reach_vlm_call` (`completion_checker.py:685-829`),
   which shows the model the current RGB + mask + depth + the perceiver's description, plus the
   reference photo, and asks for `{"Match": 0|1}`.

So: **the ORIENTATION leg and the TURNING leg are decided deterministically from the instruction text
with zero VLM calls; the ROAD leg and the FINISH leg are decided entirely by image-to-photo matching.**
Hold that asymmetry — §2.4 and §4 turn on it.

### 1.1 The three episode terminators the harness does not model

`SimWorldGym/simworld_gym/envs/traffic_world.py:365-386` ends the **whole episode** — not the leg — on
any of: the agent leaving the union-of-road-rectangles polygon; 20 consecutive identical positions;
`steps_count >= max_steps`; **or `substeps_count >= subtask_max_steps`**, a *per-leg* budget set at each
successful `-1` to 10 (ORIENTATION) / 50 (TURNING) / `int(L1_to_next_waypoint / 100)` (ROAD, FINISH).
And `step(-1)` on a leg the env grades **FAIL** sets `terminated = True` immediately.

None of this reaches the harness as anything but a bare `terminated` flag, so `notes` is empty and the
run looks like a clean exit. 26/50 verdicts this run have empty `notes`.

---

## 2. What actually happened (causal chain, with evidence)

### 2.1 Why the episodes end — reconstructed, not guessed

Replaying `_get_terminated` step-by-step over each episode's `env_step_log.jsonl` against that map's
`progen_world.json` road polygon and `gt.json` criteria:

| terminator | n/50 | note |
|---|---:|---|
| **BAD_COMMIT** — a `-1` the env graded FAIL | **16** | 9 on ROAD, 4 on FINISH-at-leg-1, 2 on FINISH-at-leg-3, 1 on TURNING |
| **OUT_OF_WORLD** — left the road polygon | **11** | walked off the end of a street |
| **SUBTASK_BUDGET** — `substeps >= subtask_max_steps` | **11** | 10 of them on leg 1 |
| **RAN_OUT** — harness broke (STUCK / NO_ACTION / TOO_MANY_ERRORS) | 7 | the only harness-visible exits |
| **FROZEN_20** — 20 identical positions | 5 | the harness's own stuck detector fired on **none** of them |
| `max_steps = 672` | **0** | largest episode: 247 steps |

**43/50 episodes are ended by the environment for a reason the harness cannot observe.** The step
budget is never the constraint; the *stop decision* is.

Worked example, `ep_20_10_task_dist_11_0_1` (3 steps, 2 VLM calls, `notes` empty): spawn
[-18300, 17645.8]; leg 0 ORIENTATION committed by the cardinal burst at step 1. That commit set
leg 1's budget to `int(L1([-18300,17800]) / 100) = int(154/100) = 1`. The agent was *already* 154 cm
from leg 1's target — 13× inside the 2000 threshold — and needed only to emit `-1`. It emitted one
`Move_Forward`, `substeps_count` hit 1 ≥ 1, and the episode was over.

### 2.2 The agent reaches the goal and walks through it — 29/45 on leg 1

For every episode, the minimum L1 from the trajectory to each graded waypoint (`gt.json`, read for
*analysis* only):

| leg | never reached | reached & committed | **reached & walked through** |
|---|---:|---:|---:|
| ROAD (45 eps) | 10 | **6** | **29** |
| FINISH (50 eps) | 47 | 0 | 3 |

Median distance travelled *after* the last in-window step, on the 29: **13 134 cm** (≈26 forward
steps, ≈ two-thirds of a city block). Worst: `ep_20_9_task_dist_11_0_1`, 50 500 cm.

`ep_20_7_task_dist_21_0_1` stood **9 cm** from leg 1's waypoint and then walked 22 000 cm past it.

### 2.3 The arrival channel is position-blind — measured, three ways

**(a) The planner's `Match` carries no position information.** Aligning every `plan` call in
`vlm_trajectory.jsonl` with the agent's true position at that step:

| | inside the leg-1 window | outside |
|---|---:|---:|
| decisions taken there | 73 | 684 |
| `Match == 1` | **3 (4.1 %)** | 18 (2.6 %) |
| `Actions` contained `-1` | **3 (4.1 %)** | 12 (1.8 %) |

Likelihood ratio **1.6**. A detector that fires 4 % of the time where it should and 2.6 % where it
should not cannot gate a one-shot decision, no matter how it is arbitrated downstream.

**(b) The planner is a constant function.** 996/1042 plan replies (**96 %**) contain "has not yet
reached" / "continue forward"; 729/1042 (**70 %**) cite lighting or appearance mismatch. The reference
photos are systematically brighter than the live render (mean pixel 135.6 vs 95.3 over 20 episodes),
and the model says so out loud: *"the current view is dim and dark, with a green sky… the expected view
is bright and sunlit with an orange sky."*

**(c) The smoking gun.** `ep_20_3_task_dist_11_0_1`, leg 1, target [61700, 17800]:

| step | true L1 to waypoint | planner `Summary` |
|---:|---:|---|
| 4 | 1520 | *"…continuing to move north along a dimly lit urban street… The current view remains significantly darker and lacks the detailed architectural features of the expected target building, confirming that the agent has not yet reached the intersection."* |
| **7** | **20** | **byte-identical to step 4** |
| 10 | 1480 | **byte-identical to step 4** |

The text does not change as the agent traverses the entire window and stops 20 cm from the waypoint.
Three steps later it walked out of the road polygon and the episode ended (14 steps, `notes` empty).

**(d) The gate lets through exactly the wrong claims.** Of 15 planner `-1`s that survived the gates:
the 3 raised in-window all **passed** (3/3); of the 12 raised out-of-window, 9 became **fatal**
commits. The arbitration rewrite achieved what NOTES promised — it stopped vetoing good claims — but
it is equally permissive toward bad ones, and a `-1` is a one-shot, episode-ending bet.

**And a motion floor cannot separate them either.** Translations executed on the leg before the commit:

- successful non-orientation commits: `[2, 2, 2, 2, 2, 12, 27, 32, 60, 63, 99]`
- fatal commits: `[2, 3, 6, 8, 12, 12, 14, 18, 23, 24, 25, 43, 71, 73, 82, 227]`

Fully overlapping. A floor at 8 translations removes 3 fatal commits at zero cost to successes — real,
but small; there is no threshold that does the job.

### 2.4 The one leg that works is the one that never asks the VLM

Leg 2 (TURNING) is decided by `active_perception.should_emit` branch (a''): parse "Turn left/right" out
of the static instruction, rotate 90°, commit. **5 of the 6 episodes that passed leg 1 also passed leg 2.**
Checking the geometry over all 45 ROAD task configs: the leg-1 waypoint lies inside the leg-2
position-area in **45/45**, and the leg-2 yaw is exactly ±90° from the leg-0 yaw in **45/45** (20 left,
25 right). Leg 2 is arithmetically pinned to leg 1 — the deterministic channel has already solved it.

The single leg-2 failure is instructive: `ep_20_19_task_dist_21_0_1` committed at y = **678** against a
box floor of y = **700**. It missed by **22 cm**, because `_TURN_APPROACH_FORWARDS = 2` spent 1000 cm of
approach from a pose that was already inside the box.

### 2.5 The sampling rate is wrong for the window

The window is ≈4000 cm ≈ 8 forward steps wide. The planner emits `[0, 0, 0]`: **782 of 992** consecutive
plan-call gaps are exactly 3 env steps. So the agent gets a **median of 2** decisions inside the window
it must stop in — and in 12 episodes only **1**. Even a perfect per-frame detector would be sampled
into unreliability at this cadence.

### 2.6 The stuck detector is structurally dead

`harness.py:293` computes `forward_count = sum(1 for a in chosen_actions if a == 0)` over the **last
burst only**, and requires `forward_count > 3`. With the planner's modal burst being 3 forwards, the
condition can never be true no matter how many consecutive bursts are blocked. Consequence: the env had
to kill 5 episodes on its 20-identical-positions rule, and **none of the five** has `STUCK` in `notes`.
`ep_20_22_task_dist_21_0_1` mashed forward for 24 steps into a pedestrian (`human: 20`) and covered
1372 cm.

### 2.7 A live prompt bug that erases the cue the task is made of

`obs['object_mask']` is exactly **4 colours**. Measured over sampled frames (bottom 28 % of the image,
i.e. the ground in front of the robot):

| colour | actual class | what the harness tells the model |
|---|---|---|
| `[255,255,0]` yellow | **sidewalk / crosswalk** | "ground/road" |
| `[0,127,255]` blue | **roadway / driveway surface** | **"other"** |
| `[128,0,128]` purple | buildings | buildings ✓ |
| `[0,255,0]` green | sky | SKY ✓ |

The legend appears three times — `modules/prompt_assembler.py:77-78`, `:132`, and
`modules/completion_checker.py:754` — and all three collapse sidewalk and roadway into one label and
throw the roadway class into "other". Leg 1's stop condition is *literally* "stop at the intersection",
i.e. "stop where the sidewalk ends and the roadway begins", and the harness tells the model those are
the same class.

### 2.8 …and the cue it erases is a working arrival sensor

Measured on the bottom band: the split is near-binary. `ep_20_3_11` reads `yellow = 1.000` at steps 1,
4, 7 (sidewalk) and flips to `blue = 1.000` at step 10 (standing in the intersection). Taking **the
first frame of leg 1 whose ground band is majority roadway** as a commit trigger, over all 45 ROAD
episodes and at the *current* 1-frame-per-3-steps cadence:

| trigger | fires in-window | fires 2009–3163 cm out (≤2 steps) | fires far-field | never fires |
|---|---:|---:|---:|---:|
| ground band under the robot (rows 72–100 %, >50 % road) | 14/45 | — | 15 | 16 |
| forward band (rows 55–75 %, **>5 %** road) | **16/45** | **8/45** | 8 | 13 |
| **incumbent: the hint-match channel** | **6/45** | — | 9 fatal | — |

Median true L1 at detection: **927 cm** — comfortably inside the 2000 threshold. Every far-field
misfire is a *later* intersection reached after the agent had already overshot, and every "never fires"
is the 3-step burst skipping the crossing frame (§2.5) — both are addressable, and neither is a
property of the cue.

---

## 3. Failure breakdown

Per episode, the dominant cause of the lost SubtaskSR:

| cause | eps | share |
|---|---:|---:|
| **leg 1 reached but not committed** → agent overshoots, then dies on OOB / leg budget / freeze | **29** | 58 % |
| **fatal false `-1`** ends the episode outright | **16** | 32 % |
| leg 1 never reached (wandered off-route from step 1) | 10 | 20 % |
| harness-side exit (STUCK / NO_ACTION) | 7 | 14 % |

(the first three partition the 45 ROAD episodes; the fourth overlaps).

Sub-metrics consistent with the above: distance_progress 0.173 (the walker moves, it just moves past);
static_collision 2.86 and red_light 8.5 (an agent crossing intersections it should have stopped at);
90.1k tok/ep and 744 s/ep, of which the majority is spent after the stop point was passed.

Cross-check against the numeric pre-pass: `score.json` `per_position_SR[1] = 0.12` = 6/50 leg-1 passes,
matching the 6 counted from the traces; `per_position_SR[2] = 0.111` = 5/45, matching the 5 TURNING
passes. `read_audit.json`: **PASS**, 15 paths, no forbidden/escalated/undeclared reads.

---

## 4. Binding constraint — classified

**VLM-CEILING on the channel the harness chose to decide with, wrapped in three HARNESS-BUGs that
prevent the better channel from being used.**

The ceiling is real and now measured three times over three iters at three operating points: the
frozen Qwen3-VL cannot answer "am I standing at the place in this photo" from a live frame versus a
differently-lit reference. The parent's diagnosis established the prompt *wording* is not a lever (two
far-apart wordings, flat ROC, the model reciting whichever clause it landed on). This iter added the
*evidence* — mask + depth + description — to the same call, which was the one axis left open, and leg 1
went 0.02 → 0.12, still half the anchor's 0.24. The correct reading is not "the evidence did not help";
it is that **enriching a question the model answers from appearance priors cannot fix a question that
is about geometry.** The channel's ceiling is LR ≈ 1.6 and the architecture bets the whole episode on it.

The fix must therefore **scaffold the decision, not re-ask it** — exactly as leg 2 already does. Leg 2
is the existence proof sitting inside this very harness: it replaced "does this look like the turn?"
with "the instruction says turn right, so rotate right", and it now passes 5/6 conditional on leg 1
with zero VLM calls. Leg 1 has the same structure available: its stop condition is a *geometric* event
("the intersection"), that event is directly encoded in a sanctioned observation channel
(`obs['object_mask']`, the sidewalk→roadway class flip), and that cue is measured here at 16/45
in-window against the incumbent's 6/45.

The three harness-side blockers, each independently fixable:
- **the legend is wrong** (§2.7) — the harness actively tells the model the cue does not exist;
- **the cadence is wrong** (§2.5) — 3-step bursts sample an 8-step window twice;
- **`-1` is not priced as fatal** (§2.3d) — 16/50 episodes are spent on a bet the harness treats as a
  retryable probe, and the gate that is supposed to price it has LR 1.6.

This is **not** TASK-GAP: the walker reaches leg 1's window in 35/45 and leg 2's is a single rotation
away. The route-following works. The stopping does not.

---

## 5. Concrete changes to try (the gradient — ranked by leverage)

### A — Fix the segmentation legend. `modules/prompt_assembler.py:77-78` and `:132`; `modules/completion_checker.py:754`
Three strings. Replace *"yellow = ground/road, … blue = other"* with the measured truth:
**`[255,255,0]` yellow = SIDEWALK/CROSSWALK (where you walk), `[0,127,255]` blue = ROADWAY/DRIVEWAY
surface (a street you are crossing), purple = buildings, green = sky.** Add one sentence to both the
perceive prompt and the arrival prompt: *"If the ground immediately in front of you has changed from
sidewalk to roadway, you have stepped into an intersection."*
**Predicted:** the perceiver's `Description` — which is the planner's **only** input (the planner is
text-only) — starts carrying the arrival event in words, so the `Match` channel gets a chance to
correlate with position for the first time. Zero risk, three lines, and a prerequisite for B.
*Why it should work:* §2.7/§2.8 — the cue is present in every frame, discriminative, and currently
described to the model as a class that does not matter.

### B — Make leg 1's commit a geometric detector, not a photo match. New `road_surface_arrival()` in `modules/completion_checker.py` (beside `_named_side_has_building`, :543-570); called from `harness.py:~430`, **before** `proactive_commit_check`
Compute the roadway-class fraction in a forward band of `obs['object_mask']` (rows ≈0.55–0.75, full
width). On a non-orientation, non-FINISH leg, latch `commit_done = True` the **first** time that
fraction crosses a small threshold. Latch **once per leg** (`self._road_fired[cur_idx]`) so a later
intersection cannot re-trigger it — that single latch removes all 8 far-field misfires in §2.8's table.
Keep the existing VLM path as the fallback when the mask channel is missing.
**Predicted:** leg-1 pass **0.12 → 0.25–0.35** (16/45 in-window at the current cadence, before C;
8 more fire within 2 forward steps of the window and would land inside it with C). Because leg 2 is
5/6 conditional on leg 1 (§2.4), leg-2 pass should track leg 1 almost 1:1 → SubtaskSR ≈ 0.45–0.50.
*Constraints:* express the band and threshold as **fractions of frame dimensions**, never absolute cm
(the lint WARNs literals ≥ 2000, and this is exactly the map-tuning it guards against); re-verify the
4-colour palette from the candidate's own probe before shipping rather than trusting this note; and
degrade to the current behaviour when `obs['object_mask']` is absent.

### C — Stop emitting 3-step bursts once the leg is near its stop condition. `harness.py:561`
`chosen_actions = [a for a in planner_actions if a in allowed]` executes the planner's whole list
blind. Truncate it to **one translation** once the leg's perceive has reported *any* roadway class in
the forward band (B's precursor, below the commit threshold), or once the leg's translation count
passes a modest floor. Keep bulk bursts for the long empty stretch — that is where they pay.
**Predicted:** turns the median 2 in-window decisions into 6–8, which is what converts B's 13
"never fires" into detections. Costs extra perceive+plan calls only on the approach; §2.5 shows the
current cadence is the reason a per-frame cue cannot be sampled reliably.

### D — Price `-1` as the fatal, one-shot bet it is. `modules/completion_checker.py:181` + a FINISH precondition
(i) `K_MIN_FORWARD_BEFORE_PLANNER_COMMIT: 1 → 8` **translations on the current leg**. Measured on this
run: removes 3/16 fatal commits, costs **0/11** successful ones (§2.3, the two lists). (ii) Gate any
FINISH-leg commit on B's road-surface detector having fired at least once on that leg — a FINISH is
always past at least one intersection, and 4 of the 16 fatal commits were FINISH-at-leg-1 claims raised
19 000–30 000 cm out, with no crossing behind them.
**Predicted:** +3 to +6 surviving episodes, each of which then gets B's detector on the next
intersection instead of being dead. Small, cheap, and strictly additive to B.
*Note:* this is `K_MIN_FORWARD_BEFORE_PLANNER_COMMIT` (the planner path's floor), **not**
`K_REACH_FORWARD_STREAK` — the latter is on the exhausted list and stays there.

### E — Repair the stuck detector, then use it. `harness.py:290-300`
`forward_count` must count **consecutive translations across decisions**, not within the last burst;
with a modal burst of 3 the current `> 3` test is unreachable (§2.6). Track a running counter reset by
any non-translation action or any `Δ moving_distance ≥` one step-length, and trip on ~3 consecutive
blocked translations. **Predicted:** recovers the 5 FROZEN_20 episodes (currently silent total losses)
and cuts the collision counters that drive static_collision 2.86 / human 20.

### F — Make the turn approach conditional. `modules/active_perception.py:124`
`_TURN_APPROACH_FORWARDS = 2` is unconditional and cost `ep_20_19_task_dist_21_0_1` a leg-2 pass by
22 cm. With B in place the leg-1 commit is raised **at** the crossing — i.e. already inside the leg-2
box — so the approach should be **skipped entirely when leg 1 was committed by the road-surface
detector**, and kept only when leg 1 was committed by the VLM path (which fires early). One boolean
threaded from the checker to the active-perception latch.
**Predicted:** +1–2 leg-2 passes, and removes a systematic overshoot that will get *worse* as B makes
leg-1 commits more frequent.

### G — Reclaim dead machinery and un-blind the trace
`request_side_scan` / `scan_reach` fired **once** in 50 episodes for 0 commits (`SIDE_SCAN_MISS` ×1),
after firing 4 times for 0 commits in the parent. It is dead as gated: either re-gate it on B's
detector (scan the named side *at* the crossing, where the building actually is) or delete it and
reclaim the budget. Separately — and this matters for every future iteration on this channel —
`vlm_trajectory.jsonl` contains **1254 `perceive` + 908 `plan` and zero arrival calls**, even though
`notes` records 9 `REACH_COMMIT` / 6 `XCHECK_BLOCK` / 1 `SIDE_SCAN_MISS`. Every VLM call made inside
`CompletionChecker` is invisible to the trace reader. Append them to `llm_calls` with a `stage`
(`"reach"` / `"xcheck"` / `"scan"`) at `completion_checker.py:829`, so the next diagnosis can measure
the arrival call directly instead of inferring it from notes.

---

## 6. Exhausted — do NOT repeat

**Established this run:**
- **The motion floor is not an arrival discriminator.** Successful non-orientation commits occur at
  `[2,2,2,2,2,12,27,32,60,63,99]` translations, fatal ones at `[2,3,6,8,12,12,14,18,23,24,25,43,71,73,82,227]`.
  Only the ≤8 region is free (§5-D); do not search for a threshold above it, there isn't one.
- **The cross-check arbitration from this iter is correct — keep it, and stop working on it.**
  `MAX_XCHECK_REFUSALS_PER_SUBTASK = 1` + `HOLD_PROBE` did exactly what NOTES predicted: **0** in-window
  claims vetoed (vs 16 in the parent), collisions 5.12 → 2.86, cost 97.2k → 90.1k. The residual gap to
  the anchor is not in the arbitration.
- **Adding evidence to `_binary_reach_vlm_call` did not lift it out of the ceiling.** Mask + depth +
  description on the same call: in-window yes-rate 4.1 % vs out-of-window 2.6 %, LR 1.6. Together with
  the parent's two wording operating points, **the hint-photo-match channel is closed as a primary
  commit decider at any threshold, with any prompt, over any of its four sanctioned evidence channels.**
  Do not iterate on it again — replace which channel decides (§5-B).
- **Do not raise the step budget or chase `max_steps`.** Zero episodes reached `max_steps = 672`; the
  largest was 247. Every budget-shaped loss is the env's *per-leg* `subtask_max_steps`, which is a
  function of distance-to-waypoint and is therefore a *stopping* problem, not a *budget* problem.

**Carried forward, re-confirmed here:**
- **Do not re-word the arrival criterion**, in any direction — now three operating points, flat ROC.
- **Do not use whole-frame building-class fraction** as an arrival feature (flat at ~0.28 everywhere).
  *Distinct from §5-B*, which uses the **ground class in a band**, a different feature on a different
  class — but validate it on the candidate's own probe before shipping rather than assuming.
- **Do not touch the ORIENTATION cardinal burst.** 50/50 again, 0 VLM calls.
- **Do not tune `K_REACH_FORWARD_STREAK` / `K_REACH_INTERVAL` / `MAX_REACH_FIRES_PER_SUBTASK`.**
  Coverage was never the binding term.
- **Do not re-run `N_REACH_CONFIRMATIONS_REQUIRED = 2`.**
- **Do not ship a bare depth-polarity flip**; do not re-enable `_strongly_far` / `_looming_state` as gates.
- **Do not read** `agent_location`, `target_location`, `destination`, `correct_robot_location`,
  `ideal_path`, `gt.json`, `current_instruction`, `env.unwrapped._agent_rotation`,
  `env.subtask_max_steps`, or `info['agent_collision']`. Everything in §5 decides from
  `obs['object_mask']`, `obs['depth']`, `obs['rgb']`, `info['moving_distance']`,
  `info['agent']['agent_rotation']` and the static `task_spec['subtasks']` text only.
  *(The geometric facts in §2.1/§2.2/§2.4 were derived from `gt.json` and the env source **for this
  analysis only** — none of them may be hard-coded into a policy.)*
- **Avoid numeric literals ≥ 2000** in candidate code; express distances as forward-action counts or
  frame fractions.

---

## 7. Provenance

- Candidate source: `$OUT_DIR` — `harness.py` (700 L) + 7 modules (2 474 L).
- Score: `…/cand_r2_0_spatial-grounding_r002/score.json` (50/50 episodes, `complete: true`).
- Read audit: `…/cand_r2_0_spatial-grounding_r002/read_audit.json` — **PASS**, 25 071 reads over 15
  declared paths, 0 forbidden / 0 escalated / 0 undeclared.
- Traces: `$TRACE_DIR`, 50 episodes × {`vlm_trajectory.jsonl`, `llm_calls.jsonl`, `env_step_log.jsonl`,
  `actions.jsonl`, `gt.json`, `verdict.json`, `frames/`}.
- Termination taxonomy (§2.1) reconstructed by replaying
  `simworld_gym/SimWorldGym/simworld_gym/envs/traffic_world.py:302-386,615-716` over each episode's
  `env_step_log.jsonl` and that map's `progen_world.json` road polygon.
- Palette and arrival-detector measurements (§2.7-2.8) computed over the `object_mask` images saved in
  each episode's `frames/`, aligned to true position by step index.
- Prior gradient read: `…/cand_r2_0_spatial-grounding_r002/source/diagnosis.md` (iter 1) and this
  candidate's `NOTES.md`.
