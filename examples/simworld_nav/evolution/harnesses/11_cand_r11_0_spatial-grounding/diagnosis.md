# Diagnosis — SimWorld-MMNav (hard) iter 4 / `cand_r004_0_spatial-grounding` ("the motion channel")

**Headline:** `SR 0.0`, **`SubtaskSR 0.355`**, `per_position_SR [1.0, 0.18, 0.1556, 0.0]`,
`distance_progress 0.1844` (best in the chain to date), `static_collision_mean 5.6`,
`red_light_violation_mean 9.88`. Against the parent (`cand_r1_0…_r001`: 0.285, pos1 **0.02**) this is
**+0.07 SubtaskSR and a 9× recovery of position 1** — the two reverts did exactly what the parent
gradient predicted. Against the anchor (0.360, pos1 0.24) it is a wash.

**Binding constraint (one sentence):** the harness now reaches the graded success window on
**40 of 57** ROAD/FINISH legs and converts only **9** of them, because *every* arrival readout it
owns — the planner's self-report, the binary REACH judge, and even the perceiver's free-text
"I am at an intersection" — has a Match=1 rate that is **flat or inverted in true distance**, so the
commit decision is uncorrelated with arrival; this is a **VLM-CEILING on the arrival channel**, and
it is a ceiling on *language over raw RGB*, not on the harness, because a zero-VLM geometric readout
measured on the same 788 frames **is** distance-monotone (§5-A).

---

## 1. What this harness does (mechanism)

`orchestrator.py:run_episode` drives one perceive→plan→gate→execute loop per decision over seven modules,
graph byte-identical to the parent:

| stage | file:line | role |
|---|---|---|
| STUCK backstop | `orchestrator.py:305-313` | `memory.forward_stall_run() >= 8` → end episode (**new this iter**) |
| ActivePerception short-circuit | `orchestrator.py:330-336` | cardinal burst (a), turn-approach latch (a″), red idle, anti-spin, **blocked-forward escape ladder (b′, new)**, momentum (d) |
| Perceiver | `orchestrator.py:339-348` | 1 VLM call, 4 images: `rgb`, hint, `object_mask`, `depth` → text description |
| Planner | `orchestrator.py:360-375` | 1 VLM call, text only → `Actions[]` + `Match` + rolling summary |
| `verify()` | `completion_checker.py:607` | honours a planner `-1` iff `Match=1` **and** `forward_streak ≥ 1` (`K_MIN_FORWARD_BEFORE_PLANNER_COMMIT=1`, `:168`) |
| cross-check | `orchestrator.py:447` → `completion_checker.py:815` | re-asks the **same** `_binary_reach_vlm_call` before honouring a planner `-1`; cap 3/leg, verdict memo; cap-exhaustion reverted to `True` (`:905`) |
| proactive REACH | `orchestrator.py:469` → `completion_checker.py:732` | fires at `forward_streak ≥ 4` then every 5 forwards, cap 12 (`:83,96,114`); Match=1 → inject `commit_burst` |
| side-scan | `orchestrator.py:504` → `:932,:971` | yaw toward the instruction's named side, re-ask, restore on miss |
| `EMPTY_FALLBACK` | `orchestrator.py:577` | nothing survived the gates → walk one forward |

**Every** commit path in the harness funnels through a single function,
`completion_checker._binary_reach_vlm_call` (`:646`), whose criterion (`sys_prompt`, `:686`) this
iter **reverted byte-for-byte to the anchor's identity phrasing**. That revert is the reason the
score moved. It also means one judge's ROC is the ceiling for the whole harness.

This iter's *new* lever was the **motion channel**: `spatial_memory.forward_stall_run()` (honest
replacement for the forbidden `info["agent_collision"]`), an escape ladder in
`active_perception.py:715-734` (strafe ×2 → rotate ×2 → yield), and the rewired STUCK backstop.

---

## 2. What actually happened (causal chain, with evidence)

### 2.1 The decisive trace — `ep_20_7_task_dist_21_0_1`: 65 consecutive forwards straight through the waypoint

ROAD waypoint 8 991 cm (L1) ahead; per-leg env budget `int(8991/100) = 89` substeps. The agent
emits **Move_Forward 65 times in a row**, with the REACH judge firing 11 times en route:

| step | L1 to waypoint | judge verdict |
|---|---|---|
| 8 | 5 491 | `{"Match": 0, "Reason": "Different lighting and building details"}` |
| 14 | 2 491 | `{"Match": 0, "Reason": "Different building colors and no visible lantern or electronics sign"}` |
| 19 | **9 cm** | *(no call — the schedule is keyed on forward count, not on arrival)* |
| 20 | 509 | `{"Match": 0, "Reason": "Different building colors and lighting conditions"}` |
| 26…62 | 3 509 → 21 509 | `Match: 0` ×8 |

The planner's rolling summary is the same sentence 11 times — *"The agent has moved forward 18/24/…
times … the current view still lacks the target landmarks — a red lantern and a blue electronics
store sign — on the right, indicating the agent has not yet reached the intersection."* It never
entertains having passed it. At step 56 the perceiver reports *"a flat, dark, featureless ground
surface … no visible buildings, landmarks or structures"* — the robot has walked **off the city
grid** (final pose `y = -1309`, the street network starts at `y ≈ 0`) — and still emits forwards
until `TrafficEnv._get_terminated` (`traffic_world.py:367`) kills the episode for leaving the road
polygon. Subtask 1 scores 0 after standing **9 cm** inside a 2 000 cm window.

### 2.2 The harness vetoed the only three correct arrival signals it ever received

`ep_20_19_task_dist_21_0_1` step 26, **L1 = 478 cm**:

- perceive: *"a wide, open paved area with a yellow ground/road surface, **indicating the agent has reached an intersection**. On the left, a large building with white stone walls and classical architectural features — tall columns, ornate…"*
- plan: *"The agent has **successfully reached the intersection** where the gray and green museum with a grand neoclassical design is visible on the left. The current view **matches the expected landmark description**…"* → emits `-1`, `verify()` passes.
- `cross_check_planner_commit` → `_binary_reach_vlm_call` → **`{"Match": 0, "Reason": "Different building designs and lighting conditions"}`**, twice at the same step. `XCHECK_BLOCK`.
- Episode walks 47 more steps and dies 235 m past the waypoint. Zero ROAD credit.

Identical shape at `ep_20_23_task_dist_11_0_1` step 10 (L1 = 878): plan *"successfully reached the
intersection and is aligned with the target building on the left… green-tinted glass facade,
dark-framed rectangular windows, metal structural elements, and a large billboard"* → xcheck
`Match: 0, "Different building features and lighting conditions"` → episode dead 9 steps later.

Across 50 episodes the planner raised **3** arrival claims while inside the window (out of 91
in-window plan calls) and **all 3** were killed by the cross-check or ran out before the commit
landed.

### 2.3 Why the judge says no: a 2.7× photometric domain shift between the live frame and the reference

Measured over all **788** ROAD-leg perceive frames (each carries `rgb` and the `robot_view_*` hint):

```
mean luminance   live RGB = 54.7      reference hint = 146.6      ratio 2.68x
 live RGB  p5/p50/p95 =  21 /  49 / 110
 hint      p5/p50/p95 = 109 / 153 / 185
```

The distributions barely overlap: the hint's 5th percentile is the live frame's 95th. The live RGB's
modal pixels are `[2,2,2]`, `[3,2,1]`, `[1,1,1]` — near-black. And the judge *tells us this is what
it is keying on*: **221 of 381 negatives (58 %)** name lighting —
`"Different building features and lighting conditions"` ×59,
`"Different buildings and lighting conditions"` ×59, `"Different lighting and building features"`
×19, `"Different lighting and building details"` ×14, `"Different building facades and lighting
conditions"` ×14, `"Different building colors and lighting conditions"` ×14, …

This is **not** the already-refuted "add an illumination-invariance clause to the prompt" axis
(parent §6). The *pixels* differ by a factor of 2.7 before the model ever sees them.

### 2.4 The FINISH channel is the inverse failure: 8 of 12 legs die on an instant false commit

| FINISH legs (12) | outcome |
|---|---|
| 8 | committed `-1` at **21 284 / 22 488 / 25 516 / 25 547 / 28 440 / 31 462 / 32 586 / 59 789 cm** — all after only **8–22** substeps of the leg. A failed `-1` terminates the episode (`traffic_world.py:653`), so each is an instant death. |
| 2 | walked the leg and **arrived** — `ep_20_4_task_dist_21_0_1` reached **27 cm**, `ep_20_4_task_dist_11_0_1` reached 767 cm — and **never committed** |
| 2 | never got within 30 m |

`per_position_SR[3] = 0.0` is therefore *two* different bugs, and the 8 instant commits separate
cleanly from the 2 true arrivals on an honest, oracle-free statistic: false commits all fired at
**≤ 22** substeps travelled on the leg; the true arrivals at **88** and **143**.

### 2.5 The iter-4 motion channel works, and is second-order

Implemented correctly and measurable: blocked forwards (Δodometry < 1 cm) fell from the parent's
**281 / 3 118 = 9.0 %** to **180 / 3 128 = 5.8 %**; the strafe chooser fired 18 times and **18/18**
produced real displacement (7 `Move_Left`, 11 `Move_Right`); STUCK fired 4× and in every case the
leg was genuinely entombed (closest approach already 23 steps in the past). But substeps are **not**
the binding resource: only **12 of 57** legs died of budget exhaustion, while **29** died holding an
unclaimed free pass. Reclaiming ~100 substeps across 50 episodes bought `distance_progress`
0.1256 → 0.1844 and no SubtaskSR. `static_collision_mean` *rose* 5.12 → 5.60 (anchor 1.78) because
collisions are produced by the 230 m over-walks grinding along facades, which the ladder does not
address.

---

## 3. Failure breakdown

**Episode terminal cause (50 episodes):**

| cause | n | note |
|---|---|---|
| left the road polygon / env bounds | **18** | walked off the city grid, `traffic_world.py:367` |
| failed `-1` commit → instant death | **16** | `traffic_world.py:653`; 8 of these are the FINISH legs of §2.4 |
| per-leg substep budget exhausted | **12** | `substeps_count >= int(L1_at_entry/100)`, `traffic_world.py:645-649` |
| harness STUCK backstop | 4 | this iter's new predicate, all 4 legitimate |
| MAX_STEPS / parse failure / crash / socket | **0** | loop hygiene is clean |

**Leg-level conversion (the primary loss):**

| | n |
|---|---|
| ROAD/FINISH legs reached | 57 |
| …that **entered** the 2 000 cm L1 success window | **40** |
| …that converted to a pass | **9 (22 %)** |
| ROAD only: entries → passes | 38 → 9 (24 %) |
| legs that never entered the window | 17 (10 of them FINISH) |
| **steps spent walking away from a window the agent already stood in** | **1 388 of 3 500 = 40 % of all env steps** |

Worst over-walks, all un-committed, all with the waypoint already satisfied:
`ep_20_7_21` 9 cm → 47 steps out · `ep_20_8_21` 6 cm → 44 out · `ep_20_19_21` 22 cm → 47 out ·
`ep_20_24_11` 37 cm → 43 out · `ep_20_17_21` 42 cm → 87 out · `ep_20_16_21` 65 cm → 108 out ·
`ep_20_2_21` → 152 out.

**The arrival judge's ROC, by true L1 distance (401 two-image calls):**

| L1 (cm) | n | Match=1 | rate |
|---|---|---|---|
| 0 – 500 | 14 | 5 | 0.357 |
| 500 – 1 000 | 16 | 4 | 0.250 |
| 1 000 – 2 000 | 15 | 0 | **0.000** |
| 2 000 – 5 000 | 64 | 1 | 0.016 |
| 5 000 – 10 000 | 88 | 2 | 0.023 |
| 10 000 – 20 000 | 93 | 0 | 0.000 |
| **> 20 000** | 111 | 8 | **0.072** |

TPR in-window 9/45 = **0.20**; FPR out 11/356 = 0.031 — the anchor's operating point, restored
exactly as §4-A of the parent gradient predicted (it quoted 21 % / 1.6 %). But the shape is the
story: the **> 200 m bucket has a higher positive rate than the 1 000–2 000 cm in-window bucket**.
The judge's 6.5× nominal lift is an artefact — most of its positive mass sits exactly where a commit
is instantly fatal, which *is* §2.4.

**Every language channel is equally blind.** Same 50 episodes, ROAD legs:

| readout | in-window rate | 5–20 km | > 20 km |
|---|---|---|---|
| planner arrival claim (`Match=1` or `-1` in `Actions`) | 0.033 | 0.026 | 0.039 |
| perceiver text asserts "reached / at an intersection" | 0.443 | 0.453 | 0.341 |
| binary REACH judge | 0.200 | 0.023 | 0.072 |

The perceiver's intersection *language* is completely uninformative (0.44 vs 0.45) — do **not**
build a commit gate on it.

---

## 4. Binding constraint — classify it

**Primary: VLM-CEILING on the arrival channel — and it is a ceiling on *language over raw RGB*,
which the harness can scaffold around.**

Three independent language readouts over the sanctioned observation all have arrival rates flat or
inverted in true distance (§3). Seven prompt reformulations have now been priced across this chain
and all lost; this iter's revert restored the best of them and it still converts only 22 % of window
entries. The frozen Qwen3-VL-30B cannot answer "am I there yet" from a night-dark egocentric frame
against a daylight reference. **Re-wording is closed. The fix must change what the decision is
computed from.** Two concrete, never-tried channel changes exist:

1. the comparison is being made across a **measured 2.68× photometric domain shift** that the model
   itself names in 58 % of its refusals (§2.3) — a pixel-level pre-processing defect, not a prompt
   defect;
2. a **zero-VLM geometric readout of the same frames is distance-monotone** where every language
   readout is flat (§5-A) — the harness already carries the JET decoder and segmentation-class
   machinery to compute it and has never used them for the commit decision.

**Secondary: TASK-GAP — no "I have passed it" state.** 1 388 of 3 500 env steps (40 %) are spent
walking away from a satisfied waypoint, and the over-walk is *terminal*: 18 of 50 episodes end by
leaving the road polygon. The parent gradient flagged this (§5-F) and this candidate explicitly
deferred it ("the largest open population, deliberately deferred" — NOTES.md). It is now the single
largest measured loss.

**Secondary: HARNESS-BUG, small but free.** `cross_check_planner_commit` (`:815`) routes to the
*same* `_binary_reach_vlm_call` with the *same* images at temperature 0 as the gate it is checking,
so it adds no independent evidence — it only re-asks. Measured: 22 veto events, **20** out-of-window
(genuine saves from instant death) and **2** in-window (`ep_20_19_21` 478 cm, `ep_20_23_11` 878 cm,
both destroying the leg). It is net-positive and must not simply be deleted, but it cannot be
*improved* while both sides of the gate share one blind judge. (5 of the 401 judge calls are
literally duplicate image pairs at the same step — pure waste, a footnote.)

**Not the constraint:** loop hygiene (0 MAX_STEPS, 0 parse failures, 0 crashes, 0 socket deaths),
ORIENTATION (50/50 at zero VLM cost), the TURNING latch (**7/9 legs converted**, keep verbatim),
locomotion (40/57 legs reach the window), cost (82 k tokens/ep, under budget), or the module graph.
Anti-cheat is clean: `read_audit.json` shows 0 forbidden / 0 undeclared reads.

---

## 5. Concrete changes to try (the gradient — ranked by leverage)

### A — Build a **non-VLM geometric arrival channel** from `object_mask` + `depth`, and let it drive the commit. `modules/completion_checker.py` (new helper beside `_depth_gray:346` / `_named_side_has_structure:514`), consumed at `orchestrator.py:469`

This is the one measurement in this diagnosis that found distance signal where the VLM has none.
Over the same 788 ROAD-leg frames, in the horizon band (rows 0.45–0.70 H, outer 25 % of columns each
side), using the already-identified classes (`object_mask` ground `[255,255,0]`, building
`[128,0,128]`) and the JET nearness proxy `R − B` from `obs['depth']`:

| L1 (cm) | n | `min(groundL,groundR)` | `max(bldgL,bldgR)` | `min` side-strip `R−B` (far ⇒ high) |
|---|---|---|---|---|
| 0 – 2 000 | 79 | **0.555** | **0.220** | **+16.6** |
| 2 000 – 5 000 | 123 | 0.452 | 0.315 | −1.8 |
| 5 000 – 20 000 | 360 | 0.485 | 0.326 | +1.1 |
| > 20 000 | 226 | 0.468 | 0.292 | +2.2 |

The lateral-depth statistic is **sharply peaked at the waypoint and ≈ 0 everywhere else** — this is
the geometry of an intersection: the building rows on both sides stop and the corridor opens
laterally. Untuned operating points, on frame fractions and class colours only (no numeric literal
≥ 2000, no map-tuned constant):

- `min(groundL,groundR) ≥ 0.55 AND max(bldgL,bldgR) ≤ 0.20` → in-window **0.25**, out **0.052**, lift **4.9×**
- `… AND min side-strip (R−B) ≥ the frame's own median` → in-window **0.13**, out **0.013**, lift **≈10×**

**Predicted effect:** this is the first readout whose positives are *not* concentrated in the far
tail, so unlike the VLM judge it can be trusted as a *positive* trigger. Two ways to spend it, in
increasing risk:
1. **Replace the firing schedule.** Today `proactive_commit_check` fires on `forward_streak ≥ 4`
   then every 5 forwards (`:83,:96`) — distance-blind, which is why `ep_20_7_21` made no call at
   step 19 (9 cm) and called at step 20 (509 cm). Fire the (unchanged, frozen) REACH judge **when
   the geometry says "intersection"**. Cheap, strictly additive, keeps the judge's 20 % TPR but
   concentrates its 12 fires on the frames that matter. Expect pos1 0.18 → ~0.30.
2. **Make it a commit channel in its own right** on ROAD legs (ROAD is graded on **position only** —
   `traffic_world.py:412-417` — no identity judgment is required to pass it), gated so it can fire
   at most once or twice per leg. The ~10 % conjunct is already a better in-window detector than the
   planner (0.033) and is free. Expect pos1 0.18 → 0.35–0.45 if the 29 walked-out legs convert at
   even half the detector's TPR.
   *Do not* also apply it to FINISH, which additionally needs the exact cardinal (§5-D).

### B — **Photometrically normalise both images** before `_binary_reach_vlm_call` encodes them. `modules/completion_checker.py:646-672` (the `numpy_to_base64` calls at `:668-669`)

The criterion string is frozen and must stay frozen; this changes the **input**, not the question.
Live frames average 54.7 luminance against the hint's 146.6 (2.68×, §2.3) and 58 % of refusals name
lighting. Insert, between loading and base64-encoding, a defensive normaliser applied **identically
to both** images: convert to grayscale (or L-channel), apply CLAHE / per-image histogram
equalisation, re-stack to 3 channels — or histogram-match the hint onto the current frame. Degrade
to the current behaviour on any exception or on a degenerate (flat) histogram.

**Predicted effect:** removes the confound the model explicitly cites, and it is the *only*
remaining way to raise the judge's 20 % TPR without touching the prompt. If the TPR moves to even
0.35 the whole harness moves with it, because every commit path shares this one function. Risk: the
FPR may also rise — so land it **with** §5-C so the far tail stays guarded. Measure the ROC table of
§3 again; the acceptance test is "the 0–1 000 cm buckets rise and the > 20 000 cm bucket does not".
**Explicitly not the exhausted axis**: parent §6 forbids an illumination *clause in the prompt*,
which was refuted; nobody has touched the pixels.

### C — Give the far-tail guard a channel that isn't the blind judge. `modules/completion_checker.py:815-930` + `:168`

The 8 fatal FINISH commits (§2.4) all fired at **≤ 22 substeps into the leg**; the 2 true arrivals
at **88** and **143**. `K_MIN_FORWARD_BEFORE_PLANNER_COMMIT = 1` (`:168`) cannot separate them, and
the xcheck cannot either (its own positive rate is *highest* beyond 200 m). Replace the
motion floor with a **leg-type-aware** one expressed in forward-action counts (no cm literal):

- a FINISH leg's destination is always ≥ 1 block beyond the TURNING leg, so require either a
  materially larger forward count **or** — better, and reusing §5-A — **at least one geometric
  intersection crossing within this leg** before any arrival claim on a FINISH leg is honoured;
- keep the ROAD floor at 1 (it is correct: ROAD waypoints can be reached in a handful of forwards —
  `ep_20_3_11` passed after 9 substeps).

**Predicted effect:** converts 8 instant deaths into 8 episodes that keep ~300 substeps of budget.
No direct pos3 gain (the agent is nowhere near the destination), but it is the precondition for any
FINISH progress at all, and for the 4 two-subtask episodes (`ep_20_10_21`, `ep_20_18_11`,
`ep_20_21_11`, `ep_20_5_11`, which died at steps 11/20/24/13) it restores a real shot at position 1.
Zero downside: the gate it tightens is **0 for 8** on this subset.

### D — An over-walk backstop grounded in §5-A, not in a streak heuristic. `modules/spatial_memory.py` + `modules/active_perception.py` + `orchestrator.py`

The parent sketched this (§5-F) as "stop forwards once `forward_streak` exceeds the largest streak
at which an arrival call was near-positive" — too weak to act on, because no call is ever
near-positive. With §5-A there is a real event to count: **the lateral opening the agent walks
through**. Track intersection crossings per leg in `SpatialMemory` (rising edge of the §5-A
predicate). A ROAD instruction says *"stop at **the** intersection"* — so once the leg has crossed
**more than one** opening without committing, the agent is past the waypoint. Then:
stop emitting `Move_Forward`, route to the side-scan path (`orchestrator.py:504`), and if that also
declines, commit anyway rather than continue — on a ROAD leg a commit from inside the last opening
is a coin flip whereas another 47 forwards is a certain zero.

**Predicted effect:** attacks the 1 388 wasted steps and the 18 off-map deaths directly. Causal
reason: the loss is not that the agent fails to reach the window (it reaches it 40/57 times) but
that nothing in the harness can ever conclude "behind me". Highest risk of the set — it converts
silent misses into explicit wrong commits, each of which is terminal — so **land A first and
re-measure**, then gate D on "the leg has crossed ≥ 2 openings", never on a bare step count.

### E — Keep verbatim; do not spend an iter on these

`active_perception` cardinal burst (a) — ORIENTATION 50/50, zero VLM cost, four iters running.
Turn-approach latch (a″, `_TURN_APPROACH_FORWARDS = 2`) — **TURNING converted 7/9 legs this run**;
it works. `commit_burst` (`:568`) — still essentially unexercised (FINISH never passed), template
only. The escape ladder + `forward_stall_run` + STUCK backstop shipped this iter — correct, measured
(9.0 % → 5.8 % refused forwards), and second-order; leave them alone rather than tuning
`_MAX_ESCAPES_PER_SUBTASK` / `_ESCAPE_SIDE_MIN_MARGIN` for the remaining 3 pp.

**One env fact worth encoding if §5-D is attempted** (`traffic_world.py:645-649`, `:683-686`): the
per-leg budget is `int(L1_at_leg_entry / 100)` substeps and the `-1` itself burns one, so a leg with
budget *B* allows only *B−1* movement actions — and `ep_20_10_task_dist_11_0_1` opened its ROAD leg
already **154 cm** from the waypoint, i.e. budget 1, **zero** usable movement actions. The only
passing play was an immediate `-1`, which `K_MIN_FORWARD_BEFORE_PLANNER_COMMIT = 1` and
`K_REACH_FORWARD_STREAK = 4` make structurally impossible. Do **not** read `env.subtask_max_steps`
(forbidden, laundered `target_location`) — but a geometric detector allowed to fire at
`forward_streak = 0` recovers this degenerate class honestly.

---

## 6. Exhausted — do NOT repeat

- **Do not re-word, re-shape or "compromise" the arrival criterion.** `sys_prompt` at
  `completion_checker.py:686` is **frozen**. Seven reformulations priced; this iter's revert to the
  anchor's identity phrasing restored the anchor's exact operating point (20 % TPR / 3.1 % FPR,
  §3). The ceiling is in the *channel*, not the words — §5-A and §5-B are the ways out.
- **Do not add an illumination-invariance clause to any prompt.** Already refuted (parent §6); the
  refusals simply migrate. §5-B is pixel pre-processing, which is a different lever entirely — do
  not conflate them and do not skip it on this bullet's authority.
- **Do not tune `K_REACH_FORWARD_STREAK` / `K_REACH_INTERVAL` / `MAX_REACH_FIRES_PER_SUBTASK`**
  (`:83,:96,:114`). Run to death in iters 006–010 and re-confirmed here: coverage is not the term.
  11 calls fired on `ep_20_7_21`'s leg and 11 said no. *Re-keying the schedule on geometry (§5-A.1)
  is not the same as tuning its constants.*
- **Do not treat a cost cap as evidence.** Reverted this iter (`:905`); the revert is part of why
  the score recovered. Leave it.
- **Do not build a commit gate on the perceiver's intersection language.** Measured flat: 0.443
  in-window vs 0.453 at 5–20 km (§3). Tempting and worthless.
- **Do not delete `cross_check_planner_commit` outright.** 20 of its 22 vetoes were genuine saves
  from an instantly fatal commit; only 2 were false. Fix its *evidence* (§5-B/§5-C), not its
  existence.
- **Do not touch the ORIENTATION cardinal burst, the turn-approach latch, or `commit_burst`.** §5-E.
- **Do not "repair" the motion channel further.** It is correct and its headroom is ~3 pp of
  forwards; only 12 of 57 legs die of the substep budget. Spend the iter on the arrival channel.
- **Do not read** `agent_collision`, `agent_location`, `target_location`, `destination`,
  `correct_robot_location`, `correct_robot_position_area`, `ideal_path`, `gt.json`,
  `criteria.dist_threshold`, `env.subtask_max_steps`, or `current_instruction`.
- **Avoid numeric literals ≥ 2000** (lint flags them as map-tuned). §5-A's operating points are all
  frame fractions, class colours and per-frame medians — keep them that way.

---

## 7. Provenance

- Candidate: `$OUT_DIR` = `…/prop_r004_fe2b86/cand_r004_0_spatial-grounding` — `orchestrator.py` (673 ln)
  + 7 modules; this iter's diff vs parent = 4 files (`orchestrator.py`, `modules/spatial_memory.py`,
  `modules/active_perception.py`, `modules/completion_checker.py`).
- Eval: `cand_r4_0_spatial-grounding_r004/score.json`, subset `search_hard_50`, **50/50 episodes
  complete**, 0 missing, 0 skipped; `read_audit.json` verdict **PASS** (0 forbidden, 0 undeclared,
  0 escalated reads; 465 reads/episode over 14 declared paths).
- Traces: `$TRACE_DIR` (50 episodes) — `trajectory.json`, `verdict.json`, `env_step_log.jsonl`,
  `llm_calls.jsonl`, and the saved `frames/*.png` the agent actually saw.
- Measurements in this document: 3 500 env steps; 57 ROAD/FINISH legs; **401** two-image arrival
  calls bucketed by true L1 distance; 1 071 planner calls; **788** ROAD-leg frame quadruples decoded
  for the luminance and `object_mask`/`depth` statistics of §2.3 and §5-A.
- Env semantics quoted from `SimWorldGym/simworld_gym/envs/traffic_world.py` —
  termination `:365-386`, per-leg budget `:636-649`, counters `:683-686`, subtask grading `:388-462`
  (`dist_threshold = 2000`, L1, `ord=1`; FINISH additionally needs a one-sided 45° yaw match),
  failed-`-1` termination `:653`, `info['success']` `:309-313`.
- Chain context: anchor 0.360 · r001 0.285 · r002 0.330 · r003 0.360 · **r004 0.355** (this).
- **Anti-cheat:** ground-truth state (`gt.json`, `env_step_log` coordinates, `dist_threshold`) was
  used **only** to compute the rates in this diagnosis. Every fix in §5 decides from `obs['rgb']`,
  `obs['depth']`, `obs['object_mask']`, `info['moving_distance']`,
  `info['agent']['agent_rotation']` and the static `task_spec['subtasks']` text — the same channels
  the candidate already declares. §5-A's thresholds are frame fractions, segmentation class colours
  and per-frame medians, carrying no episode-specific or held-out constant.
