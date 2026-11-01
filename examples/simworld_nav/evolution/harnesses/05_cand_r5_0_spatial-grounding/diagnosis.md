# Diagnosis — SimWorld-MMNav (hard, `search_hard_50`) / `anchor_nav`, arm `hard5010_adaptive_k4c1`

**Headline:** SR **0/50** · SubtaskSR **0.360** · per-leg **[ORIENT 50/50, MOVE 12/50, TURN 5/45,
REACH 0/45]** · per-type **ROAD 12/45 · TURNING 5/45 · FINISH 0/50** · 88 k tok/ep · 800 s/ep ·
`over_budget: false` · `distance_progress_mean` 0.145.

**Binding constraint (one sentence):** the arrival recognizer is asked about **1.10 frames inside a
window that contains 6.61 usable frames**, so the episode-level commit probability collapses to
`1 − (1 − 0.205)^1.10 = 0.22` — which is, to two digits, the observed ROAD pass rate — and the
missing factor is a **harness sampling policy** (`K_REACH_INTERVAL` + the burst executor), not the
frozen model's ceiling.

> **Scalar hygiene.** 50 episodes, temp 1.0, one run each. Every number below is a per-frame or
> per-step measurement pooled over all 50 episodes from `env_step_log.jsonl`, `vlm_trajectory.jsonl`
> and `gt.json`. Oracle coordinates were used **for measurement only** (§7). `per_position_SR[1]`
> pools two leg types: 45 episodes are ORIENTATION→ROAD→TURNING→FINISH, 5 are ORIENTATION→FINISH.
>
> **Relation to prior gradients.** `cand_r3_…/source/diagnosis.md` diagnosed this same eval and
> established the *reach-but-no-commit* headline; r1–r4 have since been evaluated (0.285 / 0.33 /
> 0.36 / 0.355 vs anchor 0.36), and `cand_r4_…/source/diagnosis.md` adjudicated their four edits.
> This pass does **not** restate that. It adds: (a) the two-factor decomposition of the commit
> probability and the measurement that **coverage, not recall, is the fixable factor**; (b) the
> proof that the recognizer is **deterministic per frame**, which kills the obvious
> vote/self-consistency proposal; (c) the **per-leg-type confusion matrices** (ROAD prec 0.90 vs
> FINISH prec 0.00) and an exact, free, text-only leg discriminator; (d) the discovery that **two
> episodes stood 16 cm and 27 cm from the destination** and were never asked; (e) two newly priced
> **refutations** (§6.4, §6.5).

---

## 1. What this harness does (mechanism)

`orchestrator.py::run_episode`, one iteration per *decision* (not per env step). Seven modules; the
commit is decided in exactly four places.

| # | call site | role in the outcome |
|---|---|---|
| 1 | `active.should_emit` (`orchestrator.py:301`) | **short-circuits perceive+plan** when it returns a list. Owns (a) the ORIENTATION cardinal burst `[rot…, -1]`, (a'') the relative-turn burst `[rot, -1]`, (a') the red-signal idle, (b) anti-spin, and (d) the **cap-2 forward-momentum** cadence. |
| 2 | `perceiver.perceive` | 1 VLM call, 4 images (RGB, hint, seg mask, JET depth) → free-text `Description`. **2713 in / 214 out tok, 4.1 s.** |
| 3 | `planner.plan` | 1 text-only VLM call → `{Summary, Match, Actions}`. **1241 in / 230 out tok, 2.9 s.** |
| 4 | `completion.verify` (`:391`) | honours a planner `-1` only if `forward_streak ≥ 1` on a non-orient leg. |
| 5 | `completion.cross_check_planner_commit` (`:413`) | binary arrival VLM call gating a planner `-1`; **fails OPEN** once `MAX_XCHECK_FIRES_PER_SUBTASK = 2` is spent. |
| 6 | `completion.proactive_commit_check` (`:430`) | **the main commit path.** Fires at `forward_streak ≥ 4`, then every `K_REACH_INTERVAL = 5` forwards, cap 12/leg: one binary arrival call `(live RGB, hint) → {Match}`. **1041 in / 21 out tok, 0.5 s.** |
| 7 | `completion.request_side_scan` / `scan_reach` (`:458`) | depth-gated physical yaw toward the named side, re-ask, yaw back on miss. |

**Cadence, measured.** 918 perceive + 918 plan + 348 arrival calls over 3014 env steps ⇒ one
perceive+plan per **3.28 env steps** (the cap-2 momentum burst: PP → m1 → m2), and one arrival call
per **2.6 perceive calls**.

**A structural gap that everything below turns on: this harness has no leg-type discrimination.**
`grep -n "INTERSECTION\|_is_finish" modules/completion_checker.py` returns nothing outside comments.
ROAD and FINISH legs get the identical prompt, the identical gates, the identical caps — while their
measured operating points are opposite (§2.3).

---

## 2. What actually happened (causal chain, with evidence)

### 2.1 The agent *gets there*. 40/50 episodes drive through the leg-1 pass window.

For each episode, closest L1 approach to `evaluation_of_subtask[1].correct_robot_location` during the
leg-1 walk, threshold 2000:

| | eps |
|---|---|
| entered the window (L1 ≤ 2000) and **passed** the leg | **12** |
| entered the window and **failed** the leg | **28** |
| never entered the window | 10 |

Median closest approach among the 28 failures: **139 cm**. Best: **9 cm**
(`ep_20_7_task_dist_21_0_1`, walk-step 17 — then it walked **23 500 cm** further and died).
Distribution of the "overshoot after closest approach" among the 28: 0 / 2 776 / 3 000 / 3 275 /
… / 47 900 / 53 870 / **65 079** cm.

Navigation is **not** the bottleneck. Saying `-1` is.

### 2.2 The commit probability is a product of two factors, and only one is the model's fault

Restricting to the 38 ROAD-leg windows and measuring what happens *inside* the window:

| quantity | measured |
|---|---|
| window depth | **8.3 env steps**, of which **6.61 are forward steps** (median 8) |
| perceive calls fired inside the window | **2.25** (median 2) |
| **arrival calls fired inside the window** | **1.10** (median 1; **6 episodes get zero**) |
| per-fire in-window `Match=1` rate | **9/44 = 0.205** |

```
P(commit | entered window)  =  1 − (1 − 0.205)^1.10  =  0.22
observed                    =  9 ROAD passes from the REACH path / 38 windows = 0.24
```

The model reproduces the outcome to two digits. Now read the same formula as a headroom table:

| arrival calls per window | source of that number | P(commit) |
|---|---|---|
| **1.10** | **today** (`K_REACH_INTERVAL = 5`) | **0.22** |
| 2.25 | one call per perceive (drop the interval gate) | 0.40 |
| 6.61 | one call per **forward step** in the burst | **0.75** |

**The 6× gap between 1.10 and 6.61 is pure harness policy.** `completion_checker.py:653-655` spaces
fires 5 forwards apart = **2500 cm of travel**, against a pass window that is only ~4000 cm deep; and
`orchestrator.py:525` executes the momentum burst without ever re-consulting the checker between forwards.
The window is sampled roughly once, by construction.

### 2.3 The recognizer is **deterministic per frame** — so resampling is worthless and more *distinct* frames is the only lever

10 pairs of arrival calls in this run share a byte-identical prompt (same frame, same hint, same
instruction). **10/10 agree** (all `Match: 0`). Examples: `ep_20_17_task_dist_11_0_1` step 14 → 0,0;
`ep_20_24_task_dist_11_0_1` step 8 → 0,0; `ep_20_19_task_dist_21_0_1` step 26 → 0,0.

Yet the answer *does* vary across nearby frames of the same scene — `ep_20_9_task_dist_11_0_1` has
`m1=2, m0=11`; `ep_20_3_task_dist_21_0_1` has `m1=1, m0=10`; `ep_20_1_task_dist_11_0_1` has
`m1=1, m0=6`. The variance is **positional, not sampling noise**.

**Consequence for the proposer:** a k-vote / self-consistency / temperature-ensemble wrapper on the
same frame buys **exactly nothing** (this is why §5-A is "more frames", not "more votes"). Conversely
it is why more frames works: consecutive frames are genuinely independent draws.

### 2.4 The recognizer is two different classifiers on the two leg types — and the harness treats them as one

Fire-level confusion, truth = `L1(pos, leg.correct_robot_location) ≤ 2000`:

| leg | fires | `Match=1` | TP | FP | FN | TN | **precision** | recall |
|---|---|---|---|---|---|---|---|---|
| **ROAD** | 305 | 10 | 9 | 1 | 34 | 261 | **0.90** | 0.21 |
| **FINISH** | 43 | 7 | **0** | **7** | 1 | 35 | **0.00** | 0.00 |

Every one of the 7 FINISH `Match=1` events fired at **L1 7 778 – 58 789 cm** — up to **588 forward
steps** from the destination — because the destination tower is *visible* from across the map:

```
ep_20_10_task_dist_21_0_1  step 10  L1 = 58 789   -> -1, episode over (622 substeps unspent)
ep_20_3_task_dist_21_0_1   step 73  L1 = 35 733
ep_20_5_task_dist_11_0_1   step 10  L1 = 32 586
ep_20_16_task_dist_11_0_1  step 18  L1 = 26 273
ep_20_21_task_dist_11_0_1  step 20  L1 = 22 988
ep_20_4_task_dist_11_0_1   step 48  L1 = 15 733
ep_20_9_task_dist_11_0_1   step 99  L1 =  7 778   (it had been at 3 778)
```

**A free, exact discriminator already exists in the sanctioned surface.** Over all 348 arrival calls:
instructions on **ROAD legs contain the literal phrase "at the intersection" in 305/305**; on
**FINISH legs, 0/43** (they read *"Proceed forward until reaching … stop and face …"*). This is
`task_spec['subtasks']` text — no oracle, no coordinates, 100 % separation.

### 2.5 FINISH is **starved, not hopeless** — two episodes stood 16 cm and 27 cm from the destination and were never asked

Only 10 of 50 episodes ever reached a FINISH leg. Closest approach on that leg:

| episode | finish-leg steps | start L1 | **min L1** | end L1 | outcome |
|---|---|---|---|---|---|
| `ep_20_18_task_dist_11_0_1` | 147 | 32 516 | **16** | 36 484 | **never committed**; 1 arrival call fired in-window |
| `ep_20_4_task_dist_21_0_1` | 143 | 60 027 | **27** | 10 973 | **never committed**; **0** arrival calls fired in-window |
| `ep_20_9_task_dist_11_0_1` | 67 | 18 338 | 3 778 | 7 778 | false commit at 7 778 |
| other 7 | 6–21 | 25 k–62 k | 15 k–58 k | — | false commit / out of budget |

`ep_20_4_task_dist_21_0_1` walked to **27 cm** from the destination and the harness **never asked the
question there**. That is §2.2's coverage defect, on the leg that scores 0/45.

Note the accounting: FINISH is always the **last** leg, so a false FINISH commit forfeits only the
FINISH leg, which was failing anyway. **The 7 FINISH false positives cost ≈ 0 SubtaskSR.** FINISH is
0/45 because the agent rarely arrives there and is never asked when it does — not because it
commits early.

### 2.6 TURNING: the turn is always in the right direction; **7/7 failures are a sub-two-step position shortfall**

All 12 reached TURNING legs were handled by the closed-form burst at `active_perception.py:430-440`:
**`nfwd = 0, nrot = 1`, then `-1`** — zero translation. Reconstructing yaw from the action log
(calibrated against 2609 observed forward displacements: yaw 0→+x, 90→+y, 180→−x, 270→−y):

```
required orientation achieved : 12 / 12      <-- the turn logic is correct, always
inside correct_robot_position_area : 5 / 12
pass                               : 5 / 12
```

`inArea` predicts `pass` **12/12**. The 7 failures are outside the box by **78, 356, 452, 506, 627,
692, 980 cm** — every one of them **< 2 forward steps (1000 cm)**, and every one of them *short* along
the travel axis. All 5 passes had shortfall 0. This independently reconfirms r4's §5-B (E1 approach
latch) from the anchor's own traces; r4 additionally verified the structural reason on 181/181 hard
task_configs (the ROAD waypoint sits 500 cm inside the TURNING box while ROAD tolerates 2000 cm of
undershoot) and has a 1/1 live conversion.

### 2.7 The planner is a *worse* arrival detector than the recognizer, and it leaks past a gate that fails open

| detector, ROAD legs | precision | recall | proposals |
|---|---|---|---|
| binary arrival call | **0.90** | 0.21 | 10 / 305 fires |
| planner `-1` in `Actions` | **0.36** | 0.12 | 28 / 788 plan calls |

Commit provenance and outcome:

| leg | source | pass | fail |
|---|---|---|---|
| ROAD | arrival `Match=1` | **9** | 1 |
| ROAD | planner `-1` (survived xcheck) | 3 | **5** |
| TURNING | turn burst | 5 | 7 |
| FINISH | arrival `Match=1` | 0 | 7 |

The 5 planner leaks share one signature — **the xcheck said no, then the cap let the same claim
through three steps later**:

```
ep_20_17_task_dist_11_0_1  step 11  planner -1, L1=  574   xcheck fires -> Match 0 -> XCHECK_BLOCK
                           step 14  planner -1, L1= 2074   xcheck fires -> Match 0 -> XCHECK_BLOCK
                           step 17  planner -1, L1= 3074   cap spent -> return True -> COMMIT, FATAL
ep_20_24_task_dist_11_0_1  step  8  blocked (L1=  ~1.5k) ; step 11 cap spent -> COMMIT at L1 3537
ep_20_18_task_dist_21_0_1  step 15  blocked (L1=  388)   ; step 21 cap spent -> COMMIT at L1 2888
ep_20_19_task_dist_21_0_1  step 26  blocked (L1=  478)   ; step 32 cap spent -> COMMIT at L1 2022
```

`cross_check_planner_commit` (`completion_checker.py:748-749`) returns `True` on cap exhaustion. A
spent **cost** budget is being read as **evidence of arrival**, and because the cap is per-leg it
fails open exactly on long legs — i.e. exactly when the agent is most likely past the waypoint. Four
of these five die at L1 2022–3537 having been at L1 22–574 a few steps earlier.

### 2.8 The harness's own STUCK detector is dead code; the env has a wedge terminator the harness does not model

`orchestrator.py:280`: `forward_count = sum(1 for a in chosen_actions if a == 0)` — `chosen_actions` is the
**current burst**, which is `[0]` on a momentum tick. `forward_count > 3` is therefore unreachable in
the common path; STUCK fired in **3/50** episodes. Meanwhile `traffic_world.py:371-382` terminates the
episode when the **last 20 positions are all within 1 cm**, which killed 4 episodes outright
(`ep_20_0_task_dist_21_0_1` — 20 straight forwards into a building with `moving_distance` frozen at
56 672 and `building` collisions climbing 1→20; `ep_20_22_task_dist_21_0_1` — same against a
pedestrian at `moving_distance` 1 371.5). 269 / 3014 env steps (**8.9 %**) produced zero displacement.
`spatial_memory.py:97` already exposes `forward_count_since_boundary()` and it is unused.

### 2.9 The live perceive prompt ships two contradictory segmentation legends, and the wrong one wins 70 % of the time

`prompt_assembler.py:64` builds the system prompt from `perception_template(strip=False, …)`, which
contains the baseline's *"Green = trees, purple = buildings, yellow = sidewalks/crosswalks, black =
driveways"*, and then **appends** a `SENSOR LEGEND (authoritative …)` block at char 3083 of 4509
saying *"green = SKY … there is no 'trees' and no 'driveway' class"*. Both are in every one of the 918
perceive prompts. The appended override loses:

- **644 / 918 (70 %)** perceive responses mention "tree(s)".
- It flips mid-episode on the same scene: `ep_20_0_task_dist_21_0_1` step 11 — *"buildings (purple),
  ground/road (yellow), and sky (green)"*; step 40 — *"Trees with green foliage are visible along the
  street, and the sky is a dark green"*; step 43 — *"there are green trees along the street"*.

The description feeds the planner, which is one of the two commit channels. This is a
prompt-composition bug, not a wording preference: the fix is to **remove** the contradictory line,
not to append a louder correction.

---

## 3. Failure breakdown

**Episode termination cause (n = 50):**

| cause | eps | detail |
|---|---|---|
| **Fatal bad `-1`** | **20** | 7 TURNING (position shortfall < 1000 cm), 6 ROAD (4 of them at L1 2022–3537), 7 FINISH (L1 7 778–58 789) |
| Env per-leg substep deadline / global cap | 23 | 9 hit `int(L1/100)` exactly; the agent had walked past the waypoint and kept going |
| **Env 20-identical-positions wedge** | 4 | harness never noticed (§2.8) |
| Harness STUCK | 3 | the only 3 times the dead predicate fired |

**Per-leg attribution (the thing the score is made of):**

| leg | rate | dominant cause | class |
|---|---|---|---|
| ORIENTATION | 50/50 | closed-form cardinal burst, 0 VLM calls | **solved — do not touch** |
| ROAD | 12/45 | in-window arrival calls = 1.10 of 6.61 available (§2.2) | **HARNESS-BUG** (coverage) + VLM-CEILING (0.205/fire) |
| TURNING | 5/45 | 7/7 = position shortfall 78–980 cm; `[rot, -1]` has zero translation (§2.6) | **HARNESS-BUG** |
| FINISH | 0/45 | 35 never reach the leg; of the 10 that do, 2 arrive and are never asked (§2.5) | **TASK-GAP** downstream of ROAD/TURNING |

Parse failures: 1 episode (`NO_ACTION_RETURNED`). Illegal actions: 0. Crashes / socket deaths: 0.
Read-audit: **0 forbidden reads, 0 undeclared reads** on all 50 episodes.

---

## 4. Binding constraint — classified

**Primary: HARNESS-BUG (sampling coverage).** §2.2 is a closed-form account of the ROAD rate:
`1 − (1 − 0.205)^1.10 = 0.22` vs observed 0.24. The recall term 0.205 is the frozen model's ceiling
and the chain has spent seven attempts on it (§6.1). The **exponent is ours**: `K_REACH_INTERVAL = 5`
spaces fires 2500 cm apart inside a ~4000 cm window, and the burst executor (`orchestrator.py:525`) never
re-consults the checker between the momentum forwards. Raising the exponent from 1.10 to 6.61 costs
**1041 input tokens and 0.5 s per extra call** — the arrival call is 9.1 % of input tokens and ~7 % of
the latency of a perceive+plan pair, so a 6× increase is ≈ +38 % tokens and ≈ +3 % wall-clock.

**Second: HARNESS-BUG (undifferentiated leg types).** One prompt, one gate set, one cap for a
classifier whose precision is 0.90 on ROAD and 0.00 on FINISH (§2.4) — with a 305/305 vs 0/43 exact
text discriminator sitting unused in `task_spec['subtasks']`.

**Third: HARNESS-BUG (a veto that fails open).** The xcheck cap converts a correct refusal into a
three-step delay (§2.7); 4 of its 5 leaks are fatal commits at L1 2022–3537 after correct blocks at
L1 388–574.

**Fourth: HARNESS-BUG (TURNING translation).** `[rot, -1]` commits from the ROAD stopping point;
7/7 failures are < 2 forward steps short (§2.6). Already adjudicated **CORRECT — ship it** by r4's E1
with a 1/1 live conversion.

**VLM-CEILING is real but is *not* the top of the list.** 0.205 per-frame recall with a 2–4× hint/live
illumination gap, deterministic per frame (§2.3), seven refuted reformulations (§6.1). Treat it as a
**fixed constant to be amortised over more frames**, not as something to argue with.

**TASK-GAP: the agent has no notion of "I have gone too far."** It walked 23 500 – 65 079 cm past the
waypoint in the 28 failures and never changed tactic. Real, but it should be attacked **after** the
coverage fix, because the coverage fix changes the miss distribution it would be tuned against.

---

## 5. Concrete changes to try (the gradient — ranked by leverage)

### A — Ask the arrival question once per forward step, not once per 5. `orchestrator.py:525` + `modules/completion_checker.py:96,653-655` — **the whole thesis of this diagnosis**

Two edits, smallest first; ship **A2 only if A1 lands**, so the exponent is attributable.

**A1 (cheap, 2 lines): delete the interval gate.** `completion_checker.py:653-655`

```python
# WAS: space fires K_REACH_INTERVAL=5 forwards apart -> 2500 cm between samples,
# against a pass window that is only ~4000 cm deep (diagnosis 2.2). The spacing was
# tuned when the per-leg cap was 2; the cap is now 12 and is NOT the binding term
# (median fires per ROAD leg = 6, cap reached in 0/38 windows).
# if last is not None and forward_streak - last < K_REACH_INTERVAL:
#     return False
```
Raises in-window fires **1.10 → ~2.25** (bounded by the perceive cadence). Predicted
`P(commit | window) 0.22 → 0.40`.

**A2 (the real fix): fire inside the burst executor.** In `orchestrator.py`'s `for chosen_action in
list(chosen_actions):` loop (`:525`), after a `chosen_action == 0` step lands and `obs` is refreshed,
call `completion.proactive_commit_check(...)` with the updated `obs`; on `Match=1`, break the burst
and execute `-1`. This is the only way to reach the 6.61 frames — the momentum burst currently walks
1500 cm between decisions.

- **Why it works, mechanistically:** §2.3. Identical frames give identical answers (10/10), so votes
  are worthless; *different* frames are independent draws, so the OR over `n` frames is
  `1 − 0.795^n`. The harness has been taking `n = 1`.
- **Predicted:** ROAD `12/45 → 25–30/45`; SubtaskSR **0.36 → 0.47–0.52** before any downstream leg.
- **Cost, measured:** arrival calls 348 → ~2000 ⇒ +34 k tok/ep (88 k → ~122 k, `over_budget` was
  `false` with headroom) and +~17 s/ep on 800 s. **Verify against the arm's cost cap before shipping
  A2**; A1 alone is ~+9 k tok/ep.
- **Quantified FP risk:** ROAD fire-level specificity is **261/262 = 0.9962**. At 6× density expect
  ~6 far-field ROAD false commits (vs 1 today) against ~19 gained passes. Net strongly positive, but
  **not free** — pair with A3.
- **A3 (guard, ship with A2): require 2 `Match=1` on *consecutive* frames.** Note this is **not**
  the refuted `N_REACH_CONFIRMATIONS_REQUIRED = 2` of iter 8: that demanded two positives **2500 cm
  apart** and could not be satisfied inside a 4000 cm window; at 500 cm spacing it is a 1-step
  confirmation. **Honest caveat: this cannot be priced offline from these traces** (no two
  consecutive-frame arrival calls exist in the run). Ship it behind a flag, or ship A2 bare first and
  read the FP count.

### B — Split the arrival policy by leg type. `modules/completion_checker.py` (new module-level predicate, used in `proactive_commit_check:606` and `cross_check_planner_commit:689`)

```python
_INTERSECTION_RE = re.compile(r'\bat the intersection\b', re.IGNORECASE)

def _is_intersection_leg(instruction) -> bool:
    """ROAD legs say 'stop at the intersection with …'; FINISH legs say
    'Proceed forward until reaching … stop and face …'. Measured on this arm:
    305/305 ROAD calls match, 0/43 FINISH calls match. Pure task_spec text."""
    return isinstance(instruction, str) and bool(_INTERSECTION_RE.search(instruction))
```

Then, on **non**-intersection (FINISH) legs:
1. keep density high (§2.5: two episodes arrived at 16 cm / 27 cm and were **never asked**), and
2. put the *approach* conjunct only here — e.g. require that the leg's arrival call has previously
   returned `Match=0` at least `k` times, or that `forward_streak` exceeds the streak at which the
   first `Match=1` of the leg occurred. **Do not** reuse the ROAD gates.

- **Why:** §2.4. One classifier, two operating points (0.90 vs 0.00 precision), currently one policy.
- **Predicted:** FINISH `0/45 → 2–4/45` from the two never-asked arrivals plus the density fix; the
  7 FINISH false commits are **score-neutral** (§2.5) so the upside here is real but modest — it is
  ranked below A and C because FINISH is starved upstream.
- **Anti-cheat:** `task_spec['subtasks']` text only. No coordinates, no `gt.json`, no leg-type field.

### C — Approach latch before the relative-turn commit. `modules/active_perception.py:430-440`

Emit `[0] * _TURN_APPROACH_FORWARDS + [rot, -1]` instead of `[rot, -1]`, with
`_TURN_APPROACH_FORWARDS = 2`, stall-guarded off `moving_distance`.

- **Why:** §2.6 — 7/7 TURNING failures are shortfalls of 78–980 cm; 5/5 passes have shortfall 0;
  `inArea` predicts `pass` 12/12; direction is correct 12/12.
- **Independently adjudicated:** r4's §5-B marks this **CORRECT — ship it** with a 1/1 live
  conversion (`ep_20_9_task_dist_21_0_1`, 432 cm outside → inside → TURNING PASS) and the structural
  proof on 181/181 hard task_configs.
- **Predicted:** TURNING `5/45 → ~12/45`, SubtaskSR +0.035. Compounds with A (more ROAD passes ⇒ more
  TURNING legs reached).
- **Do not raise above 2** — r4 measured 45/181 tasks with only 1500 cm of overshoot tolerance.

### D — Make the xcheck cap mean "cost spent", not "arrival confirmed". `modules/completion_checker.py:748-749`

Cache the verdict instead of deferring: memo the last `Match` for `(cur_idx, forward_streak_bucket)`
and, on cap exhaustion, **return the memoized verdict** rather than `True`.

- **Why:** §2.7. In 4/5 leaks the xcheck had *already answered 0* at L1 388–574, three steps earlier.
  Returning the remembered `0` blocks the fatal commit **for free** (no VLM call).
- **This is not r1's E3.** r4 measured E3's blanket `return False` as **harmful** (`XCHECK_BLOCK`
  18 → 117, four in-window commits refused with no evidence, 35-deep lock-in loops). The memo is the
  half of E3 that r4 explicitly kept. Returning a *remembered answer* is evidence; returning a
  *blanket False* is not.
- **Predicted:** removes 4–5 fatal commits. On its own it converts fatal→deadline (score-neutral on
  the leg), but it is what makes A2's extra samples *reachable* — today the episode is dead before
  they happen.

### E — Delete one legend line; stop appending a correction. `modules/prompt_assembler.py:64`

Call `perception_template(strip=True, …)` (or strip the `Green = trees … black = driveways`
sentence) and keep only the `SENSOR LEGEND` block already written at `:65-79`.

- **Why:** §2.9 — both legends ship in all 918 prompts and the wrong one wins **644/918 (70 %)**.
- **Predicted:** small and hard to attribute alone, but it is free, it is a strict removal, and the
  description is an input to the planner commit channel. **Ship it bundled with A1, not as a
  candidate of its own.**

### F — Make STUCK real. `orchestrator.py:280`

```python
forward_count = memory.forward_count_since_boundary()   # accessor exists, spatial_memory.py:97
```
- **Why:** §2.8 — the current expression reads the 1-element burst, so the predicate is unreachable;
  4 episodes died to the env's own 20-identical-positions terminator instead (`traffic_world.py:371`).
- **Predicted:** recovers up to 269 wasted substeps (8.9 % of all env steps) and 4 episodes' worth of
  remaining budget. Low ceiling, near-zero risk.

### G — (after A–D land) Over-walk backstop for the TASK-GAP

Once A changes the miss distribution, revisit: track `forward_streak` since the leg began and, past
the largest streak at which any arrival call in this leg was even close, stop emitting forwards and
switch to the side-scan path. **Do not read `env.subtask_max_steps`** — it is `int(L1_to_target/100)`,
i.e. laundered `target_location`. Lowest confidence; re-measure first.

---

## 6. Exhausted — do NOT repeat

1. **Do not re-word, re-shape, or re-criterion the arrival question.** Six offline reformulations
   plus graded confidence (pre-hard chain), then r1's E4 priced on all 50 episodes: in-window TPR
   **21 % → 1.7 %**, commit precision 57 % → 14 %, reason-string entropy 97 distinct/348 →
   33 distinct/420 with 81 % verbatim echoes of the prompt's own disqualifier. **Seven refutations.**
2. **Do not add a vote / self-consistency / temperature ensemble on the same frame.** **New this
   pass:** 10/10 byte-identical prompts return byte-identical verdicts (§2.3). Zero information.
3. **Do not touch ORIENTATION.** 50/50 here, 20/20 and 2/2 in earlier arms, 0 VLM calls.
4. **Do not gate the commit on a "the target building fills the frame" seg-mask feature.** **New this
   pass, priced on all 130 FINISH-leg mask frames:** the centre-cone building fraction at L1 ≤ 2000 is
   **0.089–0.240 (median 0.153)** versus **0.000–0.430 (median 0.260)** at L1 > 8000 — the signal runs
   **backwards** (close-up views are dominated by a single facade's sky-gap, far views by a wall of
   distant blocks). Fully inverted, not merely weak.
5. **Do not gate on the perceiver saying "intersection".** **New this pass:** 747/788 ROAD-leg
   perceive responses (**95 %**) mention an intersection/junction/crosswalk; `P(mention | in window)`
   = 0.906 vs `P(mention | out of window)` = **0.953**, precision **0.103**. Anti-informative.
6. **Do not use the planner's `-1` as the primary commit channel**, and do not "fix" it by loosening
   `verify()`. ROAD precision 0.36 / recall 0.12 vs the arrival call's 0.90 / 0.21 (§2.7); it
   contributed 3 passes and **5 of the 20 fatal commits**.
7. **Do not re-tune `K_REACH_FORWARD_STREAK` / `MAX_REACH_FIRES_PER_SUBTASK`.** Run to death in both
   directions across iters 006–010. The cap is **not** binding here: median fires per ROAD leg = 6
   against a cap of 12, reached in **0/38** windows. `K_REACH_INTERVAL` is the one that binds (§5-A1)
   — keep the two distinguished when reporting.
8. **Do not make the xcheck cap `return False`** (r1's E3): `XCHECK_BLOCK` 18 → 117, four in-window
   commits refused with no evidence (one at 149 cm), 35-deep `XCHECK_BLOCK; EMPTY_FALLBACK` lock-in
   loops. §5-D is the memo, which r4 kept.
9. **Do not ship a bare depth-polarity flip**, and note `_strongly_far` is already **disabled**
   (`completion_checker.py:382`) after measuring `near_frac = 0.000` on 24/24 decoded frames — a live
   polarity fix with the current `_CONE_ROW` would veto **every** commit.
10. **Do not read** `agent_location`, `target_location`, `destination`, `correct_robot_location`,
    `correct_robot_position_area`, `ideal_path`, `gt.json`, `current_instruction`, the analog
    `_agent_rotation`, `agent_collision`, or **`env.subtask_max_steps`** (= `int(L1_to_target/100)`).
    Avoid numeric literals ≥ 2000 in candidate code (lint WARNs them as map-tuned); express distances
    as forward-action counts.
11. **Chain state:** anchor 0.36 · r1 0.285 · r2 0.33 · r3 0.36 · r4 0.355. Four rounds, no move.
    Per-leg across all five: ORIENT pinned 1.00, ROAD 0.12–0.24, TURNING 0.11–0.16, **FINISH 0.00 on
    every one.** Bundling ≥ 3 edits per candidate is what cost r1 its attribution — **ship §5-A1+E
    together, then A2+A3, then C, then D** as separable candidates.

---

## 7. Provenance / anti-cheat

- Candidate: `$OUT_DIR` = `simworld_gym/simworld_config/harnesses/nav` (`orchestrator.py` 613 loc +
  7 modules, 2705 loc total). Byte-identical to `anchor_nav_r000/source/` except for this file.
- Traces: `runs_adaptive/nav_hard__hard5010_adaptive_k4c1/arm_hard5010_adaptive_k4c1/anchor_nav/traces/`
  — 50 episodes, 3014 env steps, 2184 VLM calls (918 perceive / 918 plan / 348 arrival).
- Env semantics quoted from `SimWorldGym/simworld_gym/envs/traffic_world.py:365-385` (terminators:
  out-of-bounds, 20-identical-positions, `steps_count >= criteria.max_steps`,
  `substeps_count >= subtask_max_steps`) and `:631-690` (per-leg budget assignment).
- Yaw model calibrated empirically, not assumed: 2609 forward steps, yaw 0→+x (298), 90→+y (853),
  180→−x (797), 270→−y (661), 4 off-diagonal.
- Pixel measurements: 130 FINISH-leg segmentation frames decoded from `traces/*/frames/`.
- `read_audit.json`: **0 forbidden reads, 0 undeclared reads** across all 50 episodes.
- Oracle fields (`gt.json`, `env_step_log` positions) were used **only to measure** this diagnosis.
  Every mechanism recommended in §5 decides from `obs['rgb']`, `obs['object_mask']`, `obs['depth']`,
  `info['moving_distance']`, `info['agent']['agent_rotation']`, `task_spec['subtasks']` text, and
  `env.unwrapped.current_subtask` only.
