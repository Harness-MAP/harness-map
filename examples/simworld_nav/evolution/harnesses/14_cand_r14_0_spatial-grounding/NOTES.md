# NOTES — `cand_r14_0_spatial-grounding` (iter 14, parent `cand_r13_0_spatial-grounding`)

## axis

**Token-level implementation of the FINISH position question.** No node is
added, no edge is rerouted between modules, no schedule constant is inherited
from a different file: what changes is **the literal tokens of the question the
gate asks on the terminal leg, the shape of the answer it asks for, and the two
small rules that read that answer.** The component graph is byte-identical to
the parent's — same 10 nodes, same `run_episode` control flow, same ROAD / TURN
/ ORIENT paths.

Three token-level edits, one idea:

* **The question changes.** `_FRONTAGE_SYS` ("is a facade close enough that it
  fills a large part of the frame?" — one image, no reference, **0 positives in
  29 calls**) is retired on FINISH and replaced by `_ARRIVAL_SYS`: the same
  two-image call the chain already trusts, re-worded to ask *how much more
  street the robot must cover to draw level with the building in the reference
  photo*.
* **The answer's TYPE changes.** `{"Here": 0|1}` becomes
  `{"Match": 0|1, "Prox": 0..3}`. A "no" now carries **how far**, which is what
  makes the next two rules possible at zero extra cost.
* **Two rules read the new field.** `Prox` keys the *schedule* (the gate asks
  every 6 translations when the answer is "far down the street" and **every
  translation** when it is "a few steps away"), and the license needs
  `FINISH_CONFIRM_STREAK = 2` consecutive qualifying readings instead of one.

Bundled, as the diagnosis asked (F6, *"bundle it, don't spend an iter on it"*):
the plan call retries once when it is cut off mid-JSON.

## module map (★ = changed this iter)

| module | role | this iter |
|---|---|---|
| `modules/arrival_gate.py` | the `-1` LICENSE | ★ the FINISH leg's question, answer type, schedule and confirmation rule |
| `modules/planner.py` | perceive→action list | ★ **bundled bug fix only** — one retry on `finish_reason == "length"` |
| `orchestrator.py` | entrypoint `run_episode(env, task_spec, vlm_clients, trace_dir=None)` | ★ **one expression**: the commit's trace label can now read `FIN_REACH` |
| `modules/road_geometry.py` | ROAD position conjunct (iter 13) | **untouched — byte-identical** |
| `modules/completion_checker.py` | legacy recognizer + schedule constants | **untouched — byte-identical** |
| `modules/viewpoint_controller.py` | leg typing, TURN approach, commit pose | **untouched — byte-identical** |
| `modules/{active_perception,spatial_memory,prompt_assembler,perceiver,affordance_estimator,__init__}.py` | — | **untouched — byte-identical** |
| `AGENT.md`, `rules.md`, `__init__.py` | prose priors + package | **deliberately byte-identical** (third iter running) |

*(`cmp`-verified: 12 of the 15 pre-existing files, including both prose priors,
are byte-for-byte the parent's.)*

## evolvable levers

| lever | kind | where | status |
|---|---|---|---|
| **the FINISH position question's wording** | prose-in-code | `arrival_gate._ARRIVAL_SYS` | ★ **NEW** — 0/20 → **65 %** in-window on the same archived frames (table below) |
| **the answer's schema** (`Here` binary → `Match` + `Prox` 0-3) | code | `_arrival_call`, `_extract_prox` | ★ **NEW** — the grade is what the two rules below consume |
| **`Prox`-keyed ask interval** | code | `K_FINISH_INTERVAL_BY_PROX` | ★ **NEW** — replaces a uniform clock that went mute after ~30 000 cm of a 20-50 km leg |
| **k = 2 consecutive qualifying readings** | code | `FINISH_CONFIRM_STREAK` | ★ **NEW** — MC over the 31 real FINISH legs: in-window commits 8.4 % → **10.6 %**, >10 km commits 25.1 % → **1.0 %** |
| **the `Prox == 2` tie-breaker call** | prose-in-code | `_REACHED_SYS`, `_reached_call` | ★ **NEW** — paid only on the ambiguous grade; keeps 6/7 in-window `Prox 2`, kills 7/8 far-field |
| **no per-leg cap on FINISH** (48 = runaway guard) | code | `MAX_FINISH_ARRIVAL_FIRES_PER_SUBTASK` | ★ **NEW** — 9/10 ball-entering episodes hit the old cap of 12, two of them *before* arriving |
| plan-call truncation retry | code | `planner._finish_reason` + retry | ★ **NEW (F6)** — 23/1 650 calls, one of them killed an episode holding 3 banked legs |
| `_FRONTAGE_SYS` wording | prose-in-code | `arrival_gate` | **CLOSED** — 0/29 live, 0/20 on the probe; three wordings tried across two iters. It is now unreachable from both call sites |
| plain `_IDENTITY_SYS` on FINISH | prose-in-code | `arrival_gate` | **SUPERSEDED on FINISH** — 25 % in-window on the probe vs the new call's 85 % Match, measured on the same frames. **Untouched on ROAD** (90.9 % live) |
| `Prox` = 1 | code/const | — | **dead grade** — the model never emits it; do not tune around it |
| licensing at `Prox >= 2` with no tie-breaker | code | — | **MEASURED AND REJECTED** — 85 % in-window but 22.5 % beyond 10 km; forecloses the leg before arrival |
| `Prox == 3` alone | code | — | **measured**: 50 % / 2.5 %. The shipped disjunction (`3` or `2`+tie-breaker) is 65 % / 2.5 % — strictly better |
| k = 3 confirmations | code | `FINISH_CONFIRM_STREAK` | **MEASURED AND REJECTED** — overshoots (4.7 % in-window) |
| the ROAD channel (geometry + identity) | code | `road_geometry`, `arrival_gate` | **do not touch** — 0.26 → 0.58 this run, exactly as priced |
| `MAX_CLAIMABLE_CROSSINGS` 2 → 1 | code/const | `road_geometry` | **REFUTED by the r13 run** (all 6 fatal ROAD commits were edge #1; all 4 edge-#2 claims passed) |
| **F4: crop the identity images to the named side** | code | `_identity_call` | **still open** — the ranked fix for the 5 long-ROAD fatal commits; an image-prep axis, deliberately not mixed with this one |
| **F5: let the geometric veto spend one ask per leg** | code | `proactive_commit` | **still open** — cost 7 ROAD legs this run |
| **an arrival CONTROLLER** (slow / stop / back up) | code | `orchestrator.py` burst executor | **still open** — every in-ball frame in the run is a `Move_Forward` |

## change

**1. `_ARRIVAL_SYS` replaces `_FRONTAGE_SYS` on the FINISH leg.** The parent's
FINISH position conjunct asked one image, with no reference photo, whether *"a
facade is close enough that it fills a large part of the frame … the robot is
at the near end of the block"*. It answered **0 on all 29 calls**, three of them
from inside the success ball, and its own reasons say why: *"building facade is
distant and not close enough to fill the frame"* (asked at L1 **484 cm**). That
sentence is **true** of the frame. A robot 484 cm from its destination is
walking *along* the street with the building *beside* it, half out of frame; the
frontal composition the question demands only exists after
`commit_pose_prefix` turns the body, which happens **after** the commit this
call gates. The question was unanswerable by construction.

The replacement keeps everything the chain has measured to work — two images,
no leg text, same encoder, same temperature, same 1024-token answer budget —
and changes only the sentences: the reference photo grounds *which* building,
the prompt states that the destination *"will usually appear off to one side, at
a steep angle, or partly cut off by the frame edge … when the robot is closest
it is the LEAST centred"*, and `Prox` is defined on the **current photo's street
geometry** with an explicit ban on comparing the two photos' camera distances
(round 1 of the probe showed the model will otherwise answer *"same building
visible from a closer, side angle"* — a comparison between the pictures, not a
statement about where the robot is standing — and that failure mode alone
produced all 8 of its far-field false licenses).

**2. The answer's type carries distance, so the schedule can be closed-loop.**
`K_FINISH_INTERVAL_BY_PROX = {0: 6, 1: 3, 2: 1, 3: 1}`: the gate is cheap when
the model says the building is down by the vanishing point and asks on **every
translation** once it says "a few steps away". The parent spread a flat 12 asks
at a fixed interval of 5 over a 20-50 km leg, which is why 9 of the 10 episodes
that physically entered the success ball had **hit the cap**, two of them
standing at L1 **27 cm** and **188 cm** with no arrival question left to ask.
The per-leg cap is gone (48 is a runaway guard, never binding at the measured
spacing); the *budget* no longer decides the *verdict* — the same bug class as
iter 13's F2, failing silent instead of failing open.

**3. A license needs two consecutive qualifying readings.** A qualifying
reading is `Match == 1 AND (Prox == 3 OR (Prox == 2 AND the tie-breaker says
1))`. The tie-breaker (`_REACHED_SYS`) is a second two-image call, spent **only**
on a `Prox == 2` frame, that asks the position question directly with its base
rate stated. One qualifying reading is not enough because a FINISH leg walks in
through the 2-4 km band — where the rule still fires 43 % of the time — before
it reaches the 2 000 cm ball, so a first-fire rule commits one band early.
Because a `Prox >= 2` reading re-asks on the very next translation, the
confirmation costs about **one forward step**, which is also roughly the
distance it needs to move to be right.

**4. Bundled (F6): the plan call retries once when it runs out of output
budget.** 23 of 1 650 plan calls returned `finish_reason: "length"` and parsed
to nothing, each burning a whole perceive+plan cycle (~188 k output tokens
run-wide), and one killed `ep_20_11_21` by `TOO_MANY_ERRORS` while it held
three banked legs. The budget is **not** lowered (that would truncate the 1 627
healthy calls too); a run-on is retried once with the format demand restated
and a short budget. The retry is unreachable on a call that parsed.

**Not changed, on purpose:** both prose priors; every ROAD / TURN / ORIENT path
(`RoadGeometry`, the geometric veto, the event-keyed identity schedule, the
cardinal burst, the turn-approach latch, `_IDENTITY_SYS`, `_JUNCTION_SYS`);
`commit_pose_prefix`; `completion_checker`'s schedule constants; the iter-13
fail-closed cross-check cap; the whole burst executor.

## why

**The directive** is a search-space steer toward the region exemplified by
`cand_r011_0` and `cand_r004_0`, naming **token-level implementation** as the
aspect to vary. That is the exact complement of what iter 13 did (it added a
module — a component-graph move — and this iter's diagnosis confirms that bet
paid: ROAD 0.26 → 0.58, TURNING 0.2667 → 0.5778, SubtaskSR 0.400 → 0.550, the
chain's best). So this iter leaves the graph alone and moves the tokens, and
the binding constraint happens to be exactly a token problem.

**The gradient.** The diagnosis's one-sentence binding constraint: SR is pinned
at **exactly 0.00** by *"a single dead conjunct — `_FRONTAGE_SYS` was called 29
times and answered `Here=0` all 29 times, including 3 calls made from inside the
success ball (L1 = 484 / 640 / 1144 cm) … all 31 reached FINISH legs ended
without the harness ever emitting a single `-1`."* It classifies this as
**HARNESS-BUG, not VLM-CEILING**: on those three frames identity had already
returned `Match=1` and `commit_pose_prefix` would have supplied the exact graded
yaw. Its two top-ranked fixes are **F1** (delete the dead conjunct, but *"do not
let it land on the legacy 24 % recognizer"*) and **F2** (*"make the arrival
budget leg-type-aware; stop going silent on FINISH"*). This candidate is both,
executed as a question rewrite rather than a deletion — because deleting the
conjunct outright leaves the far-field identity false positives (5.9 %) with
nothing in front of them, and ~15 asks per leg at 5.9 % forecloses almost every
leg before it arrives.

The diagnosis also **forbids** two neighbouring moves and this proposal obeys
both: a segmentation-geometry position channel for FINISH does not exist (it
replayed 1 241 archived FINISH masks: *"the destination is a point along a
continuous built-up facade — there is no geometric signature"*), and burst
truncation is not the FINISH bottleneck (*"the frames exist; the budget and the
conjunct are what fail"*). F3 (physically turning to face before asking) is the
other named route to the same leg; it is **not** taken, because F3's own caveat
is that iter 6 measured a face-scan at 0/128, and the measurement below shows
the pose was never the problem — **the sentence was**. F3 costs 2 env substeps
and a call per candidate frame; this costs 0 substeps and the probe says it
works. If the live run disagrees, F3 is still there, and its prior is now better
informed.

### Measured before shipping

`~/miniconda3/envs/simworld/bin/python`, live model on `:8000`
(`Qwen3-VL-30B-A3B-Instruct`), temperature 0, **120 frames drawn from the
parent's own 50-episode archive**: every FINISH-leg perceive frame inside the
success ball (all 20, across 9 episodes) plus 30 / 30 / 40 sampled from the
2-4 k, 4-10 k and >10 km bands. Each frame is replayed with **its own** archived
RGB and **its own** per-subtask reference image. Oracle L1 is used *only to
label frames, offline, after a finished run*, exactly as the diagnostician's
probes do; no coordinate appears in the shipped code.

**Round 1 — the three candidate questions on the same frames:**

| rule | <2 km | 2-4 km | 4-10 km | >10 km |
|---|---:|---:|---:|---:|
| `_FRONTAGE_SYS` (the parent's conjunct) | **0/20 = 0 %** | 0/30 | 2/30 | 2/40 |
| plain `_IDENTITY_SYS` `Match=1` | 5/20 = 25 % | 13 % | 10 % | 5 % |
| new `_ARRIVAL_SYS` `Match=1` | **17/20 = 85 %** | 60 % | 37 % | 25 % |

The middle row is the diagnosis's §2D anomaly (FINISH identity 37.5 % vs ROAD
90.9 %, attributed to the off-axis pose) reproduced offline at 25 %. The bottom
row is the **same frames, same pose, same images** — only the sentences differ.
**The pose was not the bug.** That is the single most useful number this iter
produced, and it is why F3 is deferred rather than taken.

**Round 2 — tightening the position field** (the round-1 grade collapsed to
`{0, 2}` and its far-field positives all read *"same building visible from a
closer, side angle"*, i.e. a comparison between the two photographs):

| rule | <2 km | 2-4 km | 4-10 km | >10 km | legs covered |
|---|---:|---:|---:|---:|---:|
| round-1 grade, `Prox >= 2` | 70 % | 50 % | 27 % | 20 % | 7/9 |
| V1 (shipped wording), `Prox >= 2` | 85 % | 63 % | 47 % | 22.5 % | 8/9 |
| V1 `Prox == 3` | 50 % | 23 % | 10 % | **2.5 %** | 5/9 |
| V2 (frame-position wording), `>= 2` | 70 % | 50 % | 30 % | 15 % | 8/9 |
| V3 (binary "reached?") | 60 % | 40 % | 17 % | 7.5 % | 7/9 |
| **SHIPPED: V1 `3` OR (V1 `2` AND V3)** | **65 %** | 43 % | 17 % | **2.5 %** | **7/9** |

**Round 3 — the per-leg consequence.** Per-frame precision is not the
objective; *where the leg commits* is. Monte-Carlo (4 000 trials) walks each of
the **31 real FINISH legs' own perceive-frame sequences** from the archive,
applies the shipped `Prox`-keyed schedule (translations converted to decision
points at the run's measured median burst of 3) and resamples each reading's
outcome from the measured per-band distribution:

| confirmation rule | commits in-window | 2-4 k | 4-10 k | >10 km | never | asks/leg |
|---|---:|---:|---:|---:|---:|---:|
| k = 1 | 8.4 % | 15.7 % | 26.8 % | 25.1 % | 23.9 % | 12.1 |
| **k = 2 (shipped)** | **10.6 %** | 10.3 % | 7.1 % | **1.0 %** | 71.0 % | 19.8 |
| k = 3 | 4.7 % | 4.3 % | 1.9 % | 0.1 % | 89.0 % | 20.8 |
| arm, then any `Prox >= 2` | 11.6 % | 12.4 % | 16.7 % | 7.8 % | 51.5 % | 17.0 |

k = 2 is taken over the nominally-better "arm then `Prox >= 2`" (11.6 %) because
the simulation's independence assumption flatters non-consecutive rules the most
— under correlated far-field positives they degrade toward k = 1, while k = 2's
1.0 % far-field rate has the most headroom.

**The priced prediction.** 10.6 % of 31 FINISH legs commit in-window ⇒ **≈ 3.3
successes, SR 0.00 → ≈ 0.065** (diagnosis F1+F2 predicted 0.08-0.12; ceiling is
0.20, the 10 episodes that physically enter the ball). `per_position_SR[3]`
0.00 → **≈ 0.10**, primary **0.550 → ≈ 0.575**. ROAD and TURNING should be
unchanged within noise — their code is byte-identical. Cost: FINISH gate calls
go 12 → ≈ 20 arrival + ≈ 4 tie-breaker per leg, so ≈ +12 calls × 31 legs ≈ **+370
calls** against the run's 3 744, partly offset by F6's 23 recovered cycles
(~188 k output tokens) and by the 29 frontage calls that no longer happen.

**Risk, stated honestly.**
(a) **71 % of legs never commit** under the shipped rule. That is not a
regression — the parent's figure is 100 % — but it means the dominant failure
mode is unchanged in *kind*, and the remaining headroom to the 0.20 ceiling is
in the arrival **controller** (nothing in `orchestrator.py` can slow, stop or back
up; every in-ball frame in the run is a `Move_Forward`), which is the next
gradient and a different axis.
(b) **The Monte-Carlo assumes per-frame independence.** Real readings on
consecutive frames of the same leg are correlated; if far-field positives
cluster, k = 2's 1.0 % understates the true foreclosure rate. The `GATE(fin_*)`
ledger is instrumented to settle this directly: `fin_armed` counts readings that
qualified but were not confirmed, and the commit's L1 is recoverable per leg
from the trace.
(c) **Recall-first is correct ONLY on FINISH and is applied only there.** A
premature `-1` on the terminal leg forfeits a leg already scoring 0.00 with
nothing downstream; banked legs survive it. The ROAD gate's conservatism is
untouched, and the FINISH branch is routed out of `proactive_commit` **before**
the geometric channel and the shared clock, so no ROAD/TURN behaviour can
inherit it.
(d) **The 2-4 km band still fires at 43 %** and is the proximate cause of the
10.3 % of legs that commit one band early. Nothing in a single frame separates
"2 500 cm away" from "1 500 cm away" reliably — the three tightenings tried all
hold the in-window : 2-4 km ratio near 1.5 : 1. Fixing that needs motion, not
wording: it is the controller in (a).
(e) **F4 (crop the identity images to the named side) is NOT taken**, though it
is the diagnosis's highest-leverage ROAD fix. It is an image-preparation axis
and mixing it with a FINISH-leg question rewrite would make neither measurable.
It is the obvious next iter.

**Anti-cheat: unchanged and unextended.** The new calls read `obs['rgb']`, the
per-subtask reference image, `obs['object_mask']` (via the pre-existing free
side veto) and the static `task_spec` instruction text — the same four inputs
the parent's FINISH path read. No coordinate, no `gt.json` field, no
`agent_location` / `target_location` / `ideal_path` / `current_instruction`, no
analog `_agent_rotation`, no `subtask_max_steps`. Every new constant is a grade
(0-3), a count of readings, or a count of translations; there is no centimetre
anywhere in the shipped code. The oracle L1 values in the tables above are
proposer-side labels attached to frames from a **finished** run, used offline to
score a replay; `read_audit` surface is identical to the parent's.

### Verified

Loaded through the evaluator's own `spec_from_file_location(f"{pkg}.harness",
…)` path, `~/miniconda3/envs/simworld/bin/python`:

* `validate_harness.py`: **OK**, signature unchanged. The 3 WARNs are the
  pre-existing `max_tokens` literals (2048 / 8192 / 8192) — unchanged in number
  and value; the retry's budget is 1024 and adds no literal ≥ 2000.
* `cmp`: `AGENT.md`, `rules.md`, `__init__.py`, `modules/__init__.py` and
  **8 of the 10 modules** are byte-identical to `$PARENT_DIR`.
* The shipped `_ARRIVAL_SYS` and `_REACHED_SYS` strings are asserted
  **byte-identical to the V1 / V3 strings the probe measured**, so the tables
  above describe the code that ships.
* **Unit, 45 checks, all passing**: the `Prox` parser (fenced, chatty,
  truncated, absent, out-of-range); one qualifying reading arms and does not
  commit; two consecutive commit; a non-qualifying reading resets the streak and
  the next pair re-arms; `Prox 2` buys exactly one tie-breaker call and `Here=0`
  blocks; `Prox 1` buys none; `Match=0`, an unparsed grade, and a dead client
  never license; the `Prox`-keyed intervals are 6 / 3 / 1 / 1 exactly; a
  turn-reset streak re-arms the spacing instead of going mute; there is no
  12-ask cliff and the runaway guard binds at 48; the FINISH cross-check needs
  one reading and its cap still fails **closed** (2 calls, then replay);
  **ROAD / TURN / ORIENT are bit-for-bit the parent's behaviour** — the free
  geometric veto still costs 0 calls, a geometric edge still commits through
  `_IDENTITY_SYS`, and `_FRONTAGE_SYS` is never called on any path.
* **End-to-end `run_episode`, 18 checks, all passing**, against a fake env +
  fake client: a 4-leg episode banks ORIENT → ROAD → TURN and commits FINISH via
  `FIN_REACH`; a far FINISH leg never commits and keeps asking past the old
  12-cap; a `Match=0` leg stays bounded by the guard; a garbage answer is safe;
  a planner `-1` on a far FINISH is blocked (`XCHECK_BLOCK`); a truncated plan
  call retries exactly once and yields actions (no `NO_ACTION_RETURNED`); a
  healthy plan call never retries.

## files

- `modules/arrival_gate.py` — `_ARRIVAL_SYS`, `_REACHED_SYS`;
  `FINISH_PROX_LICENSE`, `FINISH_PROX_CONFIRM`, `FINISH_CONFIRM_STREAK`,
  `K_FINISH_INTERVAL_BY_PROX`, `K_FINISH_INTERVAL_DEFAULT`,
  `MAX_FINISH_ARRIVAL_FIRES_PER_SUBTASK`; `_extract_int_field`,
  `_extract_prox`; `_arrival_call`, `_reached_call`, `_finish_reading`,
  `_finish_proactive`; `_last_prox` / `_fin_streak` / `last_channel` state;
  the FINISH branch at the top of `proactive_commit` and inside
  `allow_planner_commit`
- `modules/planner.py` — `_finish_reason`; the `_ask` closure and the single
  retry on a truncated plan call; `PLAN_RETRY_TRUNC` marker in `reason`
- `orchestrator.py` — one expression: the commit label reads `arrival.last_channel`
  before falling back to `GEO_REACH` / `REACH_COMMIT`
- `NOTES.md` — this file

*(unchanged, `cmp`-verified byte-identical to the parent: `AGENT.md`,
`rules.md`, `__init__.py`, `modules/__init__.py`, and
`modules/{spatial_memory,prompt_assembler,perceiver,affordance_estimator,active_perception,completion_checker,viewpoint_controller,road_geometry}.py`)*
