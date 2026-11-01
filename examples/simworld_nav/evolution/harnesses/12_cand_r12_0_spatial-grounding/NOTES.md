# NOTES — `cand_r012_0_spatial-grounding` (iter 12, parent `anchor_nav`)

## axis

**Token-level implementation of the arrival decision: the ANSWER SPACE stops
being a boolean and becomes an ORDERED LADDER, and the `-1` license becomes a
statement about the ladder's TRAJECTORY instead of about one answer.**

The anchor has exactly one question behind every non-orient `-1`
(`completion_checker._binary_reach_vlm_call`): *"Match 1 if the same building,
intersection, or scene is clearly visible and prominent in the current view; 0
otherwise."* One bit, over a **visibility** criterion, while the env grades
**arrival**. This chain has priced **seven re-wordings** of that bit and **five
re-schedulings** of the call; `diagnosis.md` §6 closes both. Nobody has changed
what the model is allowed to *say*.

`modules/arrival_scale.py` (`ArrivalScale`, the 8th module) replaces the bit
with a forced choice over five ordered labels —

```
NOT_VISIBLE  <  DISTANT  <  APPROACHING  <  ALONGSIDE  <  BEHIND
```

— each defined by **frame geometry** (where a thing's base sits, whether its top
is clipped by the frame edge, whether roadway still separates it from the
camera, whether the break in the frontage has passed under the camera). Nothing
absolute is named: `obs['depth']` is per-frame min-max normalised and carries no
scale (§6), so every rung is frame-relative and scale-free.

Three things follow that a boolean recognizer **structurally cannot express**:

* **"I see it and I am not there yet" becomes a first-class token.** The anchor's
  model, shown the landmark from 300 m, can only say `1` (fatal) or `0` (which
  throws away the identity claim it just made) — which is why its in-window
  refusals are worded *"different building colors and lighting"*: identity is
  the only subject the vocabulary gives it. `DISTANT` costs it nothing.
* **SUSTAIN** — a leg's *first* high rung does not commit; the rung must survive
  to the next look. Measured: the first high rung **leads** true arrival by
  500–1 500 cm, and on a ±2 000 window that is the difference between a pass and
  an instantly fatal `-1`.
* **OVERSHOOT** — `BEHIND` commits too, if the leg had already reached
  `APPROACHING`. This is the *"window entered and left"* latch the gradient
  names as the only route out of F1: 28/50 episodes stood inside the graded
  window and 26 walked out the far side with no vocabulary in which to say so.
  A ±2 000 window at 500 cm per forward is eight steps wide, so "just past it"
  still scores.

The schedule is the parent's **byte-for-byte** — `K_REACH_FORWARD_STREAK`,
`K_REACH_INTERVAL`, `MAX_REACH_FIRES_PER_SUBTASK`,
`MAX_XCHECK_FIRES_PER_SUBTASK` are **imported** from `completion_checker`, not
re-declared and not retuned (§6 runs that axis to death). The only thing that
moved is the shape of the answer and what the license reads off it.

## module map (★ = new / rewired this iter)

| module | role | this iter |
|---|---|---|
| `orchestrator.py` | entrypoint `run_episode(env, task_spec, vlm_clients, trace_dir=None)` | ★ 5 wiring points (xcheck gate, proactive gate, side-scan verdict, FINISH commit pose, ledger into `notes`) |
| `modules/arrival_scale.py` | **ordinal arrival ladder; owns the non-orient `-1` license + the FINISH commit pose** | ★ **NEW MODULE** |
| `modules/completion_checker.py` | the binary recognizer | **untouched** — demoted to `ArrivalScale`'s degradation path |
| `modules/active_perception.py` | cardinal burst, relative-turn burst, red idle, anti-spin, momentum | **untouched** (the cardinal burst is 50/50 — do not touch) |
| `modules/{spatial_memory,prompt_assembler,perceiver,planner,affordance_estimator}.py` | — | untouched |
| `AGENT.md`, `rules.md` | prose priors | untouched |

## evolvable levers

| lever | kind | where | status |
|---|---|---|---|
| **ordinal answer space** (5 ordered rungs, forced choice) | code | `arrival_scale.STAGES` + `_ladder_call` | **NEW** — the iter's axis |
| **INTERSECTION rungs = the STREET CROSSING** | prose-in-code | `_RUNGS_INTERSECTION` | **NEW, A/B-measured 8/16 in-window vs 1/16 far** (see §measured) |
| **FINISH rungs = frontage occupancy** | prose-in-code | `_RUNGS_FINISH` | **measured 0/16 in-window** — an honest dead end, shipped for its 1/16 far rate; see §risk |
| **SUSTAIN license** (prev look ≥ APPROACHING) | code | `licensed` | **NEW** — took out-of-window first-fires 7/12 → **0/12** |
| **OVERSHOOT rung** (`BEHIND` + prior `APPROACHING`) | code | `licensed` + `_peak` | **NEW** — the "entered and left" latch, previously only a wish in r006/r009's lever tables |
| **instruction prose deleted from the message** | code | `_ladder_call` | **NEW** — the 40-word leg text now only ROUTES the vocabulary, never acts as evidence |
| **one ordinal read per frame, shared by all gates** | code | `observe` memo on `(leg, frame_seq)` | **NEW** — kills F8 (9 duplicate calls on identical frames) |
| **cost cap no longer approves** | code | `allow_planner_commit` | **NEW** — the parent returns `True` on a spent cap (F5), which vetoed two in-window claims and rubber-stamped a third 1 500–3 000 cm past, fatally, 3× |
| **FINISH commit pose** (quarter-turn to the named side before `-1`) | code | `commit_prefix` | **NEW** — `per_position_SR[3]` was 0.000 *by construction*; correct in 2/2 testable parent episodes |
| **leg taxonomy** (3-way from static text) | code | `leg_kind` | **NEW** — 760/760 legs on all 199 `hard/` configs |
| **ledger** (`SCALE(...)` rung histogram + branch counts in `notes`) | code | `ledger` | **NEW** — the next diagnosis reads counts instead of inferring them |
| arrival-look schedule (streak / interval / caps) | code/const | `completion_checker`, imported | **frozen on purpose** — §6 |
| ORIENTATION cardinal burst | code | `active_perception` (a) | **do not touch** — 50/50, zero VLM calls |
| `_named_side_has_structure` / `_side_band_near` / `_looming_state` vetoes | code | `completion_checker` | **unwired from the new path** — §6 measures them inert/noise; left in place for the degradation route and a future A/B |
| **TURN approach** (spend ~2 forwards before the turn burst) | code | `active_perception:432` | **still open** — gradient §5-B, priced 5/12 → 11/12. Deliberately deferred: it is a MOTION axis and would confound this iter's attribution |
| **stuck detector scoped to the leg, not the burst** | code | `orchestrator.py:~285` | **still open** — gradient §5-F; `forward_count > 3` is unreachable when 750/918 bursts are `[0]` |
| **geometric roadway-band decider** (`object_mask`) | code | — | **still open** — gradient §5-C. This iter asks the VLM about the junction instead; if the ladder's INTERSECTION rung holds up, the two are substitutes and the cheaper one should win |
| **a FINISH arrival channel that is not a forward two-image call** | — | — | **the named open problem** — §risk |

## change

The gradient is a **search-space steer** toward `cand_r009_0` / `cand_r006_0`
with **token-level implementation** named as the aspect to vary. Both exemplars
are explicitly *graph/wiring* candidates — r006 added a POSE node and left the
recognizer byte-for-byte, r009 re-shaped the component graph (one node split
into a conjunction of two, one edge deleted). Varying the *token level* away
from them therefore means leaving the graph nearly alone and moving what is
actually **in the request**: its answer vocabulary, its subject, and which
elements of the message are present. r009's own token-level move was *subtractive*
(delete the `Subtask instruction:` element, split one question into two). This
iter's is *structural within the answer*: **one call, five ordered tokens, and a
license that reads the sequence of tokens rather than any single one.**

Five wiring points in `orchestrator.py`, each a re-point rather than a new edge:

1. `cross_check_planner_commit` → `scale.allow_planner_commit`, which **blocks**
   on a spent budget instead of approving (F5).
2. `proactive_commit_check` → `scale.proactive_commit`, same schedule, new
   license.
3. the side-scan's post-yaw frame → `scale.judge_frame`, so the harness has
   **one** arrival criterion instead of a second looser one hanging off that
   path.
4. the commit materialiser prefixes `scale.commit_prefix(instruction)` on FINISH
   legs — in-place rotation, so it cannot cost the location half of the grade
   and can only add the facing half that is currently always missing. Skipped
   when the side-scan already yawed the body onto that side.
5. `scale.ledger()` is appended to `EpisodeResult.notes`.

**Degradation, deliberately.** Every unavailability — no client, no hint, no
RGB, an unparseable label — returns `DEGRADE` and the harness calls the
**parent's own gate**. A dead second channel makes this harness behave like its
parent; it can never make it a harness that cannot commit. That is exactly how
the iter-5 depth gate died (a permanently-false predicate that fired twice in 50
episodes).

## why

Attacking F3 at the answer-space level is the one reading of "the arrival
question is the binding constraint" the chain has not tried, and §4 classifies
the ROAD arrival decision as a **VLM ceiling that must be scaffolded, not
re-prompted** — an ordinal vocabulary *is* scaffolding on the answer rather than
a re-wording of the criterion (which §6 forbids, and which I did not touch:
`completion_checker.py:560` is byte-for-byte the parent's). The SUSTAIN and
OVERSHOOT rules are the payoff: they are claims about a *sequence* of rungs, and
they are the first mechanism in this chain that can represent "I am past it",
which is the shape of the dominant failure (F1, 26/50 episodes).

### measured before shipping

Live `Qwen3-VL-30B-A3B-Instruct` on `:8000`, temperature 0, frames from this
chain's own traces. **Oracle-side L1 is used only to LABEL frames NEAR
(< 2 000 cm) / FAR (> 10 000 cm) for scoring — exactly as `diagnosis.md`'s own
probes do. Nothing measured here is read by the harness.** Labelled pool: 12 969
(live frame, leg reference photo, true L1) triples recovered from 12 trace sets,
leg index derived by counting accepted `-1` steps in `env_step_log.jsonl`.

**The first vocabulary I wrote was refuted, so I replaced it.** Asking about the
*reference subject's* occupancy scores **1/16 in-window vs 1/16 far** — no signal
at all, because at a ROAD waypoint the named building sits ~90° off the travel
axis and simply is not in a forward frame (*"the reference building is visible
but small and far down the street"* at 238 cm and 139 cm). Asking about the
**street crossing** — which a forward camera always sees, and which *is* the ROAD
leg's grading point (*"stop at the intersection with X on your left"*) — scores
**8/16 vs 1/16**. This chain had asked about the building 395 times and about the
junction zero times.

| ROAD vocabulary A/B (16 in-window legs / 16 far legs, one frame each) | fires ≥ ALONGSIDE, NEAR | FAR |
|---|---|---|
| reference-subject occupancy (my first draft) | 1/16 | 1/16 |
| **street crossing, with the reference photo — SHIPPED** | **8/16** | **1/16** |
| street crossing, no reference photo | 2/16 | 0/16 |

For reference the parent's binary criterion, re-run on the same frames, fires
2/16 NEAR / 1/12 FAR, consistent with its 0.21 measured in-window recall.

**The one-shot license was then refuted too.** Replaying 12 window-entering ROAD
legs frame-by-frame (one look per distinct position, far → near) and recording
the true L1 at the **first** fire:

| license rule | first fire in-window (< 2 000) | first fire OUT of window | never fires |
|---|---|---|---|
| one-shot (`≥ ALONGSIDE`) | 5/12 | **7/12 (each instantly fatal)** | 0 |
| **SUSTAIN (prev look ≥ APPROACHING) — SHIPPED** | **5–6/12** | **0/12** | 6–7 |
| two consecutive `≥ ALONGSIDE` | 3–4/12 | 0/12 | 8–9 |

(rows are stable across look strides 2 and 3; at stride 1 — a look every single
forward, finer than the harness's real cadence — SUSTAIN is 4/12 in-window and
3/12 out. SUSTAIN dominates "two consecutive" at every stride, so the obvious
`k = 2` confirmation knob is **not** the right rule and I did not ship it.)

**FINISH is an honest dead end and I am reporting it as one.** Three wordings
priced on 32 labelled destination frames — strict frontage occupancy, a loose
"dominates one side of the frame" variant, and a variant *told which side of the
frame to look at* — all fire **0/16 in-window**. At 353 cm the forward camera
answers `NOT_VISIBLE`, because the destination is named *"on your right"* and the
robot is still pointing along the street. r009 measured 0/7 independently. The
strict wording ships because it is the one that also fires only 1/16 FAR: the
parent produced **seven** far-field FINISH commits at 7 778–58 789 cm, each of
which ended its episode, and this rung set does not.

**End-to-end, live, through the shipped code path** (not the probe's copy of the
prompt): 3/3 in-window ROAD frames → `ALONGSIDE` → licensed; 2/2 frames at
~50 000 cm → `NOT_VISIBLE` → blocked; and a real leg walked far → near commits at
**L1 = 1 156 cm, inside the window**, after correctly blocking two unsustained
`BEHIND` claims at 2 156 cm and 1 606 cm.

### predicted

* `per_position_SR[1]` (ROAD) **0.24 → ~0.35–0.45** — 28/50 episodes enter a
  window and the measured conversion is 5–6/12 at zero out-of-window fires.
* **3 fatal xcheck commits removed** (the cap no longer approves) and **7 fatal
  far-field FINISH commits removed**, which returns the earlier-leg credit those
  episodes forfeited.
* `per_position_SR[3]` (FINISH): the pose prefix makes it *possible* for the
  first time in the chain (it was 0.000 by construction), but the FINISH
  recognizer measures 0/16 in-window, so the honest expectation is **still near
  zero** — now for a reason the ledger will name rather than a structural one.
* **SubtaskSR 0.360 → ~0.42–0.47.** SR stays low; it needs all four legs.
* Token cost roughly flat: one ladder call replaces one binary call, the
  per-frame memo removes the duplicates, and `max_tokens` is unchanged.

### verified

`~/miniconda3/envs/simworld/bin/python`, loaded through the evaluator's own
`spec_from_file_location(f"{pkg}.harness", …)` path:

* `validate_harness.py`: **OK**, signature unchanged. The 3 WARNs are the
  pre-existing `max_tokens` literals (2048 / 8192 / 8192); the new module adds
  **no** literal ≥ 2000.
* **46/46 fake-env assertions pass**, covering: label parsing (fences,
  lowercase, prose-wrapped, bare-label, unknown label → `None`); the taxonomy;
  every branch of `licensed`; `ALONGSIDE` sustained → `REACH_COMMIT`; `DISTANT`
  forever → never commits and keeps walking; `DISTANT,APPROACHING,BEHIND` →
  commits on the overshoot rung; `BEHIND` with no prior approach → blocked **and
  cannot self-license through the peak it just set** (a real bug the suite caught
  and the fix is `_peak` never recording `BEHIND`); FINISH `on your right` →
  `[Rotate_Right, -1]` and `on your left` → `[Rotate_Left, -1]` immediately
  before the commit; unparseable label → the parent's binary gate decides;
  planner `-1` blocked below the rung and honoured at it; a spent xcheck cap
  blocking rather than approving; one call shared by both gates on one frame; no
  client / no hint → the episode still runs and the scale is inert; a 4-leg
  episode passing all four legs; and the ORIENT cardinal burst still passing leg
  0 with zero ladder calls.

### risk, stated honestly

1. **FINISH will probably still score ~0.** Four wordings across two iters now
   measure 0 in-window recall on a forward two-image call; the pose prefix fixes
   the *grading* half but something must still decide *arrival*. The named open
   problem for the next iter is a FINISH arrival channel that is **not** a
   forward two-image call — and note §6 closes face-then-ask (0/20 faced-view
   in-window recall, 9/9 fatal), so it is not that either.
2. **SUSTAIN trades coverage for safety.** 6–7 of 12 window-entering legs never
   fire at all. That is the safer direction — a leg that does not commit keeps
   walking, a leg that commits 2 500 cm early is dead — but if the eval shows
   REACH_COMMIT counts collapsing, the lever to move is the sustain *rung*
   (`S_SUSTAIN_RUNG`), not the schedule.
3. **The crossing question can fire at a wrong junction.** Its FAR rate is 1/16,
   but a city block has many crossings and nothing in the rung set identifies
   *which*. The reference photo is in the message as the only identity anchor,
   and the 8/16-vs-1/16 gap says it is doing some work — but this is the most
   likely source of a surprise, and the ledger's rung histogram will show it.
4. **Attribution is slightly impure**: the FINISH pose prefix is a motion edit,
   not a token edit. It ships anyway because without it the FINISH half of the
   vocabulary is unmeasurable — leg 3 is 0.000 by construction — and it is 6
   lines behind a `leg_kind` test. The TURN approach (gradient §5-B, the other
   cheap motion win) is **deliberately left out** so this iter measures one axis.

## files

* `modules/arrival_scale.py` — **new** (the ordinal ladder, the license, the
  leg taxonomy, the FINISH commit pose, the ledger)
* `modules/__init__.py` — export `ArrivalScale`
* `orchestrator.py` — 5 wiring points + the module-list docstring
* `NOTES.md` — this file
