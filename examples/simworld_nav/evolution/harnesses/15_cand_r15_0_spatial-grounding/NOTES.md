# NOTES — `cand_r15_0_spatial-grounding` (iter 15, parent `cand_r13_0_spatial-grounding`)

## axis

**The FINISH `-1` licence: its position conjunct stops being a per-frame VLM question and
becomes a free sequential stopping rule over the gate's own answer history.**

Token-level, in the sense the steer asks for — the component graph is byte-for-byte the
parent's. Ten modules, the same call sites, the same stages in `run_episode`. **No file
added, no module introduced, no stage added to or removed from the loop, no prose prior
edited.** Every behavioural change is code *inside* `modules/arrival_gate.py`: one system
prompt deleted, one new piece of per-leg state, two new predicate methods, and the two
decision paths rewired onto them.

This is deliberately a different token-level direction from the two exemplars. r004 moved the
**motion channel** (how the harness learns a forward did not happen). r011 moved the
**arrival channel's input modality** (geometry instead of language), which this lineage then
rode to its best SubtaskSR. This one moves the **evidence horizon of the licence**: every
conjunct this gate has ever used — `_binary_reach_vlm_call`, `_IDENTITY_SYS`, `_JUNCTION_SYS`,
`_FRONTAGE_SYS`, `RoadGeometry` — is a function of **one frame**. The FINISH licence is now a
function of **the whole sequence of answers the gate has collected on this leg**, which is
state the harness was already generating and throwing away.

## module map (unchanged graph; ★ = internals changed this iter)

| module | role | this iter |
|---|---|---|
| `orchestrator.py` | entrypoint `run_episode(env, task_spec, vlm_clients, trace_dir=None)`; perceive→decide→act loop | comment only (the FINISH-conjunct narration at `:522` was stale) |
| `modules/arrival_gate.py` | owns the LICENSE for every non-orient `-1` | ★ **the whole change** — see below |
| `modules/road_geometry.py` | free ROAD position channel (`object_mask` rising edge) | untouched — 70 % of asks moved in-window, 97 % licence→pass |
| `modules/completion_checker.py` | recognizer + schedule constants, imported by the gate | untouched (no constant in it retuned) |
| `modules/viewpoint_controller.py` | leg typing, turn latch, commit pose prefix | untouched |
| `modules/active_perception.py` | cardinal burst, escape ladder, anti-spin | untouched |
| `spatial_memory` / `prompt_assembler` / `perceiver` / `planner` / `affordance_estimator` | per-step pipeline | untouched |
| `AGENT.md` / `rules.md` | prose priors | **untouched — the stopping rule is harness state, never shown to the agent** |

## evolvable levers

| lever | kind | where | status |
|---|---|---|---|
| per-leg identity-answer **log** (`_ident_log`, `_note_identity`) | code | `arrival_gate` | **NEW** — the first multi-frame state any conjunct reads |
| FINISH **stopping rule** (`_finish_history_ok`, `_prior_zero_run`) | code | `arrival_gate` | **NEW** — replaces the frontage veto; free (no call, no substep) |
| `M_FINISH_PRIOR_ZEROS` | const | `arrival_gate` | **NEW** = 4, read off the parent run's 368 FINISH answers (see §why) |
| `MAX_IDENT_FIRES_FINISH` | const | `arrival_gate` | **NEW** = 24; FINISH-only re-pacing, interval 5 untouched |
| planner cross-check on FINISH | code | `allow_planner_commit` | ★ **held to the same stopping rule**; may no longer defer or fall back to legacy |
| `_FRONTAGE_SYS` + the FINISH branch of `_place_call` | prose-in-code | `arrival_gate` | **DELETED — CLOSED.** 29/29 `Here: 0` live, including 3/3 in-window at 484 / 640 / 1 144 cm |
| `_named_side_has_structure` on the **proactive** accept path | code | `arrival_gate` | **DELETED** — 1 measured loss (a correct commit at 113 cm), 0 measured saves. Kept on the tighten-only cross-check path |
| text-free `_IDENTITY_SYS` | prose-in-code | `arrival_gate` | **frozen — do not touch.** 91 % in-window ROAD recall; the only arrival signal in the harness |
| `RoadGeometry` operating point (`ARRIVAL_FRAC`, band geometry) | const | `road_geometry` | **frozen** — measured-flat plateaus |
| `MAX_CLAIMABLE_CROSSINGS = 2` | const | `road_geometry` | untouched — 2 named ROAD legs lost to it (diagnosis F5); deliberately deferred so it does not confound this measurement |
| FINISH burst truncation (≤2 translations/iteration) | code | `orchestrator.py` | untouched — diagnosis F3(b); it changes `RoadGeometry`'s sampling cadence, so it must not ship with this |
| persistent wedge counter / overshoot flag | code | `orchestrator.py`, `spatial_memory` | untouched — diagnosis F6, the largest remaining open population (19/31 FINISH legs leave the road polygon) |
| ORIENT cardinal burst · TURNING approach latch | code | `active_perception`, `viewpoint_controller` | **do not touch** — 50/50 and 90 % conditional |

## change

### 1. The constant-`False` conjunct is deleted, not re-worded

`_FRONTAGE_SYS` was the one required position conjunct on a FINISH leg. Over the parent run it
answered `Here: 0` on **29 of 29** live asks, in 29 near-identical sentences ("buildings are
distant and not close enough"), **including on all three frames where the text-free identity
call had correctly recognised the destination from 484 / 640 / 1 144 cm**. A required conjunct
that is identically `False` is not a weak signal to tune — it is a closed gate, and it is the
whole reason the run emitted **zero FINISH `-1` in 31 attempts** and scored SR 0.00 with
`per_position_SR[3] = 0.00` while banking `[1.00, 0.58, 0.578, 0.00]` on the other three legs.

It is **deleted from the file**, along with the FINISH branch of `_place_call` and the
`instruction` argument that only existed to select between the two wordings; `_place_verdict`
now returns `None` on a FINISH leg before it can reach any call site. Leaving a live-looking
dead question in place is exactly how this one survived two iterations — the parent diagnosis
had already written "the pose is the bug, not the strictness" and it shipped anyway.

Re-wording is not the alternative: four free geometric substitutes (named-side building-class
fraction, named-side depth-near fraction, building base row, the `RoadGeometry` corridor
detector) were probed offline on all 1 244 archived FINISH frames labelled by true distance,
and all four are flat or **anti-discriminative**. There is no better one-frame answer to "am I
at the frontage?" available to this harness.

### 2. The replacement changes the kind of the question

A FINISH leg gets **exactly one `-1`** — `step(-1)` on a rejected claim terminates the episode —
so FINISH is a **one-shot stopping problem**, not a recall problem. And the evidence the gate
holds is not one frame: it is the *sequence* of identity answers it collected walking down the
block. The parent run contains that sequence in the clear, on the episode that should have
scored (`ep_20_18_task_dist_11_0_1`): eleven `Match=0` answers from 29 516 cm down to 2 516 cm,
then `Match=1` at **484 cm** ("same building wall and tree visible"), then `Match=0` again at
3 484 cm. The detector is sharp and well-localised; what was missing is a licence that can read
"first positive after a sustained run of negatives" as the arrival event.

So `ArrivalGate` now keeps `_ident_log[cur_idx]` — the per-leg, in-order log of its own identity
answers, fed by **both** call sites (unparsed answers are not evidence and are not recorded) —
and `_finish_history_ok` licences a FINISH `-1` on the first `Match=1` preceded by at least
`M_FINISH_PRIOR_ZEROS = 4` consecutive `Match=0` answers on the same leg. It costs no token and
no substep: it is bookkeeping over calls the harness already makes.

The guard is **FINISH-only** by construction. The same rule on a ROAD leg destroys 26 of 30
in-window licences, because `RoadGeometry` already restricts ROAD to about one ask per leg —
there is no run of zeros to accumulate there, and that is the correct design, not an oversight.

### 3. Two consequences that have to ship with it

* **The schedule is re-paced on FINISH** (`MAX_IDENT_FIRES_FINISH = 24`, interval unchanged at
  `K_REACH_INTERVAL = 5`). A stopping rule reads a sequence, so the sequence has to cover the
  leg. At the inherited cap of 12 the gate stopped asking at translation ≈ 59 on FINISH legs of
  52–199 translations and landed **8 of 368 asks (2.2 %) inside the acceptance window** — about
  0.8 asks on a leg that enters it at all, and zero asks over the back half of every long leg.
  The interval is deliberately *not* touched: it is the spacing at which the `m` statistics were
  measured. Cost is bounded and roughly self-funding — ~+12 asks on the longest FINISH legs,
  against 29 frontage calls removed, on a run whose `over_budget` flag was `false`.
* **The planner cross-check may no longer licence a FINISH `-1` by deferral.** Without this,
  deleting the frontage veto would have routed FINISH into `_legacy_verdict` (the identity
  question *with* the leg text — the 14 %-recall composition iter 9 removed), i.e. replaced the
  tightest gate in the module with the loosest one on the only leg where a wrong `-1` costs the
  episode. The planner has no FINISH arrival signal at all: its own `Match` is **0/19** in-window
  and **0 of 37** of its FINISH `-1` proposals were in-window (closest 4 403 cm). On FINISH that
  channel is now held to the identical stopping rule.

### 4. One free veto removed from the accept path

`_named_side_has_structure` fired **exactly once in 50 episodes** and the one time it fired it
killed a correct in-window ROAD commit at **113 cm** (`ep_20_14_21`, `veto_side=1`): 1 measured
loss, 0 measured saves. It is removed from `proactive_commit`'s accept path and **kept** on
`allow_planner_commit`, whose contract is "may only ever tighten".

### 5. Verification done before shipping

The module loads, `run_episode`'s signature and `EpisodeResult`'s seven fields are intact under
the evaluator's package-style load, and the new predicate is unit-tested: 3 zeros → blocked;
4 zeros → licensed; a leg-opening positive → blocked; a tail broken by an earlier positive →
blocked; 4 fresh zeros after an earlier positive → licensed; `None` answers not recorded; legs
independent. End-to-end on fakes: a FINISH leg answering `0,0,0,0,1` commits on the fifth ask
and on no earlier one; a FINISH leg answering `1,0,0,0,0` never commits; FINISH asks continue
past 12 and stop at 24; the FINISH paths raise if they ever reach `_place_call` or
`_legacy_verdict` (they do not); the ROAD path still commits on `place=1` and refuses `place=0`.
Read-audit surface unchanged — the new state is the gate's own answers.

## why

**The gradient is a search-space steer** toward the r011 / r004 region with the **token-level
implementation** named as the productive aspect: a change that lives *inside* existing
components rather than in the component graph. r011's own NOTES frame it as "no file added, no
module introduced, no stage added to or removed from the loop" — that is the constraint this
iter holds to, and the whole diff is one module.

**The merits are the diagnosis's own §4 and F1**, which classify the binding constraint as a
HARNESS-BUG rather than a model ceiling: "a function that is constant-`False` in a conjunction
is not a weak signal to be improved — it is a bug". The task is gated entirely on the FINISH
leg (0/31, the only leg below 0.55), 10 of 31 FINISH legs physically enter the ±2 000 cm
acceptance window and 24 of 31 come within 4 000 cm, and the body is already being steered there
— the gate simply refuses to stop it. The identity half works: 91 % in-window on ROAD, and on
FINISH it produced a clean `0 → 1 → 0` sweep across the window.

**The operating point is replayed, not guessed.** Counterfactual over the run's own 368 recorded
FINISH identity answers, "commit on the first qualifying `Match=1`":

```
m=0   in-window 2   far-field 11   never-commits 18
m=1   in-window 2   far-field 10   never-commits 19
m=4   in-window 2   far-field  7   never-commits 22
m=6   in-window 2   far-field  6   never-commits 23
```

Recall is flat in `m`; precision is not. `m = 4` is free, strictly dominates committing on the
raw first positive, and removes the leg's opening ask — the commonest far-field false positive,
4 of the 11 — outright. I took 4 rather than 6 because the ask schedule changes underneath it
in the same iter: a denser sequence makes a long zero-run cheaper to accumulate, but it also
makes the *first* in-window positive arrive earlier in a leg, and 4 is the value whose in-window
count is robust to that shift in both directions.

**Prediction (priced from the replay, which is a floor, not an estimate).** F1 alone converts
`ep_20_18_11` (484 cm) and `ep_20_24_21` (1 144 cm): `per_position_SR[3]` 0.00 → **≈ 0.044**,
SR 0.00 → **≈ 0.04** — the first non-zero SR this lineage has had. The re-pacing is what can
beat the floor: it roughly doubles FINISH asks (≈ 12 → ≈ 24/leg) and extends them over the back
half of the leg, where the destination of a long leg actually is, so in-window asks should go
8 → ≈ 18 and the number of legs that get *any* in-window ask should roughly triple. Combined,
**`per_position_SR[3]` → 0.06–0.16, SR → 0.04–0.14**, SubtaskSR 0.550 → **0.565–0.59**. The side
veto deletion adds ≈ +1 ROAD leg (`per_position_SR[1]` 0.58 → ≈ 0.60).

**Why the downside is close to zero, stated precisely.** The parent treats FINISH as the most
dangerous place to commit; that has the asymmetry backwards. A rejected `-1` does **not** erase
banked legs (`subtask_passed[pre_idx]` is already `True` before the claim), and the FINISH leg
is worth exactly 0.00 today — so a premature FINISH commit forfeits only a continuation that in
**31 of 31** observed legs produced nothing, and 19 of those 31 ended by walking out of the road
polygon anyway. The one real cost is the one-shot opportunity, and that is precisely what `m`
protects.

**Risks, named.**
(a) `m = 4` was fitted on answers produced at interval 5 under a 12-ask cap; at 24 asks the zero
run accumulates faster, which should only help, but the far-field column of the table is no
longer exactly the one that will be realised. (b) Uncapping raises cost ≈ +8–15k tok/ep on a
145 863 tok/ep baseline with `over_budget: false` — if the next run flips that flag, lower
`MAX_IDENT_FIRES_FINISH` before touching `m`. (c) The gate can now emit far-field FINISH `-1`s
where it previously emitted none, so episode-level terminations will shift from "walked off the
map" to "rejected commit"; that is the intended trade and should be read as a change of failure
*mode*, not a regression, unless `per_position_SR[3]` stays at 0. (d) The side-veto deletion is
a second, independent edit in the same module — it is one line, it is attributable by the
`veto_side` counter disappearing from the ledger, and its measured effect was 1 loss / 0 saves,
but it is not strictly axis-pure.

**Deliberately NOT taken, to keep the measurement clean:** `MAX_CLAIMABLE_CROSSINGS` 2 → 3 (F5,
+2 named ROAD legs), k-of-n self-consistency on far-field ROAD identity (F4), FINISH burst
truncation (F3(b) — it changes `RoadGeometry`'s sampling cadence, and the ROAD operating point
was measured at the current cadence), the persistent wedge counter and the overshoot flag (F6),
and the gate-call trace labelling (F7). F6 is the largest open population left and should be
next if this lands.

## files

- `modules/arrival_gate.py` — the change: `_FRONTAGE_SYS` deleted; `_place_call` junction-only
  (`instruction` arg dropped); `_place_verdict` returns `None` on FINISH; `_ident_log` state +
  `_note_identity` / `_prior_zero_run` / `_finish_history_ok` added; `_identity_call` takes
  `cur_idx` and logs; `proactive_commit` FINISH branch + FINISH ask cap + side-veto removal;
  `allow_planner_commit` FINISH branch; module docstring updated.
- `orchestrator.py` — comment only (`:522`, the stale `_FRONTAGE_SYS` narration at the
  `proactive_commit` call site). No code changed; `run_episode` signature and `EpisodeResult`
  verified intact.
- `NOTES.md` — this file.
