# NOTES — `cand_r008_0_spatial-grounding` (iter 8, parent `anchor_nav`)

## axis

**Code / component graph — a POSE-owning node that runs *ahead of*
`ActivePerception` instead of intercepting it.**

New module `modules/pose_script.py` (`PoseScript`), wired at **one** point in
`run_episode`: the top of the decision body, *before* `active.should_emit` is
consulted. It makes no VLM call and holds no opinion about appearance; it owns
*where the body stands* when a pose-graded commit fires. Its one implemented
program is the **TURN leg's approach into the junction**, released by a
**turn-down (peak) detector** over a windowed trace of the lateral
segmentation band.

`completion_checker.py`, `active_perception.py`, `perceiver.py`, `planner.py`,
`spatial_memory.py`, `affordance_estimator.py`, `prompt_assembler.py`,
`AGENT.md` and `rules.md` are **untouched**. Every ROAD / ORIENT / FINISH
commit path is byte-identical to the anchor.

## module map (★ = new / rewired this iter)

| module | role | this iter |
|---|---|---|
| `agent.py` | entrypoint `run_episode(env, task_spec, vlm_clients, trace_dir=None)`; perceive→decide→act loop | ★ **one** wiring point (pose consulted before AP) + one per-leg trace note |
| `modules/pose_script.py` | **POSE program per leg**; lateral-band trace + turn-down detector; the TURN-leg approach | ★ **NEW MODULE** |
| `modules/active_perception.py` | cardinal burst, relative-turn burst, red idle, anti-spin, momentum | **untouched** — its `[rotate, -1]` burst is *delayed*, never captured |
| `modules/completion_checker.py` | the appearance verdict behind every `-1` | **untouched** |
| `modules/{spatial_memory,prompt_assembler,perceiver,planner,affordance_estimator}.py` | — | untouched |
| `AGENT.md`, `rules.md` | prose priors | untouched |

## evolvable levers

| lever | kind | where | status |
|---|---|---|---|
| **TURN-leg approach program** (forwards along the pre-turn heading before the burst) | code | `pose_script.PoseScript.step` | **NEW** |
| **turn-down (peak) detector** — relative fall from a windowed peak | code | `pose_script.SideBandTrace.opened` | **NEW** — replaces the absolute-threshold form the gradient measured as a no-op |
| band-trace window / drop / support (`8` / `0.35` / `0.12`) | code | `pose_script` constants | tunable; the drop is the one worth sweeping |
| approach floor / cap (`2` / `3`) | code | `pose_script` constants | **bounds of the measured plateau**, not a fit — both ends measure 11/12 |
| leg identity = AP's own parsers (`turn_leg_direction`) | code | `pose_script` | **NEW** — deliberately *not* a second taxonomy; verified 199/199 |
| band geometry = `completion_checker`'s measured constants | code | `pose_script` import | **NEW** — single source of truth, no re-typed literals |
| ROAD / FINISH pose programs | code | `pose_script.step` (returns `None`) | **deliberately empty** — one leg per candidate |
| forward-facing arrival recognizer (criterion / evidence / schedule / caps) | code | `completion_checker` | **CLOSED this iter** — 6 prior candidates, none beat the anchor |
| ORIENTATION cardinal burst | code | `active_perception` (a) | **do not touch** — 50/50; `step` returns `None` for it by construction |
| red-signal idle during the approach | code | `agent.py` wiring | **open** — AP is not consulted for the ≤3 approach decisions (see §risk) |
| overshoot / "I have passed it" state | code | `spatial_memory` | **still open** |

## change

The env grades the four legs on three different physical quantities
(`traffic_world.py:396-462`): ROAD on **L1 position only**, TURNING on **yaw
*and* a junction box**, FINISH on **position *and* yaw**. The anchor answers
all three with one question — "does the forward RGB frame look like the hint
photo?" — and has **no component whose job is to put the body somewhere before
the commit fires**. That missing responsibility, POSE, is the region
`cand_r006` opened (0.370, the lineage best, TURNING 0.111 → 0.200). The
directive steers here and names the **token-level implementation** as the
aspect to vary, so this candidate re-implements the same responsibility three
ways differently and scopes it to one leg.

**1. Delay, don't intercept (the wiring).** r006's `ViewpointController`
*captured* AP's `[rotate, -1]` burst into `_pending_turn_burst`, held it, and
re-issued it through an idempotence latch — so AP's own once-per-instruction
latch had to be reasoned about from two modules at once, across four wiring
points. `PoseScript` never sees that burst. It is consulted **before** AP and
simply owns the opening decisions of the leg: while its program is live it
returns `[Move_Forward]` and AP is not called at all; when the program
finishes it returns `None`, and AP's relative-turn burst fires on its *first*
consultation of the leg — unmodified, un-latched, byte-identical to the anchor
— from a body that is now inside the junction. There is no hand-back protocol
to get wrong and `active_perception.py` is not edited. One call site, not four.

**2. A turn-down detector over a windowed trace, not a threshold on one
frame (the estimator).** r006 released the turn when the turn-side building
fraction fell below a fitted `0.10`. The diagnostician then measured that same
band on **743** real frames: the absolute form passes 96.2 % of in-window and
93.5 % of out-of-window frames — it discriminates *nothing* — and in-window
side fraction is actually **lower** (mean 0.434) than out-of-window (0.583),
so no absolute floor can separate them. Its verdict: *"the inverted form is the
real signal — the named-side building fraction has just fallen from its running
maximum"*. So `SideBandTrace` keeps a rolling per-side trace of the band and
`opened()` fires on a **relative** fall (`current ≤ 0.65 × windowed peak`,
peak ≥ 0.12, ≥ 3 samples). The trace is pushed on **every decision of the
episode**, including the legs the module does not act on: the TURN leg is only
a handful of decisions long, so a leg-scoped statistic would have no peak to
fall from, while an 8-decision window reaches back into the tail of the
preceding ROAD leg — travelled on the *same* heading the TURN leg starts on, so
the samples are directly comparable. Scale-free (segmentation, never
`obs['depth']`, whose per-frame min-max JET has no absolute scale and killed
the previous lever) and fit-free.

**3. Reuse over re-derivation (the interfaces).** The leg predicate
`turn_leg_direction` is AP's *own* `_parse_relative_turn` / `_parse_target_cardinal`,
imported, not a second regex set: the decision being made is "delay the burst
AP is about to emit", so it must agree with AP **by construction** rather than
by coincidence — two independent taxonomies over the same text is a drift bug
waiting for a phrasing they disagree on. Likewise the band's geometry and
building class come from `completion_checker`'s probe-measured
`_NAMED_SIDE_ROW` / `_NAMED_SIDE_COL_FRAC` / `_NAMED_SIDE_BUILDING`, imported
rather than re-typed, so there is one operating point in the harness instead of
two that can diverge. Both imports have local fallbacks for a standalone load.

**Scope: the TURN leg alone.** The gradient is explicit that this must ship
unbundled — `cand_r001` shipped the turn change together with a ROAD-commit
rewrite, ROAD collapsed 0.24 → 0.02, only one episode ever reached TURNING and
the turn change was **unobservable**; r006 bundled it with a face-scan
re-point and a FINISH commit ban and ROAD went 0.24 → 0.20. So `step()` returns
`None` on ORIENT, ROAD and FINISH legs (updating only its trace), the FINISH
commit ban and the face-scan re-point are deliberately **not** taken, and every
`-1` route in `completion_checker.py` is left exactly as the anchor has it.
Attribution this iter is therefore one leg wide.

## why

§2.3 of the diagnosis: **12/12** TURNING commits fired at
`substeps_in_leg == 1` — `[rotate, -1]` on the *first* decision of the leg with
**zero** translation since the ROAD waypoint. In **7/7** traced failures the
**yaw error was 0** (the rotation was always right) and the body was **78–980
cm** — one to two 500 cm forwards — short of the junction box. The exact replay
converts **5/12 → 8/12** (one extra forward) **→ 11/12** (two) **→ 11/12**
(three), decaying past five, and the already-earned ROAD window holds in 12/12
at ≤ 2 because an in-place rotation preserves position. So `[2, 3]` is the
measured plateau: the floor stops the detector releasing at a point known to be
worse, the cap stops a silent detector walking out the far side, and the peak
detector only ever chooses *where on* the plateau to land. The leg's own budget
is 50 steps against ~5 used, so the forwards are free.

TURNING is also the gate to FINISH, so the same change puts more episodes on the
last leg. Predicted: TURNING 0.111 → ~0.22–0.24 with ROAD held at the anchor's
0.24 (r006 reached 0.20 on TURNING while paying 0.04 of ROAD for it), i.e.
SubtaskSR ≈ 0.39–0.40 against the anchor's 0.360 and r006's 0.370. SR stays
gated on FINISH, which no candidate has yet scored and which this one
deliberately does not touch.

**Risk, stated honestly.** (a) AP is not consulted for the ≤3 approach
decisions, so the red-signal idle cannot fire there. Bounded: three decisions
per episode, on a secondary metric whose detector fired ≤5 times in 3014 steps.
(b) If the ROAD leg happened to commit *past* the junction, the approach walks
further past it — but the measured sign is the opposite in 7/7, and the cap
bounds the cost at 3 forwards. (c) The relative-drop constants are a physical
argument, not a fit; the floor guarantees the detector can only choose between
two values that both measure 11/12, so a mis-tuned drop costs at most the
difference between 2 and 3 forwards. (d) The trace spans a leg boundary; if a
ROAD-leg side-scan left the body yawed the samples would not be comparable, but
the scan restores the heading within the same decision, so the frame at the top
of the next iteration is always on the travel heading.

**Anti-cheat: unchanged and unextended.** The new module reads only the static
`task_spec['subtasks'][i]['instruction']` text (through AP's parsers) and
`obs['object_mask']` — the sanctioned channel the perceiver already receives.
No coordinate, no `gt.json` field, no `agent_location` / `target_location`, no
analog `_agent_rotation` yaw, no `subtask_max_steps`, no oracle enters any
decision path. No new read paths at all.

**Verified before shipping** (`~/miniconda3/envs/simworld/bin/python`, loaded
through the evaluator's own `spec_from_file_location(f"{pkg}.harness", …)` path):
- `validate_harness.py`: **OK** — `run_episode(env, task_spec, vlm_clients,
  trace_dir=None) -> EpisodeResult` exported unchanged; the 3 WARNs are the
  pre-existing `max_tokens` budgets (2048 / 8192 / 8192); the new module
  introduces **no** numeric literal ≥ 2000;
- leg predicate on the **real corpus**, all 199 `hard/` `task_config.json`
  files: the 181 four-leg tasks tag exactly leg 2 as TURN and the 18 two-leg
  tasks tag none — `(4, ('-','-','TURN','-')) ×181`, `(2, ('-','-')) ×18`,
  199/199; and the parser is confirmed to be `active_perception`'s own;
- detector unit tests: steady band 0.65/0.66/0.64 → not open; a fall to 0.30 →
  open; a band that never exceeds 0.05 → not open (no support); no mask at all
  → not open (fails to the cap, i.e. to the plateau, never to zero motion);
- fake-env `run_episode` rollouts on a synthetic 4-leg task, against the **same
  fixtures run through the parent** for a byte-level comparison:
  * anchor TURN leg → `[Rotate_Right, -1]` (zero forwards — the measured bug);
  * candidate, band steady → `[0, 0, 0, Rotate_Right, -1]`, note
    `TURN_APPROACH_CAP_3`;
  * candidate, band collapses at the junction → `[0, 0, Rotate_Right, -1]`,
    note `TURN_APPROACH_CUE_2`;
  * candidate, `object_mask` absent → `[0, 0, 0, Rotate_Right, -1]` (cap);
  * exactly **one** rotation in every case, correct direction for "Turn right";
  * ORIENT (`-1` on step 0, `subtask_success[0] = True`), the ROAD leg's
    forwards and commit, the FINISH leg, and a 2-leg task with no TURN leg are
    **action-for-action identical** to the parent.

## files

- `modules/pose_script.py` — **NEW**: `turn_leg_direction` (leg identity via
  AP's parsers), `side_band_fraction` (band via `completion_checker`'s
  constants), `SideBandTrace` (rolling trace + `opened()` turn-down detector),
  `PoseScript.step` (the TURN-leg approach program, `release_reason` /
  `approach_forwards` for traces)
- `modules/__init__.py` — export `PoseScript`; docstring 7 → 8
- `agent.py` — import + instantiate `PoseScript`; the single wiring point
  (`pose.step` consulted before `active.should_emit`, its result short-
  circuiting the AP consultation); one per-leg `TURN_APPROACH_{CUE,CAP}_{n}`
  note; module-list docstring 7 → 8
- `NOTES.md` — this file

*(unchanged: `AGENT.md`, `rules.md`, `__init__.py`,
`modules/{spatial_memory,prompt_assembler,perceiver,planner,affordance_estimator,active_perception,completion_checker}.py`)*
