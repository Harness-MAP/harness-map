# NOTES — `cand_r010_0_spatial-grounding` (iter 10, parent `cand_r2_0_spatial-grounding`)

## axis

**The arrival channel for an intersection-stop leg is SEGMENTATION GEOMETRY, not the hint photo.**

Token-level, in the sense the steer asks for: the component graph is byte-for-byte the
parent's — the same seven modules, the same call sites, the same wiring in `run_episode`.
**No file added, no module introduced, no stage added to or removed from the loop.** Three
files differ (`modules/completion_checker.py` +194 lines, `modules/prompt_assembler.py` 14,
`orchestrator.py` 9, the last of which is one note label). Everything that moved is code *inside*
components.

The unifying idea, and it is one sentence: **a leg whose stop condition is a geometric event
is being decided by an appearance question.** The ROAD leg's instruction is *"Continue
straight and stop at the intersection with \<building\> on your left"*; the harness answers it
by asking the frozen VLM whether a live frame "matches" a differently-lit reference photo — a
channel the diagnosis measured at 4.1 % yes in-window vs 2.6 % out (**LR 1.6**), now closed
across three wordings and four evidence sets. Meanwhile the intersection itself is written
into a sanctioned observation the harness already ships and already attaches to two prompts:
the ground class in `obs['object_mask']` flips from sidewalk to roadway when the cross street
opens ahead. This candidate gives that cue the `-1` license on exactly the legs it can decide,
and leaves every other leg untouched.

This is the move that already produced the only reliable leg in this lineage. Leg 2 (TURNING)
passes 5/6 conditional on leg 1 with **zero** VLM calls because iter 6 replaced "does this
look like the turn?" with "the instruction says turn right, so rotate right". Leg 1 has the
same structure available; it had simply never been read.

## module map (unchanged graph; ★ = internals changed this iter)

| module | role | this iter |
|---|---|---|
| `orchestrator.py` | entrypoint `run_episode(env, task_spec, vlm_clients, trace_dir=None)`; perceive→decide→act loop | ★ one note label (`ROAD_ARRIVAL` vs `REACH_COMMIT`) so the trace can attribute the commit to a channel. No control flow touched. |
| `modules/spatial_memory.py` | scratchpad: per-step action + odometry ring | untouched |
| `modules/prompt_assembler.py` | composes the perceive prompt (4 channels + legend) | ★ the segmentation legend now names the real classes (see change 4) |
| `modules/perceiver.py` | 1 VLM call: 4 images → free-text description | untouched |
| `modules/planner.py` | 1 text-only VLM call → `{Summary, Match, Actions}` | untouched |
| `modules/affordance_estimator.py` | legal-motion filter | untouched |
| `modules/active_perception.py` | cardinal burst, turn-approach latch, red idle, anti-spin, momentum | **deliberately untouched** — the cardinal burst is 50/50 and the turn latch is the best thing in this lineage |
| `modules/completion_checker.py` | owns every `-1` | ★ the new detector + the three gates that now consult it |

## evolvable levers

| lever | kind | where | status |
|---|---|---|---|
| **roadway-class arrival detector** | code | `_roadway_band_fraction`, `road_surface_arrival` | **NEW this iter** — owns the `-1` on every ROAD leg |
| **leg-type router** (which channel owns which leg) | code | `_is_intersection_stop_instruction` | **NEW this iter** — verified exact on all 796 legs of the hard set |
| arrival-call **evidence set** (mask + depth + description) | code | `_binary_reach_vlm_call` | kept verbatim — now unreachable on ROAD legs; still the FINISH channel |
| arrival-call **criterion wording** | prose-in-code | `_binary_reach_vlm_call` sys_prompt | **CLOSED, do not touch** (three operating points, flat ROC) |
| cross-check arbitration (refusal allowance / cost cap / accept memo) | code | `cross_check_planner_commit` | kept verbatim — now reached only on FINISH legs |
| segmentation **legend** | prose | `prompt_assembler` ×2, `completion_checker` ×1 | ★ corrected this iter — it had denied the existence of the cue |
| blocked-commit HOLD response | code | `orchestrator.py` `HOLD_PROBE` | kept verbatim; now also absorbs a geometric veto of a bare planner `-1` |
| terminal-pose composer (`commit_burst`) | code | `completion_checker` | kept verbatim — ROAD commits stay a bare `-1`, FINISH keeps its quarter turn |
| turn-approach latch (`_TURN_APPROACH_FORWARDS`) | code | `active_perception` | untouched, but see §why — it is now the next thing to look at |
| **FINISH leg** (0.00 for ten iters) | code | `completion_checker` + side scan | **the open axis, deliberately NOT taken** — it would confound this measurement |
| stall recovery / the dead stuck detector (`forward_count` over one burst) | code | `orchestrator.py:~300` | untouched — known-dead, next-largest open population |

## change

**1. The detector** (`_roadway_band_fraction`, new; `modules/completion_checker.py`). The
fraction of `obs['object_mask']` that is roadway class, over the center forward band
(rows 0.45–0.70 of H, columns 0.30–0.70 of W — the travel corridor). Returns `None`, not
`0.0`, when the mask is absent or malformed, so a missing sensor can never be read as "no
intersection". Matches the class in **both channel orders** ((0,127,255) and (255,127,0)):
unlike the purple building class the existing accept-validator leans on, the roadway colour
is not palindromic, and nothing in the sanctioned surface guarantees RGB over BGR.

**2. The license** (`road_surface_arrival`, new). Three-valued, and the third value is the
point of it: `None` = "this channel does not own the leg" (falls through to the unchanged VLM
path); `True` = commit; `False` = not yet, *and the VLM path stays suppressed*. On a leg
defined by a junction, "I have not seen the junction" is a positive statement about the world,
not a missing answer — that half is what removes the far-field false commits. Latches once per
leg: a `-1` the env grades FAIL ends the episode, so a leg may spend exactly one geometric
claim. Inputs: `obs['object_mask']` and the static `task_spec` instruction text. No VLM call,
no token, no coordinate.

**3. Three gates consult it, and the wiring does not move.** `proactive_commit_check` asks it
**first**, ahead of every cost gate, because those gates are calibrated for a channel that no
longer decides this leg. `cross_check_planner_commit` asks it first too, so a planner `-1` on
a ROAD leg is answered by the same detector and the VLM cross-check is not spent at all —
measured on the parent run, of the 15 planner `-1`s that reached that function, 3 were raised
in-window and 9 out of it were fatal, and all 3 good ones sit in episodes where the detector
also fires in-window. `request_side_scan` is suppressed on an owned leg (1 fire / 0 commits in
50 episodes last run, 4 / 0 the run before; it stays live on FINISH, which is what it was
built for). The **leg router** `_is_intersection_stop_instruction` is four conjuncts, each
excluding a leg type owned by a channel that already works: not ORIENTATION (cardinal burst),
not a turn verb (the turn latch), not FINISH (graded on facing), and the word *intersection*
must be present.

**4. The legend stops denying the cue exists.** Three strings told the model
*"yellow = ground/road, blue = other"*, collapsing the two ground classes the task is built
out of and throwing the roadway into "other". Re-probed on this candidate's own traces, the
palette is: `(255,255,0)` **sidewalk/crosswalk**, `(0,127,255)` **roadway/driveway**,
`(128,0,128)` buildings, `(0,255,0)` sky, `(0,128,0)` trees. All three now say so, and the two
perceive-side strings add the sentence that makes it actionable: *yellow ahead turning into
blue ahead means you are entering an intersection.* The **Match criterion is byte-identical** —
the exhausted list forbids re-wording it, and this is the sensor legend, not the criterion.

**Not changed, on purpose:** the recognizer prompt criterion, its evidence channels, the
confirmation counts, every schedule constant, `active_perception`, and the FINISH path.

## why

**The gradient.** `diagnosis.md` §4 classifies the binding constraint as a VLM-CEILING on the
channel the harness chose to decide with, wrapped in harness bugs that keep the better channel
out; §5-B ranks the roadway-class detector first by leverage; §6 forbids another iteration on
the hint-photo channel at any threshold, wording or evidence set. The search-space steer asks
for the *token-level* region (r004, r007): change what happens inside the components, not the
graph. This is that — one new predicate, one new method, three call sites, no new module.

**The measurement, made twice.** First as a probe over the parent run's saved
`object_mask` frames (45 ROAD legs, aligned to true position by step index, ground truth read
for *analysis only* and hard-coded nowhere), then again by **replaying the shipped
`road_surface_arrival` itself** frame-by-frame over those same legs, at the parent's own
~1-decision-per-3-env-steps cadence. Both agree exactly:

| | in-window (<2000) | fires early | never fires |
|---|---:|---:|---:|
| **this detector** | **24/45** | 11 | 8 |
| incumbent hint-match channel | 6/45 | 9 (all fatal) | — |

Median true L1 at the commit: **1106 cm**, comfortably inside the 2000 threshold. All 8 legs
where it never fires already fail leg 1 today, so suppressing the VLM path on them costs
nothing measurable. The 11 early fires are the price, against 16 fatal commits in the parent.

**It is not a tuned constant.** Every `(row band, threshold)` pair from (0.30–0.55, 0.03)
through (0.50–0.75, 0.03) scores 22–25 in-window — a broad plateau, so what is encoded is
"roadway occupies a few percent of the forward view", not a knife-edge. Both halves of the
episode set move the same way (9 in / 7 early, 15 in / 4 early). Every number is a fraction of
frame dimensions; the lint reports no new literal ≥ 2000.

**Predicted:** leg-1 pass **0.12 → 0.45–0.53**. Leg 2 is arithmetically pinned to leg 1 (the
leg-1 waypoint lies inside the leg-2 box in 45/45, and leg 2 passes 5/6 conditional on leg 1),
so leg-2 pass should track it to ≈0.40 → **SubtaskSR ≈ 0.45–0.50** from 0.330. Cost should
*fall*: the ROAD leg, which is where the episodes spend their steps, no longer makes arrival
VLM calls at all.

**What to watch next, in order.** (i) The 11 early fires — a distant crossing visible down the
street. A growth/approach test over the leg's own running baseline was tried here and did not
separate them (23/12 vs 24/11); the more promising handle is the cadence, since the 3-forward
burst still samples an 8-step window ~2 times. (ii) `_TURN_APPROACH_FORWARDS = 2` is now
mis-set: it was calibrated for leg-1 commits raised *early* by the VLM path, and a commit
raised *at* the crossing is already inside the leg-2 box — 1000 cm of extra approach can only
overshoot it (this is exactly how the single leg-2 failure lost by 22 cm). (iii) The FINISH
leg, 0.00 for ten iterations, is now the largest untouched population and the only leg still
decided entirely by the closed channel.

**Anti-cheat.** The detector reads `obs['object_mask']` (sanctioned perception) and the static
`task_spec['subtasks']` text. No `agent_location`, no `target_location`, no `gt.json`, no
`current_instruction`, no analog yaw. The waypoint geometry quoted above was computed from
`gt.json` **for this write-up only** and appears nowhere in the code. `validate_harness.py`
passes with 3 pre-existing `max_tokens` WARNs and no new ones.

**Verified before shipping.** Imported through the evaluator's own `_import_harness` path:
`run_episode(env, task_spec, vlm_clients, trace_dir=None) -> EpisodeResult`, signature and
`EpisodeResult` unchanged. The leg router classifies **796/796 legs of the hard task set**
exactly right (True on all 181 ROAD legs, False on every ORIENTATION / TURNING / FINISH).
Unit checks on the detector: fires on a junction frame, silent on an all-sidewalk frame,
handles BGR, latches once per leg, and returns `None` — not `False` — for a missing mask, a
malformed mask, `obs=None`, a FINISH leg and an ORIENTATION leg. Gate checks: on a ROAD leg
both `proactive_commit_check` and `cross_check_planner_commit` return the detector's verdict
with **zero** VLM calls and zero fire-counter spend, and both behave byte-identically to the
parent on FINISH / ORIENTATION legs. Two fake-env dry runs of the full `run_episode`: the
agent walks the ROAD leg and commits the step the junction appears (`ROAD_ARRIVAL`, 1 perceive
+ 1 plan per decision and no third call), the turn latch then fires unchanged, and the
junction-at-spawn case commits immediately.

## files

- `modules/completion_checker.py` — `_ROADWAY_CLASS_RGB/_BGR`, `_MASK_CLASS_TOL`,
  `_ROAD_BAND_ROW/_COL`, `_ROAD_ARRIVAL_FRAC`, `_roadway_band_fraction`, `_TURN_VERB_RE`,
  `_is_intersection_stop_instruction`, `CompletionChecker.road_surface_arrival`,
  `_road_arrival_fired` / `last_commit_channel` fields; call sites at the top of
  `proactive_commit_check`, `cross_check_planner_commit`, `request_side_scan`; the
  `last_commit_channel` label on the VLM accept; the segmentation legend in
  `_binary_reach_vlm_call`.
- `modules/prompt_assembler.py` — the segmentation legend, in the perceive system prompt and
  in the mask attachment caption.
- `orchestrator.py` — one note label (`ROAD_ARRIVAL` vs `REACH_COMMIT`) at the existing REACH
  commit site. No control flow changed.
