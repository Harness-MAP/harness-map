# NOTES — `cand_r11_0_spatial-grounding` (iter 11, parent `cand_r4_0_spatial-grounding`)

## axis

**The component graph — a NEW node.** `modules/road_geometry.py` adds an **8th module**,
`RoadGeometry`, and `run_episode` grows a new loop stage that drives it. This is the structural /
wiring axis the steer asks for, and it is the first time in this chain that a module has been
*added* rather than rewritten: iters 1–10 all kept the 7-module graph byte-identical and moved code
*inside* components (r003's NOTES: "no file was added, no module introduced, no stage added to or
removed from the loop"; r004: "same seven modules, same wiring, no file added").

The unifying idea: **every arrival readout the harness has ever owned is language over raw RGB**,
and the parent diagnosis measured all three of them to be flat or inverted in true distance — the
binary REACH judge's positive rate is *higher* beyond 200 m than inside the 1 000–2 000 cm pass
window (§3). The new node answers the same question from **geometry** instead, on channels the
harness already decodes but has never let decide anything, and it is wired as a **peer** of the
CompletionChecker rather than a helper inside it: it owns per-leg state, and three separate consumers
read from it.

## module map (★ = new / rewired this iter)

| module | role | this iter |
|---|---|---|
| `harness.py` | entrypoint `run_episode(env, task_spec, vlm_clients, trace_dir=None)`; perceive→decide→act loop | ★ new sensor stage + new commit stage + 3 call-site args |
| `modules/road_geometry.py` | **★ NEW** — non-VLM arrival channel; per-leg intersection-crossing state machine over `obs['depth']` + `obs['object_mask']` | ★ added |
| `modules/completion_checker.py` | owns every `-1` | ★ `geometric_arrival`, the FINISH crossing floor, the geo-keyed look schedule, `_depth_gray` perf |
| `modules/spatial_memory.py` | odometry + step / boundary bookkeeping | untouched |
| `modules/prompt_assembler.py` | composes the live VLM prompt | untouched |
| `modules/perceiver.py` | 1 VLM call: frames → description | untouched |
| `modules/planner.py` | 1 VLM call: description → actions | untouched |
| `modules/affordance_estimator.py` | legal-motion filter | untouched |
| `modules/active_perception.py` | deterministic short-circuit rules | untouched |
| `AGENT.md` / `rules.md` | prose priors | **untouched — the new channel never enters the prompt** |

## evolvable levers

| lever | kind | where | status |
|---|---|---|---|
| frame-median depth as an "open scene" statistic | code | `road_geometry._FRAME_OPEN_MIN` | **NEW** — the one distance-discriminative readout found (in-window 0.59 vs 0.15 out, lift 4.0×) |
| lateral building-class fraction | code | `road_geometry._SIDE_BUILDING_MAX` | **NEW** — the second conjunct; both sides must have stopped being a wall |
| arm → rising-edge → first-crossing state machine | code | `RoadGeometry.update` | **NEW** — what turns a detector into a commit channel |
| zero-VLM ROAD commit | code | `CompletionChecker.geometric_arrival` + `harness.py` | **NEW** — the harness's only model-free commit path |
| geo-keyed REACH look schedule | code | `proactive_commit_check(geo_event=…)` | **NEW** — reallocates the 12-call budget, never grows it |
| FINISH intersection-crossing floor | code | `K_MIN_CROSSINGS_BEFORE_FINISH_COMMIT` | **NEW** — diagnosis §5-C, re-expressed in crossings instead of action counts |
| photometric normalisation before `_binary_reach_vlm_call` | code | `completion_checker:~700` | **untouched — the next obvious lever** (diagnosis §5-B: a measured 2.68× luminance shift the judge names in 58 % of its refusals) |
| the "I have passed it" backstop (stop forwards after ≥2 crossings) | code | would consume `RoadGeometry.crossings` | **untouched — now cheap to build**, diagnosis §5-D; deliberately deferred until §5-A is measured in a real run |
| `_binary_reach_vlm_call` sys_prompt | prose-in-code | `completion_checker:~700` | **frozen** — 7 reformulations priced and lost (diagnosis §6) |
| ORIENTATION cardinal burst · turn-approach latch · escape ladder · `commit_burst` | code | `active_perception`, `completion_checker` | **do not touch** — diagnosis §5-E, all working |

## change

**One new component, three consumers, no prose edited.**

**1. `modules/road_geometry.py` — `RoadGeometry` (new).** Per frame it computes two statistics and
nothing else:

- `frame_open` — the **median of the JET-decoded depth over the whole frame**, on the decoder's own
  0 (nearest) … 255 (farthest) index. `obs['depth']` is normalised *per frame*, so this is not a
  distance; it is a shape statistic — "how much of this frame sits in the far part of whatever range
  this frame spans". Walking a block, a facade fills the view at close range and the median sits low;
  standing in an intersection the cross street and sky open up and it rises.
- `side_building_max` — the larger of the two lateral bands' building-class (purple) fractions in the
  horizon band (rows 0.45–0.70 H, outer 25 % of columns). An intersection needs **both** building rows
  to have stopped.

A per-leg state machine turns that detector into a commit channel. A ROAD leg *begins* standing in
the intersection the previous leg just committed in, so the raw predicate is true on almost every
leg's first frame. Hence: the leg must first be seen **closed** (armed), only a **rising edge** after
arming counts as a crossing, and only the leg's **first** crossing may commit — at most once. The
total crossing count per leg is kept and exported.

**2. `harness.py` — a new loop stage** (`road.update(obs, cur_idx)`), placed inside the perceive
branch rather than at the top of the loop *on purpose*: those are exactly the frames the operating
point was measured on, and sampling the ActivePerception short-circuits too would make the edge
detector denser than anything that has been measured. It returns `{ok, event, crossings, arrival}`.

**3. Three consumers, all in `CompletionChecker` so every `-1` is still authorised by the module that
owns commits:**

- **`geometric_arrival` → a zero-VLM ROAD commit** (new stage in `run_episode`, placed *before* the
  proactive REACH check). ROAD legs only — a ROAD subtask is graded on **position alone**
  (`traffic_world.py:412-417`), so standing in the right intersection *is* the pass criterion and
  there is nothing for a language model to add. ORIENTATION, TURNING and FINISH are excluded by
  construction (`_is_road_instruction`, new).
- **`proactive_commit_check(geo_event=…)` → the look schedule is re-keyed on geometry** (diagnosis
  §5-A.1). The parent's stride is in forward *actions* and therefore distance-blind: on
  `ep_20_7_21` it fired 11 judge calls and skipped the frame where the robot stood **9 cm** from the
  waypoint. An opening now waives the streak floor and the stride spacing — but **never** the
  12-call cap, so this reallocates the budget rather than enlarging it, and the action-stride
  schedule stays live as the no-regression floor on legs the sensor cannot read.
- **`verify(leg_crossings=…)` + the same check as a second accept-validator inside
  `proactive_commit_check` → the FINISH far-field floor** (diagnosis §5-C). A FINISH destination
  always lies at least one block past the turn, so an arrival claim raised with no opening behind the
  robot is geometrically impossible. Fails open on non-FINISH legs and whenever the count is
  unavailable; it **defers** a commit, never forbids one.

**4. `_depth_gray` — performance only, output proven byte-identical.** iter 11 adds a fourth caller
of the JET decoder on every decision; unoptimised it cost **212 ms/frame**. The nearest-LUT search
now runs over the frame's distinct colours (a few hundred) instead of its ~12 000 downsampled pixels,
with the residual kept a per-pixel mean by weighting each unique colour with its count:
**186 ms → 9.5 ms, 19.7×**, and the three pre-existing depth callers get the same speedup.

**Not changed, on purpose:** the recognizer prompt and its criterion (frozen, 7 reformulations
priced), `AGENT.md`, `rules.md`, the planner gate, the cross-check, the side-scan, `commit_burst`,
the ORIENTATION burst, the turn-approach latch and the iter-4 motion channel. The geometric channel
is **never shown to the agent** — it is harness state, not a prompt.

## why

The gradient is a search-space steer toward the anchor/r003 region with **structure/wiring (the
component graph)** named as the productive aspect. Every prior iter in this chain, including both
exemplars, held the 7-module graph fixed; adding a node is the one direction along that axis that has
not been tried, and the diagnosis independently ranks the content of that node first.

The merits are the diagnosis's own §4: the binding constraint is a **VLM-CEILING on the arrival
channel**, and it is explicitly a ceiling on *language over raw RGB* rather than on the harness —
three independent language readouts (planner claim 0.033 in-window / 0.039 at >20 km; perceiver
"I am at an intersection" 0.443 vs 0.453; the binary judge 0.20 vs 0.072 in the far tail) are all
flat or inverted in true distance. Re-wording is closed. **The fix has to change what the decision is
computed from**, and §5-A is the one measurement in the diagnosis that found distance signal where
the VLM has none.

The harness reaches the graded window on **40 of 57** legs and converts **9**; 1 388 of 3 500 env
steps (40 %) are spent walking away from a waypoint already satisfied, and 18 of 50 episodes die by
walking off the city grid. Nothing in the parent can ever conclude "that was the intersection" — this
node can.

**Measured before shipping, on the parent's own 50-episode traces.** All numbers below come from
replaying the parent's **saved `object_mask` / `depth` frames through the shipped
`RoadGeometry` + `CompletionChecker` code**, not through a prototype; the true distances come from
the traces and are used for scoring the replay only — they never enter the harness.

- **The predicate.** Over 788 ROAD-leg frames, frame-median depth ≥ 190: in-window **0.593**,
  out-of-window **0.147** (lift 4.0×). The lateral-building conjunct sharpens the per-leg
  precision. The parent's own ground/building-only conjunct reproduces at 0.21 / 0.06 (lift 3.5×) —
  real but much weaker, which is why the depth statistic carries this change.
- **The commit channel.** Run through the state machine on the 45 ROAD legs: fires on **35**, and
  **22 of those fires land inside the pass window** (precision 0.63). The fires are sharply bimodal —
  74 … 2 219 cm, then nothing until 19 134 cm — i.e. the misses are legs where the agent was already
  hopelessly far, and only **2** of them were legs the parent actually converted. 10 legs never fire,
  so the existing REACH path keeps its shot there. **Expected ROAD conversion 9 → 25 legs.**
- **Why the 13 false commits are nearly free.** A failed `-1` is terminal, but an unconverted ROAD leg
  is *also* terminal for everything downstream: the legs are sequential, so TURNING and FINISH
  successes are a strict subset of ROAD successes (the parent's pos2 = 7/50 against ROAD 9/50). An
  episode that fires geometrically out-of-window was going to score 0 on this leg and every leg after
  it regardless — the parent lost those same episodes to walking off the map instead. With TURNING
  converting 7/9 behind a ROAD pass, pos1 0.18 → ~0.45 should pull pos2 0.156 → ~0.30 with it.
- **The FINISH floor.** Of the parent's 12 FINISH legs, the 8 that died on an instant false commit
  ended with **0 crossings** (6 of 8) or 1; **both** legs that walked the leg and actually arrived had
  **2**. So the floor vetoes 6 of the 8 instant deaths and false-vetoes neither real arrival — the gate
  it tightens is 0-for-8 on this subset.

Three things the diagnosis marks exhausted are avoided by construction: the arrival criterion string
is untouched (§6), no illumination clause enters any prompt (§6), and
`K_REACH_FORWARD_STREAK` / `K_REACH_INTERVAL` / `MAX_REACH_FIRES_PER_SUBTASK` are all unchanged —
re-keying the schedule on geometry is explicitly *not* the same as tuning its constants (§6). The
largest remaining lever, §5-B (photometrically normalising both images before the judge encodes
them), is deliberately left for the next iter so this one's attribution is clean.

**Anti-cheat.** The node reads `obs['depth']` and `obs['object_mask']` — both on
`evaluator_taint._ALLOWED_PATHS` and both already read by the parent — plus the static instruction
text from `task_spec['subtasks']`. No coordinate, no ground-truth field, no `info` read at all. Every
threshold is a frame fraction, a segmentation class colour, or a position inside the frame's **own**
normalised depth range; the largest numeric literal added is `190.0` (a 0–255 colormap index), well
clear of the lint's 2 000 flag. `validate_harness.py` passes with no new warnings.

**Verified before shipping** (`~/miniconda3/envs/simworld/bin/python`):

- `validate_harness.py` **OK** — `run_episode(env, task_spec, vlm_clients, trace_dir=None)` and
  `EpisodeResult`'s seven fields unchanged under the evaluator's `spec_from_file_location` path;
  anti-cheat scan clean; the only literal warnings are the parent's three pre-existing token budgets.
- **43/43 unit assertions** on the shipped code: the two statistics and all five fail-silent branches;
  the state machine (starts-in-an-opening does **not** fire, arming, rising edge, one arrival per leg,
  second crossing gives `event` but not `arrival`, per-leg independence, an unevaluable frame does not
  advance the edge detector); `geometric_arrival` admitting ROAD and rejecting TURNING / FINISH /
  ORIENTATION / empty / `None`; `_is_road_instruction` + `_is_finish_instruction` on all four
  generator templates; the crossing floor's four branches; `verify`'s old 5-arg call site still
  working; and `geo_event` waiving the floor and the stride but **not** the 12-call cap.
- **Trace replay**: 951 real frames; the text leg-classifier agrees with the ground-truth leg type on
  **951/951** (788 ROAD, 163 FINISH, zero disagreements); at most **1** geometric commit per leg;
  **0** geometric commits on non-ROAD legs.
- **`_depth_gray` equivalence**: byte-identical on 200 real depth frames and on all four fallback
  branches (odd shape, missing key, `None`, tiny frame), 19.7× faster.
- **Three end-to-end `run_episode` dry runs** against a fake env: a walled block followed by an
  opening commits at the opening and passes the leg (`GEO_COMMIT`, first `-1` at action 7, after the
  walled stretch); a leg that is open the whole way **never** commits (arming never happens); a blind
  depth channel degrades to exactly the parent's behaviour (`geo reads=0`, no commit).

## files

- `modules/road_geometry.py` — **NEW**: `RoadGeometry`, `read_geometry`, `_Reading`, `_LegState`,
  `_lateral_bands`, `_building_frac`, and the seven operating-point constants
- `modules/__init__.py` — export `RoadGeometry` (the 8th node)
- `modules/completion_checker.py` — `K_MIN_CROSSINGS_BEFORE_FINISH_COMMIT`; `_is_road_instruction`;
  `_finish_crossing_ok`; `CompletionChecker.geometric_arrival`; `verify(leg_crossings=…)`;
  `proactive_commit_check(geo_event=…, leg_crossings=…)` (floor + stride waiver, crossing
  accept-validator); `_depth_gray` unique-colour inner loop (perf only, output byte-identical)
- `harness.py` — import + instantiate `RoadGeometry`; the new sensor stage before the Perceiver; the
  new geometric ROAD commit stage before the proactive REACH; `leg_crossings` / `geo_event` threaded
  into `verify` and `proactive_commit_check`; `GEO_COMMIT` note + a per-episode sensor-coverage token;
  module-map docstring
- `NOTES.md` — this file

*(unchanged: `AGENT.md`, `rules.md`, `__init__.py`, and
`modules/{spatial_memory,prompt_assembler,perceiver,planner,affordance_estimator,active_perception}.py`
— all verified byte-identical to the parent.)*
