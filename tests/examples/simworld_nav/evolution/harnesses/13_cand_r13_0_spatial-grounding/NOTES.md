# NOTES — `cand_r13_0_spatial-grounding` (iter 13, parent `cand_r9_0_spatial-grounding`)

## axis

**Component graph: a NEW, FREE, NON-VLM node supplies the POSITION half of the
arrival question, and it keys the schedule of the one VLM question that works.**

`modules/road_geometry.py` adds a **10th module**, `RoadGeometry`, and
`run_episode` grows a new sensor stage that drives it. Three graph edits, one
idea:

* **+ node / + stage** — `RoadGeometry` reads the roadway class out of
  `obs['object_mask']` once per loop iteration and runs a per-leg **rising-edge
  state machine**: "the body has just crossed into an intersection". Zero
  tokens, zero env substeps.
* **↻ edge** — on an **intersection-stop ROAD leg** the gate's position
  conjunct re-points from `ArrivalGate._place_call` / `_JUNCTION_SYS` (the VLM
  junction question) to `RoadGeometry.place_verdict`. Three-valued, r010-style:
  `None` = not owned → the VLM question runs exactly as before; `True` = edge
  here; `False` = owned and not here → **free veto, no token spent**.
* **↻ edge** — the *identity* call's schedule stops being clock-keyed on owned
  legs and becomes **event-keyed**: `K_REACH_FORWARD_STREAK` / `K_REACH_INTERVAL`
  are waived on a geometric edge and the call is otherwise **not made at all**.
  The gate's one good question moves from "every 5 translations" to "at the
  junction".

Plus the one HARNESS-BUG the diagnosis isolated (§2E / F2): the cross-check cap
no longer **fails open**.

The unifying sentence: **the harness had a good `which` channel and no `where`
channel.** `_identity_call` (text-free, two images) is leg-level and measured at
**93 %** in-window recall — it says *which landmark* and saturates at 1 across
whole legs. `_JUNCTION_SYS` is the position conjunct and measured at **29 %**,
with 18 of its 19 positives across 50 episodes citing one disjunct ("a crosswalk
is at or just in front of the robot") — a cue at a fixed half-block **offset**
from the grading point, and one that cannot tell the target junction from any
other (3 positives at 17 780 / 20 078 / 22 842 cm). Geometry knows *where* but
not *which*; identity knows *which* but not *where*. This iter is the first time
the chain has had **both halves**, and it is the first time the position half is
not a sentence.

## module map (★ = new / rewired this iter)

| module | role | this iter |
|---|---|---|
| `agent.py` | entrypoint `run_episode(env, task_spec, vlm_clients, trace_dir=None)` | ★ 5 sites: `RoadGeometry` imported + instantiated + handed to the gate; the new sensor stage inside the perceive branch; a `GEO_REACH` vs `REACH_COMMIT` note label; `road.summary()` appended to `notes`; docstring 9 → 10 nodes |
| `modules/road_geometry.py` | **the position conjunct** — roadway-class band fraction + per-leg rising-edge state machine | ★ **NEW MODULE** |
| `modules/arrival_gate.py` | the `-1` LICENSE | ★ `geo=` field; `_place_verdict` asks geometry first; `proactive_commit` gets the free veto + the event-keyed schedule; `allow_planner_commit` gets the free veto + the **fail-closed** cap |
| `modules/__init__.py` | package exports | ★ export `RoadGeometry`; docstring 9 → 10 nodes |
| `modules/completion_checker.py` | legacy recognizer (gate's degradation path) + planner-`-1` verify | **untouched — byte-identical** |
| `modules/viewpoint_controller.py` | TURN approach + FINISH commit pose | **untouched — byte-identical** |
| `modules/active_perception.py` | cardinal burst, relative-turn burst, red idle, anti-spin | **untouched — byte-identical** |
| `modules/{spatial_memory,prompt_assembler,perceiver,planner,affordance_estimator}.py` | — | **untouched — byte-identical** |
| `AGENT.md`, `rules.md` | prose priors | **deliberately byte-identical** — the agent's prompt must not move while the decision graph does, or the two are not separable |

*(verified by `cmp`: 8 of the 11 pre-existing source files and both prose priors
are byte-for-byte the parent's.)*

## evolvable levers

| lever | kind | where | status |
|---|---|---|---|
| **roadway-class band fraction** (the detector) | code | `road_geometry.roadway_band_fraction` | **NEW** — measured 77.3 % of in-window ROAD frames vs 25.9 % beyond 4 000 cm |
| **rising-edge state machine** (detector → event) | code | `RoadGeometry.update` | **NEW** — the edge is what is claimable; the plateau is not (edges 31 : 19 IN : FAR, all in-run frames 52 : 59) |
| **tri-valued ownership** (`None` / `True` / `False`) | code | `RoadGeometry.place_verdict` | **NEW** — `False` is a *positive* statement, and it vetoes for free |
| **event-keyed identity schedule** | code | `arrival_gate.proactive_commit` | **NEW** — the clock is replaced by the cue on owned legs; strictly fewer identity calls, all of them at a junction |
| **free veto before the cost cap** | code | `arrival_gate.allow_planner_commit` | **NEW** — a mid-block planner `-1` is refused for 0 tokens |
| **cap bounds CALLS, not the VERDICT** | code | `arrival_gate.allow_planner_commit` | **NEW (F2)** — was 6/15 fatal commits incl. all 4 FINISH deaths |
| operating point `ARRIVAL_FRAC = 0.03` | code/const | `road_geometry` | **measured this iter**, threshold sweep in §why; broad plateau, not a knife edge |
| `MAX_CLAIMABLE_CROSSINGS = 2` | code/const | `road_geometry` | **measured this iter** — 25/45 → 31/45 leg coverage for 7 extra far-field opportunities; **3 is the obvious next value, not an axis** |
| band geometry `(0.45-0.70) × (0.30-0.70)` | code/const | `road_geometry` | measured flat across 0.40-0.65 … 0.50-0.75 — **do not spend an iter on it** |
| "arming" (require a closed reading before edge #1) | code | `road_geometry` | **MEASURED AND REJECTED** — 24/45 vs 25/45; a ROAD leg does not generally start in a junction |
| **env-step-rate sampling of the detector** (diagnosis F1's pacing half) | code | `agent.py` burst executor | **still open, deliberately deferred** — see §risk (c); the measured operating point is a *rising edge at loop cadence* and a denser cadence moves it |
| **the "have I passed it?" self-referential call** | code | would be a new `road_geometry` consumer | **still open** — gradient F4; now cheap, the frame history and an edge counter both exist |
| **FINISH leg** (`_FRONTAGE_SYS`, 0/12) | code | `arrival_gate` + `viewpoint_controller` | **still open and still 0.00** — gradient F3 (ask from the faced pose). Not taken: it is a *pose* axis and would confound this measurement |
| text-free `_IDENTITY_SYS` | prose-in-code | `arrival_gate` | **keep, do not touch** — 93 % in-window, the chain's biggest perception win |
| `_JUNCTION_SYS` wording | prose-in-code | `arrival_gate` | **CLOSED** — the model collapses every wording onto the crosswalk cue; the *question type* is what changed here |
| ORIENTATION cardinal burst · TURN approach latch | code | `active_perception`, `viewpoint_controller` | **do not touch** — 50/50 and 92 % conditional conversion |
| `k = 2` confirmations on a `-1` | code | — | **CLOSED** — gradient F5: 19 legs produced exactly one `place == 1` and **zero** produced two |

## change

**1. `modules/road_geometry.py` — `RoadGeometry` (new).** Per frame it computes
one statistic: the fraction of the **forward travel corridor** (rows 0.45–0.70 H,
columns 0.30–0.70 W — fractions, no pixel literal) of `obs['object_mask']` that
carries the **roadway class**. Walking a block, that corridor is sidewalk; when
the cross street opens it floods with roadway. That is the intersection — the
literal grading point of a ROAD leg ("Continue straight and stop at the
intersection with `<building>` on your left") — in closed form. It returns
`None`, never `0.0`, on a missing or malformed mask, so an absent sensor can
never be read as "no intersection". The class is matched in **both channel
orders**: unlike the purple building class the existing side veto leans on, the
roadway colour is not palindromic and nothing in the sanctioned surface
guarantees RGB over BGR.

A per-leg state machine turns the detector into an **event**. Standing in a
junction keeps the raw predicate true for several consecutive readings, and the
measured operating point is the **rising edge**, not the plateau. Only the first
`MAX_CLAIMABLE_CROSSINGS = 2` edges of a leg are claimable; a leg may not claim
before the body has translated at all (`GEO_MIN_TRANSLATIONS = 1`).

The **router** is two conjuncts: `viewpoint_controller.leg_type(…) == LEG_ROAD`
and the word *intersection* present. Verified against this run's own traces:
**40/40** distinct ROAD instructions contain the word and **no** FINISH
instruction does; "Turn left at the intersection" also names one but classifies
as `LEG_TURN` and is excluded, because it is graded on **heading** and already
owned by the turn-approach latch. ORIENT, TURNING and FINISH are untouched.

**2. `agent.py` — a new sensor stage.** `road.update(obs, instruction,
cur_idx)` runs at the top of the **perceive branch**, on the same frame the
perceiver sees. That placement is deliberate and is the whole reason the number
in §why transfers: those are exactly the frames the operating point was measured
on. Putting it on the `ActivePerception` short-circuit frames, or at env-step
rate inside the action burst, would make the edge detector denser than anything
measured, and a rising-edge rule is sensitive to its sampling rate in a way a
threshold rule is not.

**3. The gate asks geometry before it spends a token.** `_place_verdict`
consults `RoadGeometry.place_verdict` first and only falls through to
`_place_call` on a `None`. More consequentially, `proactive_commit` now asks it
*before the schedule*: a `False` returns immediately (`geo_block`), so on an
owned leg the identity call is **not spent on frames that cannot license a
commit**; and a `True` **waives** the `K_REACH_FORWARD_STREAK` floor and the
`K_REACH_INTERVAL` spacing, because a geometric edge is a one-shot event and
waiting out an interval simply discards it. The parent asked its arrival
question ~1.6 times while crossing a window 8 forward steps wide and then walked
1 500–3 500 cm blind before asking again. The answer here is not to ask *more*
(F1) but to ask in the *right place* — and because the keying cue is free, that
costs **fewer** calls, not more: in the dry run where the leg has one edge and
identity answers 0, the whole ROAD leg spends **one** identity call where the
parent's schedule allowed twelve.

**4. The cross-check cap fails closed (F2).** `allow_planner_commit` used to
`return True` once `_xcheck_fires >= 2`, so the **third** planner `-1` on a leg
executed with no gate at all — the run shows the signature 10 times
(`XCHECK_BLOCK; EMPTY_FALLBACK; XCHECK_BLOCK; EMPTY_FALLBACK; …`) and it
produced **6 of the 15 fatal commits, including all four FINISH deaths** at
24 389 / 31 370 / 32 527 / 44 789 cm, two of them discarding an episode that had
already banked three legs. A budget bounds **calls**, not the **verdict**: the
cap now replays `_xcheck_memo`, the last verdict actually computed on that leg,
and on a FINISH leg a `-1` may never pass by deferral even if no verdict exists.
The free geometric veto is asked *before* the cap, so on an owned leg the cap is
now reached only by edges that were worth a call.

**Not changed, on purpose:** both prose priors, every schedule constant
(`K_REACH_*`, `MAX_REACH_FIRES_PER_SUBTASK`, `MAX_XCHECK_FIRES_PER_SUBTASK` — all
still imported from `completion_checker`, none retuned), the `_IDENTITY_SYS` and
`_JUNCTION_SYS` and `_FRONTAGE_SYS` wordings, the `_named_side_has_structure`
veto, the legacy degradation edge, ORIENT, TURNING, and the whole FINISH path.

## why

**The directive** is a search-space steer toward the region exemplified by
`cand_r011_0` and `cand_r004_0`, with **structure/wiring (the component graph)**
named as the aspect to vary. r011's productive difference is precisely this
shape: *"the first time in this chain that a module has been added rather than
rewritten"* — a non-VLM geometric channel wired as a **peer** of the arrival
owner, with its own per-leg state and several consumers. This iter varies the
graph in that direction and adds the thing that region has and this lineage does
not: a decision channel that is **not a sentence**. It is not a copy of either
exemplar — r011 and r010 made geometry the *license* and had no identity
conjunct at all; here geometry is **one conjunct of two**, and the other conjunct
is the 93 %-recall text-free identity call that only this lineage has.

**The gradient.** §4 classifies the binding constraint as **TASK-GAP**: *"the
harness now recognises arrival but still has no arrival controller … it samples
a 29 %-recall binary roughly 1.6 times while crossing a window that is 8 forward
steps wide"*, and 28/50 legs walk through the window and keep going a median
13 008 cm. §2C shows *why* the 29 % cannot be raised by wording: every positive
collapses onto one crosswalk cue at a fixed offset, and §6 closes junction
wording A/B explicitly — *"the question type is what must change, not the
phrasing."* This candidate changes the question type from language to geometry,
and takes F2 with it because F2 is what stops the extra commit activity from
leaking through an ungated hole.

**Measured before shipping** (`~/miniconda3/envs/simworld/bin/python`, on the
**parent's own 50-episode run**: the archived `object_mask` PNG of all **1 326**
perceive calls, scored against the true L1 distance to that leg's graded
waypoint. The distances are oracle fields read **offline to label frames**,
exactly as the diagnostician's §2A probe does — nothing here is read at eval time
and no coordinate appears in the shipped code).

*Per-frame discrimination, ROAD legs, `band_fraction > 0.03`:*

| | n | fires |
|---|---:|---:|
| in-window (≤ 2 000 cm) | 75 | **77.3 %** |
| 2 000–4 000 cm | 75 | 29.3 % |
| > 4 000 cm | 532 | 25.9 % |

*Per-leg, first rising edge of each of the 45 ROAD legs, at the parent's own loop
cadence — this is the shipped rule, and the threshold was swept, not assumed:*

| threshold | first edge IN | 2–4 k | > 4 k | never |
|---|---:|---:|---:|---:|
| 0.02 | 17 | 5 | 21 | 2 |
| **0.03** | **25** | 4 | 12 | 4 |
| 0.05 | 19 | 5 | 13 | 8 |
| 0.08 | 16 | 8 | 10 | 11 |
| 0.12 | 14 | 9 | 7 | 15 |

25/45 = 56 % against the live gate's 29 % per-ask and the run's actual ROAD pass
rate of **0.26**. The 0.03 peak is broad in the band geometry too (every row band
from 0.40–0.65 through 0.50–0.75 scores 22–25), so what is encoded is "a few
percent of the forward corridor is roadway", not a map-tuned constant.
**Arming** was measured and rejected (24/45, i.e. it costs a leg and buys
nothing). Claiming the first **two** edges instead of one takes leg coverage
25/45 → **31/45** for 7 extra far-field opportunities, which the identity
conjunct then filters at its measured 39 % far-field pass rate. Claiming all
frames of a run instead of its edge is strictly worse (52 : 59 IN : FAR against
the edges' 31 : 19), which is why only the edge is claimable. FINISH was measured
too and is **excluded on evidence**: its first edge lands in-window **0/17** and
beyond 4 000 cm **16/17** at every threshold, because a FINISH leg *begins* in
the intersection the TURNING leg just committed in.

*And then the **shipped module itself** was replayed frame-by-frame over those
same archived frames, through the evaluator's own
`spec_from_file_location` import path* — not the probe script, the code that
ships:

```
ROAD legs                     45
claimable edges raised        53   →  IN 30 · 2-4k 4 · >4k 19
legs with ≥1 in-window claim  30/45
legs that never claim          5/45
FINISH: place_verdict is None on every frame of every FINISH leg   ✓
```

(30/45 rather than the probe's 31 because `GEO_MIN_TRANSLATIONS` suppresses the
one leg whose first edge was its very first reading — by design.)

**The priced prediction.** 30/45 ROAD legs now get an in-window commit
*opportunity* (the parent's gate licensed one on 11/38 asks), each guarded by the
identity conjunct at its measured 93 % in-window / 39 % far-field rates: ≈ 28
in-window commits and ≈ 7–9 premature ones, i.e. `per_position_SR[1]` 0.26 →
plausibly **0.45–0.60**, carrying TURNING at the measured **92 %** conditional
rate (0.2667 → ≈ 0.45). F2 returns the 2 episodes that were thrown away with 3
legs banked. Token cost goes **down**, not up: the geometric veto answers most
frames for free, and the 595 gate calls per 50 episodes collapse toward ≈ 2
identity calls per owned leg. Note the two failure directions are **not**
asymmetric here: a fatal premature `-1` and a never-committed leg both score the
leg 0 and both forfeit everything downstream, so on ROAD the right objective is
recall, which is what the two-edge claim budget buys.

**Risk, stated honestly.**
(a) **5/45 legs never raise an edge** and, because `False` on an owned leg is a
veto, those legs now get *no* arrival check at all where the parent would have
asked the junction question (at 29 %). That is the price of the tri-valued
ownership and it is ≈ 1 leg of expected loss against the ≈ 15 the detector
is expected to add; the `GEO(no-readings/below=…)` ledger will name them.
(b) The 19 far-field claimable edges are all *before* the window (edge #1 of a
leg), so a far-field commit **forecloses** that leg rather than merely wasting a
call. The identity conjunct is the only thing standing in front of them, at a
measured 39 %. If the ledger shows `geo_1` high and `GEO_REACH` landing far, the
lever is `MAX_CLAIMABLE_CROSSINGS` 2 → 1 (25/45 coverage, 12 far-field
opportunities), not a new axis.
(c) **F1's pacing half is deliberately NOT taken.** The diagnosis's top-ranked
fix also truncates the action burst to one forward per iteration so the gate
re-decides every 500 cm. That would change the detector's sampling cadence, and
every number above is measured at the loop cadence — a rising-edge rule read 3×
denser raises edge #1 earlier and nothing in the archive can price it (only
perceive frames have an archived mask). It is now the single cheapest open lever
and it needs this iter's channel to exist first.
(d) FINISH stays at 0.00 and this iter does not pretend otherwise: the geometric
channel is measured useless there, `_FRONTAGE_SYS` is still 0/12, and F3 (ask it
from the faced pose) is the named route — left for the next iter so it is not
confounded with a ROAD-leg measurement.

**Anti-cheat: unchanged and unextended.** The new module reads
`obs['object_mask']` — a sanctioned observation this harness already ships to
the perceiver on every call and already reads in `_named_side_has_structure` —
and the static `task_spec` instruction text, and nothing else. No coordinate, no
`gt.json` field, no `agent_location` / `target_location`, no `ideal_path`, no
`current_instruction`, no analog `_agent_rotation`, no `subtask_max_steps`. Every
constant in `road_geometry.py` is a **fraction of the frame** or a count; there
is no centimetre and no map geometry anywhere in it. The oracle distances in §why
are proposer-side labels for an offline probe over the trace archive of a
finished run; they are not read at eval time and appear nowhere in the shipped
code. `read_audit` surface is unchanged from the parent.

**Verified** (`~/miniconda3/envs/simworld/bin/python`, loaded through the
evaluator's own `spec_from_file_location(f"{pkg}.harness", …)` path):

* `validate_harness.py`: **OK**, signature unchanged; the 3 WARNs are the
  pre-existing `max_tokens` literals (2048 / 8192 / 8192) — the new module adds
  **no** literal ≥ 2000.
* `cmp`-verified byte-identity of `AGENT.md`, `rules.md`, `__init__.py` and all 8
  untouched modules against `$PARENT_DIR`.
* **Unit**: the router accepts ROAD and rejects TURN / FINISH / ORIENT / `None`;
  the band fraction is exact on a synthetic mask and returns `None` on a missing
  / malformed one; the state machine raises an edge on a 0→1 transition, **not**
  on a plateau, re-arms after a dip, and refuses edges 3 and 4 (`edge_over_cap`);
  `place_verdict` is `False` under the translation floor, `None` on a missing
  mask, `None` on FINISH and on TURNING.
* **Gate**: mid-block → free veto with **zero** VLM calls; edge + identity 1 →
  commit at `forward_streak = 2` (floor and interval waived) and the junction VLM
  question is **never** called; edge + identity 0 → blocked; a FINISH leg still
  runs the VLM frontage question; the xcheck cap replays `False` four times for
  exactly **two** identity calls (`xcap_replay = 2`); with no verdict ever
  computed, FINISH blocks and ROAD defers.
* **End-to-end** `run_episode` dry runs against a fake env + fake client, 7 of
  them, all passing: ORIENT's cardinal burst still passes leg 0 with zero gate
  calls; a four-leg episode commits ROAD via `GEO_REACH` and carries TURNING; a
  ROAD leg with identity 0 never commits and spends **one** identity call; a leg
  whose mask never crosses the threshold spends **zero** arrival calls; a
  two-leg (ORIENT, FINISH) episode still reaches the frontage question; a
  rejected `-1` ends the episode cleanly with its banked legs intact; and the
  `GATE(...)` / `GEO(...)` ledgers appear in `notes` on every run.
* **Replay of the shipped module** over the parent's 1 326 archived mask frames,
  table above.

## files

- `modules/road_geometry.py` — **NEW**: `_ROADWAY_CLASS_RGB/_BGR`,
  `_MASK_CLASS_TOL`, `_BAND_ROW`, `_BAND_COL`, `ARRIVAL_FRAC`,
  `MAX_CLAIMABLE_CROSSINGS`, `GEO_MIN_TRANSLATIONS`, `_INTERSECTION_RE`,
  `is_intersection_stop_leg`, `roadway_band_fraction`, and `RoadGeometry`
  (`update`, `event_live`, `place_verdict`, `summary`)
- `modules/arrival_gate.py` — `geo` field + `_xcheck_memo`; `GEO_MIN_TRANSLATIONS`
  imported; `_place_verdict` geometry-first; `proactive_commit` free veto +
  event-keyed schedule; `allow_planner_commit` free veto + fail-closed cap
- `modules/__init__.py` — export `RoadGeometry`; docstring 9 → 10 nodes
- `agent.py` — import + instantiate `RoadGeometry` and hand it to the gate; the
  new sensor stage inside the perceive branch; the `GEO_REACH` note label;
  `road.summary()` into `notes`; module-list docstring 9 → 10
- `NOTES.md` — this file

*(unchanged, `cmp`-verified byte-identical to the parent: `AGENT.md`, `rules.md`,
`__init__.py`, and
`modules/{spatial_memory,prompt_assembler,perceiver,planner,affordance_estimator,active_perception,completion_checker,viewpoint_controller}.py`)*
