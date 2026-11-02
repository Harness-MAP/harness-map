# NOTES — `cand_r003_0_spatial-grounding` (iter 3, parent `anchor_nav`)

## axis

**The REACH look-schedule — rewritten inside `completion_checker.py`, nothing else touched.**

Token-level only, in the sense the steer asks for: the component graph is byte-for-byte the
parent's (`SpatialMemory → PromptAssembler → Perceiver → Planner → AffordanceEstimator /
ActivePerception → CompletionChecker`), the same seven modules, the same call sites, the same
wiring in `run_episode`. **No file was added, no module introduced, no stage added to or removed
from the loop, and `agent.py` was not edited at all.** What changed is the code *inside* one
component: two new methods on `CompletionChecker`, three new per-leg state fields, one extracted
probe function, and the gate block at the top of `proactive_commit_check`.

The unifying idea: the parent treats the arrival check as a **poll** — a fixed stride of forwards,
fired blind, identical at 30 km and at 1 m. This candidate treats it as an **event-driven window**:
the leg watches its own view for the physical cue the instruction names, and *where the looks go*
becomes a decision the harness makes rather than a constant. The recognizer question, its evidence
channels, and the commit criterion are **untouched** — only the sampling schedule moved.

This is deliberately a *different* token-level direction from the r001 exemplar. That candidate
rewrote **what a commit is** (commit-as-pose, tightened criteria); the diagnosis records it at
ROAD 12/50 → 1/50. This one rewrites **when the question is asked** and leaves the criterion alone.

## module map (unchanged graph; ★ = internals rewritten this iter)

| module | role | this iter |
|---|---|---|
| `agent.py` | entrypoint `run_episode(env, task_spec, vlm_clients, trace_dir=None)`; perceive→decide→act loop | untouched |
| `modules/spatial_memory.py` | odometry + step / boundary bookkeeping | untouched |
| `modules/prompt_assembler.py` | composes the live VLM prompt | untouched |
| `modules/perceiver.py` | 1 VLM call: frames → description | untouched |
| `modules/planner.py` | 1 VLM call: description → actions | untouched |
| `modules/affordance_estimator.py` | legal-motion filter | untouched |
| `modules/active_perception.py` | deterministic short-circuit rules | untouched |
| `modules/completion_checker.py` | owns every `-1` | ★ arrival-window cue + three-mode REACH schedule |

## evolvable levers

| lever | kind | where | status |
|---|---|---|---|
| named-side band **fraction** probe | code | `_named_side_building_frac` | **NEW this iter** — extracted from the boolean veto; two consumers, two questions |
| arrival-window cue (running-median drop) | code | `CompletionChecker._update_arrival_window` | **NEW this iter** — schedules looks, never commits |
| three-mode look schedule (near / far / sweep) | code | `CompletionChecker._reach_schedule` | **NEW this iter** |
| far-field budget reserve | code/const | `FAR_FIRE_RESERVE = 4` of cap 12 | **NEW this iter** |
| blocked-commit fallback walks the agent 1500 cm out of the zone | code | `agent.py:495-519` `EMPTY_FALLBACK` | untouched — **the next obvious lever** (diagnosis §5-A, "cannot cause a false commit") |
| xcheck cap-exhaustion returns `True` (the lottery) | code | `cross_check_planner_commit` | untouched — diagnosis §5-B, pairs with §5-A |
| relative-turn commit with zero translation | code | `active_perception` rule (a″) | untouched — diagnosis §5-C, worth +4–7 TURNING legs |
| RANGE-vs-IDENTITY split in the arrival prompt | prose-in-code | `_binary_reach_vlm_call` sys_prompt | untouched — deliberately: it is the lever r001 moved when ROAD collapsed to 1/50 |
| `_strongly_far`, `request_side_scan`/`scan_reach`, `N_REACH_CONFIRMATIONS_REQUIRED` | code | `completion_checker` | untouched dead code — 0 successful commits in 50 eps |
| ORIENTATION cardinal burst | code | `active_perception` (a) | **do not touch** — 50/50 at zero VLM cost |

## change

**The fixed-stride REACH poll becomes an event-driven, three-mode schedule.** One coherent
mechanism, entirely inside `modules/completion_checker.py`.

**1. A fraction probe where there was a boolean** (`_named_side_building_frac`). The parent read the
named-side lateral band of `obs['object_mask']` and immediately collapsed it to one bit ("is this
side empty?"). The band computation is extracted intact and now returns the raw fraction, or
`None` when it cannot be evaluated at all (the instruction names no side, or the mask is
missing/odd) — the old boolean conflated "nothing there" with "cannot say" into a fail-open `True`.
`_named_side_has_structure` is rewritten as a two-line wrapper over it, so the accept-validator's
behaviour is **bit-for-bit what it was** (verified: `None → True`, `frac > 0.02`).

**2. An arrival event, read off the agent's own recent view** (`_update_arrival_window`). The ROAD
instruction always names the cue: *"stop at the intersection with \<building\> **on your left**."*
Walking a block, that side is a solid wall of building-class pixels; at the intersection the cross
street opens and the fraction drops. The test is a drop against the **leg's own running median** —
`frac < GAP_DROP_RATIO(0.6) × median(samples before this one)`, after `GAP_WARMUP_SAMPLES(3)` of
history — so it carries no absolute threshold, no map-tuned constant and no coordinate; the
normalizer is the agent's last few frames. Firing latches a dense window spanning
`GAP_WINDOW_FORWARDS(12)` forwards, and a later drop **re-arms** it rather than being swallowed by
the first. The cue is sampled on *every* decision that reaches the checker, including the ones that
will not fire a look, so the median is warm by the time it matters. Returns `None` — distinct from
`False` — when the cue is unavailable for the leg.

**3. Three modes, and the far field no longer eats the budget** (`_reach_schedule` + the gate block
of `proactive_commit_check`). The single `K_REACH_INTERVAL = 5` stride is replaced by a
`(streak_floor, stride)` pair chosen per decision:
*near* (window open) → stride **1**, i.e. a look on every decision (~1500 cm apart under the
momentum burst), floor dropped to 2 so a short leg is covered too; *far* (cue live, no event yet) →
the **wider** stride 8, and only `FAR_FIRE_RESERVE = 4` of the 12-call cap may be spent here, so the
far field cannot consume a budget the window will need; *sweep* (cue unavailable, **or** the far
reserve spent with no event ever firing) → the **parent schedule verbatim**. That third mode is the
no-regression floor: on a leg the cue cannot see, or a leg whose gap never opens, coverage is what
it was, only shifted slightly later. The total cost cap (`MAX_REACH_FIRES_PER_SUBTASK = 12`) is
unchanged — this reallocates the budget, it does not enlarge it.

**Not changed, on purpose:** the recognizer prompt, its evidence channels, the commit criterion, the
accept-validator, the xcheck, the planner gate, and `agent.py`. The cue **never commits** — it only
decides where the looks go, and the recognizer keeps the last word.

## why

The directive is a search-space steer toward the r001 region with **token-level implementation** as
the productive aspect: keep the component graph fixed, move the code inside components, vary the
direction rather than copy it. The look-schedule is the one theme that pushes hard along that axis
while being disjoint from what r001 did — it rewrites control flow in one component, adds two
methods, three state fields and seven constants, and touches neither the graph nor any model-facing
string.

The gradient makes it the right theme on the merits, and the arithmetic is the diagnosis's own.
Over 305 ROAD-leg REACH calls, `P(Match=1)` is **0.21 inside the 2000 cm pass window and 0.005
outside it**, yet **86 % of the calls are spent outside** — the 2500 cm stride across a 4000 cm
window buys a median of **one** in-window look, and `0.21 × ~1 shot` reproduces the observed
ROAD 12/45 exactly (§2.2). The budget is not what binds: 7 of 12 fires are typically unspent at leg
end. Placement is. Concentrating ~3–4 looks in the window instead of 1 takes detection to
`1 − 0.79³′⁵ ≈ 0.55`; the diagnosis predicts **ROAD 12/45 → ~20–25/45** (§5-D1). The risk of the
extra looks is measured and small: ROAD specificity is **99.6 %** (1 false accept in 263 out-of-window
calls). And the *reserve* half should cut the other way on the 7 fatal FINISH commits (§2.6), which
were all far-field accepts ("visible ahead", 15–58 km) — far looks drop from up to 12 to 4.

The cue itself is the diagnosis's §5-D2, taken **only** as a look-scheduler, which is exactly the
role it was measured to be fit for: it fires in 28/45 episodes, median first fire at **1 880 cm**
(the zone edge), and enriches in-window call density 14 % → 25 %, but at 32 in-window vs 95
out-of-window fires it is **far too noisy to commit on**. It is never asked to.

Three things the diagnosis marks exhausted are avoided by construction: no extra evidence channels
into the recognizer (cand_r2: ROAD 12 → 6), no tightening of the ROAD commit criterion (cand_r1:
ROAD 12 → 1), and no depth predicate (the env normalizes depth per frame; there is no scale). The
ORIENTATION burst is untouched. Anti-cheat is untouched: the cue reads only `obs['object_mask']`
plus the static instruction text and the harness's own `forward_streak`, and it self-normalizes
against the leg's own samples — no coordinate, no ground-truth field, no route read enters the path.

**Verified before shipping** (`~/miniconda3/envs/simworld/bin/python`, evaluator-style
`spec_from_file_location` load): `run_episode`'s signature and `EpisodeResult`'s seven fields are
unchanged and the module loads clean; the fraction probe round-trips a known mask and
`_named_side_has_structure` reproduces the parent on all four branches (structure / empty / no
mask / no named side); the window detector does **not** fire on a flat block walk (10 frames, zero
fires — warm-up respected, no self-comparison), fires on the gap, stays open across the span,
closes after it and re-arms on a second drop, and returns `None` (never `False`) when the cue is
unavailable; the three schedule modes return the right `(floor, stride)` including the
reserve-exhausted fallback; and end-to-end against a stub VLM a simulated block-walk-with-gap fires
at forward-streaks `[6, 15, 24, 27, 30, 33, 36, 39]` — three sparse far looks, then **four
consecutive looks through the gap** — while a leg with no named side reproduces the parent's
`[6, 12, 18, 24, 30, 36]` exactly, and a `Match=1` still commits.

## files

- `modules/completion_checker.py` — `_named_side_building_frac` extracted (+ `_named_side_has_structure`
  rewritten as a wrapper); `K_REACH_INTERVAL_FAR` / `K_REACH_INTERVAL_NEAR` / `K_REACH_STREAK_FLOOR_NEAR` /
  `FAR_FIRE_RESERVE` / `GAP_WARMUP_SAMPLES` / `GAP_DROP_RATIO` / `GAP_WINDOW_FORWARDS` / `GAP_HISTORY_MAX`;
  `_side_frac_hist` / `_window_opened_at` / `_far_fires` fields; `CompletionChecker._update_arrival_window`;
  `CompletionChecker._reach_schedule`; the gate block of `proactive_commit_check` rewritten; `Tuple` import
- `NOTES.md` — this file

*(unchanged: `agent.py`, `AGENT.md`, `rules.md`, `__init__.py`, and
`modules/{spatial_memory,prompt_assembler,perceiver,planner,affordance_estimator,active_perception}.py`)*
