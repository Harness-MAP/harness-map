# NOTES — `cand_r5_0_spatial-grounding` (iter 5, parent `anchor_nav`)

## axis

**Component graph / wiring — a new module and a new edge into the actuation stage.**

`modules/burst_monitor.py` (`BurstMonitor`) is added as the **8th module**, and `harness.py`'s
burst executor is rewritten from a fixed `for` loop over `chosen_actions` into a **queue** that
consults the monitor after *every* translation it executes and can truncate its own remainder
into a `-1`. Until now the arrival channel was reachable from exactly one place in the graph —
the *decision* frame, after perceive+plan — and the actuation stage was a sink. It is now a
node with an edge back into the arrival channel.

This is a structure change, not a constant change: no threshold, cap, interval, criterion
string, prompt, evidence set or accept-validator of the existing path was re-tuned.
`K_REACH_INTERVAL` (5), `MAX_REACH_FIRES_PER_SUBTASK` (12), `K_REACH_FORWARD_STREAK` (4),
`MAX_XCHECK_FIRES_PER_SUBTASK` (2), `N_REACH_CONFIRMATIONS_REQUIRED` (1) are all **byte-identical
to the anchor** — §6.1/§6.7 of the gradient close those, and this iter does not reopen them.

## module map (★ = new, ☆ = wiring changed)

| module | role | this iter |
|---|---|---|
| `harness.py` | entrypoint `run_episode(env, task_spec, vlm_clients, trace_dir=None)` | ☆ burst executor is a truncatable queue; monitor consulted per translation; decision-level arrival check deduped against the monitor's frame |
| `modules/burst_monitor.py` | **in-burst arrival sampling policy** | ★ NEW — when to ask, per-leg budget, per-leg-**type** acceptance rule |
| `modules/completion_checker.py` | owns every `-1`; owns the arrival *question* | ☆ +`binary_reach()` public delegate for the monitor. Nothing else touched |
| `modules/active_perception.py` | cardinal burst, red idle, anti-spin, cap-2 momentum | untouched (§6.3 — ORIENTATION is 50/50) |
| `modules/{spatial_memory,prompt_assembler,perceiver,planner,affordance_estimator}.py` | — | untouched |
| `AGENT.md` / `rules.md` / `__init__.py` | contract + package pin | unchanged (`modules/__init__.py` gains the one export) |

## evolvable levers

| lever | kind | where | status |
|---|---|---|---|
| **arrival sampling density** (asks per pass-window) | code / wiring | `BurstMonitor.check` ← burst executor | **NEW this iter** — 1.10 → ~6.61 asks per ROAD window |
| **leg-type split of the acceptance rule** | code | `BurstMonitor.is_intersection_leg` + arm-then-commit | **NEW this iter** — text-only, 305/305 vs 0/43 separation |
| duplicate-frame suppression | code | `monitor.last_sampled_serial` vs `len(actions_log)` | **NEW this iter** — a repeat ask on an already-sampled frame is provably 0 bits |
| arrival **question** (criterion, evidence, decoding) | prose-in-code | `_binary_reach_vlm_call` | **CLOSED — do not touch** (7 refutations, §6.1) |
| vote / self-consistency on one frame | — | — | **CLOSED** (10/10 identical prompts → identical verdicts, §6.2) |
| xcheck cap-exhaustion semantics (`return True`) | code | `cross_check_planner_commit:748` | untouched — §5-D memo is the next separable candidate |
| TURNING approach latch (`[0]*2 + [rot,-1]`) | code | `active_perception:430` | untouched — §5-C, next separable candidate |
| stuck detector reading only the current burst | code | `harness.py:280` | untouched — §5-F, free but low ceiling |
| contradictory seg legend in the perceive prompt | prose-in-code | `prompt_assembler:64` | untouched — §5-E, bundle later |
| ORIENTATION cardinal burst | code | `active_perception` (a) | **do not touch** — 50/50 |

## change

**The harness asks the arrival question once per *decision*; the agent moves once per *env
step*; and those are not the same clock.** `ActivePerception`'s cap-2 momentum cadence means
one perceive+plan per **3.28 env steps**, and `harness.py`'s burst executor walked that whole
burst without ever re-consulting `CompletionChecker`. Measured on the parent's own 50-episode
run: a ROAD pass window is **6.61 forward frames deep** and receives **1.10 arrival calls** (6
episodes get **zero**); the in-window per-frame accept rate is **0.205**, so

```
P(commit | entered window) = 1 − (1 − 0.205)^1.10 = 0.22      observed ROAD 0.24
```

— the closed form reproduces the score. The recall term 0.205 is the frozen model's ceiling and
seven reformulations of the question have been refuted. **The exponent is the harness's**, and
it is set by the *shape of the component graph*, not by a constant: there is simply no node in
the graph that can ask a question about a frame produced mid-burst.

So the executor became a queue and a new module was hung off it. After every translation lands
and `obs` is refreshed, `BurstMonitor.check(...)` may spend one arrival sample on that fresh
frame; on an accept it replaces the remainder of the burst with `[-1]`, so the commit happens
**at the frame the recognizer fired on** rather than 500–1500 cm further down the street. The
monitor owns only *policy* — when to ask, under what per-leg budget, with which acceptance rule.
The *question* stays in `CompletionChecker`, reached through a new `binary_reach()` delegate
that changes no prompt, image, temperature or parse, so a single sample sits at exactly the
anchor's operating point and the extra frames are comparable draws.

Three properties make this the right shape rather than "more calls":

1. **Only distinct frames carry information.** 10 pairs of byte-identical arrival prompts in the
   parent run returned byte-identical verdicts (10/10), while nearby frames of the same scene
   disagree — the variance is positional, not sampling noise. A vote wrapper buys nothing; an OR
   over `n` *consecutive* frames is `1 − 0.795^n`. The monitor is the only place in the graph
   where those frames exist. Predicted `P(commit | window)` 0.22 → ~0.75.
   The same fact is used *defensively*: `last_sampled_serial` suppresses the decision-level
   `proactive_commit_check` when it would re-ask about the frame the monitor just sampled (the
   common case — a momentum burst ends on a forward), recovering part of the cost for free.
2. **The floor is preserved, so no new zero-motion failure mode.** The monitor refuses to sample
   below `K_REACH_FORWARD_STREAK` — the constant that exists because the VLM cheerfully declares
   Match=1 on the first frame of a new leg. Every inapplicable case (no hint, no client, no RGB,
   orientation leg, budget spent, API error, unparseable verdict) returns False: the monitor can
   only ever **add** a commit opportunity, never veto one the old path would have taken.
3. **ROAD and FINISH get two rules, because they are two classifiers.** Fire-level precision is
   **0.90 on ROAD** and **0.00 on FINISH** — all 7 FINISH accepts fired 7 778–58 789 cm out,
   because the destination tower is visible from across the map, i.e. on a FINISH leg "I can see
   it" is true long before "I am at it". Six-fold density under one policy would just
   false-commit earlier. The legs are separated by an exact, free, sanctioned text test —
   ROAD instructions contain *"at the intersection"* in **305/305** arrival calls, FINISH in
   **0/43** — and then: a validated ROAD accept commits immediately; a FINISH accept only
   **ARMS** the destination, and the commit needs a later accept after
   `FINISH_APPROACH_FORWARDS = 6` more translations ("I recognised it, then I walked toward
   it" — the closest honest approach signal available without reading a coordinate). This is
   aimed squarely at the two starved episodes that stood **16 cm** and **27 cm** from the
   destination and were asked **1** and **0** times there, on a leg that scores 0/45.

Cost, priced from the parent's measured cadence: arrival calls ≈ 348 → ~2 000, i.e. **+34 k
tok/ep (88 k → ~122 k, `over_budget` was `false` with headroom)** and **+~17 s on 800 s/ep**,
minus the deduped decision-level calls. Bounded by `MAX_INBURST_FIRES_PER_SUBTASK = 20` per leg
(median ROAD leg has ~15 translations, so it is a runaway backstop, not an operating point).
Known, priced risk: ROAD fire-level specificity is 261/262 = 0.9962, so ~6× density implies ~6
far-field false ROAD commits against ~19 expected gained passes. Shipped bare — the
consecutive-confirmation guard was deliberately **not** added, because it cannot be priced from
these traces (no two consecutive-frame arrival calls exist in them) and bundling it would make
the FP count unreadable.

## why

The directive is a search-space steer toward the region exemplified by
`prop_r002/cand_r002_0_spatial-grounding`, with **structure / wiring (the component graph)**
named as the productive aspect to vary. That candidate moved everything *inside* the existing
seven nodes and left the graph alone; this one varies the named aspect in the qualitative
direction the graph was never explored in — it **adds a node and an edge**, and specifically the
edge from the actuation stage back into the arrival channel, which is the one stage in the loop
that had no path to a decision.

The gradient points at the same place from the other side. `diagnosis.md` §4 classifies the
binding constraint as **HARNESS-BUG (sampling coverage)** and ranks §5-A first, with §5-A2 ("fire
inside the burst executor") called *"the only way to reach the 6.61 frames"*; §5-B (the leg-type
split) is second. Those two are the same object under this iter's framing — one is *where* the
question can be asked, the other is *which* question should be accepted once it is asked
densely — so they ship as one coherent module rather than two attributable halves of a policy
that is incoherent when split (6× density on FINISH without the arm rule is strictly worse than
today). Everything the gradient marks exhausted is untouched: no re-wording (§6.1), no votes
(§6.2), no ORIENTATION (§6.3), no seg-mask commit feature (§6.4), no "intersection" mention gate
(§6.5), no planner-as-primary-channel (§6.6), no re-tuned `K_REACH_*` / `MAX_REACH_*` (§6.7), no
`return False` xcheck (§6.8), no depth-polarity flip (§6.9). §5-C / §5-D / §5-E / §5-F are left
whole for separable candidates — §6.11's explicit instruction after r1 lost its attribution to a
3-edit bundle.

Anti-cheat: unextended. The monitor reads `obs['rgb']`, `obs['object_mask']`, the subtask's
`visual_hint_path` image, this leg's own executed-translation count, and the static
`task_spec['subtasks']` instruction text. No `agent_location`, `target_location`,
`correct_robot_location`, `gt.json`, `ideal_path`, `current_instruction`, analog `_agent_rotation`
or `subtask_max_steps`. Distances are expressed as forward-action counts, never centimetres; the
module's only new numeric constants are 1, 20 and 6. `validate_harness.py`: **OK** — the 3 WARNs
are the pre-existing `max_tokens` budgets, unchanged.

**Verified before shipping** (`~/miniconda3/envs/simworld/bin/python`, loaded through the
evaluator's own `spec_from_file_location(f"{pkg}.harness", …)` path):
`run_episode(env, task_spec, vlm_clients, trace_dir=None)` signature and `EpisodeResult`'s seven
fields unchanged. Unit: the leg discriminator separates both instruction templates; a sample
below the motion floor costs **0** VLM calls; a validated ROAD accept returns True on the first
sample at the floor; a FINISH leg arms at streak 4 and commits at streak 10, not before; the
per-leg budget stops at exactly 20 samples; missing hint / missing client / non-Mapping `obs`
each return False with **0** calls. End-to-end on a fake env + fake client, 3 legs
(ORIENTATION → ROAD → FINISH): with `Match=1` the episode closes in 17 steps with actions
`[-1, 0,0,0,0, -1, 0×10, -1]` — the ROAD leg commits **inside** the burst at the motion floor
(note `INBURST_REACH_COMMIT` ×2 in `notes`), and the FINISH leg walks the 6 arming translations
before committing; with `Match=0` the agent walks to `MAX_STEPS` and every one of the 23 arrival
calls lands on a **distinct** frame — the decision-level check never re-asks a frame the monitor
already sampled.

## files

- `modules/burst_monitor.py` — **NEW**: `BurstMonitor.check()`, `is_intersection_leg()`,
  `INBURST_SAMPLE_EVERY` / `MAX_INBURST_FIRES_PER_SUBTASK` / `FINISH_APPROACH_FORWARDS`
- `harness.py` — `BurstMonitor` import + per-episode instance; burst executor rewritten as a
  truncatable queue (`burst_queue` / `qi` / `inburst_commit`); monitor consulted after every
  translation; `INBURST_REACH_COMMIT` note; decision-level `proactive_commit_check` deduped
  against `monitor.last_sampled_serial`; module docstring 7 → 8 modules
- `modules/completion_checker.py` — `+ binary_reach()` public delegate (no behaviour change)
- `modules/__init__.py` — export `BurstMonitor`
- `NOTES.md` — this file

*(unchanged: `AGENT.md`, `rules.md`, `__init__.py`, `modules/{spatial_memory,prompt_assembler,perceiver,planner,affordance_estimator,active_perception}.py`)*
