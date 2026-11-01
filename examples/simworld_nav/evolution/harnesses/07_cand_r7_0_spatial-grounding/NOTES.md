# NOTES — `cand_r007_0_spatial-grounding` (iter 7, parent `cand_r2_0_spatial-grounding`)

## axis

**The arrival question is re-asked on DISPLACEMENT, and nowhere else.**

Token-level, same component graph. Seven modules, same wiring, no file added, no module
introduced, no stage added to or removed from `run_episode`'s loop. Every behavioural edit
is inside `modules/completion_checker.py`; `orchestrator.py` changes only comments that the edit
made false. What moves is the one thing the gradient leaves open — not *what* the arrival
call is asked or *what it looks at* (both axes are now measured closed), but **when it is
asked, how often, where, and when its answer expires.**

Four mechanisms in this file used to answer that question separately, each in its own
currency: a fixed stride (`K_REACH_INTERVAL`), a fire quota (`MAX_REACH_FIRES_PER_SUBTASK`),
a refusal allowance (`MAX_XCHECK_REFUSALS_PER_SUBTASK`) plus a cost cap
(`MAX_XCHECK_FIRES_PER_SUBTASK`), and a bucketed accept memo (`XCHECK_CACHE_STREAK_BUCKET`).
All five constants are **deleted**. Two mechanisms remain, and they are the same idea applied
to the proactive path and the planner path: a **refilling look-budget** clocked by the leg's
own executed translations, and a **standing answer** that expires after the agent has walked
`XCHECK_REARM_FORWARDS` translations. The evidence upgrade that paid for none of this is
reverted, and the leg where the call has never once been right no longer spends any of it.

## module map (unchanged graph; ★ = internals changed this iter)

| module | role | this iter |
|---|---|---|
| `orchestrator.py` | entrypoint `run_episode(env, task_spec, vlm_clients, trace_dir=None)`; perceive→decide→act loop | comments only — four blocks that described the deleted caps/evidence now describe the rate |
| `modules/spatial_memory.py` | scratchpad: per-step action + odometry ring | untouched |
| `modules/prompt_assembler.py` | composes the perceive prompt (4 channels + legend) | untouched |
| `modules/perceiver.py` | 1 VLM call: 4 images → free-text description | untouched |
| `modules/planner.py` | 1 text-only VLM call → `{Summary, Match, Actions}` | untouched |
| `modules/affordance_estimator.py` | legal-motion filter | untouched |
| `modules/active_perception.py` | cardinal burst, turn-approach latch, red idle, anti-spin, momentum | **deliberately untouched** — the latch is 5/6 in-box vs the anchor's 5/12 and is the best thing in this lineage |
| `modules/completion_checker.py` | owns every `-1` | ★ allocator, FINISH gate, 2-image revert, arbitration |

## evolvable levers

| lever | kind | where | status |
|---|---|---|---|
| arrival-look **allocator** (token bucket over translations) | code | `_spend_reach_look` + `proactive_commit_check` | **NEW this iter** — replaces stride + quota |
| **FINISH-leg gate** on the forward look | code | `proactive_commit_check` (`_is_finish_instruction`) | **NEW this iter** — 0-for-91, all 5 accepts fatal |
| arrival-call **evidence set** (mask + depth + description) | code | `_binary_reach_vlm_call` | **REVERTED — CLOSED, measured negative** (0.143 vs 0.209 at +126 % tok) |
| arrival-call **criterion wording** | prose-in-code | `_binary_reach_vlm_call` sys_prompt | **CLOSED, do not touch** (two operating points, flat ROC) |
| cross-check **standing answer** (expires on displacement) | code | `_xcheck_standing`, `XCHECK_REARM_FORWARDS` | **NEW this iter** — replaces cap + refusal allowance + accept memo |
| blocked-commit HOLD response | code | `orchestrator.py` `HOLD_PROBE` / `MAX_HOLDS_PER_SUBTASK` | kept verbatim — it does what it says (0 cm vs 1 500 cm); it simply may no longer buy a commit |
| terminal-pose composer / bare REACH `-1` | code | `commit_burst`, `orchestrator.py` REACH path | kept verbatim — correct, and now that FINISH forward looks are gone it is exercised only on ROAD |
| turn-approach latch | code | `active_perception` | kept verbatim |
| **FINISH pose** (`request_side_scan` → yaw → `scan_reach`) | code | `completion_checker` + `orchestrator.py:~505` | **the open axis, deliberately NOT taken this iter** — see §why |
| overshoot / "I have passed it" state | code | `spatial_memory` + `completion_checker` | untouched — would confound this measurement |
| stall recovery / strafe ladder | code | `orchestrator.py` STUCK + `affordance_estimator` | untouched — 24 % of episodes end wedged, next-largest open population |
| `_named_side_has_structure` as a free veto | code | `completion_checker:~1005` | untouched — inert (0/16), but it is the only uncounted veto left in the file |

## change

**1. The look is a RATE, not a quota** (`_spend_reach_look`, new; `proactive_commit_check`).
The two knobs it replaces were both uniform over a leg whose length this harness cannot know,
and they failed at the two opposite ends of that unknown length. `K_REACH_INTERVAL = 5` asked
the arrival question **0.97 times per pass through the ~8-translation-wide pass window** — 35
in-window fires across 266 in-window steps in the 36 episodes that physically entered the leg-1
window — so the leg converted at recall-per-*question* (0.167 observed) rather than
recall-per-*window*. At the other end `MAX_REACH_FIRES_PER_SUBTASK = 12` ran long legs **dry**:
12 fires × 5 translations is ~36 000 cm of uniform coverage, and 4 of 5 long legs spent the
entire allowance 15–35 km short of the destination — three episodes then stood 211 / 719 /
1 646 cm from it with *zero looks left to spend*. No single quota can be right at both ends.
A bucket is: stock starts at `REACH_BUCKET_CAPACITY = 4` so a short leg is sampled densely from
its first eligible frame; one look is earned back every `REACH_REFILL_FORWARDS = 4` translations
so a leg of *any* length still has a look in hand at its far end; `REACH_MIN_SPACING = 2` stops
the opening stock being dumped into one stretch of road. The only input is `forward_streak`, the
harness's own count of translations executed since this leg began — monotone, self-derived, no
coordinate, no distance, no privileged field. Measured on the real object: a 200-translation leg
now fires 53 looks, the last at translation 200 (the old schedule's last was at 60 and it was
mute for the remaining 140); the steady-state gap is exactly 4; a leg that does not move never
fires twice; buckets do not leak across legs; and an idle stretch cannot bank a burst.

**2. The FINISH leg stops buying looks with this call** (`proactive_commit_check`). Over every
FINISH-leg firing of the previous run: **0 accepts in 15 fires below 5 000 cm of the destination,
5 accepts in 73 fires above 15 000 cm.** It has never once been right on a FINISH leg, and those
5 accepts are 5 of the 6 fatal commits (15 396 – 44 711 cm). The cause is structural, not a
threshold: the FINISH instruction puts the target *"on your left/right"* and the reference photo
was captured **facing** it, so the forward camera at the destination does not contain the
reference view — traced at 719 cm from the target, the recogniser and the perceiver
independently and *correctly* report a non-match. A call whose every accept is a false positive
is not a detector, it is a cost. The predicate is the existing `_is_finish_instruction`, verified
199/199 against the static `task_config.json` instruction text in `hard/`, and it gates only this
channel's *unprompted injection*: a FINISH commit remains reachable through the planner's own
`-1` + cross-check path. Under change 1 this is also what keeps the budget honest — the longest
legs in the task are FINISH legs, and they were the ones the new rate would have spent most on.

**3. The question is asked over two RGB frames again** (`_binary_reach_vlm_call`). This is a
revert, not a third evidence set. The parent attached the segmentation mask, the JET depth
rendering and the perceiver's description of the same frame, with the probe-verified sensor
legend. A/B'd against the anchor at the *identical* criterion, the same model and the same 50
tasks, it is measured **negative**: ROAD in-window recall **5/35 = 0.143 vs 9/43 = 0.209**,
near-miss (2 000–5 000 cm) false positives **4/61 vs 0/52**, at **2 371 tok/call against ~1 050**
— +126 % per call and +11.6 points of the whole episode budget. Not one of 388 `Reason` strings
cited depth hue or the segmentation layout; every one was a pure RGB identity statement. That
budget is what pays for change 1: at ~1 050 tok/call the same money buys ≈875 firings instead of
388. `description` is still *accepted* by the three call sites and deliberately ignored, so
`orchestrator.py` needs no signature edit; the now-dead `_aux_b64` helper is **deleted**, not left
inert, because an unread helper is how three iters of dead schedule logic survived in this file.

**4. The cross-check's answer stands until the agent moves** (`cross_check_planner_commit`).
Every previous answer to *"what happens when the allowance runs out"* is now refuted by
measurement: `True` (anchor) rubber-stamped the Nth claim with no grounding; `False` (iter 1)
made an exhausted budget a free unbounded veto (117 blocks / 50 episodes, 62 with no VLM call
behind them); and `return walked` (the parent's motion floor) is refuted here — the floor was
**already satisfied at the moment of the refusal in 5/5 traced instances**, so a refusal merely
postponed the commit by one decision and then waved it through: **5 fires → 1 pass, 4 fatal**,
each at the exact position the recogniser had refused one decision earlier. It could not have
gone otherwise: a block is answered with `HOLD_PROBE` (net-zero yaw, zero translation), so the
next decision perceives the identical frame, and perceive and plan both run at temperature 0.2.
The same claim, from the same pose, on the same evidence, now simply gets **the same answer**,
for free, with no VLM call spent — and once the agent has walked `XCHECK_REARM_FORWARDS = 4`
translations the evidence genuinely is new, the standing answer expires, and the question is
asked again. One rule replaces three mechanisms and two constants; the cross-check is now
rate-limited by displacement at the same cadence the proactive look draws at, so **no budget
decides a commit anywhere in this file any more**. A transient API error still defers (`True`)
and deliberately does *not* become a standing answer.

## why

The directive is a search-space steer toward the `prop_r004` region with **token-level
implementation** named as the productive difference: keep the component graph, move the code
inside it. Every edit above is inside one existing component — one new private method, four
state fields replaced by three, five constants deleted and four added, and not one line of
control flow added to or removed from `run_episode`.

The diagnosis classifies the binding constraint as **HARNESS-BUG: a firing *schedule* mismatched
to the geometry it is sampling**, and prices it precisely. The recogniser is a *usable* detector
on ROAD legs again after the parent's revert — in-window 0.143 against far-field 0.010, a
likelihood ratio of ~14 — and the arithmetic reproduces the observed score exactly:
36 entering episodes × (1 − 0.857^0.97) ≈ 5.0 expected passes against 5 in-window REACH accepts
observed. When the model of the failure reproduces the score to two decimals, **the lever is the
exponent, not the base**: fires inside the window, not another wording. Change 1 moves the
exponent (≈0.97 → ≈2.5 in-window questions per entering episode ⇒ per-window recall
1 − 0.857^2.5 ≈ 0.32), change 3 pays for it, and change 2 stops the longest legs in the task from
eating the increase in a place where the call is 0-for-91.

Changes 2 and 4 are also the *hedge* on change 1. Denser sampling multiplies false positives as
well as true ones, and the diagnosis prices that risk at ≈2 extra fatal ROAD commits; changes 2
and 4 remove 5 + 4 = **9 measured fatal commits** between them, so the iter can afford the
coverage it buys. All four are one axis in the sense that matters for attribution — they are the
same edit to the same channel's *timing* — and the three axes the previous run closed (criterion
wording, evidence set, the planner channel as an arrival detector) are all left alone.

**What is deliberately NOT taken.** The FINISH *pose* — retriggering `request_side_scan` on the
look cadence instead of the unusable `_side_band_near` depth band, so the agent yaws to the named
side and asks from the pose the reference photo was taken from — is the diagnosis's only
realistic path to SR > 0, and it is left for the next iter on purpose. It is a second axis (where
the body points, not when the question is asked), it needs `orchestrator.py`'s scan branch as well as
this file, and shipping it alongside change 1 would make the coverage measurement
uninterpretable: leg-3 movement and leg-1 movement would be confounded in a single number.
Change 2 is the half of it that is measurement-free (removing a call that has never been right
cannot cost recall) and it leaves the mechanism in place for that iter to re-trigger.

Anti-cheat: unchanged and unextended. Both new rules are clocked by `forward_streak` — the
harness's own count of translations it executed since the current leg began. The FINISH gate
reads the static `task_spec['subtasks']` instruction text through the existing predicate. No
coordinate, no `agent_location` / `target_location` / `destination` / `ideal_path`, no `gt.json`
field, no `env.subtask_max_steps`, no `info['agent_collision']`, and no oracle state enters any
decision path. No new numeric literal ≥ 2000 — every constant here is a count of *actions*.
`validate_harness.py`: **OK**, `run_episode(env, task_spec, vlm_clients, trace_dir=None) ->
EpisodeResult` exported; the 3 WARNs are the pre-existing `max_tokens` budgets.

**Verified before shipping** (`~/miniconda3/envs/simworld/bin/python`, loaded through the
evaluator's own `spec_from_file_location(f"{pkg}.harness", …)` path, so `from .modules import …`
resolves exactly as it will under eval):
*allocator* — a 200-translation leg fires 53 looks with a steady-state gap of exactly
`REACH_REFILL_FORWARDS`, first at translation 4 and last at 200 (never starves); a stationary leg
cannot fire twice; `REACH_MIN_SPACING` holds; a fresh leg starts full and no state leaks across
`cur_idx`; an idle stretch cannot bank more than capacity. *FINISH gate* — a FINISH instruction
spends **zero** VLM calls while the same checker fires normally on the ROAD instruction, and
ORIENT / below-floor still short-circuit as before. *Revert* — the arrival call emits exactly
**2** images, and neither the JET legend, the segmentation legend nor the description text
appears anywhere in the request. *Arbitration* — refuse → re-claim from the same pose → re-claim
3 translations later all return `False` on **one** VLM call total; the 4th translation re-arms it
and an accept commits on the 2nd call; an accept likewise stands for free; a transient error
defers without becoming a standing answer; legs do not leak. *End-to-end* — two fake-env episodes
driven through `run_episode`: with the recogniser refusing everything the ORIENT cardinal burst
still commits on action 0 and leg 1 draws 17 looks over its walk; with it accepting, the episode
commits `-1` on ORIENT, a **single bare** `-1` on the ROAD REACH path (no double-composed pose),
passes the TURNING leg through the untouched latch, and walks the FINISH leg to `MAX_STEPS` with
**zero** forward REACH calls — exactly the designed behaviour.

## files

- `modules/completion_checker.py` — module docstring constants list; `K_REACH_INTERVAL` /
  `MAX_REACH_FIRES_PER_SUBTASK` deleted → `REACH_BUCKET_CAPACITY` / `REACH_REFILL_FORWARDS` /
  `REACH_MIN_SPACING`; `MAX_XCHECK_FIRES_PER_SUBTASK` / `XCHECK_CACHE_STREAK_BUCKET` /
  `MAX_XCHECK_REFUSALS_PER_SUBTASK` deleted → `XCHECK_REARM_FORWARDS`; `_aux_b64` deleted;
  dataclass fields `_per_subtask_reach_fires` / `_per_subtask_xcheck_fires` /
  `_xcheck_verdict_cache` / `_xcheck_refusals` → `_reach_tokens` / `_reach_refill_mark` /
  `_xcheck_standing` (`_last_reach_streak` re-purposed to spacing only); new `_spend_reach_look`;
  `_binary_reach_vlm_call` evidence block reverted to the anchor's 2-image call;
  `proactive_commit_check` gate rewritten (bucket + FINISH suppression);
  `cross_check_planner_commit` arbitration rewritten (standing answer)
- `orchestrator.py` — **comments only** (four blocks describing the deleted caps / the evidence
  upgrade); no behavioural change, signature and control flow untouched
- `NOTES.md` — this file

*(unchanged: `AGENT.md`, `rules.md`, `__init__.py`, `modules/{spatial_memory,prompt_assembler,perceiver,planner,affordance_estimator,active_perception}.py`)*
