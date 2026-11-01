# NOTES — `cand_r002_0_spatial-grounding` (iter 2, parent `cand_r1_0_spatial-grounding`)

## axis

**The arrival channel: what it decides FROM, and who arbitrates when it says no.**

Token-level, same component graph. Seven modules, same wiring, no file added, no stage
added to or removed from `run_episode`'s loop — only the code *inside* `CompletionChecker`
and the four harness lines that consume it. Two levers move, both inside the one channel
that owns every `-1`:

1. **evidence** — `_binary_reach_vlm_call` stops deciding from two RGB frames. It now sees
   the segmentation mask, the JET depth rendering and the perceiver's description of the
   same frame, with the probe-verified sensor legend the perceiver already ships.
2. **arbitration** — a refusal from that call no longer holds an unbounded veto, a cost cap
   no longer decides anything at all, and a blocked commit no longer walks the agent away
   from the place it claimed.

The parent's own diff is *reverted* where the A/B condemned it (the criterion wording) and
*repaired* where it is right but mis-wired (the terminal pose). `active_perception.py` is
untouched: the turn-approach latch is the one edit of the parent's four with a positive
data point, and the gradient says keep it verbatim.

## module map (unchanged graph; ★ = internals changed this iter)

| module | role | this iter |
|---|---|---|
| `orchestrator.py` | entrypoint `run_episode(env, task_spec, vlm_clients, trace_dir=None)`; perceive→decide→act loop | ★ REACH path raises a bare `-1` (composer called once); `description` threaded into both arrival calls; a vetoed `-1` answered with an in-place HOLD instead of a forward |
| `modules/spatial_memory.py` | scratchpad: per-step action + odometry ring | untouched |
| `modules/prompt_assembler.py` | composes the perceive prompt (4 channels + legend) | untouched — but its legend is now the source the arrival call reuses |
| `modules/perceiver.py` | 1 VLM call: 4 images → free-text description | untouched |
| `modules/planner.py` | 1 text-only VLM call → `{Summary, Match, Actions}` | untouched |
| `modules/affordance_estimator.py` | legal-motion filter | untouched |
| `modules/active_perception.py` | cardinal burst, turn-approach latch, red idle, anti-spin, momentum | **deliberately untouched** (gradient §5-D) |
| `modules/completion_checker.py` | owns every `-1` | ★ arrival-call evidence set; criterion reverted; cross-check arbitration rewritten |

## evolvable levers

| lever | kind | where | status |
|---|---|---|---|
| arrival-call **evidence set** (mask + depth + description) | code | `_binary_reach_vlm_call` user_content + legend | **NEW this iter** |
| arrival-call **criterion wording** | prose-in-code | `_binary_reach_vlm_call` sys_prompt | **reverted to anchor — CLOSED, do not touch** (two operating points, flat ROC) |
| cross-check **refusal allowance** | code | `MAX_XCHECK_REFUSALS_PER_SUBTASK`, `_xcheck_refusals` | **NEW this iter** — bounds the veto by evidence events, not budget |
| cap-exhaustion semantics | code | `cross_check_planner_commit` | **neither `True` nor `False`** now — falls through to the motion floor |
| accept memo | code | `_xcheck_verdict_cache` | **narrowed** to `True`-only (a cached refusal is self-fulfilling) |
| blocked-commit response | code | `orchestrator.py` `HOLD_PROBE` / `MAX_HOLDS_PER_SUBTASK` | **NEW this iter** — net-zero in-place hold |
| terminal-pose composer | code | `completion_checker.commit_burst` | kept; **double-apply fixed** at the REACH call site |
| turn-approach latch | code | `active_perception` | kept verbatim |
| overshoot / "I have passed it" state | code | `spatial_memory` + `completion_checker` | untouched — largest open population (40 eps walked a median 18 357 cm past the window) |
| stuck detector wired to history | code | `orchestrator.py:~290` | untouched — still counts only the current burst |
| planner `Match` in the Output Format | prose | `planner.reasoning_template` | untouched — ~12 % of replies drop it, and the planner channel is now load-bearing |
| ORIENTATION cardinal burst | code | `active_perception` (a) | **do not touch** — 50/50 |

## change

**1. The arrival call decides from four channels, not two** (`_binary_reach_vlm_call`).
This one function is stages 5, 6 and 7 of the loop — every commit in the harness passes
through it — and until now it saw exactly two RGB frames: the current view and the
reference photo. No segmentation, no depth, no description. The perceive call one stage
earlier is handed all four channels, and on the same frames, the same model and the same
steps it is measurably the better arrival detector (first planner `-1` of a leg: in-window
7/12; this call: in-window accept rate 0.205 at best). The criterion it is asked is a
question about *geometry* — "the same place", "prominent" — and RGB alone does not carry
the geometry: a facade 400 cm away and the same facade 40 000 cm away differ mostly in
depth and in how the building class is laid out across the frame. So `obs['object_mask']`
and `obs['depth']` (with the `depth_map`/`depth` key fallback that `prompt_assembler`
already needs, because the env ships it under `depth`) are attached to the current view,
the perceiver's `Description` of the same frame is attached as text, and the system prompt
gains the *sensor legend* — the probe-verified JET polarity (blue = nearest, both extremes
dark, per-frame normalised, never metric), the real 4-class palette, and the explicit ban
on reading brightness as distance. Without that legend the model reads the depth frame
backwards; with it, "still down the street" and "level with it" are finally *distinguishable
in the evidence* rather than only in the wording. New helper `_aux_b64` fails soft, so a
frame missing a channel degrades to exactly the two-image call it was before.

**2. The criterion is reverted to the anchor's, byte-for-byte.** The parent replaced
"the same building, intersection, or scene is clearly visible and prominent" with a
frame-geometry "are you STANDING AT it" test. The A/B on the same 50 tasks settled it in
both directions at once: in-window accept rate collapsed 9/44 → 1/58 while the far-field
false-positive rate did not move by a single call (6/138 → 6/139), i.e. the likelihood
ratio inverted from 4.8 to 0.40 — the call became *more* likely to say "arrived" 15 000 cm
away than at the waypoint. And in both runs the model quoted whichever clause it landed on
back verbatim in its own `Reason` field (57/57 in-window refusals, 6/6 far-field accepts).
Two far-apart wordings, one flat ROC: the wording is not a lever. This is a **revert**, not
a third wording — verified equal to the anchor's string.

**3. A budget no longer decides a commit; refusals do, and they are bounded**
(`cross_check_planner_commit`). Both answers this branch has shipped for cap exhaustion are
wrong. `True` (anchor) let the Nth claim of a leg commit with no grounding at all, at
whatever position the earlier vetoes had walked the agent to. `False` (parent) turned an
exhausted budget into a free, unbounded veto: 117 blocks over 50 episodes, 62 of them with
no VLM call behind them, 16 in-window planner claims (58–1 796 cm) vetoed, **0** committed.
The new rule separates the three things that were conflated. *Inapplicable* (no hint, no
VLM, orientation leg, no RGB, transient error) still defers — an absent channel must not
tighten the gate. *Refusals* are the only thing that vetoes, and each leg grants
`MAX_XCHECK_REFUSALS_PER_SUBTASK = 1` of them: the point of a block is to buy one fresh
perceive+plan cycle from the same place, not to hold the leg hostage. *Cost-cap
exhaustion* is a statement about the budget, not about the world, so it neither vetoes nor
rubber-stamps — like a spent refusal allowance, it hands the claim to the **motion floor**
`forward_streak >= K_REACH_FORWARD_STREAK`, the constant that already guards the proactive
path and the one signal measured to separate the planner's in-window claims (all at streak
≥ 9) from its zero-motion ones (streak 0 and 3). The cap returns to the anchor's 2 — it is
a cost bound again — and the verdict memo is narrowed to cache **accepts only**, because a
memoised refusal at a position the agent has not left is self-fulfilling: the refusal is
exactly what stops it moving, so it can never expire.

**4. A vetoed `-1` holds its ground instead of walking away** (`orchestrator.py`,
`HOLD_PROBE` / `MAX_HOLDS_PER_SUBTASK`). When the cross-check revoked a commit the
planner's remaining actions were usually nothing at all, the action list came out empty,
and the `EMPTY_FALLBACK` branch injected a **`Move_Forward`** — so the veto physically
carried the agent out of the window it was standing in, 500 cm at a time (traced end to
end: 842 → 2 342 → 3 842 → … → 50 000 cm over 35 consecutive block/forward pairs). An
empty list after a *blocked commit* is now answered with `HOLD_PROBE = (Rotate_Left,
Rotate_Right)`: net zero in yaw, zero in position (rotations do not translate), and it does
not disturb `forward_streak`, so the next decision re-perceives the same place and the
claim can be re-raised from it — which, per change 3, is when it commits. Bounded at 2
holds per leg so an insistent planner cannot pirouette away the episode; past the allowance
the old forward fallback returns. `EMPTY_FALLBACK` for its original cause (a stray planner
`[-1]` at streak 0) is untouched.

**5. The terminal pose is composed exactly once** (`orchestrator.py`, REACH path). The parent's
`commit_burst` predicate is correct (199/199 FINISH legs, 0/561 others over every
`task_config.json` in `hard/`), but the REACH path pre-composed the burst *and* the trim
loop composed it again, so the surviving rotation plus the appended one made **180°** — the
intended quarter-turn toward the building became a turn directly away from it on 6/6 FINISH
commits. The REACH path now raises a bare `[-1]` and lets the single composer in the trim
loop do its job. One line; leg 3 is arithmetically pinned at 0.00 while it stands.

## why

The directive is a search-space steer toward the `simworld_config/harnesses/nav` region with
the **token-level implementation** aspect named as the productive difference: keep the
component graph, move the code inside it. Every edit above is inside an existing component,
and the largest of them literally moves this harness *back* into that region — the arrival
criterion is now the nav anchor's string, and the cap is the nav anchor's 2.

The gradient makes the direction unambiguous. The parent is a 0/50, SubtaskSR 0.285 against
the anchor's 0.36, with leg-1 passes 12/50 → 1/50 (p = 0.0018), and the diagnosis attributes
the loss to exactly two of the parent's edits and prices them: −8 leg-1 passes from the
criterion swap, −3 from the cap flip, with the +20 % token cost and the ×2.9 collision rate
falling out of the 118 fallback forwards the veto injected. The walker was never the
problem (41/50 episodes physically entered the leg-1 window); the harness lost the ability
to *stop*. Reverting 2 and rewiring 3/4 restores stopping; 5 unblocks leg 3.

Change 1 is the one genuinely new thing, and it is the only axis the measurements leave
open. The diagnosis closes wording (two operating points, invariant far-field FP rate,
clauses recited verbatim) and closes "facade frame share" as a feature (building-class
fraction is flat at ~0.28 everywhere — 0.276 in-window-refused vs 0.309 at the far-field
false positives). What it does *not* close is the evidence: it explicitly names attaching
mask + depth + description as the unshipped item, and the 4-image channel's 0.58 first-claim
precision on the same model is the existence proof that the information is in those
channels. Shipped on top of the reverts so attribution stays clean: if leg 1 recovers to the
anchor's ~0.24 the reverts did it, and anything above that is the evidence.

Anti-cheat: unchanged and unextended. The FINISH side comes from the static
`task_spec['subtasks']` instruction text, the motion floor reads only this leg's own
executed action history, the mask/depth/description are the same sanctioned observation
channels the perceiver already receives, and no coordinate, ground-truth orientation,
`gt.json` field or oracle state enters any decision path. `validate_harness.py`: **OK**, no
new numeric literal ≥ 2000 (the 3 WARNs are the pre-existing `max_tokens` budgets).

**Verified before shipping** (`~/miniconda3/envs/simworld/bin/python`, loaded through the
evaluator's own `spec_from_file_location(f"{pkg}.harness", …)` path):
`run_episode`'s signature and `EpisodeResult`'s fields unchanged; the arrival call emits 4
images + the legend + the description and degrades to the 2-image form when channels are
missing; the restored criterion string is **equal** to the anchor's; the arbitration —
one refusal then the walked planner is honoured on **one** VLM call, a zero-motion claim
stays blocked forever at a cost of one call, an accept is memoised, a transient error
defers without spending a refusal or a memo slot, and refusals do not leak across legs;
`commit_burst` still `[Rotate_Left,-1]` / `[Rotate_Right,-1]` / `[-1]`. Four fake-env dry
runs, **including the REACH path the parent's dry run missed**: a FINISH REACH commit is
`[0,0,0,0,1,-1]` — a *single* rotation — a ROAD REACH commit is `[0,0,0,0,-1]`, a
zero-motion claim cannot commit before the motion floor is met, and the headline case —
a claim raised at odometer 2 000, vetoed, held in place `[1,2]`, committed at odometer
**2 000**: the veto now costs 0 cm of displacement where the parent's cost 1 500 cm per
cycle.

## files

- `modules/completion_checker.py` — `_aux_b64` helper; `_binary_reach_vlm_call` (+`description` kwarg, criterion reverted, evidence legend, mask/depth/description attachment); `MAX_XCHECK_FIRES_PER_SUBTASK` 3→2; new `MAX_XCHECK_REFUSALS_PER_SUBTASK`; `_xcheck_refusals` field; `_xcheck_verdict_cache` narrowed to accepts; `cross_check_planner_commit` arbitration rewritten (+`description` kwarg); `proactive_commit_check` / `scan_reach` thread `description`
- `orchestrator.py` — `HOLD_PROBE` / `MAX_HOLDS_PER_SUBTASK` constants; `_hold_fallbacks` per-leg counter; `commit_blocked` per-decision flag; `description` threaded into the cross-check and proactive arrival calls; REACH path raises a bare `[-1]`; blocked-commit HOLD branch in the empty-action fallback
- `NOTES.md` — this file

*(unchanged: `AGENT.md`, `rules.md`, `__init__.py`, `modules/{spatial_memory,prompt_assembler,perceiver,planner,affordance_estimator,active_perception}.py`)*
