# NOTES — `cand_r004_0_spatial-grounding` (iter 4, parent `cand_r1_0_spatial-grounding`)

## axis

**The motion channel: how the harness learns a step did not happen, and what it does about it.**

Token-level, in the sense the steer asks for. The component graph is byte-for-byte the parent's
— the same seven modules, the same call sites, the same wiring in `run_episode`. **No file added,
no module introduced, no stage added to or removed from the loop.** What changed is code *inside*
components: one memory primitive replaces a dead accessor, one `ActivePerception` rule is
re-grounded and turned into a ladder, one `orchestrator.py` predicate is rewired to the new primitive,
and two edits of the parent's four are reverted where the A/B condemned them.

The unifying idea: the parent's harness had **no working notion of "that forward didn't move me."**
Every mechanism that was supposed to notice was reading a channel that is both forbidden and never
written (`info["agent_collision"]`), so all of them were dead by construction. This candidate
re-grounds the whole blockage story on the one sanctioned progress channel the harness already
logs — `info['moving_distance']` — and makes the *response* a graded ladder that spends the
cheapest, least destructive action first.

This is deliberately a **different** token-level direction from the r002 exemplar. That candidate
moved the **arrival channel** (what the `-1` judge sees and who arbitrates its refusals). r003
moved the **look schedule** (when the judge is asked). This one moves the **motion channel** — a
lever neither of them touched, and the one the parent's own diagnosis scores at 281 wasted
forwards and `static_collision_mean` 1.78 → 5.12.

## module map (unchanged graph; ★ = internals changed this iter)

| module | role | this iter |
|---|---|---|
| `orchestrator.py` | entrypoint `run_episode(env, task_spec, vlm_clients, trace_dir=None)`; perceive→decide→act loop | ★ STUCK backstop rewired to `forward_stall_run`; dead `last_moving_distance` latch removed |
| `modules/spatial_memory.py` | odometry + step / boundary bookkeeping | ★ forbidden collision channel deleted; `step_progress` + `forward_stall_run` added |
| `modules/prompt_assembler.py` | composes the perceive prompt | untouched |
| `modules/perceiver.py` | 1 VLM call: frames → description | untouched |
| `modules/planner.py` | 1 VLM call: description → actions | untouched |
| `modules/affordance_estimator.py` | legal-motion filter | untouched |
| `modules/active_perception.py` | cardinal burst, turn-approach latch, red idle, anti-spin, momentum | ★ rule (b′) rewritten as the odometry escape ladder + `_freer_lateral_side`; dead momentum veto deleted. **Cardinal burst (a) and the turn-approach latch (a″) untouched.** |
| `modules/completion_checker.py` | owns every `-1` | ★ two **reverts** only: arrival criterion, xcheck cap semantics. `commit_burst` untouched. |

## evolvable levers

| lever | kind | where | status |
|---|---|---|---|
| blocked-forward **primitive** (`forward_stall_run`, `step_progress`) | code | `spatial_memory` | **NEW this iter** — the honest replacement for the forbidden collision counter |
| blocked-forward **escape ladder** (strafe → rotate → yield) | code | `active_perception` (b′) | **NEW this iter** |
| sidestep **direction chooser** from JET depth | code | `active_perception._freer_lateral_side` | **NEW this iter** |
| STUCK backstop predicate | code | `orchestrator.py` `_STUCK_BLOCKED_FORWARDS` | **NEW this iter** — was dead code in every prior iter |
| arrival-call **criterion wording** | prose-in-code | `_binary_reach_vlm_call` sys_prompt | **reverted to anchor, byte-identical — CLOSED** (seven reformulations priced, flat ROC) |
| xcheck **cap-exhaustion semantics** | code | `cross_check_planner_commit` | **reverted to `True`** — a budget is not evidence |
| xcheck verdict memo / cap = 3 | code | `_xcheck_verdict_cache`, `MAX_XCHECK_FIRES_PER_SUBTASK` | **kept from the parent** — free and correct |
| turn-approach latch (`_TURN_APPROACH_FORWARDS = 2`) | code | `active_perception` (a″) | **kept verbatim** — 7/7 anchor TURNING shortfalls closable, 1/1 live conversion; do not raise above 2 |
| terminal-pose composer (`commit_burst`) | code | `completion_checker` | **kept verbatim** — 181/181 template-verified, still unexercised (FINISH reached 0×) |
| blocked-commit `EMPTY_FALLBACK` walks the agent forward | code | `orchestrator.py` | untouched — r002's HOLD_PROBE region, left free |
| over-walk / "I have passed it" state | code | `spatial_memory` + `completion_checker` | untouched — the largest open population, gradient §5-F, deliberately deferred |
| REACH look schedule (`K_REACH_*`) | code/const | `completion_checker` | untouched — run to death in iters 006–010 |
| ORIENTATION cardinal burst | code | `active_perception` (a) | **do not touch** — 50/50 at zero VLM cost |

## change

### 1. A blocked forward becomes an observable, in the module that already owns odometry

`SpatialMemory` exposed `static_collision_delta(window)`, a cumulative building/object collision
count documented as coming from `info["agent_collision"]`. It never did: `log_step` set `coll = 0`
and passed the constant straight through, and the read audit confirms the field appears in **0 of
25 089** reads. Nor can it ever be wired — `agent_collision` is a forbidden field. So the accessor
was identically `0` and both of its consumers were unreachable by construction. It is **deleted**,
not commented out; leaving live-looking dead code is exactly what let this survive three iters.

In its place, two primitives derived only from `info['moving_distance']`:

- `step_progress(back)` — the odometry delta a single logged step produced (`None` for the first
  record, which has nothing to difference against).
- `forward_stall_run(threshold_cm)` — the **trailing run of consecutive `Move_Forward` steps inside
  the current leg that the sim refused** (delta below threshold). It walks back from the most recent
  record and stops at the first non-forward action, the first forward that actually moved, or the
  subtask boundary. `0` means "the agent is moving, or is not currently trying to."

A forward that produces no displacement *is* the observable event the forbidden collision counter
was standing in for. Unit-tested: clear run → 0; two refused forwards → 2; an intervening rotate
resets it; records before a `boundary()` are excluded; empty and single-record memories → 0.

### 2. The response is a ladder, and its first rung keeps the heading

Rule (b′) previously required the last forward to have stalled **AND** `static_collision_delta > 0`
— conjunct two being the dead one, which is why 50 episodes contain 281 blocked forwards (9 % of
all 3 118 forwards; anchor 4 %), 20 runs of length ≥ 2, nine of them 19 steps long, 259 substeps
burned, with no rule able to intervene. The conjunct is deleted and the trigger re-grounded on
`forward_stall_run`.

The first escape of a leg needs a run of **2** — on the measured distribution that catches 20 of
the 22 blocked runs and fires on neither singleton, so a pedestrian stepping in front is never
acted on. Once the stretch is known to be obstructed, later escapes need **1**: waiting again just
spends substeps. Then the ladder, cheapest and least destructive first:

- **escapes 1–2 → a sidestep** (`Move_Left` / `Move_Right`, legal codes 3/4). This is the real
  change of kind. The old rule's only response was `Rotate_Left`, which silently re-aims the agent
  90° off the road it was following and never restores it — on a TURNING leg the heading is graded
  directly, and on a ROAD leg it means the next forwards walk perpendicular to the route. A strafe
  **translates without changing yaw**: it steps around the obstacle and the cardinal heading the
  ORIENTATION burst established survives intact.
- **escapes 3–4 → the iter_010 recovery rotation, verbatim**, for a corner the strafes cannot clear
  (and used immediately when the depth frame is missing or the sides are too close to call).
- **past 4 → yield**, and the STUCK backstop owns it.

Which way to sidestep is read off `obs['depth']` by the new `_freer_lateral_side`. Per AGENT.md's
probe-verified legend the depth frame is a JET rendering — near = blue, far = red — so `B − R` is a
monotone nearness proxy that needs no metric interpretation and no absolute calibration, only a
left-vs-right comparison. It averages that proxy over the outer thirds of a band spanning the
lower-middle of the frame (the region a strafe actually has to fit through, skipping sky and
immediate foreground), expressed as **fractions of the frame, not pixel literals**, and steps toward
the open side. Below a minimum asymmetry the two sides are indistinguishable and it returns `None`
rather than strafing into a coin flip — the ladder then goes straight to the rotation. Defensive
throughout: a missing, malformed or non-JET frame degrades to exactly the rotation-only behaviour
the rule had before.

The unreachable iter_008 static-collision veto inside the momentum rule (d) is deleted in the same
pass; its removal is behaviour-preserving (the delta was always 0), and the wedged-but-sliding case
it was aimed at is now covered for real by the ladder.

### 3. The STUCK backstop gets a predicate that can fire — and one that cannot misfire

`orchestrator.py` computed `forward_count = sum(1 for a in chosen_actions if a == 0)` over the
**previous decision's burst**, not the leg's history. On the `EMPTY_FALLBACK` path that list is
`[0]`, so `forward_count > 3` was never true and the detector was dead in every iter it shipped in.
The obvious repair — swapping in a leg-wide forward count — would have made it **worse than dead**:
the second conjunct differences odometry across one *decision*, and a decision that only rotates
moves the odometer 0, so the episode would terminate on the first turn after the fourth forward.
Both halves are replaced by the single primitive that states the thing directly:
`memory.forward_stall_run() >= 8`. The ladder owns runs of 2–4; reaching 8 means every rung has
already been tried and failed and the agent is genuinely entombed, so the episode ends rather than
spending its remaining budget on a wall. On a healthy leg the run is 0 and it never fires. The
now-unused `last_moving_distance` cross-decision latch is removed with it.

### 4. Two reverts (`completion_checker.py`), both mandated by the A/B, both isolated

- **The arrival criterion is restored to the anchor's, verified byte-identical.** The parent replaced
  "the same building, intersection, or scene is clearly visible and prominent" with a frame-geometry
  "are you STANDING AT it" test. Over 50 episodes on the same tasks the in-window true-positive rate
  collapsed **9/43 → 1/58** while the far-field false-positive rate did not move (**1.6 % → 1.5 %**),
  commit precision fell 57 % → 14 %, and reason-string entropy collapsed to 33 distinct strings in
  420 calls with **81 % verbatim echoes of the prompt's own disqualifier**. One function feeds every
  commit path in the harness, so a single string moved the primary metric by −0.075. This is a
  revert, not an eighth wording: the frozen model has no metric-depth grounding, so any
  *distance* phrasing degenerates to a constant `0`; only *identity* phrasing carries signal.
  Treat that literal as frozen.
- **Cap exhaustion returns `True` again.** The parent made a spent budget a standing veto, which
  makes the gate strictly *tighter* the longer a leg runs — exactly inverted, since a long leg is
  one where the agent has walked past the waypoint. Measured: `XCHECK_BLOCK` 18 → 117, 4 in-window
  planner commits refused for free (one at **149 cm** from the waypoint), 35-deep `XCHECK_BLOCK;
  EMPTY_FALLBACK` lock-in loops, and **0 of 18** in-window planner commits allowed. Only the return
  value was wrong: the cap stays at 3 and the verdict memo stays, because the memo is genuinely
  free and correct.

The parent's two *good* edits — the turn-approach latch and `commit_burst` — are kept verbatim.
Per the gradient's own instruction not to bundle a revert-risk edit with an unproven one again,
the reverts restore a measured operating point and the motion channel is the single new lever on
top of it.

## why

The directive is a search-space steer toward the r002 region with **token-level implementation** as
the productive aspect: hold the component graph fixed, move the code inside components, and vary
the direction rather than copy it. The motion channel is the one theme that pushes hard along that
axis while being disjoint from both siblings — r002 rewrote what the arrival judge *sees* and who
arbitrates its refusals; r003 rewrote *when* it is asked; neither touched what the agent does when
the ground refuses to move under it.

On the merits the gradient is unambiguous. The binding constraint this iter is a harness bug
recoverable by reverting one string literal (§4), and without that revert nothing downstream can
score — the parent's approach latch was worth ~+0.035 SubtaskSR and earned **nothing**, because the
arrival judge killed the ROAD leg upstream of it. That fixes the floor. The motion channel is the
ceiling: 17/50 episodes ended in a facade mash of ≥5 consecutive blocked forwards (anchor 8/50),
`static_collision_mean` ran 5.12 against the anchor's 1.78, and 27 of 49 failed legs have empty
`notes` — they died silently on the env's per-leg substep deadline, at least 13 within ±1 substep
of `int(L1/100)`. Every substep spent mashing a wall is a substep not spent reaching the window,
and all three mechanisms that were supposed to stop it (rule (b′), the momentum veto, the STUCK
detector) were dead code. Reclaiming those 259 substeps is a direct transfer into the leg budget,
and doing it with a **strafe** rather than a yaw means the recovery no longer costs the cardinal
heading that ORIENTATION established and TURNING is graded on.

Anti-cheat holds: every new read is `info['moving_distance']` (via `SpatialMemory`) or
`obs['depth']`, both sanctioned in `rules.md`; the depth read is a relative left-vs-right hue
comparison, never a metric distance; no privileged field enters the decision path, and the deleted
collision accessor removes the harness's only reference to a forbidden channel. `commit_burst`'s
side and the latch's turn direction still come from the static `task_spec` instruction text. No
numeric literal ≥ 2000 was introduced; distances are expressed as counts of forward actions and
frame fractions.

## verified before shipping

- `run_episode`'s signature (`env, task_spec, vlm_clients, trace_dir=None`) and `EpisodeResult`'s
  seven fields unchanged, loading under the evaluator's `spec_from_file_location` import path;
  `py_compile` clean on `orchestrator.py` and all seven modules.
- The reverted arrival block diffed **byte-identical** to `anchor_nav_r000/source` from
  `sys_prompt = (` through the end of the user turn.
- `forward_stall_run` correct on: a clear run (0), two refused forwards (2), one refused (1), an
  intervening rotate (resets), a trailing rotate (0), across a `boundary()` (leg-isolated), and on
  empty / single-record memories (0).
- `_freer_lateral_side` correct on a left-blocked frame (→ `Move_Right`), a right-blocked frame
  (→ `Move_Left`), a symmetric frame (→ `None`), a missing depth key, the `depth_map` fallback, and
  garbage input (`None`, no exception).
- The ladder end-to-end on a synthetic `ActivePerception` + `SpatialMemory` pair: no fire at run 1,
  strafe at run 2, strafe at the next single stall, rotation on rungs 3–4, `None` once capped, and
  the counter reset on a new instruction.
- Three full fake-env `run_episode` dry runs. Clear world: ORIENTATION burst `[2, -1]` then a clean
  forward march, no spurious STUCK. Walled world with symmetric depth: the ladder falls back to
  four rotations and STUCK terminates at 19 steps instead of running the full 120-step budget into
  the wall. Walled world with asymmetric depth: `[2, -1, 0, 0, 4, 0, 4, 0, 1, 0, 1, 0, 0×8]` —
  two heading-preserving right-strafes away from the near left side, then the rotation rungs, then
  the backstop.

## files

- `modules/spatial_memory.py` — deleted the forbidden collision channel (`static_collisions_cum`,
  `static_collision_delta`, the `coll` write in `log_step`); added `step_progress` and
  `forward_stall_run`.
- `modules/active_perception.py` — rule (b′) rewritten as the odometry escape ladder; new
  `_freer_lateral_side` helper and its constants (`_STRAFE_LEFT/RIGHT`, `_ESCAPE_STALL_RUN_FIRST/
  NEXT`, `_ESCAPE_STRAFES`, `_MAX_ESCAPES_PER_SUBTASK`, `_ESCAPE_BAND_*`, `_ESCAPE_SIDE_MIN_MARGIN`);
  new `_escapes_this_subtask` field + boundary reset; deleted the unreachable momentum veto.
- `orchestrator.py` — STUCK backstop rewired to `forward_stall_run` with `_BLOCKED_FORWARD_MD_CM` /
  `_STUCK_BLOCKED_FORWARDS`; removed the dead `last_moving_distance` latch; corrected the
  `EMPTY_FALLBACK` comment's account of what bounds it.
- `modules/completion_checker.py` — reverted `_binary_reach_vlm_call`'s system prompt and user turn
  to the anchor's wording; reverted `cross_check_planner_commit`'s cap-exhaustion branch to `True`
  (cap 3 and the verdict memo kept).
