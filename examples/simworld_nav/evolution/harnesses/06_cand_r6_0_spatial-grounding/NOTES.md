# NOTES — `cand_r006_0_spatial-grounding` (iter 6, parent `anchor_nav`)

## axis

**Component graph: a new node that owns POSE.**

`modules/viewpoint_controller.py` — an 8th module, `ViewpointController`, inserted into
`run_episode`'s graph between `ActivePerception` (which owns bursts) and
`CompletionChecker` (which owns appearance). It makes no VLM call and holds no
opinion about what a frame looks like; it decides **where the body stands and
where it points** — before a turn fires, before an arrival look is taken, and in
what pose a `-1` is emitted.

This is a *wiring* change, not a recognizer change. The recognizer
(`_binary_reach_vlm_call`, its criterion, its evidence set, its schedule, its
confirmation counts) is **byte-for-byte the anchor's**. `completion_checker.py`
and `active_perception.py` are not edited at all; the new node re-routes their
existing capabilities.

## module map (★ = new / rewired this iter)

| module | role | this iter |
|---|---|---|
| `orchestrator.py` | entrypoint `run_episode(env, task_spec, vlm_clients, trace_dir=None)`; perceive→decide→act loop | ★ 4 wiring points (burst gate, FINISH commit gate, face-scan trigger + heading bookkeeping, terminal-pose composer) |
| `modules/viewpoint_controller.py` | **POSE arbitration** — leg taxonomy, approach-before-turn, face-before-look, face-before-commit | ★ **NEW MODULE** |
| `modules/active_perception.py` | cardinal burst, relative-turn burst, red idle, anti-spin, momentum | **untouched** (its turn burst is now *gated*, not rewritten) |
| `modules/completion_checker.py` | owns the appearance verdict behind every `-1` | **untouched** (`request_side_scan`'s dead depth trigger is now bypassed; `scan_reach` / `_binary_reach_vlm_call` still do the judging) |
| `modules/{spatial_memory,prompt_assembler,perceiver,planner,affordance_estimator}.py` | — | untouched |
| `AGENT.md`, `rules.md` | prose priors | untouched |

## evolvable levers

| lever | kind | where | status |
|---|---|---|---|
| **leg taxonomy** (ORIENT / ROAD / TURN / FINISH from instruction text) | code | `viewpoint_controller.leg_type` | **NEW** — verified 100% on all 199 `hard/` configs |
| **approach-before-turn** (floor 2 / cap 3 forwards, released on the seg cue) | code | `gate_burst` | **NEW** — constants are a 12-sample fit, see §caveat |
| **turn-side-open cue** (`TURN_SIDE_OPEN_MAX_FRAC = 0.10`) | code | `side_building_fraction` | **NEW** — segmentation, scale-free |
| **face-before-look trigger** (cadence, not depth) | code | `request_face_scan` | **NEW** — replaces the permanently-false `_side_band_near` gate |
| face-scan cadence (`K_FACE_STREAK/INTERVAL=4`, cap 4) | code | `viewpoint_controller` constants | tunable, deliberately tight (see §risk) |
| **FINISH forward-commit ban** | code | `allow_forward_commit` | **NEW** |
| **terminal-pose composer** (single call site) | code | `commit_pose_prefix` + the harness trim loop | **NEW** |
| arrival recognizer (criterion / evidence / schedule / confirmations) | code | `completion_checker` | **CLOSED this iter — untouched on purpose** (5 prior candidates, none beat the anchor) |
| ORIENTATION cardinal burst | code | `active_perception` (a) | **do not touch** — 50/50, and `gate_burst` is a verified pass-through for it |
| stuck detector scoped to the current burst | code | `orchestrator.py:~281` | **still open** — `forward_count` counts `chosen_actions`, not history, so `>3` is unreachable when 801/918 plans are 1 action long |
| overshoot / "I have passed it" state | code | `spatial_memory` | **still open** — nothing records that the window was entered and left |
| `completion_checker.request_side_scan` | code | `completion_checker:768` | **now dead** — the trigger moved to the controller; the function is left in place rather than deleted so a future iter can A/B the two triggers |

## change

The anchor has one arrival predicate — "does the forward RGB frame look like the
hint photo?" — and applies it to three legs the env grades on three different
quantities: ROAD on **L1 position only**, TURNING on **yaw and a junction box**,
FINISH on **position and yaw**. Every one of those is a statement about *where
the body is and where it points*, and there was no component whose job was to put
the body somewhere before asking. Five candidates have now re-worked the
recognizer (evidence set, look schedule, criterion wording, in-burst sampling,
confirmation counts) and none beat the anchor's 0.360. So this iter adds the
missing node instead and leaves the recognizer alone.

`ViewpointController` first types the leg from the **static instruction text**
(`task_spec['subtasks'][i]['instruction']`). Verified exhaustively against all
199 `hard/` `task_config.json` files: the 181 four-leg tasks classify exactly
`ORIENT / ROAD / TURN / FINISH` and the 18 two-leg tasks exactly `ORIENT /
FINISH`, 199/199, no misses. Order matters — ORIENT is tested first because
"Turn to face east" contains the word *turn*, and its closed-form cardinal burst
is the one path in this harness that passes 50/50. Then it owns three decisions:

**1. approach-before-turn** (`gate_burst`, wired at the AP call site). The
instruction is *"Turn right **at the intersection**"*; `active_perception`
fired `[rotate, -1]` on the **first frame of the leg with zero forward motion**,
landing 78–980 cm outside the position box in 7/7 traced failures — with **yaw
error 0 in every one**, i.e. the rotation was always right and only the position
was wrong. The controller now *captures* that burst rather than letting it
execute, emits single forwards along the pre-turn heading, and releases the held
burst when the lateral segmentation band on the **turn side** opens (the facade
line gives way to the cross street, `frac ≤ 0.10`) past a 2-forward floor, or
unconditionally at a 3-forward cap. Capture is necessary, not incidental: AP
latches the relative-turn burst to once per instruction, so once intercepted the
burst can only come back from the controller — `_turn_released` makes the
hand-back idempotent, and afterwards the leg behaves exactly as before.

**2. face-before-look** (`request_face_scan`, replacing the harness's iter_005
scan trigger). The measured cause of the 22-episode "reached the window, never
committed, walked 10–23 km past" bucket is not the question, it is the
**viewpoint**: every ROAD/FINISH instruction ends "…on your left/right", the
reference photo is a robot view captured *facing* that building, and at the
waypoint the building is ~90° off the travel axis and outside the FOV —
recall 21% (9/43 in-window fires) against a 0.4% out-of-window FPR, which is a
*framing* failure, not a threshold failure. The mechanism that fixes it already
existed (`request_side_scan` → yaw in place → `scan_reach` → restore on a miss;
in-place turns preserve position, so a position-graded leg is unharmed, and a
miss costs 2 steps and can never be a fatal commit) and it was **switched off**:
its trigger was a depth predicate, and `obs['depth']` is a per-frame min-max
normalised JET rendering carrying no absolute scale — the module's own comments
record `near_frac == 0.000` on 24/24 real frames. It fired **2 times in 50
episodes**. The controller retriggers it on the arrival-look **cadence** instead
(motion floor 4, spacing 4 — the rhythm the forward-facing check already runs on,
which puts one look every 2000 cm against a 4000-cm-wide window), with the
scale-free segmentation band as a **veto only** (`frac < 0.02` ⇒ nothing there to
look at, and the cap is not spent), capped at 4 looks per leg.

**3. face-before-commit** (`allow_forward_commit` + `commit_pose_prefix`).
FINISH is the only leg graded on yaw as well as position, and nothing ever set
the yaw: 6 of the 7 fatal finishes were **270° off** the required heading. Two
halves:
  * the forward-facing `proactive_commit_check` may **no longer inject a `-1` on
    a FINISH leg**. That injection is the mechanism behind all 7 fatal premature
    finishes (7.8k–58.8k cm out) — down a straight city street the destination
    facade *is* "clearly visible and prominent" from tens of km away, so the
    question has no stopping power there. The face-scan path becomes the FINISH
    commit route, and it commits **from the faced pose**, which is the yaw
    criterion.
  * a FINISH commit that still arrives from the travel heading (a planner `-1`
    that survived the cross-check) gets the quarter-turn toward the named side
    prepended. This is composed in the **one** place in `orchestrator.py` where a `-1`
    is materialised, and returns `[]` when the body is already faced, so the
    rotation can never be applied twice.

**Heading bookkeeping** is the other half of the rewire and closes the hazard the
gradient flagged: on a **ROAD** leg (position-graded) a scan match **restores the
travel heading before the `-1`**, so the next leg's turn burst cannot double-rotate
off a 90°-yawed body; on a **FINISH** leg it deliberately does not.

## why

The directive is a search-space steer toward the region exemplified by
`cand_r002_0_spatial-grounding` with **structure/wiring (the component graph)**
named as the productive aspect to vary. r002 moved token-level code *inside* a
fixed 7-node graph and scored 0.330. This iter varies the graph itself: a new
node, a new edge from it to each of the three commit paths, and the arrival-look
edge re-pointed from `CompletionChecker`'s dead depth trigger to the controller's
cadence trigger — while the levers r002 moved (criterion, evidence set,
arbitration, caps) are left exactly where the anchor had them, so the two
candidates are cleanly separable.

The gradient's ranked leverage lines up with this one node. It names three sites
as *verified untouched by all five prior candidates*: `active_perception.py:431-440`
(the zero-motion turn burst, §5A), `completion_checker.py:801` (the dead
`_side_band_near` gate, §5B), and the FINISH pose (§5C) — and its own analysis
says the remaining 21% recall "must be fixed by **changing the viewpoint**…
not by asking more often". All three are pose. It also prices them: the turn
approach takes TURNING legs 5/12 → 11/12 on the measured sweep (SubtaskSR ≈ +0.03,
and 6 more episodes get to attempt FINISH); a 3× recall recovery on the 43
in-window ROAD fires converts ~8–10 legs (≈ +0.04–0.06); removing the FINISH
injection takes 7 fatal commits to ~0, and since the harness uses a mean of 60
env steps against a 672–1328 budget, those episodes simply keep walking. FINISH
is 0/45 today and is arithmetically the only route to a non-zero SR.

**Risk, stated honestly.** (a) The face-scan's out-of-window false-positive rate
*from a faced viewpoint* is unmeasured — the 0.4% figure is for the forward view,
and a generic facade head-on could plausibly match more often. On a ROAD leg a
false `-1` is fatal, so the cap stays at the gradient's recommended 4 rather than
being widened for coverage, and the empty-side veto means looks are not spent
where there is no facade. (b) `APPROACH_FLOOR_FORWARDS = 2` / `CAP = 3` are fitted
on 12 samples; both ends sit on the measured plateau (N=2 and N=3 both give 11/12,
decaying only past N=5), so the adaptive cue chooses *where on* the plateau to
land rather than whether to overshoot — but the next iter should treat them as a
fit, not a law. (c) Blocking the FINISH forward injection removes a commit path;
if the face-scan turns out not to fire on FINISH legs, leg 3 stays 0 and the
diagnosis will say so cleanly, because nothing else changed in that channel.

**Anti-cheat: unchanged and unextended.** The new module reads only the static
`task_spec` instruction text, `obs['object_mask']` (the same sanctioned channel
the perceiver already receives), and this leg's own executed-action bookkeeping.
No coordinate, no `gt.json` field, no `agent_location`/`target_location`, no
analog `_agent_rotation` yaw, no oracle enters any decision path.
`validate_harness.py`: **OK** — `run_episode(env, task_spec, vlm_clients,
trace_dir=None) -> EpisodeResult` exported unchanged; the 3 WARNs are the
pre-existing `max_tokens` budgets (2048 / 8192 / 8192), and the new module
introduces no numeric literal ≥ 2000.

**Verified before shipping** (`~/miniconda3/envs/simworld/bin/python`, loaded
through the evaluator's own `spec_from_file_location(f"{pkg}.harness", …)` path,
plus fake-env `run_episode` dry runs):
- leg taxonomy 199/199 on the real corpus (`(4,0)=ORIENT ×181, (4,1)=ROAD ×181,
  (4,2)=TURN ×181, (4,3)=FINISH ×181, (2,0)=ORIENT ×18, (2,1)=FINISH ×18`);
- ORIENT: `gate_burst` is a pass-through, and a full episode still emits the
  cardinal burst's `-1` on step 0 with `subtask_success[0] = True`;
- TURN, side stays closed: `[0,0,0, Rotate_Right, -1]` — 3 forwards then the held
  burst, released at the cap, leg graded PASS;
- TURN, side opens at the floor: `[0,0, Rotate_Left,-1]` — 2 forwards;
- ROAD scan-commit: `[0,0,0,0, Rotate_Left, Rotate_Right, -1]` — look, **restore**,
  then commit; and the episode continues into the TURN leg whose burst fires as a
  *single* rotation (`[0,0,0, Rotate_Right, -1]`), i.e. no double-rotation;
- FINISH with the recognizer answering **1 on every call**: the forward-facing
  path never commits; the only commit is `[0,0,0,0, Rotate_Left, -1]` — from the
  faced pose, no restore;
- FINISH planner-driven commit: `[0, Rotate_Left, -1]` — **exactly one** rotation
  (the terminal pose is composed once, at the single `-1` call site);
- ROAD planner-driven commit: `[0, -1]` — no pose prefix on a position-graded leg;
- missing `object_mask` fails **open** (the scan is still allowed, the turn still
  releases at the cap), so an absent channel can never silently lock a pose
  decision closed — which is exactly how the lever this replaces died.

## files

- `modules/viewpoint_controller.py` — **NEW**: leg taxonomy (`leg_type`,
  `named_side`, `turn_direction`), the lateral-band primitive
  (`side_building_fraction`), and `ViewpointController` (`gate_burst`,
  `request_face_scan`, `face_code`/`restore_code`/`note_faced`/`note_restored`,
  `allow_forward_commit`, `commit_pose_prefix`)
- `modules/__init__.py` — export `ViewpointController`
- `orchestrator.py` — import + instantiate the node; `gate_burst` wrapped around
  `active.should_emit`; `allow_forward_commit` guard on
  `proactive_commit_check`; the side-scan block re-pointed from
  `completion.request_side_scan` to `viewpoint.request_face_scan` with
  ROAD-restore / FINISH-keep-pose heading bookkeeping; `commit_pose_prefix`
  in the `-1` trim loop; module-list docstring 7 → 8
- `NOTES.md` — this file

*(unchanged: `AGENT.md`, `rules.md`, `__init__.py`,
`modules/{spatial_memory,prompt_assembler,perceiver,planner,affordance_estimator,active_perception,completion_checker}.py`)*
