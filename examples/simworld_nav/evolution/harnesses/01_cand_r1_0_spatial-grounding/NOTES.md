# NOTES — `cand_r001_0_spatial-grounding` (iter 1, parent `anchor_nav`)

## axis

**Commit geometry — the `-1` decision, rewritten inside the existing components.**
Token-level only: the component graph is byte-for-byte the one the parent had
(`SpatialMemory → PromptAssembler → Perceiver → Planner → AffordanceEstimator /
ActivePerception → CompletionChecker`), the same seven modules, the same call sites, the
same wiring in `run_episode`. No file was added, no module was introduced, no stage was
added to or removed from the loop. What changed is the *code inside* four of those
components: the control flow of one ActivePerception rule, a new composer method and
rewritten gate arithmetic inside CompletionChecker, the wording of the one instruction
CompletionChecker sends to the model, and the four harness lines that name a commit.

The unifying idea: the parent treats `-1` as an **event** — a scalar "yes" that is
emitted from wherever the agent happens to be standing, facing wherever it happens to be
facing. This candidate treats `-1` as a **pose**: a commit is a position *and* a heading,
and every path that raises one now has to produce both. Three legs of the route are
graded on geometry the parent never represented, so three mechanisms were re-implemented
against it.

## module map (unchanged graph; ★ = internals rewritten this iter)

| module | role | this iter |
|---|---|---|
| `agent.py` | entrypoint `run_episode(env, task_spec, vlm_clients, trace_dir=None)`; perceive→decide→act loop | ★ four commit sites route through `CompletionChecker.commit_burst`; `forward_streak` threaded into the cross-check |
| `modules/spatial_memory.py` | scratchpad: per-step action + odometry ring | untouched |
| `modules/prompt_assembler.py` | composes the VLM prompt from the sanctioned surface | untouched |
| `modules/perceiver.py` | 1 VLM call: 4 images → free-text description | untouched |
| `modules/planner.py` | 1 text-only VLM call: description → `{Summary, Match, Actions}` | untouched |
| `modules/affordance_estimator.py` | legal-motion filter | untouched |
| `modules/active_perception.py` | deterministic short-circuit rules (cardinal burst, relative turn, red idle, anti-spin, momentum) | ★ relative-turn rule: zero-translation burst → staged approach latch |
| `modules/completion_checker.py` | owns every `-1`: `verify` / `cross_check_planner_commit` / `proactive_commit_check` / `scan_reach` | ★ new `commit_burst` pose composer; cap semantics + verdict memo; arrival prompt criterion |

## evolvable levers

| lever | kind | where | status |
|---|---|---|---|
| terminal-pose composer | code | `completion_checker.commit_burst` | **NEW this iter** — one place decides what a commit physically *is* |
| relative-turn approach depth | code/const | `active_perception._TURN_APPROACH_FORWARDS` (=2) | **NEW this iter**, stall-guarded off odometry |
| cross-check cap semantics | code | `cross_check_planner_commit` cap branch | **flipped this iter** (`True`→`False`) |
| cross-check verdict memo | code | `_xcheck_verdict_cache`, bucket=2 forwards | **NEW this iter** |
| arrival criterion wording | prose-in-code | `_binary_reach_vlm_call` sys_prompt | **replaced this iter** (visibility → position) |
| arrival evidence set (mask/depth/description into the REACH call) | code | `_binary_reach_vlm_call` user_content | untouched — next obvious lever |
| overshoot / "I have passed it" state | code | `spatial_memory` + `completion_checker` | untouched — largest open population |
| stuck detector wired to `forward_count_since_boundary` | code | `agent.py:~280` | untouched — still a no-op |
| planner `Match` in the Output Format | prose | `planner.reasoning_template` | untouched — 12% of replies drop it |
| ORIENTATION cardinal burst | code | `active_perception` (a) | **do not touch** — 50/50 |

## change

**1. The TURNING commit now walks to the intersection first** (`active_perception.py`,
rule (a″)). The parent returned `[rotate, -1]` unconditionally on the first decision of a
relative-turn leg — rotating and committing with *zero translation*, from the ROAD
waypoint it had just committed at. The TURNING leg is graded on a position box centred on
the intersection, which lies further along the street than that waypoint, so the commit
landed outside the box by a fraction of a forward step every time. The unconditional
return is replaced by a staged latch that spends `_TURN_APPROACH_FORWARDS = 2` decisions
returning `[Move_Forward]` before returning `[rotate, -1]`. The approach is priced off
odometry: if an approach forward moves the odometer less than `_TURN_APPROACH_MIN_DELTA`
the agent is against a facade, `_turn_approach_stalls` increments, and at
`_MAX_TURN_APPROACH_STALLS = 1` the latch short-circuits to the commit rather than mashing
a wall. Phase state (`_turn_approach_for/_done/_stalls`) is keyed on the instruction string
and reset both on instruction change and on the existing subtask-boundary check, so it
cannot leak across legs. The commit latch `_turn_burst_done_for` now closes at the
*commit*, not at first entry, so the approach forwards are never mistaken for a second
burst. Still zero VLM calls — the whole approach is short-circuit path.

**2. Every `-1` is composed, not literal** (`completion_checker.commit_burst` +
four call sites in `agent.py`). The FINISH leg is graded on facing as well as position:
its instruction always says "… on your left/right, stop and **face that building**", and a
bare `-1` stops the robot looking along the street. `commit_burst(instruction)` returns
`[Rotate_Left, -1]` when the instruction names the left, `[Rotate_Right, -1]` for the
right, and `[-1]` for every other leg. `_is_finish_instruction` is a three-conjunct
predicate (not an ORIENTATION instruction ∧ names a side ∧ contains a face-the-building
clause ∧ mentions no intersection) checked against all ten generator phrasings in
`task_route_generator.py`; ROAD, TURNING and ORIENTATION templates all fall through to the
bare `[-1]`, which matters because a spurious quarter turn before a ROAD commit would
corrupt the heading the following relative-turn burst rotates from. The rotations are
in-place, so the pose prefix cannot cost a position window already earned. The three
harness paths that used to write `[-1]` as a literal (proactive REACH commit, side-scan
commit, planner-driven commit spliced at the trim loop) all go through the composer; the
side-scan path passes `already_facing_side=True` because it has physically yawed toward
the side already and a second rotation would overshoot to 180°.

**3. An exhausted verification budget now means "don't commit"**
(`cross_check_planner_commit`). The cap branch returned `True` — "defer" — so the Nth
planner `-1` of a leg committed with no grounding at all, at whatever position the earlier
(correct) vetoes had walked the agent to. It now returns `False`: the channel exists, we
simply declined to spend it, so the honest answer is *not verified*. To stop the cap from
being spent on re-asks of the same claim, verdicts are memoized per
`(subtask, forward_streak // XCHECK_CACHE_STREAK_BUCKET)` — the same question from the same
stretch of road is answered for free — and the cap is raised 2 → 3, so it now bounds
*distinct* claims at roughly the old cost. Transient API failures still defer (`True`) and
are deliberately **not** memoized, so an error cannot harden into a standing veto.

**4. The arrival question asks about position, not visibility**
(`_binary_reach_vlm_call` system prompt). The parent's criterion — "the same building,
intersection, or scene is *clearly visible and prominent* in the current view" — is
satisfied by a landmark half a kilometre down a straight street, and was hedged with a
bare "be conservative" that pulled the opposite way at the waypoint itself: one prompt,
two opposing criteria. It is replaced by a single question cashed out in frame geometry
the model can read off the image — does the facade subtend a large part of the frame or
has it swung off-axis the way a building does when you draw level with it; is the
cross-street opening beside you or receding — with "still down the street, small, near the
vanishing point" named explicitly as the disqualifying case and recognition-confidence
explicitly ruled out as grounds for a yes. Appearance drift (lighting, colour, weather,
angle) is explicitly disallowed as grounds for a *no*; the `Reason` field is kept verbatim
because it is what makes this call auditable. This is criterion **replacement**, not the
exhausted illumination-invariance re-wording.

## why

The directive is an archive-coverage steer toward an empty behaviour cell with the
**token-level implementation** axis dominant (‖Δtoken‖ 1.913 vs ‖Δstructure‖ 1.405) and an
explicit instruction to keep the component graph fixed and move the code *inside*
components. Commit geometry is the one theme that can be pushed hard along that axis: it
touches control-flow complexity in two components, adds five new numeric constants and
changes the magnitude of a sixth, and rewrites the single model-facing instruction string
the harness owns — all without adding a node or an edge.

The gradient also makes it the right theme on the merits. The diagnosis measures the
walker as adequate (36/50 episodes physically enter the leg-1 pass window) and the stopper
as the binding constraint, and it localises three *deterministic* geometry failures rather
than model-ceiling ones: the turn burst missed the position box by 78–980 cm outward on
7/7 leg-2 failures; the FINISH facing is ±90° off on 50/50 tasks, making leg 3
arithmetically unreachable; and the cross-check's `return True` on cap exhaustion plus the
forward that follows a veto converted three in-window commits into out-of-window ones.
Those are arithmetic, not inference — each is fixed by changing what a commit *is*, inside
the component that already owns it.

The one inferential piece, (4), is included because it is the same axis stated in prose:
the false-positive class quotes the visibility clause back verbatim in its own `Reason`
field, so the criterion is not a hedge to tune but a clause to delete. Anti-cheat is
untouched throughout — the FINISH side and the turn direction are read from the static
`task_spec['subtasks']` instruction text, the approach stall test reads only
`info['moving_distance']`, and no coordinate, ground-truth orientation, or oracle field
enters the decision path.

**Verified before shipping** (`~/miniconda3/envs/simworld/bin/python`): `run_episode`'s
signature and `EpisodeResult`'s fields are unchanged and the module loads under the
evaluator's `spec_from_file_location` path; `commit_burst` returns the right burst for all
ten generator phrasings plus the `None`/empty/side-scan cases; the turn latch emits
`[0], [0], [rotate, -1]` when moving, short-circuits to `[rotate, -1]` when blocked, and
degrades gracefully with `state=None`; the cross-check never rubber-stamps (8 consecutive
claims → 8 refusals on 3 VLM calls), accepts and memoizes a yes, and does not memoize an
error; and a full fake-env dry run with a stub VLM produces
`[2,-1, 0,0,0,-1, 0,0,1,-1, 0,0,0,1,-1]` — the turn leg translating twice before its
rotate+commit, and the FINISH leg committing with a Rotate_Left prefix for a
building-on-the-left instruction.

## files

- `modules/active_perception.py` — approach constants + `_approach_delta` helper + three phase fields; relative-turn rule (a″) rewritten as a staged latch
- `modules/completion_checker.py` — `_is_finish_instruction` / `_FACE_BUILDING_RE` / `_INTERSECTION_RE` / action-code constants; `CompletionChecker.commit_burst`; `_xcheck_verdict_cache` field, `XCHECK_CACHE_STREAK_BUCKET`, `MAX_XCHECK_FIRES_PER_SUBTASK` 2→3; `cross_check_planner_commit` cap branch + memo (+ `forward_streak` kwarg, defaulted); `_binary_reach_vlm_call` system prompt + closing user turn
- `agent.py` — `facing_side_already` per-decision flag; `forward_streak` threaded into `cross_check_planner_commit`; REACH / side-scan / planner-trim commit sites route through `commit_burst`
- `NOTES.md` — this file

*(unchanged: `AGENT.md`, `rules.md`, `modules/{spatial_memory,prompt_assembler,perceiver,planner,affordance_estimator}.py`, `__init__.py`)*
