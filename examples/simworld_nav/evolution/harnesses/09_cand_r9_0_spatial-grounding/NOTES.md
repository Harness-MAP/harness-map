# NOTES — `cand_r009_0_spatial-grounding` (iter 9, parent `cand_r6_0_spatial-grounding`)

## axis

**Component graph: the `-1` LICENSE is taken away from the single recognizer and
becomes the CONJUNCTION of two independent single-purpose perception nodes —
and the parent's worst edge is deleted.**

Three graph edits, one idea:

* **+ node** `modules/arrival_gate.py` (`ArrivalGate`, the 9th module): owns the
  license for every non-orient `-1`. It asks **two** questions instead of one —
  `identity(current, hint)` with the leg text **removed**, AND `place(current)`
  with no hint and no text — and commits only on their conjunction.
* **− edge** the iter-6 **face-before-look** path (`request_face_scan` → yaw →
  `scan_reach` → ±commit) is cut out of `run_episode` entirely.
* **↻ edges** both surviving commit sites (`proactive_commit_check`,
  `cross_check_planner_commit`) re-point from `CompletionChecker` to the gate;
  `CompletionChecker`'s recognizer stays in the graph as the gate's
  **degradation path**, and `ViewpointController` keeps only its TURN approach
  (this chain's one genuine gain) and its FINISH commit pose.

No constant in the chain was retuned. The gate imports
`K_REACH_FORWARD_STREAK` / `K_REACH_INTERVAL` / `MAX_REACH_FIRES_PER_SUBTASK` /
`MAX_XCHECK_FIRES_PER_SUBTASK` from `completion_checker` rather than
re-declaring them, so the **schedule is the parent's, byte-for-byte, and the
only thing that moved is the shape of the decision**.

## module map (★ = new / rewired this iter)

| module | role | this iter |
|---|---|---|
| `harness.py` | entrypoint `run_episode(env, task_spec, vlm_clients, trace_dir=None)` | ★ 4 sites: gate instantiated; xcheck re-pointed; proactive re-pointed + FINISH un-blocked; the 71-line face-scan block deleted; gate ledger appended to `notes` |
| `modules/arrival_gate.py` | **the `-1` license** — text-free identity ∧ hint-free place | ★ **NEW MODULE** |
| `modules/completion_checker.py` | `verify()` (planner-`-1` rule) + `_binary_reach_vlm_call` as the gate's fallback | **untouched** (its two decision methods are now unreferenced, left in place for A/B) |
| `modules/viewpoint_controller.py` | TURN approach + FINISH commit pose | **untouched** (`request_face_scan` / `face_code` / `restore_code` / `allow_forward_commit` now unreferenced) |
| `modules/active_perception.py` | cardinal burst, relative-turn burst, red idle, anti-spin | **untouched** |
| `modules/{spatial_memory,prompt_assembler,perceiver,planner,affordance_estimator}.py` | — | untouched |
| `AGENT.md`, `rules.md` | prose priors | **deliberately byte-identical** — the agent's prompt must not move while the decision graph does, or the two are not separable |

## evolvable levers

| lever | kind | where | status |
|---|---|---|---|
| **text-free identity call** (the images-only reach question) | code | `arrival_gate._identity_call` | **NEW** — deletes the `Subtask instruction:` element the diagnosis measured at 14 % vs 100 % in-window recall |
| **hint-free place call** (an independent second conjunct) | code | `arrival_gate._place_call` | **NEW** — junction wording on ROAD/TURN, frontage wording on FINISH |
| **conjunction as the license** (`identity ∧ place`) | code | `arrival_gate.{proactive_commit,allow_planner_commit}` | **NEW** — the diagnosis measured a single call asserting both conjuncts at 0/14; it has to be two calls |
| **legacy degradation edge** (`checker._binary_reach_vlm_call` on a `None`) | code | `arrival_gate._legacy_verdict` | **NEW** — a dead second channel degrades to the parent, never to silence |
| junction wording operating point | prose-in-code | `_JUNCTION_SYS` | **measured this iter**: 5/14 NEAR vs 1/14 FAR (A/B of 3 wordings, §why) |
| frontage wording operating point | prose-in-code | `_FRONTAGE_SYS` | **measured 0/7 NEAR, 0/12 FAR** — an honest weak spot, see §risk |
| face-before-look path | code | `harness.py` + `viewpoint_controller` | **DELETED — CLOSED, measured 0/128 on ROAD and 9/9 fatal on FINISH** |
| `allow_forward_commit` FINISH block | code | `viewpoint_controller` | **unwired** — it was a workaround for a criterion with no stopping power; the conjunction is the replacement |
| TURN approach (`APPROACH_CAP_FORWARDS = 3`) | code | `viewpoint_controller.gate_burst` | **keep verbatim** — TURNING 5/45 → 9/45, the chain's one genuine gain |
| ORIENTATION cardinal burst | code | `active_perception` (a) | **do not touch** — 50/50 |
| arrival-look *schedule* (streak / interval / caps) | code/const | `completion_checker` constants, imported | **frozen this iter on purpose** — iters 003/007 ran this axis to death |
| `side_building_fraction` as a junction cue | code | `viewpoint_controller` | **CLOSED** — 11.9 % in-window vs 16.1 % far, released the burst early 0/10 times |
| **k = 2 confirmations on a `-1`** | code | `completion_checker` + `harness` | **still open** — gradient §5-(4); deliberately deferred so the conjunction is measured alone |
| **"window entered and left" latch** | code | `spatial_memory` | **still open** — gradient §5-(5); the only named route to a non-zero FINISH |
| stuck detector scoped to history, not the burst | code | `harness.py:~293` | **still open** — gradient §5-(7), `forward_count` is unreachable at `>3` with 1-action plans |
| TURN approach conditional on ROAD arrival evidence | code | `viewpoint_controller` | **still open** — gradient §5-(6); needs this iter's evidence channel to exist first |

## change

The gradient's headline is that the binding constraint is **the arrival
question itself**, not the trigger and not the pose: *"exactly one of the three
pose levers paid… the other two re-routed the SAME visibility-criterion
recognizer through a new pose and so inherited its behaviour unchanged."* Five
candidates re-scheduled that recognizer, one re-posed it, and the diagnosis
then went and probed the recognizer directly on this run's own frames.

**1. The identity call loses the instruction text.** `_binary_reach_vlm_call`
prefixes `"Subtask instruction: <the 40-word leg text>"` to a pair of images
whose job is image identity. Probed at temperature 0 on 14 in-window / 14
far frames: **with** the text 2/14; **without** it 14/14, and the rejection
reasons flip from *"different building colors and lighting"* to actual image
comparison. The model was adjudicating prose against a photograph — a contest
it loses — instead of comparing two photographs. `ArrivalGate._identity_call`
is that call with the text element deleted and the criterion restated as image
identity ("the same structure is visible in both, allowing for a different
viewing angle or lighting"), which is what the two-image comparison actually
supports.

**2. A second, independent conjunct, because text-free identity alone is
lethal.** The same probe puts text-free identity at 6/14 on FAR frames. In a
harness where a rejected `-1` **terminates the episode** (17 of the parent's 50
deaths), recall without precision converts "reach but never commit" straight
into "commit at 20 000 cm and die". The diagnosis also measured that asking one
call to assert identity *and* arrival together scores **0/14** — the model drops
a conjunct. So the second question is a **separate call with disjoint inputs**:
one image, no hint, no instruction text, asking only *where the body is
standing*. It cannot inherit the identity call's error because it never sees
the identity call's evidence.

That question is asked per leg type (`viewpoint_controller.leg_type`, read off
the static instruction text, 199/199 verified in iter 6):

* **ROAD / TURN — "are you at the intersection?"** The ROAD leg's grading point
  literally *is* the intersection ("stop at the intersection with X on your
  left") and this harness has asked about the building 395 times and about the
  junction zero times.
* **FINISH — "are you at the frontage?"** The nine fatal FINISH commits all
  asserted *visibility* (*"is visible in both views"*, at 8 666–58 789 cm
  against a 2 000 cm threshold). Visibility is not arrival; this conjunct is
  the missing distance predicate.

**3. The face-before-look edge is deleted.** It is the single worst edge in the
parent: on ROAD **128 faced reach calls, 0 matches** (0/20 in-window against the
forward view's 8/34) for 256 substeps of pure cost — 10 % of all ROAD substeps,
12–18 % of the budget on tight legs, with 5 legs dying at `used == budget`
*exactly*; and on FINISH **9 matches in 35 calls, 9/9 fatal**, five of them
throwing away an episode that had banked 3 legs after spending 8–18 of a
173–333 substep budget. Its premise is refuted by its own numbers. Those
substeps go back to walking.

**4. FINISH is un-blocked at the proactive site.** The iter-6
`allow_forward_commit` block was a workaround for a criterion with no stopping
power, and blocking the path without fixing the question is exactly what the
gradient closes as a repeated mistake — it took forward FINISH commits 7 → 0 and
the face-scan immediately produced 9 with the same question. The question now
carries a standing-position conjunct, and `commit_pose_prefix` still supplies
the yaw the leg is graded on, so this becomes the only FINISH commit route and
it is the one with a distance predicate in it. The method stays in
`ViewpointController`, unreferenced, so a future iter can A/B it.

**Degradation, deliberately.** Every unavailability — no client, no RGB, an
unparseable answer, the runaway cap — routes to `checker._binary_reach_vlm_call`
(the parent's exact gate) rather than to a veto. A dead second channel makes
this harness behave like its parent; it can never make it behave like a harness
that cannot commit. That is precisely how the iter-5 depth gate died (a
permanently-false predicate that fired 2 times in 50 episodes), and the
`GATE(...)` ledger written into `notes` reports every branch count so the next
diagnosis can see it immediately instead of inferring it.

## why

The directive is a search-space steer toward the region exemplified by
`cand_r004_0` and `cand_r007_0`, with **structure/wiring (the component graph)**
named as the aspect to vary. Both exemplars are explicitly *"unchanged graph,
token-level internals"* candidates — r004 moved the motion channel inside a
fixed 7-node graph, r007 moved the look *rate* inside `completion_checker`.
Varying the graph qualitatively away from the parent (which **added** a node and
hung three new edges off it) therefore means changing the graph's *shape*, not
its size: this iter **removes** the parent's worst edge, **collapses** three
commit routes to one, and **splits** the single overloaded recognizer into two
narrow nodes whose conjunction is the license. Net node count +1, net edge count
down, and the one node that was doing three jobs now does one.

The gradient's own ranking is (1) drop `instr_text`, (2) add an independent
junction call and require the conjunction — *"ship (1) only together with (2)"* —
and (3) delete the face-scan path. This candidate is exactly (1)+(2)+(3) as a
single coherent package, because they are not separable: (1) alone converts
reach-but-no-commit into fatal-commit, and (3) is what pays for the extra calls
in substeps. The priced prediction is ROAD 10/45 → ~19/45 at a lower false-fire
rate, ~9 more episodes reaching TURNING where conversion is 90 %, and the 17
fatal commits falling toward the 3 that came through the forward path.

**Measured before shipping** (live `Qwen3-VL-30B-A3B-Instruct` on `:8000`,
temperature 0, frames from the parent's own traces; oracle-side distances used
only to *label* frames, exactly as the diagnostician's §4 probe does — nothing
here is read by the harness):

* junction wording, 14 NEAR (< 2 000 cm of the ROAD waypoint) / 14 FAR
  (> 10 000 cm): a hard three-cue conjunction **3/14 NEAR, 0/14 FAR**; the
  diagnosis's own row-C phrasing **2/14, 0/14**; the shipped **any-cue**
  phrasing **5/14 NEAR, 1/14 FAR**, reproducing the 6/14 : 1/14 the diagnosis
  reported. Shipped the best of the three rather than assuming the first.
* frontage wording, 7 NEAR (< 2 500 cm of the destination) / 12 FAR
  (> 15 000 cm): shipped wording **0/7, 0/12**; an any-cue frontage variant
  **0/7, 4/12**; the junction question reused on FINISH **0/7, 3/12**. The two
  looser variants are strictly worse, so the strict one ships.
* both new prompts parse on the served model on 6/6 real frames (no `None`, no
  API error), and the full `proactive_commit` path runs end-to-end on real
  frames with a real client and a real `CompletionChecker` attached.

**Verified** (`~/miniconda3/envs/simworld/bin/python`, loaded through the
evaluator's own `spec_from_file_location(f"{pkg}.harness", …)` path):

* `validate_harness.py`: **OK**, signature unchanged; the 3 WARNs are the
  pre-existing `max_tokens` literals (2048 / 8192 / 8192) — the new module adds
  **no** literal ≥ 2000;
* 12 fake-env `run_episode` dry runs, all passing: ROAD ident=1 ∧ place=1 →
  `REACH_COMMIT`; ident=1 ∧ place=0 → blocked, the agent walks on; ident=0 → the
  place call is **never spent**; place unparseable → legacy fallback decides
  (commits on legacy=1, does not on legacy=0); FINISH ident=1 ∧ place=1 →
  commit **with the quarter-turn pose prefix immediately before the `-1`**;
  FINISH place=0 → blocked; **no `SIDE_SCAN` note on any leg**; ORIENT's cardinal
  burst still passes leg 0 with zero gate calls; a planner `-1` with place=0 →
  `XCHECK_BLOCK`, with both conjuncts → allowed; and with no hint image the gate
  is fully inert and the episode still runs.

**Risk, stated honestly.** (a) The FINISH conjunct measures 0/7 NEAR, so on the
evidence FINISH will commit rarely or never and `per_position_SR[3]` may well
stay 0 — the same as the parent, but now for a reason the ledger will name. The
NEAR sample is 7 frames from a run that rarely entered the destination window,
so "0/7" is "low", not "impossible"; the real fix the gradient names for FINISH
is the window-entered-and-left latch, deliberately deferred so as not to
confound this measurement. (b) Text-free identity is the high-recall /
low-precision half of the gate; if the junction conjunct under-fires on ROAD,
the harness reverts toward reach-but-no-commit rather than toward fatal commits
— the safer of the two failure directions, and the ledger distinguishes them
(`blocked_by_place` vs `ident_0`). (c) Two extra VLM calls per arrival check
against a measured 46.3k tok/ep with `over_budget: false`, and they cost **zero
env substeps**, unlike the 256 the deleted face-scan spent. (d) `k = 2`
confirmations (gradient §5-(4)) would have blocked all 9 fatal commits and is
*not* in this candidate — one axis per iter; if the conjunction still passes a
fatal commit, that is the next lever and the traces will show it.

**Anti-cheat: unchanged and unextended.** The new module reads `obs['rgb']`, the
per-subtask hint image, `obs['object_mask']` (via the parent's unchanged
`_named_side_has_structure` veto), the static `task_spec` instruction text, and
its own call bookkeeping. No coordinate, no `gt.json` field, no
`agent_location`/`target_location`, no analog `_agent_rotation`, and no
`subtask_max_steps` enters any decision path. The oracle-side distances in §why
are proposer-side frame *labels* for an offline probe, in the trace archive of a
finished run — they are not read at eval time and appear nowhere in the shipped
code. `read_audit` surface is unchanged from the parent.

## files

- `modules/arrival_gate.py` — **NEW**: `_IDENTITY_SYS` / `_JUNCTION_SYS` /
  `_FRONTAGE_SYS`, `_extract_here`, and `ArrivalGate`
  (`_identity_call`, `_place_call`, `_place_verdict`, `_legacy_verdict`,
  `proactive_commit`, `allow_planner_commit`, `summary`)
- `modules/__init__.py` — export `ArrivalGate`; docstring 8 → 9 nodes
- `harness.py` — import + instantiate the gate (`checker=completion`); the
  xcheck site re-pointed to `allow_planner_commit`; the proactive site
  re-pointed to `proactive_commit` and the `allow_forward_commit` FINISH block
  removed; the 71-line iter-5/6 face-scan block deleted; `arrival.summary()`
  appended to `notes`; module-list docstring 8 → 9
- `NOTES.md` — this file

*(unchanged: `AGENT.md`, `rules.md`, `__init__.py`,
`modules/{spatial_memory,prompt_assembler,perceiver,planner,affordance_estimator,active_perception,completion_checker,viewpoint_controller}.py`)*
