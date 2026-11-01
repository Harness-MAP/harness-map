# SimWorld-MMNav (hard): the evolution behind the paper's best harness

This directory holds every harness the paper's best navigation run evaluated, from the anchor to
the best harness, in evaluation order, and the recipe for rendering an episode of any of them.

**The run.** `mapelites_adaptive` with k0 = 4, C = 1 and P = 2 on the hard setting (50 search
tasks, maps 20_0 to 20_24; `SubtaskSR` as the primary metric), with the Claude Code CLI as
meta-agent and `Qwen3-VL-30B-A3B-Instruct` as the policy model. The run evaluated 16 harnesses
(the anchor, one coverage-initialization founder, and 14 bred candidates); the best,
`cand_r15_0_spatial-grounding`, was the 16th. Its ancestry is the anchor (0.360) →
`cand_r6_0` (0.370) → `cand_r9_0` (0.400) → `cand_r13_0` (0.550) → `cand_r15_0` (0.555). On the
ten held-out tasks (maps 20_25 to 20_29) three replicate evaluations of the best harness scored
0.600, 0.575 and 0.525 (mean 0.567), against 0.430 ± 0.033 for five replicates of the anchor.

**Full-task successes.** Hard tasks are rarely completed end to end: across the campaign's 117
fifty-episode evaluations of all methods, only three evaluations contain a full success, two of
them in this run. `cand_r15_0` completed map 20_10, task `dist_21_0_1` (SR 0.02 over the 50
search tasks), and its sibling `cand_r14_0` completed map 20_2 and map 20_18, task `dist_11_0_1`
(SR 0.04); the third is a Meta-Harness candidate with one success. Every other evaluation in
this run, including the anchor's, has SR = 0. Per-episode outcomes are in each harness's
`score.json` under `per_episode`.

**The anchor** (`harnesses/00_anchor_nav`) runs SimWorld-Robotics' ReAct baseline agent through
the multi-module harness (perceiver, spatial memory, planner, prompt assembler, active
perception, completion checker) without changing its behaviour; the evolved harnesses add
modules such as `road_geometry.py`, `arrival_gate.py` and `viewpoint_controller.py`.

## What is in a harness directory

| file | written by | content |
|---|---|---|
| `orchestrator.py`, `modules/*.py`, `AGENT.md`, `rules.md` | the proposer | the harness as evaluated (`run_episode(env, task_spec, vlm_clients, trace_dir)` is the entry point) |
| `NOTES.md` | the proposer | its own account of the change: axis, module map, what changed and why; its first line names the parent |
| `diagnosis.md` | the diagnostician | the report on this harness's evaluation, read by the proposer when the harness is bred from |
| `score.json` | the evaluator | the search-set evaluation: `SubtaskSR`, `SR`, `per_position_SR`, and one row per episode |

The harness modules import `prompt_template` and `utils` from SimWorld-Robotics'
`baseline/single` directory through a hard-coded `_BASELINE_DIR` constant (in `orchestrator.py` and
in the modules that build prompts); point it at your checkout before running one of them. The
bundles are otherwise unchanged from the run.

## Harnesses in evaluation order

Δ is the change in `SubtaskSR` against the parent; the diff column counts changed files and
inserted/deleted lines of the harness files (notes and reports excluded). Per-position SR is the
success rate of the four subtask positions (orientation, reaching the street, reaching the
block, reaching the destination).

| # | harness | parent | SubtaskSR | SR | per-position SR (orient / street / block / destination) | Δ vs parent | diff vs parent (files, +/−) |
|---|---|---|---|---|---|---|---|
| 0 | [`00_anchor_nav`](harnesses/00_anchor_nav) *(on the path)* | — | 0.360 | 0.00 | 1.00 / 0.24 / 0.11 / 0.00 | — | — |
| 1 | [`01_cand_r1_0_spatial-grounding`](harnesses/01_cand_r1_0_spatial-grounding) | anchor_nav | 0.285 | 0.00 | 1.00 / 0.02 / 0.02 / 0.00 | -0.075 | 3, +342/−28 |
| 2 | [`02_cand_r2_0_spatial-grounding`](harnesses/02_cand_r2_0_spatial-grounding) | cand_r1_0_spatial-grounding | 0.330 | 0.00 | 1.00 / 0.12 / 0.11 / 0.00 | +0.045 | 2, +280/−116 |
| 3 | [`03_cand_r3_0_spatial-grounding`](harnesses/03_cand_r3_0_spatial-grounding) | anchor_nav | 0.360 | 0.00 | 1.00 / 0.22 / 0.13 / 0.00 | +0.000 | 1, +186/−17 |
| 4 | [`04_cand_r4_0_spatial-grounding`](harnesses/04_cand_r4_0_spatial-grounding) | cand_r1_0_spatial-grounding | 0.355 | 0.00 | 1.00 / 0.18 / 0.16 / 0.00 | +0.070 | 4, +287/−159 |
| 5 | [`05_cand_r5_0_spatial-grounding`](harnesses/05_cand_r5_0_spatial-grounding) | anchor_nav | 0.345 | 0.00 | 1.00 / 0.20 / 0.09 / 0.00 | -0.015 | 4, +305/−4 |
| 6 | [`06_cand_r6_0_spatial-grounding`](harnesses/06_cand_r6_0_spatial-grounding) *(on the path)* | anchor_nav | 0.370 | 0.00 | 1.00 / 0.20 / 0.20 / 0.00 | +0.010 | 3, +536/−11 |
| 7 | [`07_cand_r7_0_spatial-grounding`](harnesses/07_cand_r7_0_spatial-grounding) | cand_r2_0_spatial-grounding | 0.345 | 0.00 | 1.00 / 0.14 / 0.16 / 0.00 | +0.015 | 2, +247/−274 |
| 8 | [`08_cand_r8_0_spatial-grounding`](harnesses/08_cand_r8_0_spatial-grounding) | anchor_nav | 0.350 | 0.00 | 1.00 / 0.18 / 0.13 / 0.00 | -0.010 | 3, +382/−4 |
| 9 | [`09_cand_r9_0_spatial-grounding`](harnesses/09_cand_r9_0_spatial-grounding) *(on the path)* | cand_r6_0_spatial-grounding | 0.400 | 0.00 | 1.00 / 0.26 / 0.27 / 0.00 | +0.030 | 3, +594/−93 |
| 10 | [`10_cand_r10_0_spatial-grounding`](harnesses/10_cand_r10_0_spatial-grounding) | cand_r2_0_spatial-grounding | 0.535 | 0.00 | 1.00 / 0.60 / 0.49 / 0.00 | +0.205 | 3, +206/−11 |
| 11 | [`11_cand_r11_0_spatial-grounding`](harnesses/11_cand_r11_0_spatial-grounding) | cand_r4_0_spatial-grounding | 0.480 | 0.00 | 1.00 / 0.46 / 0.40 / 0.00 | +0.125 | 4, +477/−10 |
| 12 | [`12_cand_r12_0_spatial-grounding`](harnesses/12_cand_r12_0_spatial-grounding) | anchor_nav | 0.335 | 0.00 | 1.00 / 0.12 / 0.13 / 0.00 | -0.025 | 3, +777/−8 |
| 13 | [`13_cand_r13_0_spatial-grounding`](harnesses/13_cand_r13_0_spatial-grounding) *(on the path)* | cand_r9_0_spatial-grounding | 0.550 | 0.00 | 1.00 / 0.58 / 0.58 / 0.00 | +0.150 | 4, +468/−16 |
| 14 | [`14_cand_r14_0_spatial-grounding`](harnesses/14_cand_r14_0_spatial-grounding) | cand_r13_0_spatial-grounding | 0.515 | 0.04 | 1.00 / 0.48 / 0.49 / 0.02 | -0.035 | 3, +479/−9 |
| 15 | [`15_cand_r15_0_spatial-grounding`](harnesses/15_cand_r15_0_spatial-grounding) **(best)** | cand_r13_0_spatial-grounding | 0.555 | 0.02 | 1.00 / 0.58 / 0.58 / 0.00 | +0.005 | 2, +215/−53 |

## Rendering an episode

The evaluator the paper used writes a trace per episode (`traces/ep_<map>_<task>/` next to the
evaluation: `env_step_log.jsonl` with the robot's position at every step, `actions.jsonl`,
`vlm_trajectory.jsonl` with every model call and the frames it was shown under `frames/`,
`gt.json` with the destination and the ideal path, and `verdict.json` with the per-subtask
result). [`../../render/render_nav_trace.py`](../../render/render_nav_trace.py) turns such a
trace into a video: the camera frame the model last saw, the expected view at the end of the
current subtask, a top-down map with the ideal path, the destination and the robot's track, and
the step, action and distance to the target.

```bash
python examples/render/render_nav_trace.py <eval>/traces/ep_20_10_task_dist_21_0_1 cand_r15_0__20_10_dist_21_0_1.mp4 \
    --title "cand_r15_0_spatial-grounding, map 20_10, task dist_21_0_1"
```

## Videos

`videos/` holds a re-run of the task the best harness completed during the search, map 20_10
task `dist_21_0_1`, played by four harnesses of the lineage, the anchor and the best's ancestors
`cand_r9_0`, `cand_r13_0` and the best itself, `cand_r15_0`, in the simulator with the same policy
model and evaluator settings as the campaign, plus a `_progression_grid` video that tiles the four
(anchor top left, best bottom right). The campaign's own frames were not archived, and the policy
model samples at every step, so a re-run is a fresh draw of the same harness on the same task
rather than a replay of the campaign episode. The videos were rendered from the re-runs' traces
with [`../../render/render_nav_trace.py`](../../render/render_nav_trace.py): each frame shows the
camera image the model was last shown, the expected view at the end of the current subtask, the
step's action and distance, the per-subtask verdict, and a top-down map with the ideal path, the
destination and the robot's track.

**Outcomes.** The anchor's re-run reproduced its campaign episode exactly. `cand_r9_0` and
`cand_r13_0` produced the same 233-step trajectory: both drove along the street past the
destination (closest approach about 73 m, at step 120) until the step cap and ended 487 m from
it; the two share the road-following controller, and the model's answers, which differ between
the two runs, never triggered the arrival check in either. The best harness was re-run four
times: it ended 46 m, 268 m, 428 m and 36 m from the destination, each time committing to an
arrival or stalling before the destination check fired; the video shows the fourth re-run, the
closest. The last column gives the campaign's own evaluation of each harness on this task.

| video | harness | re-run shown | campaign evaluation of this task |
|---|---|---|---|
| [`ep_20_10_task_dist_21_0_1__00_anchor_nav.mp4`](videos/ep_20_10_task_dist_21_0_1__00_anchor_nav.mp4) | `anchor_nav` (eval 1) | orientation passed, then an arrival commit after 11 steps, 588 m from the destination (6 % of the way) | the same: 11 steps, 588 m |
| [`ep_20_10_task_dist_21_0_1__09_cand_r9_0_spatial-grounding.mp4`](videos/ep_20_10_task_dist_21_0_1__09_cand_r9_0_spatial-grounding.mp4) | `cand_r9_0_spatial-grounding` (eval 10) | 233 steps along the street to the step cap, 487 m from the destination (22 %) | 40 steps, 448 m (29 %) |
| [`ep_20_10_task_dist_21_0_1__13_cand_r13_0_spatial-grounding.mp4`](videos/ep_20_10_task_dist_21_0_1__13_cand_r13_0_spatial-grounding.mp4) | `cand_r13_0_spatial-grounding` (eval 14) | the same trajectory as `cand_r9_0`: 233 steps, 487 m (22 %) | 131 steps, 463 m (26 %) |
| [`ep_20_10_task_dist_21_0_1__15_cand_r15_0_spatial-grounding.mp4`](videos/ep_20_10_task_dist_21_0_1__15_cand_r15_0_spatial-grounding.mp4) | `cand_r15_0_spatial-grounding` (eval 16) | 135 steps along the route to 36 m from the destination (94 %), then an arrival commit; the destination check did not fire | completed in 135 steps |
| [`ep_20_10_task_dist_21_0_1__progression_grid.mp4`](videos/ep_20_10_task_dist_21_0_1__progression_grid.mp4) | the four above, 2×2 | — | — |

To draw again, evaluate a harness directory on map 20_10, task `dist_21_0_1` with frames kept
(`EVAL_FORCE_EPISODE=ep_20_10_task_dist_21_0_1` and `EVAL_SAVE_FRAMES=1` in the paper's
evaluator, one fresh simulator per episode) and render the trace with the command above.

