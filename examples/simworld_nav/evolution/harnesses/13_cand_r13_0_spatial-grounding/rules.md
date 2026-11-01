# rules.md — the navigation contract (output format, legal actions, anti-cheat)

The agent is a frozen Qwen3-VL-30B-Instruct controlling a quadruped in a procedurally
generated UE5 city. It reaches a destination by executing a turn-by-turn route of
subtasks (ORIENTATION → MOVE_ALONG_ROAD → TURN_AT_INTERSECTION → REACH_DESTINATION),
one VLM call per step. This file is the CONTRACT every candidate harness must keep.

## Legal action set (high-level codes → env step)

| code | action | env step |
|---|---|---|
| `0`  | Move_Forward       | 0 |
| `1`  | Rotate_Left  (−90° yaw) | 5 |
| `2`  | Rotate_Right (+90° yaw) | 4 |
| `3`  | Move_Left  (strafe) | 2 |
| `4`  | Move_Right (strafe) | 3 |
| `-1` | Subtask_Completed (COMMIT the current subtask) | −1 |

The planner emits a list of these codes per step and commits a subtask with `-1`.
No other action codes exist. `-1` is the only way to advance a subtask.

## Sanctioned observation surface — read ONLY these

Perception (per frame):
- `obs['rgb']`, `obs['depth']` (JET colormap 600×720×3, per-frame normalized: NEAR=blue, FAR=red — NOT grayscale; key is `depth`, not `depth_map`), `obs['object_mask']`
  (segmentation; crosswalk shares hue with sidewalk).
Deployment-honest state (per step, from `info`):
- `info['moving_distance']` — **odometry** (cumulative distance moved; use for progress/stuck).
- `info['agent']['agent_rotation']` — **cardinal compass** string (West / North / East / South).
- `info` landmark reference images.
Task structure:
- `task_spec['subtasks']` — the static per-subtask instruction list.
- `env.unwrapped.current_subtask` — the current subtask index (structural; the evaluator already
  knows the episode has N subtasks).

## ANTI-CHEAT — these are PRIVILEGED; reading any is an automatic FAIL (candidate discarded)

Never read the ground-truth world state. Forbidden fields (the read-audit rejects them):
- `agent_location`, `target_location` — **absolute GT world coordinates** of the agent or goal.
- `gt.json`, `ideal_path` — the ground-truth route / oracle plan.
- `current_instruction` — the raw per-step instruction field (use `task_spec['subtasks']` instead).
- `env.unwrapped._agent_rotation` analog yaw scalar — stay on the **cardinal** `agent_rotation`.

Navigate from **relative perception + odometry + the cardinal compass ONLY**. There is no shortcut:
you cannot compute a heading to the goal from absolute coordinates, because you may not read them.
Arrival is judged by the environment from true state; a harness that "knows" its position is cheating
and scores zero after the audit. Improve navigation by improving *perception, planning, and
commit-timing over the sanctioned inputs* — not by reading the answer.

## Stuck / termination
`forward_count > 3` with Δ`moving_distance` < 0.5 → the agent is wedged; break or change tactic.
