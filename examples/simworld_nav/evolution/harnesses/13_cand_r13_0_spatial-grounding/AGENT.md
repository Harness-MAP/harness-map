# AGENT.md — role, task, and verified perception facts

## Role
A frozen **Qwen3-VL-30B-Instruct** agent drives a quadruped robot through a procedurally
generated Unreal Engine 5 city, one VLM call per step. This harness is the scaffolding around
that frozen model: it assembles the prompt, parses the model's action list, tracks state, and
decides when a subtask is complete. **The model weights never change — the harness is what evolves.**

## Task
Reach a destination by executing a turn-by-turn route expressed as a sequence of subtasks:

```
ORIENTATION  →  MOVE_ALONG_ROAD  →  TURN_AT_INTERSECTION  →  REACH_DESTINATION
```

The environment judges each subtask from **true post-rollout state**; the agent commits a
subtask with action `-1` and advances. SR (all four legs) is the hard bar; SubtaskSR (fraction
of legs passed) is the graded signal. See `rules.md` for the action set and the sanctioned inputs.

## Verified perception facts (established by probe — treat as ground truth about the sensors)
- `depth` (env key; NOT `depth_map`) is a **JET‑colormapped** image (600×720×3), not grayscale.
  Distance is encoded by **hue**, low→high distance = **blue → cyan → green → yellow → red**, so
  **near = blue, far = red/orange**. Do NOT read it as brightness (JET luminance is non‑monotonic in
  distance — a mid‑distance green/yellow patch is *brighter* than a near blue one). It is NOT metric;
  use it only for coarse relative near/far by hue. (Probe‑verified: ground directly ahead reads blue.)
- `object_mask` is a segmentation image (600×720×3); **crosswalk shares its hue with sidewalk** — do
  not disambiguate them by color alone.
- `info['agent']['agent_rotation']` is a **cardinal compass string** (West / North / East / South),
  NOT a scalar yaw. Turn actions rotate ±90° between cardinals (`Rotate_Left` −90°, `Rotate_Right` +90°).
- `info['moving_distance']` is **odometry** — cumulative distance travelled; use it for progress and the
  stuck check (`forward_count > 3` with Δ < 0.5 ⇒ wedged).
- Instructions come from the static `task_spec['subtasks']` list; the subtask index is a structural
  fact (`env.current_subtask`), not a privileged observation.

## The per-step pipeline (module map)
`SpatialMemory` (scratchpad) → `PromptAssembler` (compose the VLM prompt from the sanctioned surface)
→ `Perceiver` (one VLM call: frame → description) → `Planner` (one VLM call: description → action list)
→ `AffordanceEstimator` / `ActivePerception` (look-scans, commit checks) → `CompletionChecker`
(decide the `-1` commit). Each is an evolvable lever — improve **perception, planning, and
commit-timing over the honest inputs**, never by reading the answer (see `rules.md` anti-cheat).
