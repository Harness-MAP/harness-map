# SimWorld-MMNav harness API

## `env`

A `simworld_gym/SimpleWorld` (or `TrafficWorld`) gym env, constructed by the
evaluator. The harness gets it *already reset* to the current task. Use:

- `env.step(action: int) -> (obs, reward, terminated, truncated, info)`
- `env.unwrapped` to reach the raw env if needed (avoid).

### Action codes (the `int` you pass to `env.step`)
```
-1  subtask_completed   (signal current subtask done; advances to next)
 0  move_forward        (~5 m / 500 sim units)
 1  rotate_left         (-90°)
 2  rotate_right        (+90°)
 3  move_left           (5 m strafe)
 4  move_right          (5 m strafe)
```
The full action space the env exposes is larger (look_up/down, stay, …); the
ReAct baseline restricts itself to the 6 above.

### Observation (`obs`) keys

The robot's *declared* observation space — never touched by the info firewall,
**this is what a deployed robot would see, use these freely**:

- `obs['rgb']`           : HxWx3 uint8 — the ego-RGB frame.
- `obs['depth_map']`     : HxW float — per-pixel depth in cm. **This is your
                           non-privileged substitute for `info['target_location']`**:
                           pair with `object_mask` segmentation to estimate
                           distance to landmark / intersection / building edge
                           without reading ground-truth coords. The proposal's
                           single biggest failure mode ("53% misestimate
                           distance to intersection") is what depth grounding
                           is meant to fix.
- `obs['object_mask']`   : HxWx3 uint8 — semantic colour map
                           (green=trees, purple=buildings,
                           yellow=sidewalks/crosswalks, black=driveways).
                           **Use this for landmark matching and free-space
                           detection** — the same signals that, in privileged
                           form, were `buildings_info` / `target_location`.

### `info` (after step) — **FIREWALL-REDACTED BY DEFAULT**

The evaluator firewalls `info` so the harness only sees what a real robot's
non-privileged sensors carry. Default fields (firewall ON):

- `info['agent']['agent_rotation']`    : quantized cardinal compass label
                                         (one of {North, East, South, West})
- `info['agent']['agent_orientation']` : raw yaw in deg (analog compass /
                                         IMU yaw). Use this when the cardinal
                                         label is too coarse.
- `info['agent']['agent_location']`    : **`None` placeholder** — redacted.
- `info['target']['target_location']`  : **`None` placeholder** — redacted.
- `info['moving_distance']`            : cumulative L2 path length (per-agent
                                         odometry). Direction-agnostic and
                                         goal-free — walking 200 m in a circle
                                         reports 200, distance_progress reports 0.
                                         Use this for stuck-detection and
                                         travelled-distance heuristics.
- `info['landmark_images']`            : per-subtask hint images (also in
                                         `task_spec['subtasks'][i]['visual_hint_path']`).

Redacted (not in the harness-visible `info` at all): `agent_collision`,
`buildings_info`, `current_instruction` (use `task_spec['subtasks']` instead),
`success`, `ideal_path`, and all per-step scoring counters.

**Reproducing pre-firewall numbers:** run `evaluator.py` with
`--allow-info-leak` to disable the firewall. New proposers should NOT design
against the leaked view — anything that lifts SR via `agent_location` /
`target_location` reads is partly-oracle and will collapse to ~0 under the
default firewall (see `iclr2026_submission/` no-oracle re-eval results for
the empirical demonstration).

### Spatial grounding: how to substitute for the redacted coords

The right substitute for the privileged distance gate is the depth + mask pair.
Concrete patterns proposers should try:

| Privileged read | Non-privileged substitute |
|---|---|
| `dist(agent_location, target_location)` | Identify the target landmark in `object_mask`, read the mean `depth_map` value at those pixels, threshold against `task_spec['dist_threshold_cm']`. |
| "Am I past the intersection?" | Detect crosswalk pixels (yellow class in `object_mask`) below the horizon; integrate `moving_distance` since they first appeared. |
| "Is there a building blocking me?" | Front-cone of `depth_map` < threshold AND purple pixels (buildings) dominate that cone. |
| "Have I drifted off-road?" | Foreground-row drivable-pixel fraction in `object_mask` (black=driveway) below threshold. |

These are the same signals a real quadruped's perception stack provides, so any
SR lift built on them is non-oracle and deployment-comparable.

## `task_spec`

Constructed by the evaluator from the task's `task_config.json`. The harness
sees ONLY these fields:

```
{
  "subtasks": [
    {"instruction": str, "visual_hint_path": str},
    ...
  ],
  "max_steps": int,
  "dist_threshold_cm": float,
  "compass_at_start": float
}
```

`landmark_images`, `ideal_path`, `destination`, full `robot_view_images` are
intentionally withheld — they are ground truth.

## `vlm_clients`

```python
vlm_clients = {
    "Qwen3-VL-30B-A3B-Instruct": openai.OpenAI(base_url=..., api_key="EMPTY"),
    # more models will be added later; always check
    # task_spec_dir/allowed_models.json before hard-coding a key
}
```

The OpenAI client's `chat.completions.create` is the call surface. Image
inputs go in user-message `content` as
`{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}`.

## What you return: `EpisodeResult`

```python
@dataclass
class EpisodeResult:
    success: bool                # final SR
    subtask_success: list[bool]  # per-subtask
    steps: int                   # action steps used
    actions: list[int]           # action trace
    llm_calls: list[dict]        # full call log (model, prompt, response,
                                 # latency_s, input_tokens, output_tokens)
    notes: str                   # free-form, written to the trace
```

`actions` and `llm_calls` are written verbatim by the evaluator to
`traces/iter_<NNN>/ep_<task>/{actions.jsonl,llm_calls.jsonl}`.
