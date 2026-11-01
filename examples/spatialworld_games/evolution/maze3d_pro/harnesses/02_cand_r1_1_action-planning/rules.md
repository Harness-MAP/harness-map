### Available Actions (Control Mapping):
You must choose ONE of the following precise function names for your action:
- `move_forward`: Move forward with a required `granularity` parameter. `small` moves 1 grid cell, `medium` moves 2 grid cells, and `large` moves 3 grid cells. Movement stops early if blocked by a wall.
- `move_backward`: Step backward with a required `granularity` parameter. `small` moves 1 grid cell, `medium` moves 2 grid cells, and `large` moves 3 grid cells. Movement stops early if blocked by a wall.
- `turn_left`: Turn your view 90 degrees left.
- `turn_right`: Turn your view 90 degrees right.
- `climb_up`: Ascend to the higher floor. **CRITICAL: ONLY use this action if you are standing on a stair AND see "[Q] UP" in the bottom HUD.**
- `climb_down`: Descend to the lower floor. **CRITICAL: ONLY use this action if you are standing on a stair AND see "[E] DOWN" in the bottom HUD.**

### Movement Granularity:
- Include `granularity` only when the action is `move_forward` or `move_backward`.
- Use exactly one of: `small`, `medium`, `large`.
- For `turn_left`, `turn_right`, `climb_up`, and `climb_down`, do not include a `granularity` field.

Analyze the screen carefully step by step, confirm what is directly in front of you or under your feet, and make your next move.
