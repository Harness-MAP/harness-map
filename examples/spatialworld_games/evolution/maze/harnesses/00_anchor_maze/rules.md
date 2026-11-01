### Available Actions (Control Mapping):
- `move_forward`: Move forward with a required `granularity` parameter. `small` moves 1 grid cell, `medium` moves 2 grid cells, and `large` moves 3 grid cells. Movement stops early if blocked by a wall.
- `turn_left`: Turn 90 degrees to the left (changes your DIR).
- `turn_right`: Turn 90 degrees to the right (changes your DIR).

### Movement Granularity:
- Include `granularity` only when the action is `move_forward`.
- Use exactly one of: `small`, `medium`, `large`.
- For `turn_left` and `turn_right`, do not include a `granularity` field.
