### Available Actions (Control Mapping):
You must choose ONE of the following precise function names for your action:
- `move_forward`: Move forward with a required `granularity` parameter. `small` moves 1 grid cell, `medium` moves 2 grid cells, and `large` moves 3 grid cells. Movement stops early if blocked by a wall.
- `move_backward`: Step backward with a required `granularity` parameter. `small` moves 1 grid cell, `medium` moves 2 grid cells, and `large` moves 3 grid cells. Movement stops early if blocked by a wall.
- `turn_left`: Turn your view 90 degrees left. A turn is never blocked, and it always changes what you see.
- `climb_up`: Ascend to the higher floor. It works only while you are standing on a cyan stair tile that leads up; otherwise it silently does nothing and the frame stays identical.
- `climb_down`: Descend to the lower floor. It works only while you are standing on a cyan stair tile that leads down; otherwise it silently does nothing and the frame stays identical.
- `turn_right`: Turn your view 90 degrees right. A turn is never blocked, and it always changes what you see.

### Movement Granularity:
- Include `granularity` only when the action is `move_forward` or `move_backward`.
- Use exactly one of: `small`, `medium`, `large`.
- For `turn_left`, `turn_right`, `climb_up`, and `climb_down`, do not include a `granularity` field.
- Prefer `small` unless the GROUNDED STATE block tells you the corridor ahead is clear for further: a `small` move that changes the view is exactly one cell, which is what keeps your tracked position exact.

### Climbing (this build has no stair prompt):
There is no `[Q] UP` / `[E] DOWN` text on screen, so you can never "see" that a climb is allowed. The way to find out is to stand on a cyan stair tile and simply try `climb_up`. If the view changes, you climbed. If the view is byte-identical, that climb is impossible here and the harness will mark it dead — try `climb_down` instead. **Never refuse to climb merely because you saw no prompt; the prompt does not exist.** The exit is frequently on another floor, and the only way to reach it is through a cyan stair.

Analyze the screen carefully step by step, read the GROUNDED STATE block as fact, confirm what is directly in front of you, and make your next move.
