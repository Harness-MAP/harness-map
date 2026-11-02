### Available Actions (Control Mapping):
You must choose ONE of the following precise function names for your action:
- `move_forward`: Move forward with a required `granularity` parameter. `small` moves 1 grid cell, `medium` moves 2 grid cells, and `large` moves 3 grid cells. Movement stops early if blocked by a wall.
- `move_backward`: Step backward with a required `granularity` parameter. `small` moves 1 grid cell, `medium` moves 2 grid cells, and `large` moves 3 grid cells. Movement stops early if blocked by a wall.
- `turn_left`: Turn your view 90 degrees left.
- `turn_right`: Turn your view 90 degrees right.
- `climb_up`: Go up to the floor above. It works **only while you are standing on a stair tile**; anywhere else it is a silent no-op that leaves the screen byte-for-byte identical and wastes the decision.
- `climb_down`: Go down to the floor below, under the same single condition.

### The climb rule, stated correctly:
- **There is no `[Q] UP` or `[E] DOWN` prompt on this build, and no bottom HUD of any kind.** Never
  refuse to climb because you did not see one — that text does not exist and waiting for it means
  never changing floor at all.
- The only precondition is a stair under your feet, and you cannot see that: the tile you stand on
  is not drawn. The WORKING MEMORY block states when it has been **measured** there. Climb when it
  says so and the floor has stopped showing you anywhere new.
- A climb that works can leave the screen looking unchanged, so an unchanged screen after a climb is
  not proof of failure. Do not spam the climb to find out — the memory block already tells you.

### Movement Granularity:
- Include `granularity` only when the action is `move_forward` or `move_backward`.
- Use exactly one of: `small`, `medium`, `large`.
- For `turn_left`, `turn_right`, `climb_up`, and `climb_down`, do not include a `granularity` field.
- When a gold or cyan tile is reported as the very next cell, ask for `small`: a longer stride walks
  straight over the tile you are trying to land on.

Analyze the screen carefully step by step, confirm what is directly in front of you, read the
WORKING MEMORY block for what the screens have already settled, and make your next move.
