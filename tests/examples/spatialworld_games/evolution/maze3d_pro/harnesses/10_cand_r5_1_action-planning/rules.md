### Available Actions (Control Mapping):
You must choose ONE of the following precise function names for your action:
- `move_forward`: Translate forward with a required `granularity` parameter. `small` moves 1 grid cell, `medium` moves 2, `large` moves 3. Movement stops early if blocked by a wall.
- `move_backward`: Translate backward, without turning round, with a required `granularity` parameter. `small` moves 1 grid cell, `medium` moves 2, `large` moves 3. Movement stops early if blocked by a wall.
- `turn_left`: Turn your view 90 degrees left. A turn is never blocked.
- `turn_right`: Turn your view 90 degrees right. A turn is never blocked.
- `climb_up`: Go up to the next floor. Works only while you are standing on a stair tile.
- `climb_down`: Go down to the floor below. Works only while you are standing on a stair tile.

### Movement Granularity:
- Include `granularity` only when the action is `move_forward` or `move_backward`.
- Use exactly one of: `small`, `medium`, `large`.
- For `turn_left`, `turn_right`, `climb_up` and `climb_down`, do not include a `granularity` field.

### What an action costs
You get a fixed number of decisions for the whole maze and every action — a one-cell step, a
three-cell stride, a turn, a floor change — costs exactly one of them. So distance is cheap and
**turning is expensive**, and a longer stride is never worse than a short one: it stops itself at
the wall. Prefer `large` down a corridor the MEASURED STATE block reports as clear, and spend a turn
only when you actually intend to go that way.

### Changing floor
There is no on-screen prompt for a floor change and none is needed. Use `climb_up` or `climb_down`
when, and only when, the MEASURED STATE block says there is a stair tile under your feet. The exit
can be on any floor.

Decide from the screen and the MEASURED STATE block only.
