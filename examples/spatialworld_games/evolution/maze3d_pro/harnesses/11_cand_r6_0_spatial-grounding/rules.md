### Available Actions (Control Mapping):
You must choose ONE of the following precise function names for your action:
- `move_forward`: Move forward with a required `granularity` parameter. `small` moves 1 grid cell,
  `medium` moves 2 grid cells, and `large` moves 3 grid cells. Movement stops early if blocked by a
  wall.
- `move_backward`: Step backward with a required `granularity` parameter. `small` moves 1 grid cell,
  `medium` moves 2, `large` moves 3. Movement stops early if blocked by a wall.
- `turn_left`: Turn your view 90 degrees left.
- `turn_right`: Turn your view 90 degrees right.
- `climb_up`: Go up to the floor above. **Use it only when the MEASURED STATE block says a STAIR
  tile is under your feet.**
- `climb_down`: Go down to the floor below. **Use it only when the MEASURED STATE block says a STAIR
  tile is under your feet.**

### The floor-change precondition (corrected for this build):
There is **no `[Q] UP` / `[E] DOWN` prompt** and no bottom text of any kind on this screen, and the
tile under your feet is never drawn, so nothing in the picture can tell you that a climb is
available. The MEASURED STATE line `under your feet` is the precondition: when it names a STAIR
tile, a climb from that square works; when it says ordinary floor, a climb does nothing at all and
wastes the decision. Both floors matter — roughly half the exits are not on the floor you start on.

### Movement Granularity (and what a stride costs):
- Include `granularity` only when the action is `move_forward` or `move_backward`.
- Use exactly one of: `small`, `medium`, `large`.
- For `turn_left`, `turn_right`, `climb_up` and `climb_down`, do not include a `granularity` field.
- **A stride costs exactly one decision whatever its length, and it stops by itself at a wall**, so
  a longer stride is never worse than a shorter one. `straight ahead` states how many cells a
  forward move will really travel: ask for that many (`small` for 1, `medium` for 2, `large` for 3)
  and never fewer out of caution.
- **You have a hundred decisions and a turn costs one of them just as a three-cell stride does.**
  That is the whole budget of the episode, so the question at every step is not "is this move
  legal" but "does this move buy a cell of new ground or a view I do not have". A turn that lands
  you facing a corridor you have already looked down buys neither.

Decide from the screen and the MEASURED STATE block in front of you, name the one action that
advances you, and answer with the JSON object only.
