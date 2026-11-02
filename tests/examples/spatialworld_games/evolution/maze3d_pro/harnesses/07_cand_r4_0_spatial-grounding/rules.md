### Available Actions (Control Mapping):
You must choose ONE of the following precise function names for your action:
- `move_forward`: Move forward with a required `granularity` parameter. `small` moves 1 grid cell,
  `medium` moves 2 grid cells, and `large` moves 3 grid cells. Movement stops early if blocked by a
  wall, so asking for more cells than are clear is never an error — it simply stops short.
- `move_backward`: Step backward with a required `granularity` parameter (`small`/`medium`/`large`
  = 1/2/3 cells). You cannot see behind you, so this is a blind move; prefer turning.
- `turn_left`: Turn your view 90 degrees left. A turn is never blocked, and it does not move you.
- `turn_right`: Turn your view 90 degrees right. A turn is never blocked, and it does not move you.
- `climb_up`: Go up to the floor above. **This fires only while the tile under your feet is a
  stair.** There is no on-screen prompt confirming that — the measured-state block tells you when a
  stair is underfoot, and that is the only precondition.
- `climb_down`: Go down to the floor below, under the same single precondition.

### Movement Granularity:
- Include `granularity` only when the action is `move_forward` or `move_backward`.
- Use exactly one of: `small`, `medium`, `large`.
- For `turn_left`, `turn_right`, `climb_up`, and `climb_down`, do not include a `granularity` field.
- A stride costs **one decision whatever its length**, and it stops safely at a wall, so moving
  `large` down a corridor you can see is clear is strictly cheaper than three `small` moves.
  Use `small` only when the next cell is the one you want to be standing on — stepping onto a
  stair, or onto the exit.

### Decide from the observation only:
Read the frame and the measured-state block appended to this prompt. Both describe the picture you
were given and your own action history, and nothing else. Confirm what is in front of you and what
is under your feet, then make your next move.
