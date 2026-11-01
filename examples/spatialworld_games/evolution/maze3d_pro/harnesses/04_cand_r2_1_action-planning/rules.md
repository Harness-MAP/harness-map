### Available Actions (Control Mapping):
You must choose ONE of the following precise function names for your action:
- `move_forward`: Move forward with a required `granularity` parameter. `small` moves 1 grid cell, `medium` moves 2 grid cells, and `large` moves 3 grid cells. Movement stops early if blocked by a wall.
- `move_backward`: Step backward with a required `granularity` parameter. `small` moves 1 grid cell, `medium` moves 2 grid cells, and `large` moves 3 grid cells. Movement stops early if blocked by a wall.
- `turn_left`: Turn your view 90 degrees left.
- `turn_right`: Turn your view 90 degrees right.
- `climb_up`: Ascend to the higher floor. Works only while you are standing on a stair tile.
- `climb_down`: Descend to the lower floor. Works only while you are standing on a stair tile.

### The climb rule (read this carefully — it is the one rule most often got wrong here):
- A climb changes floors **only while a stair tile is under your feet**. Off a stair it is a silent
  no-op: the screen comes back byte-for-byte identical and you have spent a decision for nothing.
- **There is no `[Q] UP` or `[E] DOWN` prompt on this build, and no text of any kind on the screen.**
  Never refuse to climb because you did not see a prompt, and never claim to have seen one. The
  UNDER YOUR FEET line in the WORKING MEMORY block is the only statement about the tile beneath you,
  and it is measured from the frames themselves.
- You step onto a stair by walking into it: when the cyan tile is the very next cell dead ahead,
  one `move_forward` at `small` puts you on it — and the cyan then vanishes from the view, which is
  normal and is exactly how arriving on it is recognised.

### Movement Granularity:
- Include `granularity` only when the action is `move_forward` or `move_backward`.
- Use exactly one of: `small`, `medium`, `large`.
- For `turn_left`, `turn_right`, `climb_up`, and `climb_down`, do not include a `granularity` field.
- Use `small` whenever the thing you are aiming at is the very next cell; a longer stride would walk
  straight over a stair or past a turning.

Analyze the screen carefully step by step, confirm what is directly in front of you and what the
WORKING MEMORY block says is under your feet, and make your next move.
