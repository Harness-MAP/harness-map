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
- **A step costs one decision whatever its length.** `small` moves 1 cell, `medium` 2, `large` 3,
  and all three cost you exactly one of your 100 decisions. The engine stops a step early at a
  wall, so asking for three cells when only one is clear simply moves you one cell — a long step
  is never worse than a short one. On a 100-decision clock with an exit that can be dozens of
  cells away, defaulting to `small` is how an episode runs out of time with the maze unsolved.
- The WORKING MEMORY block's CORRIDOR line names the longest granularity that can land, measured
  off the picture you are looking at. Ask for that one unless you have a reason not to.
- Two reasons not to, and the CORRIDOR line states both: a stair or the exit in the VERY NEXT cell
  (you want to land ON it, so `small`), and an opening in the side wall you mean to stop at.

Analyze the screen carefully step by step, read the CORRIDOR line for how far the way ahead is
open and what granularity will land, confirm what the WORKING MEMORY block says is under your feet,
and make your next move.
