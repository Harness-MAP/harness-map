### Available Actions (Control Mapping):
You must choose ONE of the following precise function names for your action:
- `move_forward`: Move forward with a required `granularity` parameter. Movement stops early if blocked by a wall.
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
- You step onto a stair by walking into it: when the DEAD AHEAD line says a cyan stair is the very
  next cell, one `move_forward` puts you on it — and the cyan then vanishes from the view, which is
  normal and is exactly how arriving on it is recognised.
- The exit is on the floor you start on about half the time, so a stair is worth spending only once
  this floor has stopped showing you anything new. The WORKING MEMORY block counts that for you and
  says when the stair is worth taking.

### Movement Granularity:
- Include `granularity` only when the action is `move_forward` or `move_backward`.
- Use exactly one of: `small`, `medium`, `large` (1, 2 or 3 grid cells).
- For `turn_left`, `turn_right`, `climb_up`, and `climb_down`, do not include a `granularity` field.
- **For `move_forward` the distance is not yours to pick and not yours to worry about.** Whatever
  you write is replaced by the clear run the harness measures off this very screen — the longest
  stride that cannot overshoot anything worth stopping for. Say `move_forward` and the right
  distance runs. Being deliberately cautious with it only throws away part of your step budget.

Analyze the screen carefully step by step, confirm what the DEAD AHEAD and OPEN BEARINGS lines say
about the way you are facing and what is under your feet, and make your next move.
