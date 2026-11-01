Movement rules:
- You can move in 6 directions: left, right, up, down, forward, backward
- Each move changes cursor position by 1 unit; a move into a grid face leaves the cursor where it is
- Space toggles the block at the cursor position: pressing it twice at one cell undoes itself

Control mapping:
- move_left(): Move cursor left (-X direction)
- move_right(): Move cursor right (+X direction)
- move_up(): Move cursor up (-Y direction)
- move_down(): Move cursor down (+Y direction)
- move_forward(): Move cursor forward (+Z direction)
- move_backward(): Move cursor backward (-Z direction)
- place_block(): Place or remove a block at cursor position

Observation tips:
- The game shows three orthographic views (Top, Front, Side)
- Gray squares in these views represent the target structure
- Blue blocks in 3D view are your placed blocks
- Highlighted cube shows cursor position
- Coordinate axes: Red=X, Green=Y, Blue=Z

Decide from the observation only:
- Never assume a block landed because you asked for it; look for it in the 3D view
- Never assume the cursor moved because you asked it to; a move into a face is a no-op and the
  picture is then completely unchanged
- The three panels are the only statement of the target. There is no other source of truth, and
  there is nothing to infer from the level number or from how long the episode has run
