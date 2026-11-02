Movement rules:
- You can move in 6 directions: left, right, up, down, forward, backward
- Each move changes cursor position by 1 unit
- Space toggles block at cursor position

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
