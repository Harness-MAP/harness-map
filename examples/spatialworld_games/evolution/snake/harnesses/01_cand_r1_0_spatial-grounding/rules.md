Movement rules:
- You can move in 6 directions: up, down, left, right, forward, backward
- Cannot make 180-degree turns (reverse direction). A reversal is REFUSED rather than fatal: the snake does not move, the move is not counted, and the next frame is identical — a wasted turn.
- Each move advances the snake exactly one cell. There is no step size to choose.

Control mapping (axis-aligned, one cell per call):
- move_right(): +X
- move_left(): -X
- move_up(): -Y (upward on screen)
- move_down(): +Y (downward on screen)
- move_forward(): -Z (into the screen)
- move_backward(): +Z (out of the screen)

What the frame actually shows:
- The frame is an oblique projection of the 4x4x4 lattice: the grey cube frame and its dots, the snake head and body, and the food. There are NO direction-indicator letters on this build and no on-screen list of currently safe moves. Do not wait for a marker and do not conclude that a direction is unavailable because nothing is drawn for it.
- Nearer cells are drawn larger, so apparent size is depth evidence. Screen-left/right and screen-up/down are not themselves game directions.
- Decide from the current frame plus your own record of the moves you have already made, and never assert a coordinate you cannot justify from the picture.
