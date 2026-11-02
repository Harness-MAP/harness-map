Movement rules:
- You can move in 6 directions: up, down, left, right, forward, backward
- Cannot make 180-degree turns (reverse direction): the engine refuses the move outright and the
  snake does not advance at all, so the turn is spent for nothing
- Each move advances the snake exactly one cell; there is no movement granularity

Control mapping (the axis each action moves the head along):
- move_right(): +X
- move_left(): -X
- move_up(): -Y (towards the top of the screen)
- move_down(): +Y (towards the bottom of the screen)
- move_forward(): -Z (into the screen)
- move_backward(): +Z (out of the screen)

Observation tips:
- This build draws NO direction-indicator letters and no legality markers of any kind. The frame
  is the rotated wire cube, the white head sphere, the green body spheres and the yellow food
  sphere, and nothing else. Do not look for letters.
- Spheres closer to the camera are drawn larger; use sphere size to separate depth from the two
  on-screen axes.
- The harness appends a [LATTICE BALLOT] block to this request listing the moves that are
  actually pressable right now, plus which move it pressed last turn and why. Answer with one of
  the listed moves: an answer outside the list is projected onto it before the key is pressed, so
  naming an unlisted move spends the turn without steering.
- Decide from the current screenshot and that block only.
