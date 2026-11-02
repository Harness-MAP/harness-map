Spatial-grounding procedure (apply this to the newest frame before naming an action):

1. Find the head and the food in the oblique projection, then place each of them on the lattice
   rather than on the screen. Count lattice cells along the three drawn cube edge directions to
   read off an integer (x, y, z) in [0,3] for both; use the relative size of the two markers as
   the depth check, because nearer cells are drawn larger. Do not convert screen-left or
   screen-up into a game direction.
2. State the signed displacement food - head on each axis separately. Only axes with a nonzero
   difference are candidate moves, and each one names exactly one function: +X move_right,
   -X move_left, +Y move_down, -Y move_up, +Z move_backward, -Z move_forward.
3. Before you name any of them, check the FACE you are standing on. A coordinate that already
   reads 0 has no room in the negative direction on that axis, and a coordinate that already
   reads 3 has no room in the positive direction. The head spawns at y=3, so -Y is the only
   vertical direction with room until you have climbed. A move with no room is not a slow move;
   it ends the episode on the spot, which is strictly worse than any detour.
4. Among the directions that both reduce a nonzero displacement and have room, prefer the one
   that also leaves room on the NEXT step: look one cell ahead and reject a direction whose
   arrival cell is cornered against two faces and your own body. If two directions survive,
   take the one whose axis has the larger remaining difference, so the later route keeps more
   freedom.
5. If no reducing direction has room, do not force it. Move along an axis that is already
   matched, pick the direction with the most room, and re-ground from the next frame. Also
   re-ground from scratch whenever the food marker jumps: a pellet has been eaten, the body is
   one segment longer, and the old displacement is dead.

Two standing constraints, both cheap to honour:
- Never name the exact reverse of your last move. The engine refuses it, so the frame will not
  change and the turn is spent for nothing.
- Keep a short mental trail of the cells you have just occupied and do not re-enter them; the
  body is still standing in them, and it is longer after every pellet.

Final check: emit exactly one of the six function names, and only one whose room you have
just justified from the current frame.
