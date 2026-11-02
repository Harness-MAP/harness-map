### Navigation Strategy (follow this priority order every step):
- **If the DEAD AHEAD line names the gold exit**, go straight for it with `move_forward`. The exit
  ends the episode the moment you enter it, and the stride you are given lands you on it. Do not
  climb and do not wander off while the exit is ahead of you.
- **If the gold exit is painted in the view but NOT dead ahead**, it is beside the way you are
  facing: turn toward an OPEN bearing and bring it in front of you.
- **If the DEAD AHEAD line names a cyan stair**, walk at it with `move_forward` — the stride is
  measured so that you arrive on the tile instead of striding over it.
- **If a stair is under your feet** (the UNDER YOUR FEET line says so), you are one keystroke from
  the other floor. Spend it when this floor has stopped showing you anything new — the WORKING
  MEMORY block counts that for you — and not before: the exit is on the starting floor often enough
  that leaving it early throws away an easy win.
- **If the way ahead is clear**, use `move_forward`. You do not choose the distance; the clear run
  measured off this screen is what runs, so a clear corridor costs one decision rather than three.
- **If the DEAD AHEAD line puts a wall in the very next cell**, `move_forward` cannot move you.
  Turn to a side the OPEN BEARINGS line calls open, and then move.
- **Never test a direction by walking into it.** The OPEN BEARINGS line has already tested both
  sides of the cell you are standing in, from this screen, and it is exact.
- **Avoid turning repeatedly in the same spot** — if you have turned several times without moving,
  commit to an open bearing and move forward.
- **Prefer a bearing this screen has not refused and you have not already left by.** A screen you
  have been shown several times means you are going in circles; break it by taking the bearing the
  WORKING MEMORY block lists as never tried from this exact screen.
- **The exit can be on either floor.** When this floor is exhausted, take a stair; after a floor
  change, explore the new level before going back.
