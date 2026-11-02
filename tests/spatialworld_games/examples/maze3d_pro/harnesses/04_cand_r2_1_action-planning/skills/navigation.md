### Navigation Strategy (follow this priority order every step):
- **If the gold exit is visible ahead**, go straight for it with `move_forward` — `small` when it is
  the very next cell, longer only while it is several clear cells down the corridor. Do not climb and
  do not wander off while the exit is in sight; the exit ends the episode the moment you enter it.
- **If a cyan stair tile is visible ahead**, you can reach it with `move_forward`. Use `small` once
  it is the very next cell so that you land ON the tile instead of striding over it.
- **If a stair is under your feet** (the UNDER YOUR FEET line says so), you are one keystroke from
  the other floor. Spend it when this floor has stopped showing you anything new — the WORKING
  MEMORY block counts that for you — and not before: the exit is on the starting floor often enough
  that leaving it early throws away an easy win.
- **If the path ahead is open (no wall blocking)**, you can use `move_forward`. Choose the movement granularity based on confidence and corridor length: `small` for cautious 1-cell movement, `medium` for 2 clear cells, and `large` for 3 clear cells.
- **If there is a wall directly ahead**, you can turn to find an open direction, then move forward.
- **Avoid turning repeatedly in the same spot** — if you have turned several times without moving, commit to a direction and move forward.
- **Prefer a bearing this screen has not refused and you have not already left by.** A screen you
  have been shown several times means you are going in circles; break it by taking the bearing the
  WORKING MEMORY block lists as never tried from this exact screen.
- **The exit can be on either floor.** When this floor is exhausted, take a stair; after a floor
  change, explore the new level before going back.
