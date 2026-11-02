### Navigation Strategy (follow this priority order every step):
- **If the gold exit is visible ahead**, go straight for it with `move_forward` at the granularity
  the CORRIDOR line names — `small` only when it is the very next cell. Do not climb and do not
  wander off while the exit is in sight; the exit ends the episode the moment you enter it.
- **If a cyan stair tile is visible ahead**, you can reach it with `move_forward`. Use `small` once
  it is the very next cell so that you land ON the tile instead of striding over it.
- **If a stair is under your feet** (the UNDER YOUR FEET line says so), you are one keystroke from
  the other floor. Spend it when this floor has stopped showing you anything new — the WORKING
  MEMORY block counts that for you — and not before: the exit is on the starting floor often enough
  that leaving it early throws away an easy win.
- **If the CORRIDOR line says cells are clear ahead, take them all in one step.** It names the
  longest granularity that can land, measured off the picture, and a step of any length costs the
  same single decision. Walking a long clear corridor one cell at a time is the most expensive
  mistake available to you: it is how the clock runs out with the maze unsolved.
- **If the CORRIDOR line says a wall fills the very next cell**, `move_forward` cannot move you at
  all — do not spend a decision on it. Turn toward one of the OPENINGS that line lists.
- **If it lists no openings at all, you are in a dead end — TURN AROUND, do not step backward.**
  A turn changes which way you are facing; `move_backward` does not. Step back and you are standing
  in the previous cell still staring at the dead end, with a clear corridor in front of you leading
  straight back into it — so you walk back in, and back out, and in again. Two turns cost the same
  two decisions and leave you facing the way out. `move_backward` is also the one move nothing on
  this screen can check for you: everything the CORRIDOR line measures is ahead of you, so a
  backward step is always a guess. Prefer a turn.
- **Head for an opening you have not been through.** The CORRIDOR line tells you which cells beside
  the corridor are empty and how far ahead they are; those gaps are the unexplored branches of this
  floor, and the stair you need is behind one of them. Prefer a gap you have not taken over a
  corridor you have already walked.
- **Avoid turning repeatedly in the same spot** — if you have turned several times without moving,
  commit to a direction and move forward.
- **Prefer a bearing this screen has not refused and you have not already left by.** A screen you
  have been shown several times means you are going in circles; break it by taking the bearing the
  WORKING MEMORY block lists as never tried from this exact screen.
- **The exit can be on either floor.** When this floor is exhausted, take a stair; after a floor
  change, explore the new level before going back.
