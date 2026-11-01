### Navigation Strategy (follow this priority order every step):
- **If the gold exit tile is the very next cell** (it fills about half the screen), `move_forward`
  with `small`. That ends the maze — nothing outranks it.
- **If a stair is measured under your feet and this floor has stopped showing you anywhere new**,
  climb. That is the whole of the floor-change decision; there is no prompt to wait for.
- **If a cyan stair tile is the very next cell and you have decided to change floor**,
  `move_forward` with `small` to step onto it. A `medium` or `large` stride walks over it.
- **If the path ahead is open (no wall filling the middle of the view)**, `move_forward`. Choose
  granularity by what you can actually see: `small` for a cautious 1-cell move, `medium` for 2
  clearly empty cells, `large` for 3.
- **If a wall is directly ahead**, turn to a bearing this screen has not already refused, then move.
- **Avoid turning repeatedly in the same spot** — if you have turned several times without moving,
  commit to a direction and move forward.
- **Prefer a screen you have never been shown before.** Revisiting is how a 100-decision budget gets
  spent on a four-step loop; the memory block counts how often this exact screen has come back.

### The two floors
The exit is on the upper floor on about half of these levels and on your starting floor on the rest,
so neither "always climb" nor "never climb" works. Work the floor you are on while it is still
showing you new screens. Once the memory block says it has stopped, the exit is probably elsewhere:
head for a cyan stair, step onto it, and climb. If the other floor goes stale too, come back — the
same stair works in both directions.

### What you cannot see, and what to do about it
The tile under your feet is never rendered, and there is no HUD giving your position, floor or
heading. Everything of that kind in the WORKING MEMORY block is measured from the screens you were
shown; it is the only place those facts exist. Read it before you decide, and never contradict it on
the basis of a picture that cannot show what it reports.
