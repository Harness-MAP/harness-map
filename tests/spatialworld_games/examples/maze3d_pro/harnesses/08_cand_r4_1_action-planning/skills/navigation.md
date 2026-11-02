### Navigation Strategy (apply in this order, using the MEASURED STATE block as ground truth):
1. **Gold ahead is the end of the maze.** If the block reports the EXIT straight ahead, move forward
   — `large` if it is 3 cells away, `medium` at 2, `small` at 1. Nothing else is worth a decision.
2. **A stair you are standing on is a floor change.** If the block says a stair is under your feet
   and this floor has been explored out, `climb_up` (or `climb_down`). Do not walk off a stair you
   came to use.
3. **A stair ahead or beside you is a tile to step onto**, not to admire: turn to face it if it is to
   one side, then `move_forward` with `small` so you land exactly on it.
4. **Walk the open corridor.** The block gives the exact number of cells a forward move will travel.
   Spend it: `large` down a corridor open 3+ cells, `medium` at 2, `small` at 1. A forward move with
   0 travel is a wasted decision — never issue one.
5. **When the way ahead is blocked, turn, do not shove.** Turn toward a side the block reports as
   open floor or as a stair, then move. Turning is free and never refused.
6. **Prefer ground you have not covered.** If the way ahead leads back where you have already been
   and another direction is still unexplored, take the unexplored one. If you have turned several
   times without moving, commit: pick an open direction and move forward.
7. **Finish a floor before leaving it.** The exit is as likely to be on this floor as on another one,
   and a floor you abandon half-explored is a floor you have to come back to.
