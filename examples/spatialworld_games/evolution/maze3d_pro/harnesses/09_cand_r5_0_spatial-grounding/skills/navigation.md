### Navigation Strategy (follow this priority order every decision):
- **The EXIT is the whole task.** If `straight ahead` names the EXIT, `move_forward` toward it and
  keep going all the way onto it — contact ends the maze.
- **If `the route` names a step, take exactly that step.** It is computed from the map the harness
  has measured, it already knows where the walls are, and it is the shortest route to the nearest
  thing worth reaching. Agreeing with it costs you nothing.
- **If `under your feet` names a STAIR tile and `this floor so far` reports no cells left with
  something unseen beside them, change floor** (`climb_up`, else `climb_down`). A floor whose cells
  are all settled and that holds no exit cannot be hiding one.
- **If a STAIR tile is ahead or beside you and this floor is finished, walk onto it first** — the
  climb only works from on top of the tile.
- **If the way ahead is open, move, and take the whole clear run in one decision** — the stride
  costs the same as a single step and stops at the wall by itself.
- **If a wall is in the very next cell, TURN; never step backward.** `move_backward` does not change
  which way you face, so it lands you in the cell behind still looking at the same wall, and the
  next forward walks straight back into it. Turning is what acquires a new view. Choose the side
  that `the cells beside the corridor` shows as `open`; if both are walls, turn either way and keep
  turning the same way until something opens.
- **Prefer ground nothing has looked at yet.** The side lines list the openings beside the corridor;
  an `open` neighbour you have not been down is where the unexplored maze is. Re-walking cells that
  are already settled cannot find the exit.
- **Never repeat a move the block says left the screen unchanged.** An unchanged screen means
  something solid refused it, and the same move from the same square will be refused again.
