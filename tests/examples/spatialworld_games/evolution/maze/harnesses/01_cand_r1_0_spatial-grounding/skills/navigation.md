### Navigation Strategy:
- **If the path ahead is open (no wall blocking you)** , you can use `move_forward` to make progress. Choose the movement granularity based on confidence and corridor length: `small` for cautious 1-cell movement, `medium` for 2 clear cells, and `large` for 3 clear cells.
- **If there is a wall directly ahead** , you can turn to find an open direction, then move forward.
- **Avoid spinning in place** — if you have turned multiple times without moving, commit to a direction and move forward.

### What the harness has already measured for you:
- **A `move_forward` that leaves the view unchanged PROVES the side ahead is a wall.** That side is then refused for the rest of the episode; proposing it again wastes the step. Pick a different heading instead.
- **You begin the level facing a wall.** The first useful action on almost every level is a turn, not a forward.
- **A dead end needs two or three turns in the SAME direction.** Never reverse a turn before the view has changed — a left after a right cancels and the pose never moves.
- **Granularity is not the thing to vary when a forward does nothing.** `small`, `medium` and `large` are all zero cells into a wall; change the *heading*, not the distance.
