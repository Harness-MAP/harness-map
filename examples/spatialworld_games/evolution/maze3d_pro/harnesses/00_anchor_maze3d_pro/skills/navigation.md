### Navigation Strategy (follow this priority order every step):
- **If the exit is visible ahead** , you can use `move_forward`.
- **If a stair tile is visible ahead** , you can use `move_forward` to step onto it.
- **If the path ahead is open (no wall blocking)** , you can use `move_forward`. Choose the movement granularity based on confidence and corridor length: `small` for cautious 1-cell movement, `medium` for 2 clear cells, and `large` for 3 clear cells.
- **If there is a wall directly ahead** , you can turn to find an open direction, then move forward.
- **Avoid turning repeatedly in the same spot** — if you have turned several times without moving, commit to a direction and move forward.
- **The exit can be on any floor.** Actively decide whether to climb up or down based on your current floor and unexplored areas. Use `climb_up` or `climb_down` when standing on a stair tile and the corresponding prompt (`[Q] UP` or `[E] DOWN`) is shown at the bottom.
