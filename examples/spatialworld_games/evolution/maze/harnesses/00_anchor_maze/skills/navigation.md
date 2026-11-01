### Navigation Strategy:
- **If the path ahead is open (no wall blocking you)** , you can use `move_forward` to make progress. Choose the movement granularity based on confidence and corridor length: `small` for cautious 1-cell movement, `medium` for 2 clear cells, and `large` for 3 clear cells.
- **If there is a wall directly ahead** , you can turn to find an open direction, then move forward.
- **Avoid spinning in place** — if you have turned multiple times without moving, commit to a direction and move forward.
