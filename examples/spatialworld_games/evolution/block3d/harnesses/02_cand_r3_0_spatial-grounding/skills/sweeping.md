How the plan spends presses, so you can tell a cheap choice from an expensive one:
- Travel is Manhattan: getting from the cursor to a cell costs the sum of the three axis gaps, and the
  axes can be closed in ANY order for the same total. That is why several moves are often equally good —
  the brief lists them as equally_short_choices and any one of them is free to pick.
- Because the order within a route is free but the route length is not, a move that does not reduce the
  gap is pure loss, and a move into a wall is pure loss twice over.
- The plan walks one height at a time and inside a height it walks whole rows, one row left-to-right and
  the next right-to-left. Consecutive cells in that sweep are one press apart. If you are picking an
  axis, picking the one that keeps you inside the current row and the current height is the one that
  stays on the cheap path.
- Finishing matters more than being tidy: the structure is checked after every press, so the episode can
  end the moment the set you have placed matches the target. Keep the presses moving toward the named
  objective.
