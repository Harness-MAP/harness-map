Spending an action well:
- One action per turn. Aim at a specific cell and name it; drifting without a target wastes presses.
- Move only along an axis where the cursor and your target cell differ. A move along an axis that
  already matches is a wasted press, and a move past the edge of the grid does nothing at all.
- Press place_block ONLY when the cursor stands on a cell you want filled. A toggle at any other
  cell either adds a block that is not in the target or removes one that is.
- If the picture is unchanged after a move, the cursor was already against that face: it is at
  0 or 5 on that axis. Use that, and move the other way.
- Finishing early is the whole game: the episode ends the instant your blocks equal the target, so
  place the certain cells first and the doubtful ones last.
