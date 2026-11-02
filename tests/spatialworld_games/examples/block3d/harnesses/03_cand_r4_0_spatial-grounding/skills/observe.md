Reading the frame:
- **Panels first**: read all three 6x6 panels before deciding anything. Each filled cell is a
  constraint; together they bound the target from above and from below.
- **Candidate cells**: a cell (x,y,z) is possible only if TOP[x][5-y], FRONT[5-z][5-y] and
  SIDE[5-z][x] are all filled. Everything outside that set is wasted effort.
- **Proved cells**: if a filled panel cell has exactly one possible cell projecting onto it, that
  cell is in the target for certain. Place those first; they can never be wrong.
- **Your own blocks**: the blue cubes in the 3D view are what you have placed. Compare them with
  the panels; a panel cell with no blue cube behind it is work still to do.
- **A frame that did not change at all** means your last action did nothing: a move into a grid
  face, or a toggle whose cube is hidden behind a nearer one. Those are different situations and
  only the first one tells you where the cursor is.
