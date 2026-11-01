Choosing the next cell:
- **Ground up**: finish the lowest z layer before starting the next one. The panels pin each
  layer's row and column support, so a layer you have finished stays finished.
- **Core before edge**: inside a layer, place the cells near the middle of the candidate set
  before the ones on its outside. When three silhouettes admit more cells than the target has,
  the surplus is on the outside of the admitted shape.
- **Shortest leg**: when several cells remain, go to the nearest one. A move and a press cost the
  same, so travel is the only part of the plan you can make cheaper.
- **Never undo by accident**: toggling a cell you already filled removes the block. If you are not
  sure the cursor is where you think it is, make a move you can verify before pressing space.
