Reading the three panels:
- A TOP square at (x, y) says the finished structure has at least one block in the column above (x, y).
  A FRONT square at (y, z) says height z contains at least one block in row y. A SIDE square at (x, z)
  says height z contains at least one block in column x.
- Those three statements are independent, which is why one structure is rarely the only answer. A cell
  whose three squares are all lit is ALLOWED; a cell that is the only allowed explanation of some lit
  square is REQUIRED.
- Required cells are safe: every structure the panels permit contains them. The harness presses those
  first, so early in an episode the objective it names is a cell that cannot be wrong.

Correcting work:
- place_block on a cell that already carries a block REMOVES it. That is how a surplus block is undone,
  and it costs a press each way, so it is worth doing only when a block is genuinely not part of the
  shape.
- You never need to remove a block to make room for another. Blocks do not interfere.
