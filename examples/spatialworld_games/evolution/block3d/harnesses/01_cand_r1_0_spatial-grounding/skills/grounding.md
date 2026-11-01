Reading the grounded-state block:

From the second turn on, these instructions end with a GROUNDED STATE block. It is not advice
and it is not a guess: the three orthographic panels have been read back square by square from
the very picture you are shown, and the cursor has been read off the highlighted cube in that
same picture. Use it like this.

- **Never re-derive a coordinate the block already states.** If it says the cursor is at
  (1,2,3) then it is at (1,2,3), whatever the isometric picture looks like from this angle.
  Spend your turn choosing an action, not re-reading the panels.
- **Build exactly the listed cells.** The structure is accepted only on an exact match: every
  listed cell filled and nothing outside the list filled. A correct block you have placed must
  stay placed.
- **place_block is a toggle.** On a cell already in the filled list it REMOVES the block. Never
  revisit a finished cell to check it; that undoes the work.
- **Never step in a dead direction.** Those moves leave the cursor where it is and throw away
  one of the decisions the episode allows.
- **Commit when you are standing on a listed cell.** The walk between commitments is already
  planned for you; the judgement that is yours is whether the cell under the cursor belongs to
  the structure. Say place_block when it does.
- **The unexplained-squares count is the progress meter.** It falls only when a block explains
  a panel square nothing else explains. If it stops falling, the next block is in the wrong place.
