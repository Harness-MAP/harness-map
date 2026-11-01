### Navigation Strategy (follow this priority order every step):
1. **Gold ahead ends it.** If the gold exit panel is ahead and the measured range reaches it, move
   forward exactly that many cells. Nothing else is worth a decision.
2. **Never spend a decision on a move the frame has already refused.** If the measured-state block
   reports that your last move covered 0 cells, that direction is a wall. Turn — do not repeat it,
   and do not try `move_backward` to escape: retreating just walks back where you came from and
   you will be facing the same wall next step.
3. **Travel in strides.** When the forward range is 2 or 3 clear cells, use `medium` or `large`.
   One decision buys up to three cells, and running out of steps is the usual way this maze is
   lost. Drop to `small` when the cell you actually want is the next one.
4. **A stair is how you change floor, and you must be standing on it.** Walk onto the cyan tile —
   stepping *onto* it, not past it — and then use `climb_up` or `climb_down`. The stair panel
   disappearing from view as you arrive is the stair being underfoot; it is the confirmation, not
   a failure.
5. **Finish a floor before you leave it.** The exit is on one specific floor and it may well be
   this one, so climb when this floor has stopped yielding anything new — the measured-state block
   says when it looks exhausted. Climbing at the first stair you meet abandons floors whose exit
   you were about to reach.
6. **Prefer unexplored ground.** Head for corridors you have not walked rather than re-walking
   ones you have; the route line in the measured-state block already points at the nearest
   unexplored opening. If you have turned several times without moving, commit to an opening and
   move forward.
