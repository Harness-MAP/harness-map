### Navigation Strategy (follow this priority order every step):
1. **The gold exit is in sight down the corridor** — go straight at it with the granularity that reaches it, and ignore every other consideration. It ends the episode.
2. **`underfoot` says you are standing on a stair, and `plan` says change floor** — press `climb_up` (or `climb_down` if that is what `plan` names). Do not wait for a prompt; this build draws none.
3. **A stair tile is in sight and the exit is not** — walk onto it. The exit is on another floor on most levels, and a floor you cannot leave is a floor you cannot win from.
4. **`ahead` says a forward move travels 0 cells** — do NOT move forward. Something solid is in the next cell, and pressing into it changes nothing whatsoever. Turn toward an opening `beside` reports, or use `move_backward` if the way behind you is ground you have already walked.
5. **`ahead` says a forward move travels N cells, with N of 1 or more** — move forward, and ask for the granularity that uses all N. The corridor is already proven clear that far.
6. **Nothing is in sight and nothing is open ahead** — keep turning the SAME way until something opens. Alternating turns is a trap: it returns you to a view you were already refused in.

### Two habits that lose episodes:
- **Repeating an action the measured block has already called refused.** A refused press is not "try harder"; it is a wall. The `mapped` line lists the directions from this exact spot that are already proven blocked — treat them as closed.
- **Never leaving the floor you started on.** The exit sits upstairs on most of these levels. If `plan` is committed to a stair, help it get there instead of exploring one more corner of this floor.
