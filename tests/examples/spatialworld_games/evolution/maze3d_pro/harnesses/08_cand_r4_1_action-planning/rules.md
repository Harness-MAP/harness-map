### Available Actions (Control Mapping):
Choose exactly ONE of these function names per decision:
- `move_forward`: move forward, with a required `granularity`. `small` = 1 grid cell, `medium` = 2,
  `large` = 3. The move stops early when something solid blocks it, and is refused outright when the
  very next cell is a wall.
- `move_backward`: step backward, with a required `granularity` (same 1 / 2 / 3 cells). You cannot
  see behind you, so prefer turning around to walking backwards blind.
- `turn_left` / `turn_right`: rotate your view 90 degrees. A turn is NEVER refused, and it is how you
  look down a corridor you have not looked down yet.
- `climb_up`: go to the floor above. **The ONLY precondition is that you are standing on a stair
  tile.** There is no key prompt on this screen to wait for — the MEASURED STATE block tells you when
  a stair is under your feet, and that is the signal.
- `climb_down`: go to the floor below, under the same single precondition.

### Movement Granularity:
- Include `granularity` ONLY for `move_forward` and `move_backward`; use exactly one of `small`,
  `medium`, `large`. Never include it for a turn or a climb.
- A move costs you one decision whatever its length and it stops safely at the first obstacle, so
  down a corridor that the MEASURED STATE block reports open for 3 or more cells, `large` is simply
  worth more than `small`. Use `small` when the next cell is the one you want to land on — stepping
  onto a stair, or onto the exit.

### Deciding
Use the measured facts: what is ahead and how far, what is immediately left and right, what is under
your feet. Never spend a decision on a move the block reports as blocked in the very next cell, and
never repeat an action that left the screen unchanged. Reply with ONLY the JSON object.
