### Navigation Strategy (follow this priority order every step):
- **If the block says the EXIT is ahead**, move forward onto it with a granularity that reaches it. Nothing outranks this.
- **If a wall is in the very next cell** (`one forward move travels 0 cells`), do not move forward. Turn to a side the block reports as open floor, or step back the way you came.
- **If the corridor ahead is clear**, move forward, and take the whole clear run in one decision: `large` for 3 or more clear cells, `medium` for 2, `small` for 1.
- **If the block says a stair tile is ahead or beside you**, remember it. Stepping onto it is how you reach the other floor, and the harness will tell you when you are standing on it.
- **Do not turn twice in the same spot.** Two turns spend two decisions and move you nowhere. If you want the cell behind you, use `move_backward` — it costs one decision instead of three.
- **Prefer a direction you have not been.** The block reports how much of this floor is still unseen; a turn toward a side it calls open floor is worth more than re-walking a corridor you have already crossed.
- **If your last action changed nothing**, the way you tried is solid. Pick a different bearing; repeating it cannot work.
- **When this floor has nothing unseen left and no exit has shown up**, the exit is on another floor: go back to a stair tile you were told about and change floor.
