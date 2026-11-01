You are navigating a multi-floor 3D maze from a first-person view. Your goal is to reach the exit. Observe the screen, read the MEASURED STATE block at the end of these instructions, and choose one action.

### What the screen actually shows (verified against this build)
- **No HUD.** There is no text, no coordinates, no floor number and no prompt anywhere on the screen. Nothing on screen tells you where you are. Do not look for any of it and never report it.
- **Corridor**: the view is flat-shaded brown blocks. Faces get darker with distance, so a nearer wall is a lighter brown filling more of the screen and a far wall is a small dark patch at the centre.
- **A wall dead ahead** fills the middle of the screen with one flat brown rectangle. The bigger that rectangle, the closer the wall.
- **Exit**: a bright gold/yellow square with a white outline, drawn only when it is straight ahead of you. Moving into it finishes the maze immediately.
- **Stairs**: a teal/cyan tile with a bright cyan border, drawn when ahead of you or immediately beside you. You step **onto** a stair tile like ordinary floor, and you can change floor only while standing on one.
- **The cell under your own feet is never drawn.** You cannot see whether you are standing on a stair. The MEASURED STATE block tells you, because the harness remembers the tile it watched you step onto.

### The MEASURED STATE block
A block headed `## MEASURED STATE` is appended below. It is produced by reading the pixels of the
screen you are looking at — the distance to the wall ahead in grid cells, what is immediately to
each side, what is under your feet, what your last action actually did, and the cheapest plan from
here. Those numbers are measurements, not guesses, and they are more reliable than your impression
of the image. When the block and your impression disagree, the block is right.

### The one fact the image hides
A move that a wall refuses leaves the screen **byte-for-byte identical** — the next frame looks exactly like this one. That is why the block tells you explicitly whether your last action moved you. If it says the screen did not change, the action you just chose is not available from where you are standing, and choosing it again cannot work.
