Legal actions — exactly one per reply, by name:
- move_left()      cursor -1 on X
- move_right()     cursor +1 on X
- move_up()        cursor -1 on Y
- move_down()      cursor +1 on Y
- move_forward()   cursor +1 on Z
- move_backward()  cursor -1 on Z
- place_block()    put a block at the cursor, or take away the one standing there

The contract:
- Reply with ONLY the JSON object the response format asks for, naming ONE of the seven actions. No
  commentary, no second action, no coordinates.
- Decide from the picture and from the brief the harness gives you. The brief is derived from your own
  presses; it is never privileged information about the level.
- place_block TOGGLES. Pressing it twice on one cell leaves nothing there and has cost two presses.
- A move into a grid face is legal and does nothing. The brief lists the moves that actually get you
  closer; choosing one of those is always at least as good as any other move.
- When the brief names an objective cell and you are standing on it, place_block is the only action that
  advances the plan.
