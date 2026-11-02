<!-- memory.md — the per-step scratchpad TEMPLATE, and a live prose lever.
     agent.py reads everything after this comment and hands it to tools/action.py:Tape as the
     format string it renders each step; tools/vlm.py appends the result to the agent's system turn
     (conversation_history[0]), which is the only per-turn text channel the engine has. The block is
     APPENDED to a stashed pristine base, never spliced in: truncating that turn at a marker deletes
     the JSON RESPONSE FORMAT contract from every later call and breaks the parser.
     Placeholders must match Tape.fields(): step, max_steps, ahead_line, left_line, right_line,
     foot_line, last_line, map_line, plan_line, moves, overrides. If {ahead_line} or {step} is
     missing the file is treated as documentation and Tape.TEMPLATE is used instead.
     Self-derived state ONLY. Never oracle state: not the exit's cell, not the floor it is on, not
     the level layout, not the true remaining distance. -->

## MEASURED STATE (read off the screen in front of you — not a guess)
- decision {step} of {max_steps}.
- straight ahead: {ahead_line}
- immediately to your left: {left_line}   |   immediately to your right: {right_line}
- under your feet: {foot_line}
- your last decision: {last_line}
- this floor so far: {map_line}
- cheapest plan from here: {plan_line}

There is no HUD, no text and no coordinates anywhere on screen. Distances above are counted in grid
cells and are exact. A translation into a wall changes nothing whatsoever, so never spend a decision
on one. A move costs one decision whether it covers one cell or three, and a turn costs one decision
and covers none.

Reply with ONLY the JSON object in the format specified above.
