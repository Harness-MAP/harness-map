<!-- memory.md — the RUNTIME working-memory template (prose lever).

     tools/plan.py:ActionPlanner.render_memory() renders the body below once per step and
     tools/vlm.py:_inject_memory appends the result to the agent's system turn
     (conversation_history[0]) — the ONE channel the engine re-reads every turn, since
     `current_prompt` is consumed by build_full_prompt on the first call only. The base system turn
     is kept whole and the block appended after it; it is never spliced, because truncating it would
     delete the JSON RESPONSE FORMAT contract from every later turn.

     Everything here is SELF-DERIVED from `frame_data.image` — the very screens the agent is shown.
     Two measurements feed it: differencing consecutive frames (did the last action move us?) and an
     exact-colour census of the current frame (the stair mask is `R == 0 and G >= 1`, the exit mask
     is `R >= 200 and 150 <= G <= 230 and B <= 60`; no other value in this renderer's 30-colour
     palette satisfies either). No engine state, no layout, no coordinates, no true distances.

     The renderable body is everything after the LAST HTML comment, and it must use exactly the
     placeholders ActionPlanner.fields() supplies: step, max_steps, view_visits, last_line,
     since_move, blocked_here, untried_here, sight_line, floor_line, mode_line, moves, overrides,
     climbs. A body with no `{step}` or with an unknown placeholder is ignored and
     plan.DEFAULT_TEMPLATE is rendered instead, so a bad edit here degrades to the built-in wording
     rather than breaking the episode. KEEP THE TWO IN STEP.
-->
## WORKING MEMORY (self-derived: measured off the screens you were shown)
- decision {step} of {max_steps}. This exact screen has been shown to you {view_visits} time(s).
- last decision: {last_line}
- decisions since the screen last changed because you MOVED: {since_move}
- PROVEN REFUSED from this exact screen: {blocked_here}
- never tried from this exact screen: {untried_here}
- what the pixels show: {sight_line}
- floor: {floor_line}
- mode: {mode_line}

How to use this:
- An action that left the screen byte-for-byte identical did not happen — something solid is there.
  Never re-propose an action listed as PROVEN REFUSED while this screen is in front of you; turn to
  a bearing you have not tried yet from this screen and then move.
- "VERY NEXT CELL, dead ahead" is a measurement, not a guess: that tile fills half the screen only
  when it is the adjacent cell. If the EXIT is there, `move_forward` with `small` granularity ends
  the maze. If a STAIR is there, one `small` step puts you on it — a longer stride would carry you
  straight over it.
- A climb does something only while a stair is under your feet, and the floor line above says when
  that is measured to be true. The tile you are standing on is never drawn, so you cannot see it;
  trust the measurement instead of looking for it.
- While this floor still has somewhere new to go, go there. Once the floor line says it has stopped
  showing you anywhere new, the exit is probably on the other floor — head for a cyan stair and
  change floor.
- Moving one cell at a time (`small`) proves what is really ahead; a longer stride is only worth
  asking for down a corridor you can already see is clear and empty.

Reply with ONLY the JSON object in the format specified above.
