<!-- memory.md — the RUNTIME working-memory template (prose lever).

     tools/plan.py:ActionPlanner.render_memory() renders the body below once per step and
     tools/vlm.py:_inject_memory appends the result to the agent's system turn
     (conversation_history[0]) — the ONE channel the engine re-reads every turn, since
     `current_prompt` is consumed by build_full_prompt on the first call only. The base system turn
     is kept whole and the block appended after it; it is never spliced, because truncating it would
     delete the JSON RESPONSE FORMAT contract from every later turn.

     Everything here is SELF-DERIVED from `frame_data.image` — the very screens the agent is shown.
     Three readings of those pixels feed it: the whole-frame fingerprint (did the last action change
     anything, compared EXACTLY), the exact-colour census in tools/sight.py (where the cyan stair
     and the gold exit are, and whether a stair is underfoot), and the GEOMETRY read in the same
     module (how far the corridor ahead runs, what closes it, and which cells beside it are open —
     all three measured 5624/5624 and 23688/23688 against the renderer). No engine state, no
     layout, no coordinates, no floor count, no distance to the exit.

     The renderable body is everything after the LAST HTML comment, and it must use exactly the
     placeholders ActionPlanner.fields() supplies: step, max_steps, view_visits, corridor_line,
     sight_line, stair_line, last_line, since_move, stale, blocked_here, untried_here, climb_line,
     mode_line, moves, overrides. A body with no `{step}` or with an unknown placeholder is ignored
     and plan.DEFAULT_TEMPLATE is rendered instead, so a bad edit here degrades to the built-in
     wording rather than breaking the episode.
-->
## WORKING MEMORY (measured off the screens you were shown)
- decision {step} of {max_steps}. This exact screen has been shown to you {view_visits} time(s).
- THE CORRIDOR: {corridor_line}
- IN THIS FRAME: {sight_line}
- UNDER YOUR FEET: {stair_line}
- last decision: {last_line}
- decisions since the screen last changed because you MOVED: {since_move}
- PROVEN REFUSED from this exact screen: {blocked_here}
- never tried from this exact screen: {untried_here}
- this floor has shown nothing new for {stale} decision(s). {climb_line}
- mode: {mode_line}

How to use this: the CORRIDOR line is MEASURED off the picture in front of you, not guessed, and the
granularity it names is the longest step that can actually land - ask for that one. A step costs ONE
decision whatever its length, and the engine stops it early at a wall, so a long step is never worse
than a short one and almost always better: you are on a 100-decision clock and the exit can be
dozens of cells away. Say `small` only for the two reasons the line itself gives you - a stair or an
exit in the very next cell, and an opening you mean to stop at. An action that left the screen
byte-for-byte identical did not happen, so something solid is there; never re-propose an action
listed as PROVEN REFUSED while this screen is in front of you, and when the corridor is shut, turn
toward an OPENING the line names rather than guessing a side. `climb_up` and `climb_down` do
something ONLY while a stair tile is under your feet - there is no on-screen prompt to confirm it,
so trust the UNDER YOUR FEET line above and nothing else. After a floor change, explore the new
level before going back.

Reply with ONLY the JSON object in the format specified above.
