<!-- memory.md — the RUNTIME working-memory template (prose lever).

     tools/plan.py:ActionPlanner.render_memory() renders the body below once per step and
     tools/vlm.py:_inject_memory appends the result to the agent's system turn
     (conversation_history[0]) — the ONE channel the engine re-reads every turn, since
     `current_prompt` is consumed by build_full_prompt on the first call only. The base system turn
     is kept whole and the block appended after it; it is never spliced, because truncating it would
     delete the JSON RESPONSE FORMAT contract from every later turn.

     Everything here is SELF-DERIVED from `frame_data.image` — the very screens the agent is shown.
     Three readings of those pixels feed it: the whole-frame fingerprint (did the last action change
     anything), the exact-colour census (is a stair or the exit in view at all), and the geometric
     decode in tools/sight.py:Scene (what stands dead ahead and how far, how far the body can walk,
     and which of the two side bearings are open). No engine state, no layout, no coordinates, no
     floor count.

     The renderable body is everything after the LAST HTML comment, and it must use exactly the
     placeholders ActionPlanner.fields() supplies: step, max_steps, view_visits, ahead_line,
     open_line, sight_line, stair_line, last_line, since_move, stale, blocked_here, untried_here,
     climb_line, mode_line, moves, overrides. A body with no `{step}` or with an unknown placeholder
     is ignored and plan.DEFAULT_TEMPLATE is rendered instead, so a bad edit here degrades to the
     built-in wording rather than breaking the episode.
-->
## WORKING MEMORY (measured off the screens you were shown)
- decision {step} of {max_steps}. This exact screen has been shown to you {view_visits} time(s).
- DEAD AHEAD: {ahead_line}
- OPEN BEARINGS: {open_line}
- ELSEWHERE IN THIS FRAME: {sight_line}
- UNDER YOUR FEET: {stair_line}
- last decision: {last_line}
- decisions since the screen last changed because you MOVED: {since_move}
- PROVEN REFUSED from this exact screen: {blocked_here}
- never tried from this exact screen: {untried_here}
- this floor has shown nothing new for {stale} decision(s). {climb_line}
- mode: {mode_line}

How to use this: the DEAD AHEAD and OPEN BEARINGS lines are read off this very screen and are
exact, so you never have to test a direction by walking into it - a `move_forward` into a wall in
the next cell, or a turn toward a walled side, is a decision spent on something already settled.
The distance of a `move_forward` is measured for you and the longest safe stride is what runs, so
never hold yourself to one cell to be careful; that only spends more of your budget on the same
corridor. Never re-propose an action listed as PROVEN REFUSED while this screen is in front of you.
`climb_up` and `climb_down` do something ONLY while a stair tile is under your feet - there is no
on-screen prompt to confirm it, so trust the UNDER YOUR FEET line above and nothing else. After a
floor change, explore the new level before going back.

Reply with ONLY the JSON object in the format specified above.
