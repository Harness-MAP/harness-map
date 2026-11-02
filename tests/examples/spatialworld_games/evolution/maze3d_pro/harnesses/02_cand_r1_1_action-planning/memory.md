<!-- memory.md — the RUNTIME working-memory template (prose lever).

     tools/plan.py:ActionPlanner.render_memory() renders the body below once per step and
     tools/vlm.py:_inject_memory appends the result to the agent's system turn
     (conversation_history[0]) — the ONE channel the engine re-reads every turn, since
     `current_prompt` is consumed by build_full_prompt on the first call only. The base system turn
     is kept whole and the block appended after it; it is never spliced, because truncating it would
     delete the JSON RESPONSE FORMAT contract from every later turn.

     Everything here is SELF-DERIVED: the planner compares consecutive `frame_data.image` frames —
     the very screens the agent is shown — and keeps its own log of what it proposed. No engine
     state, no layout, no coordinates, no distances.

     The renderable body is everything after the LAST HTML comment, and it must use exactly the
     placeholders ActionPlanner.fields() supplies: step, max_steps, view_visits, last_line,
     since_move, blocked_here, untried_here, climb_line, mode_line, moves, overrides. A body with no
     `{step}` or with an unknown placeholder is ignored and plan.DEFAULT_TEMPLATE is rendered
     instead, so a bad edit here degrades to the built-in wording rather than breaking the episode.
-->
## WORKING MEMORY (self-derived: computed by comparing the screens you were shown)
- decision {step} of {max_steps}. This exact screen has been shown to you {view_visits} time(s).
- last decision: {last_line}
- decisions since the screen last changed because you MOVED: {since_move}
- PROVEN REFUSED from this exact screen: {blocked_here}
- never tried from this exact screen: {untried_here}
- stair lock: {climb_line}
- mode: {mode_line}

How to use this: an action that left the screen byte-for-byte identical did not happen - something
solid is there. Never re-propose an action listed as PROVEN REFUSED while this screen is in front of
you; turn to a bearing you have not tried yet from this screen and then move. Moving one cell at a
time (`small`) proves what is really ahead; a longer stride is only worth asking for down a corridor
you can already see is clear. After a floor change, explore the new level before going back.

Reply with ONLY the JSON object in the format specified above.
