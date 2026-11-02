<!-- memory.md — the per-turn scratchpad TEMPLATE. tools/vlm.py renders the lines above this block from
     the harness's OWN ledger (dead-reckoned cursor, placed set, press account, plan objective) and
     injects the pair into the dialogue tail, never into the frozen system turn. Everything here must be
     self-derived: the cursor from the presses this harness returned, the plan from the decoded panels.
     Never the level's cell list, never the engine's state. -->

Use the brief like this:
- objective_now is a cell the plan has already proved belongs to the structure, or a surplus block the
  plan has already proved does not. Trust it and move.
- equally_short_choices are interchangeable. Pick one. If it names place_block, you are standing on the
  objective and placing is the whole move.
- recent_presses is there so you do not repeat a press that did nothing. If the cursor did not change,
  the direction you chose was a wall.
