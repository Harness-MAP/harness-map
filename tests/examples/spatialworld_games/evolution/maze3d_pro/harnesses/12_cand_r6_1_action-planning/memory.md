<!-- memory.md - the per-step scratchpad TEMPLATE, rendered live by tools/plan.py:Pilot.render_memory
     and appended to the agent's system turn by tools/vlm.py:inject_memory. Every field is
     self-derived: a reading of the frame the agent is shown, or an arithmetic consequence of an
     action this harness itself emitted. Never privileged state - not the exit's position, not the
     level's layout, not the true separation from the goal.
     The five placeholders below are the whole contract with Pilot.render_memory; if a placeholder
     is renamed here it must be renamed there too, and a stray brace anywhere in this file breaks
     the render (the Pilot falls back to its own built-in template when that happens). -->

## working memory (measured this step, self-derived only)
- ahead: {ahead}
- beside: {beside}
- underfoot: {underfoot}
- mapped: {mapped}
- goal the harness is committed to: {plan}

Answer with ONE action in the required JSON format. If the lines above say a forward move travels 0 cells, do not move forward.
