<!-- memory.md — the RUNTIME working-memory block.

     harness.py:memory_template() reads the text after the TEMPLATE marker below and
     tools/vlm.py:_inject_memory appends it, filled in, to the stashed pristine system turn on
     every step. `conversation_history[0]` is the one piece of text the engine re-reads each turn
     (`current_prompt` is consumed by build_full_prompt on the first call only), and the block is
     APPENDED to a whole stashed base, never spliced into it: truncating that turn would delete
     the JSON RESPONSE FORMAT contract the evaluator's parser depends on from every later step.

     The placeholders must stay in step with SightedPlanner.fields() in tools/plan.py. A name
     that does not match makes harness.py fall back to plan.DEFAULT_TEMPLATE silently, so edit
     both or neither. Every field is SELF-DERIVED — the inverted projection of the frame the
     model is already shown, plus this harness's own action history. Never put privileged state
     here (the exit's coordinates, the level layout, the engine's own progress metric or its floor
     count): the agent must decide from the observation only. -->

<!-- TEMPLATE -->
--- MEASURED STATE (read off the frame you are looking at) ---
Your last decision: {last_line}
Straight ahead: {sight_line}
Forward range: {reach_line}
Immediately adjacent: {neighbours}
Underfoot: {underfoot}
Floor: {floor_line}
Map so far: {mapped}
Route from the map: {advice}

These lines are measurements, not guesses. A move that reports 0 cells covered did not happen,
and repeating it will not happen either. Reply with ONLY the JSON object for your next action.
