<!-- memory.md — the RUNTIME working-memory template (an evolvable PROSE lever).

     tools/plan.py renders this block every step and tools/vlm.py appends it to the agent's system
     turn (conversation_history[0]), which is the only per-turn text channel the engine has. The
     placeholders must match plan.Pilot.fields() — {step} {max_steps} {ahead_line} {flank_line}
     {foot_line} {last_line} {chart_line} {plan_line}, plus {moves} {overrides} {filed} if wanted.
     If a placeholder is missing or misspelled, plan.DEFAULT_TEMPLATE is used instead and this file
     silently stops being the lever, so keep the two in step.

     Every field is SELF-DERIVED: measured off the frame the agent is shown plus the agent's own
     action history. Never put privileged state here (the exit's coordinates, the layout, the true
     distance, the floor count) — the eval voids an episode that reads any of it. -->

## MEASURED STATE (read off the screen in front of you — not a guess)
- decision {step} of {max_steps}.
- straight ahead: {ahead_line}
- the cells beside the corridor: {flank_line}
- under your feet: {foot_line}
- your last decision: {last_line}
- this floor so far: {chart_line}
- the route: {plan_line}

Distances are counted in grid cells and are exact. The tile you are standing on is never drawn, and
the screen carries no HUD, no text and no coordinates. A move into a wall changes nothing at all, so
it is never worth a decision.

Reply with ONLY the JSON object in the format specified above.
