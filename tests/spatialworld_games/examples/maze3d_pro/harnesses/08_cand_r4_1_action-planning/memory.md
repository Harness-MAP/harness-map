<!-- memory.md — the RUNTIME working-memory template (an evolvable prose lever).
     tools/vlm.py:_inject_memory renders this every step and APPENDS it to the agent's system turn
     (conversation_history[0]), which is the only per-turn text channel the engine has; the base
     turn is stashed whole and never spliced, because truncating it deletes the JSON RESPONSE FORMAT
     contract from every later call.
     Placeholders must match tools/plan.ActionPlanner.fields() exactly:
       {step} {max_steps} {ahead_line} {left_line} {right_line} {foot_line} {last_line}
       {map_line} {plan_line}
     A mismatch (or a missing {ahead_line}/{step}) silently falls back to plan.DEFAULT_TEMPLATE, so
     keep the two in step. Every field is SELF-DERIVED from the frames the agent was shown plus its
     own action history — never engine state, coordinates, the layout or the true distance. -->

## MEASURED STATE (decoded from the screen in front of you — these distances are exact)
- decision {step} of {max_steps}.
- STRAIGHT AHEAD: {ahead_line}
- IMMEDIATELY LEFT: {left_line}
- IMMEDIATELY RIGHT: {right_line}
- UNDER YOUR FEET: {foot_line}
- your last decision: {last_line}
- this floor so far: {map_line}
- what the measurements point at: {plan_line}

The tile under your feet is never drawn and there is no HUD, text or coordinate anywhere on the
screen, so the lines above are the only reliable statement of where you are. A move reported as
travelling 0 cells cannot move you at all — spend the decision on a turn instead.

Reply with ONLY the JSON object in the format specified above.
