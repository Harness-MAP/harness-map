<!-- memory.md — the per-step working-memory TEMPLATE. tools/grounding.py fills the {{TOKENS}}
     below from FRAME DIFFERENCES ONLY and harness.py appends the result to the live system
     message (conversation_history[0]) before every VLM turn. Everything here is self-derived:
     the view digest, the outcome of the agent's own last action, and the exact-colour pixel
     census of the frame the agent is already looking at. No layout file, no true pose, no
     distance-to-exit, no engine state. Reorder/reword freely — it is a prose lever; the token
     names are the contract with tools/grounding.py::Grounding.block. -->

=== GROUNDED STATE (measured from the pixels — this is fact, not inference) ===
On screen right now:
{{SIGHT}}

Your body (dead-reckoned from which actions actually changed the view):
  - facing: {{HEADING}}
  - floor: {{FLOOR}}
  - position: {{POS}}
  - step {{STEP}}; last action: {{LAST}}
  - {{STUCK}}
  - {{REVISIT}}

The four cells around you (a wall here was PROVEN by trying it and seeing an identical frame):
{{BEARINGS}}

DEAD ACTIONS — these cannot change anything from where you stand: {{DEAD}}
Recent outcomes: {{TRACE}}
Wasted so far: {{WASTE}}

How to use this block:
  1. If the gold exit is dead ahead, move_forward until you are on it. That ends the maze.
  2. Never pick a DEAD action. Repeating one burns a step and leaves the world untouched.
  3. Prefer a bearing marked UNEXPLORED over one marked OPEN; you have already been down OPEN.
  4. Prefer granularity "small": a small move that changes the view is exactly one cell, which
     keeps the position above exact. A medium/large move stops early at a wall and destroys it.
  5. A cyan stair is the only way between floors. Step onto it, then try climb_up; if the view
     does not change, that climb is impossible here and climb_down is the other option.{{REFUSED}}
