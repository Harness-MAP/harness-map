## axis: code lever — the per-step decision inside `tools/vlm.py:decide` (token level, graph frozen)

## change:
`decide` was a one-line delegation to `evaluator.baseline_step`; it is now a measured state
transition around that same delegation. Each step it reads the heading off the compass arrow's
exact ink and the three visible sides of the cell off the vertical extent of the brick ink in
three fixed column bands, folds the previous action's outcome into a per-cell map of open and
solid sides, routes over that map's own open edges to the nearest cell it has not entered, and
appends the map to `conversation_history[0]`, the only slot the engine carries forward verbatim.
The frozen model is still called exactly once a step and its proposal is still what executes —
unless the measurement refutes it, in which case the routed action is substituted and the newest
assistant entry is rewritten to the action actually taken.

## why:
The anchor gives the frozen agent one screenshot per turn and no memory, so it cannot tell a
corridor it has exhausted from one it has never entered: 80.1% of its forwards moved zero cells
and 76.0% re-pushed an edge the same episode had already seen fail. Both are recoverable without
any privileged state, because the frame answers the wall question exactly.

## files:
`tools/vlm.py`, `harness.py`, `tools/action.py`, `tools/frame.py`, `tools/__init__.py`,
`AGENT.md`, `rules.md`, `memory.md`, `skills/navigation.md`, `NOTES.md`, `traces_notes/`.

## verification:
The graph is the anchor's twelve nodes and four edges and every module compiles. The decode was
censused over all 2752 poses of the twenty shipped levels with no overlap between its rungs. On
the real evaluator with the model stubbed the mechanism is 20/20, and 0/20 blinded.
