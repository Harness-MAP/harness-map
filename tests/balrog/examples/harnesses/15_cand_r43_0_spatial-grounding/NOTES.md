# cand_r043_0_spatial-grounding

## axis:
Control-flow wiring (code): a model-free rung ladder in `agent.py`, with the rungs placed in
`tools/grounding.py` and `tools/spatial_memory.py`. This follows the steer toward r036's
structure, where the model is consulted only when every model-free rung is silent. In the parent
(r001), every step goes through `client.generate`, and the model copied `recommended` on 99.2% of
steps.

## change:
`harness.act` is now a ladder. The steps run in order: modal reflex → **route rung** → **search
rung** → model.
- **Route rung.** When `ground()` has a BFS route to a drawn `>` or to a frontier, the first hop is
  emitted directly with no model call (`reasoning = rung:route (...)`).
- **Frontier fix** (`grounding.plan_route`). A frontier must be an *unstood* square. A square we
  have stood on never counts, because its undrawn neighbours are rock. This removes the stood-frontier
  period-2 orbits (R3 ep03 83/100 steps, Quest-Medium 60%). The frontier is the nearest in BFS order.
- **Search rung** (`grounding.search_plan`). Conditions: no route, no `>` charted, `search` legal.
  The agent walks to the nearest dead end and searches there. The budget is 8 searches, keyed on the
  probed cell (`memory.searched`).
- **Temporary conviction** (`spatial_memory`). If the agent does not move and the message is
  `Really attack` / `Unknown command` / `You displace` / `You stop.`, the move `(cell, dir)` is
  convicted for 6 steps (`memory.temp`). The planner routes around `memory.blocked` (permanent plus
  live temporary convictions). This breaks the Quest-Easy `[yn]` peaceful-blocker loop (54 steps)
  without sealing a corridor for good.

Only when all rungs are silent (the least-visited fallback, or no position) does the old path run:
the grounding block, one model call, then `arbitrate`.

## why:
Diagnosis fix 5: the VLM is inert, with 0 live overrides. Skipping the call keeps primary the same,
cuts cost by about 90%, and removes temperature noise. Fix 1: the stood-frontier orbit is the binding
harness bug on R3 and Quest. Fix 4: the `[yn]` loop. r036's productive difference is exactly this
wiring, where the planner and reflex rungs answer before the model and the model is the last rung.

## files:
agent.py, tools/grounding.py, tools/spatial_memory.py, NOTES.md
