# NOTES — cand_r038_0_spatial-grounding (parent cand_r001_0_spatial-grounding)

## module map
- `agent.py` — ENTRYPOINT (`build_agent` -> `GroundedHarness.act`). This candidate turns it into a **staged
  controller**: modal reflex -> stairs -> chart route -> model fallback.
- `tools/perception.py` — parses message, cursor, map grid and adjacent prose nouns (unchanged).
- `tools/spatial_memory.py` — tracks visits, refused (cell, dir) pairs and the persistent seen-map (unchanged).
- `tools/grounding.py` — move menu, vetoes and BFS `plan_route`. The frontier predicate changed.
- `tools/action_io.py` — legal set, modal answers, reply arbitration (unchanged).
- `tools/prompting.py` + `skills/spatial.md` — prompt assembly, now used only on model-fallback steps.

## axis:
Code, the control flow and wiring, moving toward the r026 region: **the chart decides, and the model is the fallback**.

## change:
`GroundedHarness.act` was rewired from "ask the model every step, then arbitrate" into an ordered stage ladder.
- **Stage 0, modal reflex.** While a prompt is open, the frame is no longer folded into memory. The cursor sits on
  the prompt line then, which previously corrupted `last_pos` and erased `last_move`. A `Really attack` prompt now
  books the (square, bearing) that raised it as refused before answering, which breaks the `north, more, more` loop
  seen on QE/04.
- **Stage 1, stairs.** Standing on a charted `>` emits `down`.
- **Stage 2, chart route.** When `ground()` finds a genuine route (to a drawn `>`, or to an unexplored frontier),
  its first hop is emitted directly with no model call.
- **Stage 3, model fallback.** Only when there is no real route (exhausted level, room sealed by lava, Boxoban,
  unreadable frame) does the model get the grounded menu. The arbiter's veto and legal snap still apply.

`tools/grounding.plan_route` now retires stood cells from the frontier (`visits[c]>0 -> not a frontier`), and the key
is the total order `(dist, index)`. Without this, route-first would lock into the A<->B orbit.

## why:
From diagnosis (iter 1):
- The VLM copied `recommended` on 2513/2541 steps, so the planner already decided everything, at about
  301k tokens per episode.
- The losses came from planner bugs: the visited-frontier orbit (QM 62% and R3 33% ABA) and the prompt-blind
  refusal ledger (QE/04 `[yn]` loop).
- Moving to the r026 wiring lets the model spend its call only where the chart is blind. This should cut cost
  sharply while leaving primary set by the chart.
- Retiring stood frontiers removes the orbit, so exhausted levels now reach the model instead of ping-ponging.
- Expected: primary >= parent (orbit fix on QM/R3, QE/04 unlocked), cost well below the parent.

## files:
`agent.py` (act() rewired into stages, plus `_prompt_open`), `tools/grounding.py` (frontier retires stood cells;
total-order key), `NOTES.md`.
