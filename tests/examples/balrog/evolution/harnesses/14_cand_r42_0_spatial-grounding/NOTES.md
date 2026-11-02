# cand_r042_0_spatial-grounding — middleware pipeline whose innermost stage OWNS movement

## module map
- `agent.py` — `build_agent` -> `PipelineHarness`. It builds a StepContext, then runs each stage's `pre` from outer to inner. The first stage that answers short-circuits the stack, and the stages outside it still run their `post` veto. If no stage answers, the model is called, the reply is snapped to the legal set, and every `post` runs. The final action is written to `ctx.action`.
- `tools/context.py` — StepContext holds `raw`, `message`, `pos`, `cells`, `nouns`, `notes`, `trail` and `action`.
- `tools/field.py` (NEW) — `Chart`: a per-Dlvl terrain map, visit counts, and a (square, dir) refusal ledger. The ledger does not book melee: it skips a no-move when the message is a fight message or a letter is drawn on the target square. `flood` is a reverse Dijkstra whose step cost grows with visits. `fields()` returns the stairs field first, then the frontier field.
- `tools/stages.py` — the stack is `Legal > Modal > Hazard > Field`. The `Bump` and `Orbit` classes are still in the file but are no longer on the stack.
- `tools/action_io.py` — `snap` now maps 16-point compounds to their 8-point suffix or prefix, e.g. `southsoutheast` -> `southeast`.
- `tools/prompting.py` — builds the prompt from the prose priors plus the notes block.
- The prose files (AGENT/rules/skills/memory) are empty.

## axis:
Code: the wiring of the component graph, i.e. where the decision is made. This moves toward the r027 gradient-walker region, where code descends a potential field and the model only breaks ties. Here that idea is embedded as the innermost decision-owning stage of the middleware pipeline.

## change:
I added `tools/field.py` (a Chart plus a visit-weighted potential field) and a `FieldStage` as the innermost stage. `FieldStage` replaces `BumpStage` and `OrbitStage` on the stack.

In `pre()`, FieldStage folds the frame into a persistent chart and then decides in this order:
1. If it is standing on a remembered `>`, it goes `down`.
2. If a letter is drawn on an open adjacent square, it swings at it.
3. Otherwise it steps to the open neighbour with the lowest potential. It uses the stairs field first. If the stairs are drawn but no neighbour can reach them, it falls back to the frontier field.

Lethal moves and booked refusals are excluded throughout. The model is called only when the field is flat, and it then gets a least-trodden-first move menu as notes.

`agent.py` now runs the outer stages' `post` on short-circuited actions too, so HazardStage still vetoes route hops. The final action is recorded in `ctx.action`, and the refusal ledger books that action rather than the model's proposal.

## why:
diagnosis.md makes four points:
- The pipeline has no goal and no route. 18 of the 23 capped episodes are model-driven 3-, 4- or 6-cycles.
- OrbitStage's perpendicular redirect seeds those cycles.
- BumpStage books melee swings as walls, which feeds the 14 of 15 QM vetoes into rat deaths.
- `snap` passes `southsoutheast` through unmatched, which caused 51 illegal steps.

The steer points at r027's structure, where code owns movement through a field. A field whose cost grows with visits has no fixed-point 2-cycle, so this should also remove the imitation cycles, and the model is skipped on most steps. HazardStage and ModalStage, the proven gains, are kept.

## files:
- `agent.py`
- `tools/field.py` (new)
- `tools/stages.py`
- `tools/context.py`
- `tools/action_io.py`
