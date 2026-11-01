# cand_r057_0 — stage pipeline with a code-owned ROUTE stage (model = fights + residue)

## module map
| file | role | lever |
|---|---|---|
| harness.py | `PipelineHarness.act`: ledger perceive -> stages' `pre` (outer->inner, may short-circuit with NO model call) -> model -> stages' `post` (inner->outer); the trail goes to `response.reasoning` | code |
| tools/stages.py | NEW. `ModalStage` (`[yn]`->y, `--More--`->more), `MeleeStage` (letter glyph adjacent -> model owns the step), `RouteStage` (BFS hop), `ArbitrateStage` (legal projection, veto, orbit break) | code |
| tools/route.py | NEW. BFS on the `map:` block: first hop to a charted `>`, else to the nearest never-stood frontier (walkable square with an unknown neighbour). Failed edges, prose-blocked/deadly squares and diagonal moves through `+` are removed, and ties use a total order | code |
| tools/ledger.py | r047's combat-aware ledger (a swing is never a failed move; prose noun outranks glyph) + `banned` squares per level | code |
| tools/action_io.py | r047's `arbitrate` | code |
| tools/prompting.py | r047's neighbour table (`#` = "corridor or bars") | code |
| AGENT.md / memory.md | r047 priors | prose |

## axis:
Decision locus / component wiring (code). This moves the harness toward the r018 (middleware stage stack) and r047
(combat-aware ledger + code arbitration) region, and adds the stage that neither exemplar had: a route stage that
**decides** instead of advising.

## change:
I replaced the parent's `LedgerHarness` (one model call per step, advisory 8-row table, veto only on the no-move
pairs) with an ordered stage pipeline.
- On every step with no creature drawn adjacent, `RouteStage.pre` runs a BFS over the decoded map (`tools/route.py`)
  and emits the first hop toward a charted, reachable `>`. If there is none, it emits the first hop toward the
  nearest walkable square that has never been stood on and has an unknown neighbour. The model is then **not called**.
- The model is consulted only when a creature is adjacent (melee), when no route or frontier exists (exhausted chart,
  Boxoban, lava-locked Quest), or never on modal prompts, which `ModalStage` answers.
- The model's reply still passes through r047's `arbitrate`: legal projection, deadly/blocked/failed veto, orbit break.
- The ledger is r047's combat-aware version, which fixes the parent's swing-as-failure bug (diagnosis D).
- Any exception inside a stage falls through to the model, so a stage bug cannot cost the episode.
- The `build_agent` entry is unchanged.

## why:
The r029 diagnosis found the ledger displayed but ignored. With an unvisited open neighbour available, the model
picked a visited one 70–85% of the time. Period-2 orbits stayed at 0.38–0.48, and a `>` drawn in 99/100 frames on
Corridor-R3 r02 was never routed to, because the table only covers one hop.

The diagnosis ranked "make the ledger decide" first, and a BFS explorer in this run scored 0.350 (MazeWalk 10/10,
R3 4/5).

The swing veto disarmed CB-Dark and Quest-Medium in 12 + 21 frames. The combat-aware ledger fixes that, and melee is
left to the model. Tokens per episode should fall sharply because route steps make no call.

Predicted: MazeWalk 6/10 → 9–10/10, R3 0/5 → 2–4/5, others roughly flat.

## files:
harness.py (rewritten), tools/stages.py (new), tools/route.py (new), tools/ledger.py (from r047 + banned),
tools/action_io.py (from r047), tools/prompting.py (from r047), AGENT.md, memory.md, NOTES.md
