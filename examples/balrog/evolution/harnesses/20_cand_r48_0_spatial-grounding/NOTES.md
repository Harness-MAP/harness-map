# cand_r048_0_spatial-grounding — potential-field walker rewired as a STAGE table over tools/field + tools/chart

## module map
| file | role | lever |
|---|---|---|
| orchestrator.py | `GradientAgent.act` builds a step context `c` and runs `STAGES = (prompt, orient, stairs, swing, descend, consult)`; first answer wins, rung -> `reasoning`. Loads tools/ via importlib under a per-candidate name. | code |
| tools/field.py | frame decoder only: sections, `clauses()` (prose `<noun> adjacent <dirs>`), `glyph`, modal detection (incl. cursor on row 0) | code |
| tools/chart.py | NEW `Chart`: terrain/visits/bump edges/door SQUARES/bars flag; `edge()` (no diagonal through a door), `reach()`, `field(pos)` = reverse Dijkstra seeded on a REACHABLE `>` else reachable frontier | code |
| AGENT.md / rules.md | two-line priors, seen only by `consult` | prose |

## axis:
Code: structure/wiring, steered toward r041/r046. The parent's monolithic `act()` with chart state
inside the parser becomes an ordered stage pipeline over a parser module and a Chart module, and
the field's seeding now depends on reachability instead of on whether a glyph is drawn.

## change:
`tools/field.py` is now a pure parser. Map memory lives in the new `tools/chart.py:Chart`:
- **Doors are squares.** A cell that was ever drawn `+`, is named `doorway`/`open door`/`broken door` in prose, or just printed `The door opens.` joins `doors` and stays passable even when it is drawn `-`/`|`. `The door opens.` no longer convicts the edge. Diagonal moves into or out of a door are excluded.
- **Reachability-aware seeding.** `field(pos)` first floods from the hero, seeds only on a `>` inside that reachable region, and otherwise seeds on reachable frontier squares. A drawn but unreachable `>` no longer flattens the field.
- **Bars.** After a `bars` message, `#` is solid for the rest of the level.

`act()` runs `STAGES` in order:
- `prompt` answers every modal frame with `more` (else `esc`) before any geometry runs. A cursor on row 0 counts as modal. `Really attack` marks the last swung-at glyph as peaceful.
- `orient` folds the frame and computes the veto and the field.
- `stairs` emits `down`.
- `swing` attacks an adjacent foe, skipping peacefuls.
- `descend` takes the field argmin.
- `consult` calls the model only when the field is flat.

## why:
The r027 diagnosis names three code bugs that hand every lost R3 and Quest step to the model:
- door squares convicted as walls, both by glyph and by the no-move ledger (R3/00, /01, /03);
- `>`-only seeding that ignores reachability (R3, and all 10 Quest episodes);
- a dead modal reflex keyed on `y`/`esc`, which are not legal (QE/04: 93 steps on one prompt).

The steer exemplars r041/r046 reached 0.375 with exactly this wiring: a chart module plus a staged pipeline, door squares, and a reachable-first route. This candidate keeps the parent's visit-weighted field and brings it onto that structure.

Stub check: the field routes through a doorway drawn `-`, a no-move blocks only that edge, and a `[yn]` frame at y=0 returns `reflex:more`.

## files:
orchestrator.py (rewired), tools/field.py (parser only), tools/chart.py (new), tools/__init__.py (new), NOTES.md
