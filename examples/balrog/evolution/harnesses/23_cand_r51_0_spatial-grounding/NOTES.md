# cand_r051_0_spatial-grounding — prose-grounded potential-field walker (steered toward r027)

## module map
| file | role | lever |
|---|---|---|
| harness.py | `FieldHarness.act`: rungs modal → down → kick → swing → descend → consult; rung name → `reasoning` | code |
| tools/field.py | NEW `Chart`: per-Dlvl terrain/visits/blocked edges/door squares/prose-solid squares/locked doors; `flood()` = heap Dijkstra seeded `>`@0 AND frontier@25 | code |
| tools/ledger.py | prose `<noun> adjacent <dirs>` parser + `classify` (kept; now feeds the chart's per-square solidity) | code |
| tools/prompting.py | transcript + priors + NEIGHBOUR TABLE (used only on `consult`) | code |
| tools/action_io.py | legal-set parse, whole-word `mentions`, `rank_open` menu | code |
| AGENT.md / memory.md | priors, seen only on consult | prose |

## axis:
Code: where the decision is made (structure/wiring), steered toward r027. The parent asked the model
every step and arbitrated a local 8-cell table; now a charted potential field owns the step and
the model is consulted only where the field is flat.

## change:
New `tools/field.py:Chart` folds each frame's map per level. It judges passability per SQUARE, not per glyph.
- **Unknown squares:** blank is unknown and never entered.
- **Doors:** `|`/`-` are passable if the square is a known door (ever drawn `+`, the prose says `open/closed door`, or `The door opens.`). Door squares can't be entered or left diagonally.
- **Prose outranks the glyph:** a square the ledger's prose nouns call wall/stone/bars/lava/water is booked solid.
- **Bumps:** a bump convicts the (square, dir) edge. `The door opens.` never convicts an edge, and `This door is locked.` books the door as locked.

`flood()` seeds the drawn `>` at 0 and every unstood frontier at +25. An unreachable `>` therefore no longer flattens the field; this was r027's Quest/R3 failure.

`harness.py` is rewritten as a rung ladder:
1. **modal:** a prompt frame (`[yn`, `--More--`, `In what direction?`, cursor on row 0) is answered before any geometry is read. It gets the pending kick bearing, else `more`. `Really attack?` books the edge as peaceful.
2. **down:** on a charted `>`.
3. **kick:** a locked cardinal door gets `kick` plus its bearing, up to 4 tries.
4. **swing:** at an adjacent letter.
5. **descend:** the field argmin, with cardinal moves winning ties.
6. **consult:** one model call with the NEIGHBOUR TABLE and an open-move menu, reply projected whole-word.

## why:
- The r047 diagnosis: the local table can't explore. MW15 was 61% stone bumps, R3 69% revisits, the modal rung was dead (`y` is illegal), and the cost was 348k tok/ep.
- The r027 diagnosis: its field abdicated on three bugs (door convicted as a wall, `>`-only seeding, dead prompt reflex).
- This candidate combines r027's field ownership with the parent's prose-noun oracle and fixes all three bugs. It also adds the locked-door kick (+15/−3 on R3 in chain memory).
- Expected: MW 10/10, R3 ≥3/5, and a large token cut.

## files:
harness.py (rewritten), tools/field.py (new); tools/ledger.py, tools/prompting.py, tools/action_io.py reused unchanged.
