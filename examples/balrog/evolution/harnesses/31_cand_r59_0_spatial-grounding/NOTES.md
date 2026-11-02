# cand_r059_0_spatial-grounding — field walker rewired as a STAGE table over tools/ modules (+ door/Quest owners)

## module map
| file | role | lever |
|---|---|---|
| agent.py | `GradientAgent`; `STAGES = (prompt, orient, stairs, swing, door, cross, hold, kit, fetch, descend, consult)`. The first stage that answers is emitted, and its rung goes to `reasoning`. tools/ are loaded under a path-unique module name. | code |
| tools/field.py | frame decoder only: sections, `adjacent` prose clauses, raw `lang`, glyph lookup. It holds no chart state any more. | code |
| tools/chart.py | `Chart`: terrain, door SQUARES, bump-closed edges, visits, **locked-door ledger**. `edge` has no diagonal through a door. `field(pos)` is seeded only on REACHABLE `>` or reachable frontier. | code |
| tools/quest.py | Quest regime (ported from r052), armed only when `{pickup,kick,zap} ⊆ legal`. It handles silent-pile detection, one safe kick, horn-first pickup plus a letter filler, apply/zap across lava, prompt answers and the mouth hold. | code |
| AGENT.md / rules.md | consult-only prose (unchanged) | prose |

## axis:
Code, structure and wiring, steered toward r041/r046. The parent's monolithic `act()` becomes an ordered stage table, and its one-file `field.py` is split into a pure decoder (`field.py`), a chart (`chart.py`) and regime owners (`quest.py`, `door` stage).

## change:
- **Door squares instead of door glyphs.** A door square is one with a `+` drawn on it, one prose names as `doorway`/`open door`/`broken door`, or one that just opened. Door squares are passable even when drawn `-`/`|`, and diagonal moves into or out of them are illegal. "The door opens." is no longer booked as a bump.
- **Field seeded on the reachable region.** The field now seeds only on a `>` or frontier the agent can actually reach, so a `>` across lava no longer darkens it.
- **Quest stages ported from r052.** `cross`, `hold`, `kit` and `fetch` come from r052's `tools/quest.py`. An armed Quest level never calls the VLM.
- **New `door` stage.** `This door is locked.` after a cardinal step records `(square, dir)` in `chart.locked`, not in `blocked`. Standing there with the door still drawn `+`, the stage emits `kick`, and `st_prompt` answers the following `In what direction?` with that bearing.
  - Kicks are capped at `KICK_CAP=8` per door. After that the edge is convicted and the field routes around the door.
  - The door stage is armed only when `kick` is legal.
- **Stub-checked:** the field walks onto a `+`, then "This door is locked." → `kick` → `west` → `kick` → `west`.

## why:
The r027 diagnosis gives two binding constraints, both because `flood_field` goes silent and hands the episode to a random-walking VLM:
- **(a) Open door drawn `-` is SOLID.** This is 3/3 of the R3 losses.
- **(b) Lava partition with no verb owner.** This is 10/10 of the Quest losses.

The steer exemplars are r041/r046, and r052 is the realisation of that shape on this exact field-walker lineage. There, a stage table with separable chart and Quest owners lifted the lineage from 0.325–0.375 to 0.600. The door-square chart fixes (a), and the Quest stages fix (b).

The new `door` stage follows the memory "`This door is locked.` → kick" (+7/−0 on R3). It is a new owner that only this stage-slot structure can host cleanly. Expected result: R3 2→4–5/5, Quest 0→8–9/10, with MazeWalk unchanged at 10/10.

## files:
agent.py (rewired as a stage table, plus the `door` stage and kick-aim prompt), tools/field.py (decoder only), tools/chart.py (new: doors, reach-seeded field, `locked` / `lock_to_kick`), tools/quest.py (ported), tools/__init__.py, NOTES.md
