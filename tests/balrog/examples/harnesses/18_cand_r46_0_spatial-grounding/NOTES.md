# cand_r046_0_spatial-grounding — staged pipeline over tools/ modules + a Quest lava-crossing stage

## module map
| file | role | lever |
|---|---|---|
| agent.py | `LearnerAgent`: builds a per-step context `c`, runs `STAGES = (prompt, orient, stairs, fight, quest, fetch, route, learner)`; first stage to answer wins, rung -> `reasoning`. Holds the TD value learner (last stage). Loads tools/ via `importlib` under a per-candidate module name (no `sys.path` cache collisions). | code |
| tools/frame.py | observation parser (`parse_glyph_map`, `clause_index`), `glyph`, `decide_action`, STEP/ORDER/CARDINALS/LETHAL/WALL | code |
| tools/chart.py | `Chart`: drawn glyphs, doors, bump-closed edges, visits, stood; `bfs`, `route` -> (hop, 'stairs'/'frontier'/None), `toward(goals)` | code |
| tools/quest.py | NEW `Quest`: inventory letters from `x - a <thing>.`, spawn/item `pickup`, walk targets (items, then lava banks with a valid crossing), fire `apply`/`zap` ACROSS lava, answers the letter / `Improvise?` / direction prompts via the key-bound vocab words | code |
| AGENT.md | one tie-break sentence for the learner consult | prose |
| rules.md / memory.md | empty | prose |

## axis:
Code: structure/wiring (steered toward r018's middleware stage stack and r039's tools/chart.py
split). The parent's monolithic `act()` becomes an ordered stage table over three tool modules,
and the new structure hosts a new task-mechanism stage (quest + fetch) the monolith had no slot for.

## change:
Split agent.py into tools/frame.py (parsing), tools/chart.py (map memory + BFS; `route` now also
returns its kind, and `toward` routes to arbitrary goal sets), and tools/quest.py (new). `act()`
now runs `STAGES` in order; every stage abstains or answers. The existing reflexes/route/learner
behaviour is preserved stage-for-stage (prompt -> chart+credit -> stairs -> swing -> route -> learner).
Two new stages sit before `route`, gated on `Quest.armed` (`pickup` legal and a `}` charted, i.e.
Quest only): `quest` picks up at spawn and on any charted item square, then, holding a horn/wand on
a typeable letter, fires it across the shortest lava run onto new ground (aim is cardinal; ZAP_CAP 4;
after a crossing it re-fires only when the router has nothing left). `fetch` walks to uncollected
items, then to lava banks from which a crossing exists, unless a `>` is reachable. The prompt stage
answers `What do you want to use or apply? [..]` with the vocab word bound to the item's letter,
`Improvise?` with `northwest` (y), and `In what direction?` with the aim, before the generic `more`.

## why:
Diagnosis r041: the binding constraint is a TASK-GAP. MazeWalk/R3 are saturated by `chart:route`,
while Quest is 0/10 because nothing picks up the horn beside spawn (QE/01, horn adjacent south) or
applies it at the `}` river. The router returns None from step 1 on QE and the learner orbits the
pocket for 99 steps. The same item->bank->fire sequence scores QE 5/5 on hier_k8 r036. The
stage-table structure (the steer) is what makes the mechanism a separable, attributable owner.
A stub test confirmed the order pickup -> walk to bank -> apply -> letter -> improvise(y) -> aim.

## files:
agent.py (rewired), tools/__init__.py, tools/frame.py, tools/chart.py, tools/quest.py (new), NOTES.md
