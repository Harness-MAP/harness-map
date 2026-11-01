# cand_r068_0_spatial-grounding — r041 learner rewired onto r064's stage table + tools/ owners, with a `>`-attracted frontier in route()

## module map
| file | role | lever |
|---|---|---|
| harness.py | `LearnerAgent` hub; `STAGES = (prompt, orient, sokoban, stairs, swing, cross, hold, kit, fetch, route, learner)`. The first stage that answers wins, and its rung goes to `reasoning`. `load_tool()` imports `tools/<name>.py` under a path-unique key. `ChartView` exposes the chart to the tools. `route()` now elects frontiers with a stairs attractor. | code |
| tools/quest.py | Quest-pair owner: kick-split, pickup, apply/zap across lava, mouth hold. Armed on {pickup, kick, zap} ⊆ legal. | code |
| tools/sokoban.py | Boxoban owner: exact push-level A*, a plan cache, expect-and-replan. Armed on a 4-cardinal vocab plus a drawn boulder and fountain. | code |
| AGENT.md | one-sentence tie-break prose | prose |

## axis:
Code, structure/wiring (steered toward r064/r058). The parent's monolithic `act()` becomes an ordered stage table with separable `tools/` task owners (quest, sokoban). The varied piece is in the chart router: frontier election by attraction to a drawn but unreachable `>`.

## change:
- **Stage table:** the r041 parent's single-module `act()` is replaced by r064's stage table, with the loader, `ChartView`, `tools/quest.py` and `tools/sokoban.py`. The `bars` flag now makes `#` impassable once the prose names bars.
- **Frontier attractor in `route()`:** this fixes the one deterministic loss the r064 diagnosis names (§5.1).
  - The BFS records `dist` and collects every frontier it meets.
  - When a `>` is drawn but never reached, and no `}` is charted (so it never pulls toward Quest lava), the first hop goes to the frontier that minimises `(dist + cheb(f, nearest >), dist, y, x)`.
  - That key is a total order, so re-electing it every step cannot orbit.
  - With no drawn `>`, or with a reachable one, the behaviour is identical to the parent.
- **Synthetic check:** an R3/03-like frame (hero at (15,7) on a corridor, `>` 7 columns west past undrawn cells). The parent's bearing order emits `east`; this candidate emits `west`.

## why:
- The r041 diagnosis: 20/25 losses were a TASK-GAP (Boxoban, Quest) and the harness had no owner for either. The steer exemplars r058 (quest module, 0.625) and r064 (+sokoban, 0.875) fill exactly that gap through a stage table over `tools/` modules.
- r064's diagnosis names one remaining code-level bug, the drawn-unreachable `>` on R3/03. Chain memory prices that fix at +2/−0 over 125 R3 levels, with 0 changes on other tasks.
- Expected: the r064 profile (35/40) plus R3 5/5, so about 0.900.

## files:
- harness.py (replaced with the stage-table wiring; `route()` attractor)
- tools/__init__.py, tools/quest.py, tools/sokoban.py (new)
- NOTES.md
