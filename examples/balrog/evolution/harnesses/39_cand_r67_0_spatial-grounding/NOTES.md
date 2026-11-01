# cand_r067_0_spatial-grounding — middleware pipeline + ledger + BINDING chart route

## module map
| file | role | lever |
|---|---|---|
| orchestrator.py | `PipelineHarness.act`: context → ledger.observe → stage `pre` (outer→inner) → ONE model call → `snap` → stage `post` (inner→outer); trail → `response.reasoning` | code |
| tools/context.py | NEW. `StepContext`: cursor (row = y−1), `map:` grid, Dlvl, message, "<noun> adjacent <dirs>" prose, notes, trail | code |
| tools/ledger.py | NEW. per-level visits, COMBAT-AWARE failed (square, dir) edges, prose-blocked/deadly cells, `hash_is_bars` level flag, `probed(cell)` | code |
| tools/route.py | NEW. BFS over drawn passable glyphs (no walls/blank/boulder/`}`/failed edges/door diagonals): drawn `>` first, else best unprobed frontier, key `(dist−2·gain, dist, y, x)` | code |
| tools/stages.py | NEW. `LegalStage`, `ModalStage` (`--More--`→`more`), `HazardStage` (last-word veto of deadly/blocked/wall/blank/failed moves), `RouteStage` (model's move kept only if it is the route hop or a swing at an adjacent creature) | code |
| tools/action_io.py | legal-name projection `snap` (whole-word, earliest, longest) | code |
| tools/prompting.py | naive transcript + priors + stage NOTES block (replaces r029's ignored ledger table) | code |
| AGENT.md / memory.md | short priors explaining the NOTES | prose |

## axis:
Code: the decision locus and component graph. The parent's single linear path (a table plus an output veto) becomes a
middleware stack (r018's structure) with a shared perception/memory module (r047's ledger). A chart-route stage is added,
and it is binding rather than advisory. This is the steer direction: r018/r047 structure and wiring, varied by what
sits in the innermost stage.

## change:
I rewrote the parent (r029 neighbour ledger) as a stage pipeline around its one model call. New `tools/context.py` parses
each frame into a `StepContext`. New `tools/ledger.py` keeps per-level memory from the agent's own frames:
- visits per square;
- failed edges, booked only when there is NO combat message, which fixes r029's swing-as-bump disarm (diag §2C);
- cells that the prose names wall/stone/bars or lava/water, with a level-wide "`#` is bars" flag for Boxoban.

New `tools/route.py` runs a BFS on the drawn map and elects a goal every step. A drawn reachable `>` wins. Otherwise it
takes the best frontier: a reachable cell with blank neighbours that no stood-on square has probed. On MazeWalk/R3, blanks
next to where you have stood are stone, so they never count as frontier.

New `tools/stages.py`:
- `RouteStage` keeps the model's move only if it equals the route's first hop or attacks an adjacent creature (letter
  glyph). Otherwise it substitutes the hop.
- `HazardStage` vetoes moves onto walls, adjacent blanks, boulders, `}`, prose-blocked/deadly squares or failed edges.
  This happens ex ante, not after a bump (diag §2B).
- `ModalStage` pages `--More--` with `more` when that verb is legal.

With no route (Boxoban, a sealed Quest room), the model decides, subject to the hazard veto. There is still exactly one
model call per non-modal step, and the trail (`route:explore@(x,y) | model:north | route:north->west`) is written to
the reasoning column.

## why:
The r029 diagnosis gives the binding constraint: the model ignores the visits column. It skipped an unvisited neighbour
65% of the time, so orbits survive, and the fix is to move the choice into code (§5.1). Every no-move was predictable from
the map (§5.3), and swings were booked as failures (§5.2). The r018 diagnosis says its pipeline "has no goal and no route",
while chart + BFS routing reaches 0.35–0.375 elsewhere in this chain. The r047 diagnosis found its noun classifier labelled
`dark area` as open and walked into stone. Here blanks are never walkable and frontier-ness comes from the ledger's probed
set, not from prose.

Offline replay on r049's frames with a stub model: the pipeline runs on all 8 tasks without error. MazeWalk-15x15/00
elects `route:stairs` at step 16 and reaches the `>`. Boxoban elects no route once bars are named.

Predicted: MazeWalk 9x9/15x15 and Corridor-R3 up (DFS exploration plus stairs seeking), and far fewer no-moves.
Boxoban and Quest-Easy are unchanged (no mechanism). Melee tasks: the swing is kept.

## files:
orchestrator.py (rewritten), tools/context.py (new), tools/ledger.py (new), tools/route.py (new), tools/stages.py (new),
tools/action_io.py, tools/prompting.py, AGENT.md, memory.md, NOTES.md
