# cand_r039_0 — spatial chart + BFS route hint (grounded single call)

## module map
| file | role |
|---|---|
| agent.py | `ChartHarness`: observe → Chart → one model call → legal projection + veto → record the move for blocked-edge detection. Writes `route=…picked=…` to `response.reasoning` |
| tools/chart.py | NEW. `Chart`: per-level map memory (last-seen glyphs, visits, blocked (square,dir) edges, adjacent lava/water) + `plan()` BFS to a charted `>` or else the nearest frontier (never-stood-on or unseen-bordering passable square) |
| tools/prompting.py | naive transcript + priors (AGENT.md, rules.md, memory.md, skills) + `render_chart`: ROUTE line + 8-row neighbour table |
| tools/action_io.py | `legal_actions` from the system prompt; whole-word projection of the reply onto the legal set with a veto set and a fallback (no more coercion to `north`) |
| AGENT.md / memory.md | short priors explaining the ROUTE line and the chart columns |

## axis:
Code: spatial grounding through a persistent MAP MEMORY and a planned route, moving toward the
r029 region (a naive single call, grounded on input and filtered on output). r029 only knew the
eight neighbours and their visit counts, so it could not see past one hop. This candidate keeps the
glyphs of every square it has seen, then plans over that memory with a BFS that respects blocked
edges.

## change:
Added tools/chart.py and rewired agent.py around it. Each step the Chart folds in the `map:` block
(row = y−1), the cursor, `Dlvl` (a level change resets it) and the adjacent-hazard prose. When a move
leaves the cursor where it was, that (square, direction) edge is recorded as blocked. `plan()` runs a
BFS over the charted passable squares. Walls, unseen squares, `}` and creature letters are excluded,
and so are blocked edges and hazard directions. The target is a charted `>` first, otherwise the
nearest frontier. The prompt gets `ROUTE: nearest <stairs down|unexplored> is N step(s) away; first
move toward it: <dir>` plus the neighbour table. The model still picks the action. Its reply is
projected onto the legal names, with deadly and blocked directions vetoed. If no mention survives,
the BFS first hop is used, and failing that the least-visited open neighbour.

## why:
The diagnosis found that 92% of steps revisit a square, 36% are period-2 ping-pongs and 28% are
bumps. All 4 wins had `>` visible within 4 steps, and no win came from exploring. The lever it
ranked first was a map memory with a frontier/BFS route. r029 (the steer exemplar) grounded only the
immediate neighbours. A multi-hop route turns "least-visited neighbour" into "the way to the nearest
unexplored ground or the stairs you already saw", which is what breaks orbits in mazes and corridors.

## files:
agent.py, tools/chart.py (new), tools/prompting.py, tools/action_io.py, AGENT.md, memory.md, NOTES.md
