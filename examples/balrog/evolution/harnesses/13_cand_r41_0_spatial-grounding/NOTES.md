## axis:
Code, the decision wiring. A chart (BFS route) now drives each step. The value learner only walks when the chart has no route.

## change:
`act()` now goes chart-first. `observe()` keeps a bump-edge ledger and a set of door cells. A door counts if prose names it or it just opened; doors take no diagonal moves. `route()` runs a BFS to a drawn `>`, or else to the nearest unstood frontier. The learner runs only when no route exists, over glyph-admissible moves. "The door opens." is no longer counted as a bump. A `[yn]` prompt marks that foe as peaceful and emits `more`.

## why:
Per the diagnosis, 27% of moves were bumps, a drawn `>` had no pull, and doors got pinned. The steer points toward r026's chart and route wiring.

## files:
agent.py, NOTES.md, DESIGN.txt
