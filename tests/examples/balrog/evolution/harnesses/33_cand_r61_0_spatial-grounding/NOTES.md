# cand_r061_0 — chart-routed learner

## axis:
Code wiring (agent.py hub): a persistent chart + BFS route now sits ahead of the value learner, as in r026.

## change:
Adjacent prose nouns are credited to their SQUARES: wall/corner/dark area/bars block, and doorway/open door open. A bump books (square, move). `route()` runs BFS over the chart to a drawn `>`, or else to the nearest unstood frontier, and emits the first hop. The learner's scores order equal-distance ties. The learner/model runs only when no route exists, and its default q is -1 on noun-blocked bearings.

## why:
r035 diagnosis: 26–49% of steps bumped into prose-named walls, and 10 of 16 frames with `>` one hop away did not step onto it. The chart+BFS region (r026) solves both.

## files:
agent.py, NOTES.md
