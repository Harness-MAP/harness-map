# NOTES — cand_r050_0_spatial-grounding (chart-routed learner)

## axis:
Code: where the decision is made, and what the value table is grounded on. Steered toward the
r026 wiring (a remembered chart plus a BFS route decides each step). The `LearnerAgent` hub,
the TD(lambda) credit and the tie-break consult all stay, but they now own only the residue
where the chart has no route. It is still one module (`orchestrator.py`) and `build_agent` is unchanged.

## change:
New hub methods: `chart`, `passable`, `frontier` and `route`.
- `chart` folds every frame into `self.terrain`. It convicts a (square, move) edge in
  `self.blocked` on a bump (no position change after a non-swing move) and on a
  `Really attack? [yn]` frame, so a peaceful creature is not swung at again. A "bars" message
  marks `#` as impassable for the level.
- `route` runs a breadth-first search over charted passable squares. `WALL` now includes blank,
  so undrawn stone is never entered. The search targets a drawn `>`, or else the nearest
  unstood square that touches undrawn space. It returns the first hop, emitted as
  `chart:route`.
- Only when there is no route (Boxoban, the far side of the lava, exhausted levels) does the
  learner rank moves. `value()` is now glyph-grounded: a convicted edge, or a move onto a
  drawn wall, blank or lava square, starts at `BUMP_R` instead of `OPTIMISM`.
- The stairs reflex also reads the remembered terrain, because `@` covers `>`.
- The swing reflex skips convicted edges.

## why:
The r035 diagnosis names this binding constraint: optimism was placed on (square, move) pairs,
so walls were found by walking into them (41–54 bumps per 100 steps on MW15). There was also
no wall-aware route to a charted `>`, and 4 losses had `>` on screen. The BFS-chart lineages
(r026 region, r036–r048) score 0.375, with MW 10/10 and R3 4/5. On a dark stub maze this
harness reaches `>` in 21 steps with 0 model calls.

## files:
`orchestrator.py`, `NOTES.md`.
