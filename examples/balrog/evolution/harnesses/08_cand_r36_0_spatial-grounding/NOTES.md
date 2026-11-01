# cand_r036_0_spatial-grounding

## axis:
Component graph / module wiring (code). The parent r026 is ONE module (`harness.py` holds parsing,
chart, reflexes, planner and model call in a single class). Following the steer toward r001's
structure, this candidate splits it into a perception → chart → reflex → planner → arbiter
PIPELINE under `tools/`. `harness.py` is now only the orchestrator of the rung ladder. It also
fixes the r026 diagnosis HARNESS-BUGs at the module where each belongs.

## change:
New modules:
- `tools/perception.py`: the frame decoder, plus prose door facts. `open/closed door adjacent`
  seals diagonals; `doorway`/`broken door` does not.
- `tools/chart.py`: a `Chart` object holding terrain, visits, the (square, bearing) ledger, and new
  **door identity** (`doors`, `doorways`, `searched`). A `+` that is opened and redrawn `-`/`|` stays
  a door. An open door never seen as `+` is inferred geometrically: `-` flanked E/W by walkable
  cells and N/S by `|` (or the transpose). Doors are passable, and diagonal edges into or out of them
  are pruned. `door opens` clears that square's ledger. New **search rung** (`search_plan`): when
  `route` is None and no `>` is charted, walk to the nearest dead end and `search`, with a budget of
  8 keyed on the probed cell.
- `tools/reflex.py`: the modal reflex answers only with legal verbs. A `[yn]` attack prompt gets
  `northwest` (the `y` key); every other prompt, `--More--`, or a cursor on row 0 gets `more`. The
  old `esc`/`y` branches could never fire. Also adjacent-foe swing and the lethal veto.
- `tools/action_io.py`: legal set and whole-word projection. `arbitrate()` never passes an unlisted
  string through: attack/yes goes to the foe bearing, else the least-visited safe move.
- `tools/prompting.py`: prior composition and banned-move tail, unchanged in content.
- `harness.py`: orchestrator. It resets the chart on a new system prompt (reused agent) and writes
  the deciding rung into `reasoning` (`model:<pick>`, `arbiter:*`, `chart:search`, …).

## why:
The r026 diagnosis names the binding constraint as three code bugs plus a missing rung. First, an
open door is charted as a wall, which cost R3/00 and R3/01 with the `>` drawn. Second, the dead
`esc`/`y` modal reflex locked QE/00 for 87 steps and QE/04 for 94. Third, a `re.sub` pass-through
coerced 79 illegal verbs to `north`. Fourth, there was no search rung (R3/03's hidden corridor).
Open-loop replay of all 40 r026 CSVs through the new and old harness (stub model echoing the
recorded action):
- MazeWalk (10), CB-Dark (5), Boxoban (10), QM (5) and the R3 winners (R3/02, R3/04) are
  **byte-identical**.
- R3/00, R3/01 and R3/03 now stay on `chart:route` (92–99/99 steps vs ~56–76).
- Rebuilt at step 98, the chart has a route to the drawn `>` on R3/00 and R3/01 (was None).
- The QE prompt-locks are answered by the reflex (181 steps).

Expected: R3 2/5 → 4/5. QE/QM depend on the model after the prompts close.

## files:
harness.py (rewritten as orchestrator), tools/perception.py, tools/chart.py, tools/reflex.py,
tools/action_io.py, tools/prompting.py (all new), NOTES.md
