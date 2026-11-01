## axis:
Code wiring / decision ownership (steered toward the r027 region, where code owns the move and the model only fills the residue). A persistent per-level chart plus a BFS route now decide the move; the model is consulted only when no route exists.

## change:
New `tools/chart.py` holds three things: `new_chart`, `merge`, and `route`.
- `merge` folds every frame's `map:` rows into a per-level chart. Blank and `@` never overwrite a known glyph, and the chart resets when `Dlvl` changes.
- A square the hero has stood on is always passable, so an opened door drawn as `-` or `|` stays usable.
- `route` returns the first hop of a BFS to a charted `>`. When there is none, it goes to the nearest frontier: a passable, never-stood square with an uncharted neighbour. The BFS respects refused (square, direction) edges and never steps onto lava.

`harness.py:GroundedAgent.own` answers before any model call:
1. `--More--` gets `more`; an attack `[yn` prompt gets `y`.
2. Standing on a charted `>` gets `down`.
3. A letter drawn in the ring gets a code-owned swing.
4. Otherwise the route hop is taken.

When code answers, it returns an `LLMResponse` whose reasoning column names the owner (`route:stairs`, `route:frontier`, `reflex:*`) and costs zero tokens. The old grounded prompt plus `vet()` path is kept for the residue, which is Boxoban and cases with no hop.

Two bug fixes in `tools/grid.py`:
- A blank neighbour is now `stone`: impassable, forbidden in `vet()`, and printed as such. It is no longer `dark, visited 0x`.
- Foes now come only from letter glyphs on the current frame, with the prose used only to name them. Walls, corners and horns are no longer called creatures.

A "door opens" message also cancels the refused edge that the non-move would otherwise record.

## why:
The diagnosis puts the binding constraint on the missing chart and route: the model owns 83% of moves and does a local random walk.
- It ignored a drawn `>` 22 squares away on R3/02.
- It made 181 stone bumps, 27% of MW15 steps.
- It walked away from 51% of adjacent rats on CB-Dark.
- It was told to attack walls and the horn, 825 frames.

Across the chain, chart+BFS lineages score 0.325–0.375 against this candidate's 0.175. Expected effect: MW15 and R3 recover, and tokens per episode fall sharply, because most steps become model-free.

## files:
tools/chart.py (new), tools/grid.py, harness.py, NOTES.md
