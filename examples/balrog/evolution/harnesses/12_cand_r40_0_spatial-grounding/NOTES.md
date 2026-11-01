## axis
Code: where the decision is made, which is a structural change to the wiring. Following the steer toward the r027 region, a remembered chart plus a potential field now decides each move. The model is consulted only when the field is flat. Unlike r027, the chart and field sit inside the existing modules: they are methods on the `GroundedAgent` hub plus two functions in `tools/grid.py`. No new module was added and none was removed.

## change
`tools/grid.py` gains two functions:
- `fold_chart` merges every frame's `map:` rows into a per-level chart, using the cursor's convention `(x,y) = rows[y-1][x]`. It resets the chart when `Dlvl` changes.
- `flood_field` runs a reverse relaxation from a drawn `>`. When no `>` is drawn, it starts from every open square not yet stood on that touches blank rock. A square's cost grows with how often it has been stood on (`1 + 0.5*min(visits,6)`). The harness's existing `(square, bearing)` refusals are applied to every hop, not only the first.

`GroundedAgent.reflex` is a new method that runs before the model call. It works through these rungs in order:
1. Modal prompts: `--More--` gets `more`, an attack `[yn]` gets `y`, and any other prompt gets `esc`.
2. Standing on `>` gives `down`.
3. An adjacent creature gets a swing.
4. Otherwise the agent moves to the ring neighbour with the lowest `field + 0.25*visits`, labelled `field:descend`.
5. At a dead end, when the field is empty, it searches, up to 8 times per square. This rung is skipped on boulder levels such as Boxoban.

Only when no rung fires does the old path run: the grounded ring prompt, the model's reply, and `vet`. `forbidden` now has a hard tier, which adds adjacent blank rock (0.989 stone in the diagnosis). It also has a soft tier for stale pacing, and `vet` drops the soft tier before falling back. A reply made by the harness carries its rung name in `reasoning`.

## why
The r033 diagnosis traces all 8 losses caused by the harness (MW15 ×3, R3 ×5) to two gaps:
- There was no persistent chart or frontier, so the agent paced dead ends until the step cap.
- It probed stone: 28% of MW15 steps and 14% of R3 steps.

r027's structure, a field that decides and a model that breaks ties, and the r001/r036 chart-plus-BFS both scored MW15 5/5 and R3 about 4/5. A field that re-descends every step with a visit-weighted cost has no fixed-point 2-cycle. Skipping the model on steps the code decides should also cut the roughly 330k tokens per episode.

## files
orchestrator.py, tools/grid.py
