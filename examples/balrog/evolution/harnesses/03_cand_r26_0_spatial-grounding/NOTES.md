# NOTES — cand_r026_0 (one-module pilot)

## axis:
Code: the component graph and where the decision is made. The anchor was three modules
(`agent.py` + `tools/prompting.py` + `tools/action_io.py`) and asked the model every turn.
This candidate is ONE module with no `tools/` package. `build_agent` constructs the hub
`GroundedAgent`. The hub calls `parse_glyph_map`, which calls `clause_index`, and it also calls
`decide_action`. Everything else is a method of the hub. The map now drives each step, and the
model is asked only when the map has no route.

## change:
Every step goes through this order:
1. **Modal reflex.** If the prompt is `Really attack? [yn]`, answer `y`. On `--More--`, answer
   `more`. On any other open prompt (`What do you want to drink?`, `In what direction?`), answer
   `esc`.
2. **Stairs.** Standing on a charted `>`, emit `down`.
3. **Hazard veto.** A lava or water noun in the `adjacent` prose, or a `}` glyph in a neighbouring
   square, closes that bearing for this turn.
4. **Melee.** A letter glyph on an adjacent square gets swung at, by moving into it.
5. **Chart route.** Breadth-first search over the remembered floor, in a fixed bearing order. It
   targets a drawn `>`, or else the nearest UNSTOOD square that touches undrawn space. A bump
   (no position change after a move that was not a swing) closes that (square, bearing) edge for
   the rest of the level. The glyph type is never convicted.
6. **Model.** Only when no route exists (Boxoban, the far side of the Quest lava, exhausted
   levels) does the stock BALROG history go to the model. The closed and deadly bearings are
   named in the prompt. The reply is projected whole-word onto the legal list. A vetoed pick is
   replaced by the least-visited safe bearing.

The chart resets when `Dlvl` changes and on `reset()`. The map row is `cursor.y - 1`, which
matched `@` on 321 of 325 real frames.

## why:
Diagnosis iter 0: 27.8% of steps did not move, 35.3% were period-2 revisits, two Quest-Easy
deaths walked into lava that the prose had named, and two episodes locked on `[yn]` or a drink
prompt. The model was re-deciding from a transcript with no memory. A remembered map with a
bump ledger addresses the orbits and the stone bumps on MazeWalk and Corridor-R3.

## files:
`agent.py` (rewritten as one module), `tools/` removed, `NOTES.md`.
