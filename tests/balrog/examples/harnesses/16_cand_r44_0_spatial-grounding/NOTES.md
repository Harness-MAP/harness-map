# cand_r044_0_spatial-grounding

## module map
- `agent.py`: hub `GroundedAgent`, in the same shape as r026. Each step runs a modal reflex, then stairs, the hazard veto, a swing, the chart route, and finally the model fallback. It loads `tools/terrain.py` under a module name unique to its path.
- `tools/terrain.py` (NEW): square-level terrain semantics. It turns adjacent prose nouns into per-cell facts (`door`, `doorway`, `solid`, `liquid`), gives a `passable(glyph, fact, bars_level)` verdict, and an `edge_ok` rule: no diagonal into or out of a real door, or any square ever drawn `+`.

## axis:
Code: the chart's glyph→terrain wiring. Passability moves out of a fixed glyph table into a terrain module where a prose noun convicts the SQUARE and outranks the glyph. The steer's exemplars r036 and r041 both scored 0.375 against the parent's 0.325. Both won through door identity plus a modal reflex that emits a legal verb. This candidate varies that wiring as a separate semantics layer (and adds bars), and keeps the one-module hub.

## change:
- `observe()` now does four things:
  - fills `self.facts` from the adjacent prose (`open door adjacent east` → that cell is a DOOR).
  - marks every charted `+` as a DOOR.
  - on `The door opens.`, records the target as a door instead of booking the edge as a bump.
  - flags the level as a bars level when prose names `bars`.
- `passable()` delegates to `terrain.passable`. A door or doorway is walkable even when drawn `-`/`|`. Wall, stone, bars, lava or water named in prose is never walkable. `#` is a wall on a bars level (Boxoban).
- `route()` and the swing reflex prune diagonal edges through doors.
- The modal reflex answers every open prompt with the legal verb `more`, capped at 3 in a row. Before, it used `esc`/`y`, which are not in the vocabulary. A cursor on the message line now also counts as a prompt, and that frame's pos is None. A `Really attack?` prompt marks the last-swung glyph as peaceful, and the swing reflex skips it from then on.

## why:
The r026 diagnosis names the binding constraint: the glyph→terrain table is wrong in task-specific ways.
- An open door drawn `-` is a wall, which caused 3/3 R3 losses.
- `#` bars are floor, which made 44% of Boxoban-M steps bumps.
- The `esc`/`y` reflex is dead, which cost QE/00 and QE/04 ~80 steps each.

Every one of these drops the harness into the model fallback, which is net harmful. Expected: R3 2/5 → 4/5, fewer Boxoban bumps, and freed QE prompt steps. MazeWalk is unchanged, since it has no doors or bars and stays route-owned.

## files:
agent.py (edited), tools/terrain.py (new), NOTES.md
