# cand_r054_0_spatial-grounding: staged pipeline over tools/ modules (from the one-module r026 walker)

## module map
| file | role | lever |
|---|---|---|
| agent.py | `GroundedAgent`: builds a per-step context `c`, then runs `STAGES = (prompt, orient, stairs, fight, route, probe, model)`. The first stage that answers wins, and its rung goes to `reasoning`. Loads tools/ through `importlib` under a per-candidate module name. `build_agent` is the entry, unchanged. | code |
| tools/frame.py | Parser (`parse_glyph_map`, `clause_index`), `glyph`, `decide_action`, and the STEP/ORDER/CARDINALS/LETHAL/WALL/CREATURE/DOOR_NOUNS tables. A prompt counts as open when `cursor.y==0`. | code |
| tools/chart.py | `Chart`: drawn glyphs, doors, bump-closed edges, visits. Its wall table depends on the regime. Also holds `route` (to `>`, else the nearest frontier) and `probe` (a step into undrawn space in a dark room). | code |
| AGENT.md / rules.md / memory.md / skills/ | empty | prose |

## axis:
Code, the structure and wiring (search-space steer toward r041/r046). The parent's monolithic `act()` becomes an ordered stage table over two tool modules. Each owner is a separable stage that either abstains or answers. The stage slots also host the diagnosis's glyph and verb fixes, which had no single place to live in the monolith.

## change:
`agent.py` is split into `tools/frame.py` and `tools/chart.py` and rewired as `STAGES`. The stages run as follows:
- **`prompt`** answers any open prompt with `more`, the only legal verb that closes both `[yn]` and the item-letter prompts. It stops after 3 in a row. A `[yn]` raised by a move also closes that `(square, bearing)` edge, so the walk does not re-raise it. This replaces the dead `y`/`esc` branches.
- **`orient`** folds the frame into the Chart and books bumps. `The door opens.` marks a door instead of booking a bump.
- **Doors in the Chart.** Doors are:
  - every `+`;
  - prose-named doorways and open doors;
  - opened doors;
  - `-` in a `|` wall, or `|` in a `-` wall, with walkable ground on both sides.

  Doors are passable, orthogonal-only.
- **Boxoban walls.** When the legal moves are exactly the 4 cardinals (Boxoban), `#` (iron bars) and `` ` `` (boulders) are walls.
- **`probe`** is new. When `route` finds nothing, it walks to the nearest reachable square with an undrawn orthogonal neighbour and steps into it; a bump closes that edge.
- **`model`** is the last stage. When the reply matches no legal word, it takes the least-visited safe bearing and never passes raw text through.

## why:
Diagnosis r026 found these HARNESS-BUGs in the one-module walker:
- **A.** The `y`/`esc` reflex is dead code, and the model's raw `attack`/`yes` text is coerced to `north`, which freezes Quest-Easy.
- **D.** An open door `-`/`|` is charted as a wall, which cuts off R3's `>`. Dark rooms also leave no frontier.
- **E.** `#` bars are treated as floor on Boxoban (297 bar bumps, plus about 1.4M tokens of model fallback).

The steer says the productive difference in r041/r046 is the stage structure with chart routing and doors. That structure makes each fix a separate stage or Chart property that can be attributed on its own.

Stub checks:
- A `-` door in a `|` wall is inferred, and route reaches `>` through it.
- A shared wall between two floors is not inferred as a door.
- A `[yn]` frame emits `more`.
- On Boxoban, `#` blocks.
- An illegal model reply maps to a safe bearing.

## files:
agent.py (rewired), tools/__init__.py, tools/frame.py, tools/chart.py (new), NOTES.md
