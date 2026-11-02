## axis:
Code: where the decision is made. The parent let the model choose and the map only checked its choice. Now code picks the move through a per-level chart and a BFS walker, like the r027 field. The model is called only when that walker has no answer: a creature is adjacent, there is no reachable goal or frontier, or the hero can't be located.

## change:
- **New `tools/chart.py:Chart`.** It keeps a record of terrain across frames and resets when the dungeon level (Dlvl) changes. Creature letters and `@` are never stored.
- **How it judges each move from the result.**
  - No position change plus `The door opens.`: the target square is an opened door, so it is passable but can't be entered or left diagonally. This avoids r027's "door drawn `-` counts as wall" bug.
  - `solid stone`, or a bump into an undrawn square: that square is marked stone.
  - Any other bump without a fight message: the (square, direction) edge is blocked.
  - Undrawn neighbours of a square the hero has stood on are marked stone, so a known-blank square is never a frontier.
- **`Chart.plan`** runs a BFS over passable squares with 8 directions. `+` doors and opened doors can't be crossed diagonally. Lava and boulders are impassable, and `#` is too once bars are seen. It chooses in this order:
  1. The first step toward a reachable `>` (`chart:route`).
  2. Otherwise, the frontier square nearest to a drawn but unreachable `>`. This is the goal-attracted seek (`chart:seek`), and it fixes r027's bug where only `>` was used as a target.
  3. Otherwise, the nearest frontier by a total order (dist, visits, y, x), chosen again every step (`chart:explore`).
- **`agent.py`**
  - `GroundedAgent.walk` answers before the model and returns a no-token LLMResponse with the owner written in `reasoning`. It emits `down` when standing on a charted `>`.
  - The old model-plus-vet path is kept for the leftover cases. `vet` now also rejects a move into a square the chart has marked as stone.
  - Every bearing move sets `chart.pending` so the next frame can judge the result.
- **`tools/grid.py`** now also returns the decoded `rows`.
- `build_agent` and the reply format are unchanged.

## why:
The diagnosis for r033 found the model making about 88% of moves. That led to 160 bumps into solid stone and revisit rates of 52–70%, which lost the MW15 and R3 episodes. It also found that a far `>` was ignored (R3/02). The steer points toward r027's structure, where code does the navigation, and r027 got MW15 to 5/5 at 0 tokens. This version also fixes r027's two recorded field bugs: the door that opens being treated as a wall, and seeding only from `>` when it is unreachable. Expected result: MW15 and R3 wins go up, and cost drops sharply on the maze and corridor tasks.

## files:
tools/chart.py (new), agent.py, tools/grid.py
