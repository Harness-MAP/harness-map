## axis:
Code, control-flow wiring. Where the move decision is made changes, following the r027 structure: a potential field over a chart now decides moves in code, and the model is consulted only where the field is flat. Before this change the model decided every step and a one-hop arbiter patched its choice.

## change:
- **New `tools/field.py` (`Chart`).** Each frame's `map:` block (row = cursor.y−1) is folded into a per-level terrain chart. Two things are added to the chart:
  - neighbours the prose names deadly become `}`;
  - blank neighbours the prose names passable (a dark room floor) become "virtual" cells.
- **`Chart.flood`.** It runs a reverse multi-source Dijkstra, re-elected every step.
  - Sources: a drawn `>`, or when there is none, every unvisited passable frontier cell.
  - Edges the ledger booked as failed are closed.
  - The cost of entering a cell grows with its visit count, so the field has no fixed 2-cycle.
- **`harness.GroundedHarness.decide_in_code` checks, in order:**
  1. modal reflex, which now answers `more` (`y` is not a MiniHack action) and books the edge that opened the prompt as failed;
  2. `down` when standing on a charted `>`;
  3. swing at a drawn letter whose prose is not terrain;
  4. field descent over open neighbours, excluding deadly, blocked and failed ones.

  Only when none of these fires does the model get its one call, and that call goes through the unchanged prompt, table and `arbitrate` path. Code-made moves return an `LLMResponse` whose reasoning is set to `[modal|down|swing|field]`.
- **Bug fixes from the diagnosis.**
  - `ledger.classify`: `dark area` is now labelled `unknown`, not `open`, and `corner` now counts as blocking.
  - `action_io.rank_open`: undrawn or unknown squares are ranked last.

## why:
The steer points at r027's structure: the field decides and the model breaks ties. The diagnosis for r047 blames a one-hop, visit-count arbiter with no global objective, which causes the R3/CB-Dark locality and orbits. It also names three code bugs:
- `dark area` labelled open caused 310 stone bumps, 61% of MazeWalk-15x15 steps;
- the dead `[yn]` reflex froze QE/04 for 69 steps;
- items read as creatures switched off orbit-break on Quest.

Frontier or stairs descent on a chart is the mechanism behind 0.375 on the chart lineage. It also cuts model calls, and with them the cost, on navigation tasks.

## files:
agent.py, tools/field.py (new), tools/ledger.py, tools/action_io.py
