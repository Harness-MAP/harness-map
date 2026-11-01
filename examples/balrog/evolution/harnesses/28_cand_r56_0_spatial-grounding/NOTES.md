# cand_r056_0_spatial-grounding — one-module chart pilot rewired as an ordered STAGE PIPELINE

## module map
| file | role | lever |
|---|---|---|
| orchestrator.py | `GroundedAgent`: `act()` builds a per-step context `c` and walks `STAGES = (prompt, orient, stairs, fight, route, model)`; each `st_*` abstains (None) or answers `(word, rung, kind)`; `emit()` books `self.last` and writes the rung to `reasoning`. Chart = `terrain`/`visits`/`blocked`/`doors` + BFS `route`. | code |
| AGENT.md / rules.md / memory.md / skills/ | empty | prose |

## axis:
Code: structure/wiring (steered toward the r041/r046 region, which scored 0.375 against this parent's 0.325). The monolithic `act()` ladder becomes an ordered stage table on a DOOR-AWARE chart. That is the shared structural feature of both exemplars. Everything stays in one module, and no tools/ were added.

## change:
- **`act()` is now a stage table.** `act()` = parse → context `c` → first answering stage in `STAGES` → `emit()`.
- **`prompt` is the first stage.** It answers an attack `[yn]` with `northwest` (the `y` key; `y` is not in the vocab). It answers every other open prompt, and any frame whose cursor sits on the message line (`pos.y<0`), with `more`, which cancels. `esc` is kept only as the last fallback. The parent's `y` and `esc` answers were never legal, so they never fired.
- **`orient` folds the frame into the chart.**
  - A square drawn `+` becomes a door, and it stays passable when later redrawn `-`/`|`.
  - Prose `doorway`/`open door`/`broken door` also mark doors, as does the bump target after `The door opens.`; that move is no longer booked as a blocked edge.
  - The BFS forbids diagonals into or out of doors.
- **`stairs`** also fires on the `staircase down here` message.
- **`fight` and `route`** are unchanged.
- **`model`** projects the reply onto movement words plus `down`, so the fallback model can no longer open item prompts (`quaff`/`pickup` absorbed 90 steps each in the parent).

## why:
The parent diagnosis names three HARNESS-BUGs:
- **(B)** an opened door is charted as wall. This caused all 3 Corridor-R3 losses; r041/r046 carry the door set and reach R3 0.8.
- **(A)** the modal reflexes are dead because `y`/`esc` are not in the vocabulary. This caused 2 Quest-Easy absorptions of ~90 steps.
- **(A')** the fallback model emits item verbs.

The stage table is the steer's wiring. It also makes each rung a separable, attributable owner that later iterations can slot new stages into (a fight-post hold, a Quest opener, a Boxoban solver) without touching the others.

Checked by a stub smoke test:
- a route across a `+` door;
- `northwest` on an attack `[yn]`;
- the door-opens frame keeps the `-` square passable;
- a `What do you want to drink?` prompt with the cursor on the message line gets `more`.

## files:
orchestrator.py, NOTES.md
