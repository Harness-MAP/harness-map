# cand_r066_0_spatial-grounding — chart pilot rewired as a STAGE PIPELINE over tools/ modules

## module map
| file | role | lever |
|---|---|---|
| harness.py | `GroundedAgent`: builds a per-step context `c`, runs `STAGES = (prompt, orient, stairs, fight, quest, fetch, route, model)`; first stage to answer wins, rung -> `reasoning`. Loads tools/ via `importlib` under a per-candidate module name. The model is the LAST stage (no learner). | code |
| tools/frame.py | parser (`parse_glyph_map`, `clause_index`), `glyph`, `decide_action(reply, legal, allowed, banned)`; a cursor with y<0 now counts as an open prompt | code |
| tools/chart.py | `Chart`: drawn glyphs, prose-credited DOORS (passable, no diagonal edges) and BARS (impassable), bump ledger via `book()` ("The door opens." marks a door, not a bump), `bfs`/`route`/`toward` | code |
| tools/quest.py | `Quest` (ported from r046): inventory letters, pickup, walk to lava banks, `apply`/`zap` across `}`, answers letter / Improvise / direction prompts with key-bound vocab words | code |
| AGENT.md / rules.md / memory.md / skills/ | empty | prose |

## axis:
Code: structure/wiring (steered toward r041/r046). The parent r026 was a single monolithic `act()`.
It is now an ordered stage table over three tool modules, and the new structure has room for owners
the monolith had no slot for.

## change:
harness.py is split into tools/frame.py, tools/chart.py and tools/quest.py, plus an `act()` that walks
`STAGES`. The parent's behaviour is kept stage by stage: prompt reflex -> chart fold -> stairs -> swing -> route -> model.
The restructure also repairs the diagnosis's HARNESS-BUGs at their owning modules:
(A) doors: prose `doorway`/`open door` and "The door opens." make a square a passable door, and BFS forbids diagonal door edges;
(B) prompts: `y`/`esc` are not in the vocab, so a `[yn]` attack marks the foe peaceful and emits `more` (capped at 3), and the `(x,-1)` cursor is treated as a prompt, not fed to the router;
(C) prose `bars adjacent` marks those squares impassable.
The model fallback is projected onto compass moves + `search` only, so it can no longer open item prompts (`quaff`/`pickup` lockups).
New `quest`+`fetch` stages sit before `route`, gated on `pickup` legal + a charted `}`.

## why:
diagnosis r026: R3 3/3 losses are open doors charted as WALL; 2 QE lockups come from unanswerable `esc`/`y` prompts and model-emitted item verbs; and the Quest pair has no item rung. The steer says the productive difference in r041/r046 is the staged/tools wiring, which makes each mechanism a separable owner. A stub room test confirmed route -> `>` -> `down`.

## files:
harness.py (rewired), tools/__init__.py, tools/frame.py, tools/chart.py, tools/quest.py (ported from r046), NOTES.md
