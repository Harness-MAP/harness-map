# cand_r064_0_spatial-grounding — tools/ loader + a Boxoban `sokoban` stage owner (exact push-level A*)

## module map
| file | role | lever |
|---|---|---|
| harness.py | `LearnerAgent` hub; `STAGES = (prompt, orient, sokoban, stairs, swing, cross, hold, kit, fetch, route, learner)`; first stage to answer wins, rung -> `reasoning`. `load_tool(name)` imports any `tools/<name>.py` under a path-unique `sys.modules` key. Chart (`observe/passable/frontier/route`) + TD learner. | code |
| tools/quest.py | Quest-pair owner (unchanged, byte-identical): kick-split, pickup, zap/apply across lava, hold | code |
| tools/sokoban.py | NEW `Sokoban` owner: `read` frame -> floor/boulders/goals, `solve` push-level A* (cost = moves, h = sum of wall-only pull distances, dead squares pruned on walls only), `simulate`, per-level plan cache with expect-and-replan | code |
| AGENT.md | one tie-break sentence for the learner consult | prose |

## axis:
Code, structure/wiring (steered toward r046's stage table over separate tools/ modules): a generic
tools loader and a new task-mechanism module that owns its own stage slot, placed right after `orient`.

## change:
Replaced the quest-only import with `load_tool()` and added `tools/sokoban.py` as a second stage
owner. `st_sokoban` arms only when the legal moves are exactly the 4 cardinals AND the frame draws
a boulder `` ` `` and a fountain `{` (Boxoban only; every other task offers diagonals, so it is inert
there). Boulders come from the CURRENT frame and goals are ACCUMULATED across frames. If fewer goals
than boulders are drawn, the never-seen start square is added as a candidate goal. The hero square
is floor, never a boulder. The solver runs once (A* w=1 for 4 s, then w=3 for 6 s), caches the move
list, and pops one move per step via `commit()` with rung `sokoban:plan:<left>`. If the observed
(hero, boulders) != the expected state, or the goal set changes, it replans from this frame. A
failed solve is cached per state and falls through to `route`. Hygiene: once the prose names `bars`
(or "through the bars"), `passable()` treats `#` as a wall, so the fallback router stops bumping bars.

## why:
Diagnosis r058: Boxoban is 10/15 of the remaining losses (0/10). Nothing plans pushes, and `#` was
routed as floor (20–46% bumps). Offline check on the 10 real r058 Boxoban frames: all 10 solve in
<=0.54 s with valid simulated plans of 38–81 moves (all <100 steps). That includes Medium run_04,
which draws 3 `{` and solves with the start square as the 4th goal. Expected result: Boxoban
0/10 -> up to 10/10 at 0 tokens, with the other 30 episodes byte-identical (the stage never arms,
and `bars` never appears in their prose).

## files:
harness.py (loader, `sokoban` stage, `bars` flag in observe/passable), tools/sokoban.py (new), NOTES.md
