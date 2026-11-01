# cand_r058_0_spatial-grounding — charted learner as a STAGE table + a Quest-regime tool module

## module map
| file | role | lever |
|---|---|---|
| orchestrator.py | hub `LearnerAgent`. `act()` builds one context `c` and runs `STAGES = (prompt, orient, stairs, swing, cross, hold, kit, fetch, route, learner)`. The first stage that answers is emitted, and its rung goes to `reasoning`. `ChartView` exposes the learner's chart (`drawn`, `blocked`, `doors`) to tools as `terrain` / `passable` / `edge`. `goal_kind()` reports `stairs` / `frontier` / None. | code |
| tools/quest.py | **NEW** (ported from r052). The `Quest` regime: silent-pickup pile detection, one safe kick, horn-first pickup plus a letter filler, apply/zap across a cardinal lava run, typed prompt answers (`In what direction?`, `[j or ?*]`, `Improvise?`), a mouth hold, and a charted `toward` walk. Loaded under a path-unique module name. | code |
| AGENT.md | one-sentence tie-break prose for the learner consult (unchanged) | prose |
| rules.md, memory.md | empty | prose |
| DESIGN.txt | the lineage's r035/r041 notes (history) | – |

## axis:
Code, structure and wiring, steered toward r052/r044. The monolithic `act()` becomes an ordered stage table over a `tools/` module (r052's shape). I also added the missing regime owner, a separable `tools/quest.py` wired in as four stages (`cross`, `hold`, `kit`, `fetch`) between swing and route.

## change:
- **`act()` is now a stage table.** Each `st_*` returns a reply or None.
  - `prompt`: a Quest-armed level asks `Quest.answer` first, then the parent's `more` (at most 2) / `esc`.
  - `orient`: runs the parent's observe, credit and deadly steps, then `Quest.observe` and `goal_kind`.
  - `stairs`, `swing`, `route` and `learner` are the parent's code unchanged.
- **Quest stages are armed only when `{pickup, kick, zap} ⊆ legal`.** That holds for the Quest pair and for no other task, so MazeWalk, R3, CB-Dark and Boxoban run the parent's logic in the parent's order.
- **Verbs never read as bumps.** Non-movement verbs are emitted through `verb()`, which clears `pending`, so the TD learner and the edge ledger do not book them as bumps. Quest walks go through `commit()`, so a bumped quest step still closes its edge.
- **No VLM tie-break consult on an armed Quest level** (least-tried argmax instead). This cuts the QE learner-orbit token bill.
- **Dead reflex dropped:** the `y` reflex, since `y` is not in the MiniHack vocabulary.

## why:
The r041 diagnosis names a TASK-GAP. The 10 Quest losses need a verb plan: pick up the horn or wand, then fire it across the lava. The harness had no stage able to emit it (0 `pickup`/`apply`/`zap`/`kick` in 4000 steps). The steer exemplar r052 shows that a separable Quest module on a chart walker moves Quest 0/10 → 9/10 (primary 0.375 → 0.600).

This candidate transplants that module onto the learner lineage through a chart adapter instead of re-deriving the chart. Expected: roughly +8–9 Quest episodes, with the other 30 episodes unchanged.

Checked on a synthetic frame:
- With the Quest verbs offered, it goes spawn `pickup` → walk to the kick post → `kick` → walk to the bank → `apply`.
- With only move verbs offered, it takes `chart:route` exactly as the parent does.

## files:
orchestrator.py (stage-table rewire, ChartView, goal_kind, Quest stages, `lang`/`modal` in the frame parser), tools/quest.py (new), tools/__init__.py (new, empty), NOTES.md
