# cand_r065_0_spatial-grounding — stage-table chart walker + a separable Quest regime module (steer: r058/r052)

## module map
| file | role | lever |
|---|---|---|
| agent.py | hub `GroundedAgent`; `act()` walks `STAGES = (prompt, orient, stairs, fight, cross, hold, kit, fetch, route, probe, model)`; first answer wins, rung -> `reasoning` | code |
| tools/frame.py | observation parser (+ `lang`, `modal` fields now), glyph tables, whole-word reply projection | code |
| tools/chart.py | per-level map memory, regime-aware walls, door inference, BFS route/probe; now also exposes `terrain` + `edge(a,d)` (the view tools/quest.py reads) | code |
| tools/quest.py | **NEW** (ported from r058, = r052 lineage): silent-pickup pile detection, one safe kick, horn-first pickup + letter filler, apply/zap across a cardinal lava run, typed prompt answers, mouth hold, charted `toward` walk | code |
| AGENT.md / rules.md / memory.md / skills | empty | prose |

## axis:
Code — structure/wiring (search-space steer toward r058/r052): add the missing regime OWNER as a separable tool module wired into the stage table as four new stages, plus Quest-aware prompt answering and a VLM-free fallback on Quest levels.

## change:
Copied r058's `tools/quest.py` and adapted it to this lineage's `tools/chart.Chart` by adding `Chart.terrain` (the drawn map) and `Chart.edge(a,d)` (blocked-edge / passability / no-diagonal-through-door rule, identical to `bfs`). `frame.parse_glyph_map` now also returns `lang` (prose block, for the hostile check) and `modal`. In agent.py: `Quest` is loaded via `_load`, cleared with the chart on reset and on every Dlvl change; `c["quest"] = {pickup,kick,zap} <= legal` (true only on the Quest pair); `st_orient` calls `quest.observe` and computes `c["kind"]` (stairs/frontier/None from `chart.route`); new stages `st_cross`/`st_hold`/`st_kit`/`st_fetch` between `fight` and `route` delegate to the module (verbs leave `self.last=None` so they are never booked as bumps; quest walks set `last` so a bump still closes the edge); `st_prompt` asks `quest.answer` (direction / item letter / Improvise) BEFORE the generic `more`, which fixes the chain-B quaff⇄`more` cancel loop on Quest; and on an armed Quest level `st_model` never calls the VLM — it takes the least-visited open edge (`quest:wander`). Every non-Quest task (MazeWalk, R3, CB-Dark, Boxoban) runs exactly the parent's stage sequence (verified: a synthetic non-Quest frame takes `chart:route-frontier` then the model, as before).

## why:
Diagnosis r054: 20/25 losses are Boxoban + Quest, a TASK-GAP — on Quest the VLM decides 50–100% of steps and emits 0 pickup/apply/zap/kick; `more` cancels item prompts (quaff 2-cycle). r052/r058 (the steer exemplars) show this Quest module on a chart walker takes Quest 0/10 -> 9–10/10. Synthetic check: spawn `pickup` (silent -> pile) -> walk to a kick post whose ray avoids `}` -> `kick` -> walk to the horn. Expected: +8–10 Quest episodes (primary ~0.375 -> ~0.60), lower Quest token bill (no VLM on Quest), other tasks unchanged.

## files:
agent.py, tools/quest.py (new), tools/chart.py (`terrain`, `edge`), tools/frame.py (`lang`, `modal`), NOTES.md
