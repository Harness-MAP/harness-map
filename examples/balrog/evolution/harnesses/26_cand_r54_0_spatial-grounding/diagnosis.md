# Diagnosis — minihack iter r054 / cand_r054_0_spatial-grounding (stage pipeline over tools/frame.py + tools/chart.py)

**Headline:** primary = **0.375 (15/40)**, survival 0.850, step_eff 0.273, cost **214.8k tok/ep**.
Per task: MazeWalk-9x9 5/5 · MazeWalk-15x15 5/5 · Corridor-R3 4/5 · CB-Dark 1/5 · **Boxoban-M/H 0/10 · Quest-E/M 0/10**.
This is the same per-task vector as r046/r048/r051/r056/r066/r069/r072. The cleanup of prompts, doors and bars changed survival (0.775→0.850) and did not move one win.

**Binding constraint:** a **TASK-GAP**. The 20 Boxoban+Quest episodes have **no owner stage**, so they fall to `st_model`, which decides from free text with no solver. That covers 100% of Boxoban steps, 81% of Quest-Easy steps and 40% of Quest-Medium steps. Both gaps have a known mechanism that has already been measured in this run:
- `tools/sokoban.py`: r064, Boxoban 0→10/10.
- `tools/quest.py`: r052/r058, Quest 0→9–10/10.

Neither is present in this candidate.

Provenance check: the trace rungs (`chart:route-frontier`, `reflex:prompt-more`, `chart:probe`) are this candidate's rung names, so `$TRACE_DIR` IS this code. The judge's "blank Reasoning" note just reflects the model-stage rows (see §5.4).

## 1. What this harness does (mechanism)
`orchestrator.py:GroundedAgent.act` parses the frame with `tools/frame.py:parse_glyph_map`, then runs `STAGES = (prompt, orient, stairs, fight, route, probe, model)`. The first non-None answer is emitted.
- `st_prompt`: any open prompt (`cursor.y==0`) → `more`, cap 3.
- `st_orient`: folds the frame into `Chart`, books bumps, and marks doors from prose and from `door opens`.
- `st_stairs`: `down` when standing on `>`.
- `st_fight`: moves into any adjacent letter glyph.
- `st_route`: `Chart.route` BFS. It goes to a *reachable* `>`, else to the nearest unstood frontier.
- `st_probe`: steps into an undrawn orthogonal neighbour.
- `st_model`: the VLM with the naive instruction, projected whole-word onto the legal set, with a least-visited-bearing fallback.

`Chart.set_regime` turns `#` and `` ` `` into walls when the vocabulary is exactly the 4 cardinals (Boxoban). AGENT.md, rules.md, memory.md and skills are all empty.

## 2. What actually happened (causal chains, with evidence)

**A. Boxoban (10/10 lost, 6.0M of the 8.6M token bill): the model owns every step.** Every Boxoban CSV row has an empty `Reasoning` (1000/1000 = model stage). The cause:
- `set_regime` makes `#` and boulders walls.
- The lit Sokoban room has no `>` and no undrawn neighbour, so `route` returns `None, None` and `probe` returns `None`.
- `st_model` is then called on each of 100 steps with the full ~6k-token history.

The actions come out uniform: N 123 / S 137 / E 123 / W 117 on Hard. That is a random walk. The model cannot plan irreversible pushes. The episode return is −0.099 on 8/10 and +0.001 on 2/10, and no episode placed a boulder. Boxoban is an exact Sokoban (memory: push model 2746/2746), so this is a solver-owned task, not a VLM task.

**B. Quest-Easy (5/5 lost): two failure modes, both "no Quest owner".**
- **The quaff↔more 2-cycle (runs 00 and 04, ~86 of 100 steps each).** In run_00 the hero kills the fox and grid bugs (`reflex:swing`, steps 0–6). Then:
  1. It steps onto a potion, and the model emits `pickup` and then `quaff`.
  2. The next frame is `What do you want to drink? [f or ?*]`.
  3. `st_prompt` answers `more`, which gives `Never mind.`.
  4. Route and probe are None, because the left bank is fully drawn and sealed by the `}` column at x=30, so `st_model` runs again and says `quaff`.

  This repeats from step 12 to step 99: 44× `quaff` and 43× `more`, costing −0.01 each (return −0.89). The harness never learns that a verb whose prompt it cancelled is useless.
- **Model orbit (runs 01–03).** Frame 0 of run_01 reads `horn adjacent west` with `(` at (26,9) and `}` covering x=30 on every row. `route` has no frontier, so the model wanders E/W/N/S for 100 steps (27/25/23/22). It never picks up the horn, kicks the pile or zaps.

The mechanism that wins these tasks is known (memory `balrog-minihack-quest-is-provably-unwinnable…`, REFUTED section):
1. Kick the spawn pile once.
2. Pick up the horn first, then the wand.
3. Fire `zap`/`apply` across the lava using the letter-verb read from `[.. or ?*]`.
4. Answer `Improvise?` with `northwest` (y).
5. Answer the direction prompt with the bearing.

This harness's blanket `more` answer cancels every one of those prompts. The generic `st_prompt` is actively incompatible with the Quest mechanism.

**C. Quest-Medium (5/5 lost, 2 deaths).**
- `>` is charted behind the `}` column but is unreachable. `route` falls to the frontier, then to `probe`: 140 probe steps, including `north`/`south` into `It's solid stone.` 7× each on run_00. Then the model takes over.
- Runs 01 and 04 die meleeing giant rats in the open room: `reflex:swing` into `southeast` and `The giant rat bites!`. There is no mouth-hold.

**D. Corridor-R3 run_03 (the one loss): the drawn-unreachable `>`.**
- By step 60, `>` is drawn at (8,6) in the west room (`|....>.`), but the gap at x≈10–13 is undrawn, so `route` finds no reachable `>`. It picks the *nearest* frontier, which drags the hero east into the SE room (x=35–43, step 97), away from the goal.
- The episode times out at 100 with `chart:route-frontier` owning all of the last 60 steps.

This is exactly the r064 §5.1 bug that r068 fixed with the stairs-attracted frontier key.

**E. CB-Dark (1/5): deaths at steps 27–34, all in `reflex:swing` exchanges with giant rats.** Chain memory classifies CB-Dark as dice (a 500-episode census is needed), so it is not the lever.

## 3. Failure breakdown (25 losses)
| category | eps | owner stage at failure | class |
|---|---|---|---|
| Boxoban: no solver, model random-walks | 10 | `st_model` 100% | TASK-GAP |
| Quest: no acquire/kick/zap owner; `more` cancels item prompts; quaff↔more 2-cycle (QE 00/04) | 10 | `st_model` / `st_prompt` / `st_probe` | TASK-GAP + HARNESS-BUG (cycle) |
| R3: drawn-unreachable `>`, nearest-frontier drifts away | 1 | `st_route` | HARNESS-BUG |
| CB-Dark melee deaths | 4 | `st_fight` | dice |

Cost: 100% of tokens come from Boxoban (≈60 calls/ep × 6k tok) and Quest. The chart tasks run at 0 tokens.

## 4. Binding constraint
**TASK-GAP**, and it is fixable in the harness, since the mechanisms already exist as drop-in `tools/` modules in this run. The frozen VLM cannot solve Sokoban or discover a 5-step item chain, and the prose is empty, so re-wording will not help. The fix must hand those decisions to deterministic owner stages. The remaining R3 loss is a small HARNESS-BUG in `Chart.route`'s frontier key.

## 5. Concrete changes (ranked by leverage)
1. **Add a `sokoban` stage right after `orient`.** Port `prop_r064/.../tools/sokoban.py` (`Sokoban.read/solve/simulate`: exact push-level A*, walls-only dead-square pruning, plan cache with expect-and-replan).
   - Arm it on `set(moves)==CARDINALS` plus a drawn `` ` `` and `{`.
   - Emit rung `sokoban:plan:<n>`.
   - Predicted: Boxoban 0→~9–10/10 (+0.225 primary), and cost −6M tok per eval (≈−150k tok/ep), because the model is never asked on Boxoban.
2. **Add a Quest regime owner** by porting `prop_r052/.../tools/quest.py` as stages `cross, hold, kit, fetch` between `fight` and `route`.
   - Arm on `{pickup, kick, zap} ⊆ legal`.
   - `st_prompt` must first delegate to `Quest.answer`, a typed handler:
     - `[x or ?*]` → the verb whose keystroke is the letter;
     - `Improvise?` → `northwest`;
     - `In what direction?` → the lava bearing.

     Only fall back to `more` otherwise. The current blanket `more` is what makes the item chain impossible.
   - Keep: horn-first pickup, a filler object when the next letter is untypeable, one kick from a post whose ray hits a wall before any `}`, no firing once `>` is reachable, and the ≤2-lane mouth hold while a foe is named (the QM rat deaths).
   - Predicted: Quest 0→~9/10 (+0.225), survival ↑ on QM.
3. **Stairs-attracted frontier in `tools/chart.py:Chart.route`** (r068's fix).
   - When a `>` is drawn but not in `seen`, and no `}` is charted, pick the frontier minimising `(dist + chebyshev(f, nearest >), dist, y, x)` instead of the first in BFS order.
   - It is a total order, so re-electing it each step cannot orbit.
   - Predicted: R3 4→5/5 (fixes run_03), with 0 change elsewhere.
4. **Cheap hygiene:**
   - (a) In `st_model`, when the previous emitted non-move verb produced a prompt that `st_prompt` cancelled (`Never mind.`), ban that verb for the level. This kills the 86-step quaff↔more cycle even before the Quest module lands.
   - (b) Restrict model passthrough to movement verbs plus `search`/`open`/`kick`.
   - (c) Write rung `model` into `reasoning` (`out._replace(reasoning="model:"+raw[:40])`) so model-owned rows are attributable. Right now they are blank, which is what confused the judge.
   - (d) `st_probe` into `It's solid stone.` should book the *glyph* as stone. Better: skip probe when the prose names no dark room, since QM wastes ~14 probe bumps per episode.

Expected with changes 1 to 3: ≈0.875–0.90. That matches r064 (0.875) and r068/r074 (0.90), which carry exactly these modules.

## 6. Exhausted — do NOT repeat
- **Re-shaping the chart walker** (stage splits, door inference, bar walls, prompt `more`, module count or shape). This is the 8th candidate at the identical vector [0,0,1,1,0.8,0.2,0,0] (r046, r048, r051, r054, r056, r066, r069, r072). Structural refactors of the walker alone are inert on primary.
- **Prose / AGENT.md tweaks for Boxoban or Quest.** The VLM random-walks Boxoban and cannot discover the item chain.
- **CB-Dark melee axis** (retreat, recoil, hold variants). Chain memory says it is dice below 500 episodes.
- **`pray` on Quest** (refuted), and **zapping rats** (spends the charge the lava needs).
