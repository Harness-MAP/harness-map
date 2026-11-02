# Diagnosis — minihack iter r060 / cand_r060_0_spatial-grounding (r033 grounded walker + `tools/chart.py` BFS route)

**Headline:** primary = **0.350** (14/40). Survival is 0.80: 8 deaths, 4 on CB-Dark and 4 on Quest-Medium. Cost is **236k tokens per episode**, and 94% of that is Boxoban plus Quest-Easy.

| Task | Wins | Who decided the moves |
|---|---|---|
| MazeWalk-9x9 / 15x15 | 5/5 + 5/5 | `route:*` on 100% of steps, 0 tokens |
| Corridor-R3 | 3/5 | route; the 2 losses hand over to the `model` for 24 and 82 steps |
| CorridorBattle-Dark | 1/5 | route plus `reflex:swing`, 0 tokens (dice, see §6) |
| Quest-Easy | 0/5 | `model` on 90–95 of 100 steps, ~450k tok/ep |
| Quest-Medium | 0/5 | 4 deaths by giant rat at steps 22–31; 1 timeout |
| Boxoban-Hard / Medium | 0/5 + 0/5 | `model`/`grounded` on 99/100 steps, ~620k tok/ep |

**Binding constraint:** the chart+route lever works: MazeWalk is 10/10 at 0 tokens, up from the r033 parent's 0.175. But this candidate was built on the **r033 lineage**, which lacks the mechanisms the chain's elite (r058: 25/40, r059: 24/40) already has:
- a Quest owner (`tools/quest.py`), which r058 used to go Quest 10/10;
- a push planner (`prop_r008/.../tools/sokoban.py`, which solves 10/10 of these Boxoban frames offline).

On top of that, `tools/chart.py` has a real **HARNESS-BUG**: an open door drawn `-`/`|` is impassable until the hero has stood on it. That strands the route on 2/5 R3 levels.

Classification: **TASK-GAP** (Quest and Boxoban have no owner) plus **HARNESS-BUG** (door squares).

## 1. What this harness does (mechanism)

`agent.py:GroundedAgent.act`:
1. `grid.parse_glyph_map` decodes the hero square, its 8 neighbours, the lava named in the prose, and foes. A foe is a letter on the current frame.
2. `learn` records `(pos, dir)` in `refused` when a move does not change the position. On "The door opens." the harness only **discards** that refused edge.
3. `chart.merge` folds the `map:` rows into a per-level terrain dict and adds `pos` to `stood`.
4. `own` tries these in order: `--More--`→`more`; attack `[yn`→`y`; `down` on a charted `>`; **swing** at any adjacent letter (first in `ORDER`); `chart.route`. `route` is a BFS to a charted `>`, else to the nearest *never-stood passable square with an uncharted neighbour*.
5. If `own` returns nothing, it builds the full prompt: priors (AGENT/rules/skills) plus the grounded ring plus the `memory.md` trail. It calls the VLM, then `vet` swaps out lava, wall, stone, refused or stale moves.

`chart.passable` is true when a square is in `stood`, or when its glyph is in `PASSABLE` and is not `}`. `PASSABLE` contains no `-` or `|`. That missing pair is the door bug.

## 2. What actually happened (causal chains)

### 2a. Corridor-R3 run_03 and run_01: the route dies at an open door, and the VLM burns out the clock
I replayed `chart.merge`/`route` offline over the CSV frames.
- **R3/03, step 12:** `west` → "The door opens." The door at (19,3) is now drawn `-` in the west wall, and the hero stands at (20,3). The room is fully charted. The corridor squares east have all been `stood`, and a stood square is never a frontier. `route` returns `(None, None)` from step 13 on. The door the hero just opened is the only exit, and it is not passable because its glyph is `-` and it was never stood on. The model then owns **82 steps** of north/south pacing inside the room, and the episode times out at 357k tokens.
- **R3/01, step 76:** the chart draws `|>....-#######`. That is the `>` room, whose east door is an **open door `-`** with charted floor `.` on its west side and a stood corridor `#` on its east side. The stairs are 12 squares away and charted, but the BFS cannot cross the `-`, so `route` returns None. The model owns the remaining **24 steps**, walks east, and times out.
- The r060 "door opens → discard refused edge" fix is therefore incomplete: it un-refuses the edge, but the chart still rejects the target square.

### 2b. Quest-Easy 0/5: nothing owns the lava crossing
- QE run_00: the lava column `}` at x=29 splits the level. The west room (~5×5) is charted by step 3. After that, `route` has no frontier: every uncharted square lies across `}`. It returns None on **90–95 of 100 steps in all 5 episodes**.
- The model is given a movement-only ring and the "reach `>`" priors. It paces (`north` 24–33×, `southwest` 13–17×) and never emits `pickup`/`apply`/`zap`. The horn and wand are drawn in the room: step 30 shows "horn adjacent north, sky blue potion adjacent northeast".
- Each episode costs ~450k tokens and scores 0.
- r058's `quest.py` (pickup spawn → kick the pile → pickup item → zap/apply toward the lava → walk) won this task 5/5.

### 2c. Quest-Medium 4/5 deaths: the frontier route walks into the rat room
- QM run_01, steps 0–13: `route:frontier` walks the corridor to the room mouth at (33,12). Steps 14–15 step *inside* to (35,12), in a 25-wide open room holding 6 `r`.
- From step 16, `reflex:swing` owns every step. Up to 3 rats are adjacent (E/NE/SE). HP goes 16→12→10→5→3→1, and the hero dies at step 30.
- Runs 02/03/04 follow the identical script: 14 route steps, then 6–9 swings, then death at steps 22–25.
- r058's `quest:hold` (hold the corridor mouth, where only one lane is exposed) owned 91/387 QM steps and won 5/5.

### 2d. CorridorBattle-Dark 1/5: the same walk-in, but this is the known dice task
- CB/00, step 20: `route:frontier` steps past the doorway to `|.@r`, with 3 rats adjacent. `reflex:swing` then trades blows until death at step 34.
- All 4 deaths show this shape. Chain memory prices CB-Dark as dice: every hold/retreat/fight-post arm is ±noise over 100–500 episodes. r058/r059 were also 1/5.

### 2e. Boxoban 0/10: VLM residue, the entire token bill
- `route` fires once, then there is no frontier: the level is lit and fully drawn.
- `model` owns 55–88 moves and `grounded` 8–35. The directions come out uniform (north/south/east/west ≈ 25 each), which is a random walk.
- Each episode costs ~620k tokens, so Boxoban alone is 6.2M of the 9.46M total.

## 3. Failure breakdown (26 losses)

| Cause | Losses | Class |
|---|---|---|
| Boxoban: no push planner, VLM random walk | 10 | TASK-GAP (a known solver exists in-chain) |
| Quest-Easy: no pickup/zap owner, route exhausted at the lava | 5 | TASK-GAP (quest.py exists in-chain) |
| Quest-Medium: route walks into the open rat room, swing to death (4); 1 timeout | 5 | TASK-GAP (quest:hold exists in-chain) |
| CB-Dark: walk-in plus melee dice | 4 | dice; closed axis |
| R3: open door `-` impassable in `chart.passable`, then the VLM paces | 2 | **HARNESS-BUG** |

The VLM owns 1696 steps; the judge counted 538 frontier, 55 stairs and 113 swing steps. Those model-owned steps produced **zero** wins: every one of the 14 wins is route-owned. The model is not scaffolding anything. It only converts a stuck state into a timeout at ~4.5k tokens per step.

## 4. Binding constraint

**TASK-GAP (20 of 26 losses), plus one HARNESS-BUG (2 losses).**
- The navigation core is at ceiling on MazeWalk and nearly so on R3.
- Quest and Boxoban need *verbs* (`pickup`, `kick`, `zap`, push sequences) that no owner emits, and the frozen VLM will not discover them from a movement-grounded prompt. This was measured over r041–r059: 0 boxes placed via the VLM.
- The door bug is a pure code error in `chart.passable`/`merge`.

## 5. Concrete changes (ranked by leverage)

### 1. Transplant the Quest owner
Source: `prop_r058/cand_r058_0_spatial-grounding/tools/quest.py`, or r059's copy.
- **Where:** call it in `agent.py:GroundedAgent.own` **before** the swing reflex and `route`. Arm it only when `{"pickup","kick","zap"} ⊆ legal`: that is observation-only and true only on the Quest pair.
- **What to bring:** its `cross`, `hold`, `kit` and `fetch` stages, with their rung names written to `reasoning`.
- **`hold` must pre-empt `reflex:swing` and `route:frontier`.** The QM deaths come from walking *into* the room (§2c), not from swinging.
- **Predicted:** Quest 0/10 → 9–10/10, primary **+0.22–0.25**. Quest-Easy tokens fall from ~450k to ~0.
- **Cell cost:** a second tool module moves the cvt_v3_k16 descriptor away from cell 8. Measure it, but +9 wins dominates any cell bonus.

### 2. Port the Boxoban solver
Copy `prop_r008/cand_r008_0_spatial-grounding/tools/sokoban.py` as `tools/sokoban.py`.
- **Gate:** the legal bearings are exactly the 4 cardinals, and the current frame draws at least one `` ` `` and one `{`.
- **Solve from the current frame's rows, not from `chart["terrain"]`.** The chart keeps a stale `` ` `` under the hero after a push, which is the "phantom boulder" trap (0/10 with a correct solver).
- **Replan every step** (≤0.3 s) and emit `plan[0]` as `push:<len>`. If the solver returns an empty plan, fall through.
- Put this before `route` in `own`.
- **Predicted:** Boxoban 0/10 → 7–10/10, primary **+0.17–0.25**. Cost drops by ~6.2M tokens per eval, so cost per episode goes from 236k to under 60k.

### 3. Fix open doors in `tools/chart.py`
This is cheap, in-cell, and recovers the 2 R3 losses.
- **(a) Opened doors.** In `agent.py:act`, on "The door opens." compute `door = before_pos + BEARINGS[prev_action]` and add it to a new `chart["doors"]` set. `merge` must keep `doors` across frames and reset it on a `Dlvl` change. `passable` returns True for `c in chart["doors"]`.
- **(b) Already-open doors** (R3/01 case). In `passable`, a `-` or `|` counts as a door when its two orthogonal neighbours on one axis are both charted-passable, e.g. `.` on the west and `#` on the east. Treat it as passable.
- **(c) No diagonal steps into or out of a door square.** Door squares here mean `+`, `chart["doors"]`, and the inferred doors from (b). In `route`, skip a diagonal `d` when either `cur` or `nxt` is one of them. NetHack refuses these moves, and memory shows doorless prose `doorway` allows diagonals, so exclude those from the rule.
- **Predicted:** R3/01 routes 12 squares to `>`, and R3/03 walks out the opened west door. R3 goes 3/5 → 4–5/5, primary **+0.025–0.05**. MazeWalk has no doors, so it is unaffected.

### 4. Cut the useless residue
With 1–3 in place, the residue is small. Until then, every `route=None` step costs ~4.5k tokens for a random walk.
- Make `own` fall back to `vet(None, …)`'s least-visited ranking, at 0 tokens, whenever the frame draws no `>`. Consult the model only on Boxoban, if the solver fails.
- Also let a **stood** square with uncharted non-stone neighbours count as a frontier for a bounded `search` (a dead-end corridor, as in R3/03 before the door opened). Key the budget on the probed cell, per memory.
- **Predicted:** cost −30–50% with primary unchanged. That helps the `cost` tiebreak.

## 6. Exhausted — do NOT repeat
- **CB-Dark melee/hold/retreat/HP-gated disengage and fight-post variants:** the chain prices them as dice (recoil +9/−10 and +8/−11 over 100 episodes). Leave `reflex:swing` alone on CB. The Quest-Medium walk-in is handled by transplanting `quest:hold` (gated on the Quest verbs), not by a generic CB change.
- **Prompting/priors/menus for the VLM on Boxoban or Quest:** 0 boxes and 0 Quest verbs across r041–r060, and 1696 model steps here yielded 0 wins.
- **Stage-table rewiring without new mechanisms:** r041/r046/r051/r056/r059 reproduce the same vector ±1 episode.
- **Grounded-ring prose wording (AGENT.md register):** it only moves the cvt cell, not the score. r033 got 0.175 across 18 wording variants.
- **Boxoban "unwinnable from drawn count" pruning:** refuted. Prune dead corners on walls only.
