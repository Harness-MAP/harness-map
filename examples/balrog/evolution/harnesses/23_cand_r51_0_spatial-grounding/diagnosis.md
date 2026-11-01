# Diagnosis — minihack iter r051 / cand_r051_0_spatial-grounding (prose-grounded potential-field walker)

**Headline:** primary = **0.375** (15/40, se 0.077). Progression 0.35, survival 0.775, step_efficiency 0.272, cost **64.5k tok/ep**. The cost is down from 207k on r048 and 279k on r049.

| Task | Result |
|---|---|
| MazeWalk-9x9 | 5/5 |
| MazeWalk-15x15 | 5/5 |
| Corridor-R3 | 4/5 |
| CorridorBattle-Dark | 1/5 (4 deaths) |
| Boxoban-Medium | 0/5 |
| Boxoban-Hard | 0/5 |
| Quest-Easy | 0/5 (all hit the 100-step cap) |
| Quest-Medium | 0/5 (**all 5 died**, at steps 22–28) |

This is the **same per-task vector as r048** (0.375), and within noise of r049 and r050 (0.325). Navigation is saturated.

**Binding constraint:** a TASK-GAP. The 20 Boxoban and Quest episodes (half the eval) score 0 because the harness has **no mechanism for them**:
- There is no push planner.
- There is no item pickup/apply layer to cross the lava.

On every such frame the field is either goal-less (Boxoban) or flat (Quest, where the `>` sits behind `}`). The harness then wanders, or hands the turn to a model that walks into rat packs. The chain's diagnoses have prescribed both mechanisms for four rounds (r048, r049 and r050 §5), and they have still not been built.

## 1. What this harness does (mechanism)

`harness.py:FieldHarness.act` is a rung ladder, and the first rung that fires wins. Its name is written to the CSV `Reasoning` column.

1. **modal:** answers `--More--`, `[yn` or `In what direction` with the pending kick bearing, else `more`.
2. **down:** fires when standing on `>`.
3. **kick:** kicks an adjacent cardinal door booked as locked, up to 4 tries.
4. **swing:** moves into any adjacent letter glyph.
5. **descend:** takes the argmin of `tools/field.py:Chart.flood()`, a reverse Dijkstra seeded with a drawn `>` at 0 and every unstood frontier square at `FRONTIER_OFFSET=25`, with visit penalties. Only passable squares count: prose nouns and bumps convict squares and edges, and door squares are diagonal-free.
6. **consult:** one VLM call on the transcript plus the NEIGHBOUR TABLE plus the open-move menu, projected whole-word onto the legal set.

Rung census over all 40 episodes (2204 steps):

| Task | descend | consult | swing | modal |
|---|---|---|---|---|
| Boxoban-H | 500 | 0 | 0 | 0 |
| Boxoban-M | 432 | 68 | 0 | 0 |
| R3 | 276 | 0 | 0 | 0 |
| CB-Dark | 107 | 0 | 54 | 0 |
| MW15 | 155 | 0 | 0 | 0 |
| MW9 | 31 | 0 | 0 | 0 |
| Quest-Easy | 6 | **437** | 25 | 32 |
| Quest-Medium | 65 | 18 | 41 | 0 |

- **All 15 wins are model-free** (MW, R3 and CB-Dark are 100% code rungs).
- **~96% of the token bill** comes from two sources. Quest-Easy's 437 consults take ~2.03M of 2.58M total tokens (≈430k/ep). The single Boxoban-M/04 episode that fell to consult took 467k.

## 2. What actually happened (causal chains)

### A. Quest-Medium: 5/5 deaths, all by the same trajectory
This is the most important finding. All 5 episodes emit the **identical** opening, driven purely by `descend`: `e n e e s s e e e e e e n`.
- The hero starts next to the horn: `horn adjacent west`, and the `(` is at map (22,11).
- It walks the corridor east and steps from the doorway (32,12) into the lit rat room.
- The room is fully drawn, so there is no frontier, and the `>` at (53,10) sits behind the `}` column. The field goes **flat**, and at steps 13–16 the turn falls to `consult`.
- The model steps further east into the open (QM/00 rows 13–16). The pack of 5–6 `r` surrounds the hero at (35,12), where 3 or more rats have lanes onto the hero.
- `swing` then fires until HP 16 → 0. The last frames read HP 5, 4, 2, 4 and 2 of 16.

**The fatal decision is the harness's descend+consult into an open room with no goal reachable.** It is not the melee.
- The horn that solves the task (apply → freeze the lava, which also kills rats along the ray) was adjacent at step 0 and is never picked up.
- `pickup`/`apply` appear nowhere in the ladder.

### B. Quest-Easy: 437 consults, a flat field orbiting a sealed room
QE/00 shows the whole pattern:
1. Steps 0–8 are swings (fox, grid bugs).
2. From step 9 to step 99, every step is `consult` inside the spawn room (x 25–29, y 8–12). The map is fully lit, and the `}` column at x=30 separates it from the exit.
3. The model random-walks, for example `e e sw e w sw e e ne n sw …`. It re-enters the potion square (27,10) repeatedly (`You see here a sky blue potion.`).
4. The horn `(` at (26,9) sits in plain view for 90 steps.

`Chart.frontier()` is empty and the `>` is unreachable, so `flood()` returns no value for any neighbour, `scored` is empty, and `consult` fires every step at ~4.3k tokens per call.

### C. Corridor-R3/03: the one R3 loss, a drawn `>` ignored by the field
- By row 77 the map shows the `>` at (8,7) inside `|....>.`. It is separated from the charted corridor at (14,7) by 4 blank (unknown) squares.
- `flood()` seeds that `>` at 0, but it is disconnected, so it contributes nothing. **Every frontier is seeded at the same 25**, so the walker takes the nearest frontier, which is east.
- It spends steps 40–98 exploring the east rooms: a door at (35,13) `The door resists!`, and a room at x 35–43.
- It only turns back west at step 91 and runs out at the 100-step cap at (19,8). That is roughly 10 squares from the `>`.
- A secondary bug: `The door resists!` (row 77) falls into `Chart.fold`'s `else: blocked.add(pend)` branch. The door edge is convicted like a wall. It cost nothing here, but it is a misclassified no-move.

### D. Boxoban: 0/10, 932 `descend` steps with no goal
- Boxoban has no `>`. The field is pure frontier-seeking, and `` ` `` (a boulder) is not in `SOLID`, so the walker treats boulders as floor.
- It pushes them incidentally: 3–10 `With great effort you move the boulder.` per episode and 2–8 `in vain`. Those pushes deadlock the puzzle, and the episode idles to the cap.
- Boxoban-M/04 fell through to `consult` for 68 steps (467k tokens) once the frontier was exhausted.

### E. CorridorBattle-Dark: 1/5, dice
CB/00 and CB/01 are identical `descend` walks east down the corridor (steps 0–18). The swing rung then fights at (34/35,12) until death: 10 and 6 swings, with HP ending around 0. The chain has priced this task as dice (≥200 episodes are needed to resolve any melee lever), and nothing in the traces contradicts that.

## 3. Failure breakdown (25 losses)

| Category | Episodes | Mechanism |
|---|---|---|
| Boxoban: no push planner | 10 | goal-less field + boulders treated as floor → deadlocked pushes |
| Quest-Medium: walk into rat room | 5 (deaths) | `descend`/`consult` enter the room when the field is flat; no item layer |
| Quest-Easy: sealed-room orbit | 5 (timeouts) | empty frontier, unreachable `>` → 90 consults per episode |
| CB-Dark melee deaths | 4 | dice at n=5 |
| R3: drawn `>` not attracting | 1 | equal frontier seeding in `Chart.flood` |

Of the 25 losses, 20 are the missing Sokoban and Quest mechanisms.

## 4. Binding constraint

**TASK-GAP**, with a harness-side cause:
- The agent operates legally. It has 0 parse failures, and modal prompts are all handled.
- It lacks a goal-directed lever for two of the four task families.
- It is not a VLM ceiling: the model is only consulted where the harness has no plan, and it is harmful there, walking into rats in QM and wandering in QE.
- Everything that wins is code, so the fix is more code rungs, not prose.

## 5. Concrete changes (ranked by leverage)

### 1. `tools/quest.py` plus a `quest` rung placed **before `descend`** (QE+QM: up to +10 eps; cost −79%; QM deaths → 0)
Working reference: `balrog/balrog_ours/runs/adaptive_k8_c1/quest_opener_k8c1.py` (229 lines, with `exposure`, `bfs`, `bearing` and the kick/pickup/apply sequence). Chain memory records it at 5/5 QM and 2/5 QE on CRN seeds.

- **Gate:** a `}` is in `chart.terrain`, and the `>` is either not drawn or not in the connected region of the hero's square. Compute that region as a BFS over `chart.passable` from `pos`.
  - This gate is exactly the QE/QM regime, and it never fires on MW, R3 or CB (no `}`), so the 15 current wins are untouched.
- **Acquire:**
  - Walk (via chart BFS) to the item glyph (`(`, `/`, `!`, `?`) nearest the spawn, and emit `pickup`.
  - Parse `x - <desc>.` into `inv`.
  - A silent pickup means a pile: step off, then **kick once, not toward lava**, and pick up the scattered items. Both a double kick and a kick toward lava are refuted.
- **Use:**
  - At a lava bank, `apply` the horn or `zap` the wand along the cardinal with the **shortest `}` run**.
  - Answer the `What do you want to …?` item-letter prompt and the `In what direction?` prompt in this rung, before the generic `modal:more`. The modal rung currently answers them with `more`/the kick bearing.
- **Anti-death:** while the quest gate is live, **ban `descend` and `consult` from entering any square with ≥2 open neighbours when a letter glyph is drawn within 4 squares.**
  - The hero should stay in the spawn room and corridor, where the horn is.
  - This alone removes all 5 QM deaths (A): the fatal steps 12–16 all happen after the quest gate would be live.
- **Cost:** the 437 QE consults become code steps, which saves about 2.0M of the 2.58M-token bill.

### 2. `tools/sokoban.py` plus a `board` rung before `descend` (Boxoban: up to +10 eps)
Working reference: `runs/adaptive_k16_crn/candidates/prop_r014/cand_r014_0_spatial-grounding/tools/sokoban.py` (`push_plan(hero, rocks, sinks, floor, moves, seconds)`, 78 lines). There is also `runs/adaptive_k2_crn/candidates/prop_r017/*/tools/sokoban.py`.

- **Gate:** a `` ` `` and a `{` are drawn, and no `>` is drawn.
- **Decode:** `#`, `-` and `|` are walls (Boxoban `#` is bars). `` ` `` is a rock and `{` is a sink.
- **Plan:** cardinals only, g counts every move, and dead squares are pruned on walls only.
- **Replay:** replay the tape, and re-plan on `in vain` or when the boulder set differs from the prediction.
- **Guard in `field.py:Chart.passable`:** treat a `` ` `` square as solid when the board gate is live, so `descend` never makes incidental pushes (D).

Memory: 19/20 on the stub rig and 10/10 live; it wins iff the optimal plan is ≤ ~99 moves.

### 3. `tools/field.py:Chart.flood`: attract the frontier to a drawn, disconnected `>` (R3 +1, ~+0.025)
- After seeding, if some `>` is drawn but the flood from the `>` seeds does not reach `pos`, seed each frontier square `c` at `FRONTIER_OFFSET + max(|cx-gx|, |cy-gy|)`, where g is the nearest drawn `>`. Break ties in the level frame.
- **Gate:** no `}` charted, so Quest is left to change #1.
- In R3/03 this sends the walker to the frontier (14,7), which is 4 blanks from the `>` at (8,7), instead of the 50-step eastern detour. Memory prices this lever at +2/−0 over 125 R3 levels.

### 4. `tools/field.py:Chart.fold`: classify door no-moves (small; R3 robustness)
Extend the `pend` branch: `door resists`, `door is stuck` and `WHAMM` must **not** add to `blocked`. Book the target as a door and allow a retry, or a kick if `kick` is legal. Only an unexplained no-move convicts the edge (R3/03 row 77).

### 5. `harness.py` consult rung: never pay for a consult on a flat, creature-free field (cost)
Once #1 and #2 land, `consult` should almost never fire. As a backstop:
- When the field is flat and no letter glyph is drawn, emit the least-visited open neighbour (`rank_open(...)[0]`) in code, or `search` at a dead end.
- Keep the model only for frames code cannot parse.
- This removes the Boxoban-M/04 467k-token episode class. Primary is unchanged: 0 of the 15 wins used the model.

## 6. Exhausted — do NOT repeat
- **Navigation re-ranking or rewiring** (the frontier offset, visit-penalty constants, cardinal tie-breaks). MW is 10/10 and R3 is 4/5 on r036, r041, r044, r046, r048 and r051, so the result replicates. Only #3 (`>` attraction) is still open.
- **Prompt, prior or menu wording** (AGENT.md, memory.md, NEIGHBOUR TABLE text). The model decides only on losing residue frames, and it is net harmful there. Prose cannot move primary.
- **CB-Dark melee posture** (retreat, hold, recoil or HP gates) priced on 5 episodes. It is dice; price it on ≥200 episodes or leave it alone.
- **Treating creatures as a veto.** A creature is a ranking term, never a refusal. The QM anti-death rule in #1 is a *region* gate while the quest gate is live, not a creature veto on the routed path.
- **Bars-as-floor to cut Boxoban cost.** It is 0/10 either way; cost should fall through the planner.
- **Quest pickup without pile detection, a double kick, or a kick toward lava.** All are refuted in chain memory.
