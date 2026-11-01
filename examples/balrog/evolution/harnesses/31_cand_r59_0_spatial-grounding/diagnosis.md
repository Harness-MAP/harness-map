# Diagnosis — minihack iter r059 / cand_r059_0_spatial-grounding (stage table + chart.py door squares + quest.py + `door` stage)

**Headline:** primary = **0.600** (24/40). Survival is 36/40, and all 4 deaths are CB-Dark. Tokens are spent only on Boxoban (~99 `model:consult` calls per episode); every other task is harness-decided at 0 tokens.

| Task | Wins | Note |
|---|---|---|
| MazeWalk-9x9 / 15x15 | 5/5 + 5/5 | field walker, at ceiling |
| Quest-Medium | 5/5 | quest.py, at ceiling |
| Quest-Easy | 4/5 | run_04: letter trap (horn=`f`, wand=`g`, both untypeable) |
| Corridor-R3 | 3/5 | run_01 **reached `>`** but was scored 0 (final reward 0.99, see §2c). run_03: drawn `>` never aimed at |
| CorridorBattle-Dark | 1/5 | open-room giant-rat melee (dice, see §6) |
| **Boxoban-Hard / Medium** | **0/5 + 0/5** | no push planner; the VLM walks into bars 28/100 steps |

**Binding constraint:** Boxoban is 10 of the 16 losses, and it is a HARNESS-BUG (missing mechanism). No stage plans pushes, and the VLM fallback is incapable of doing so. The in-chain solver `prop_r008/.../tools/sokoban.py` solves **10/10 of this eval's Boxoban frames offline**: plans of 36–79 moves in ≤0.29 s, re-verified this iter from each episode's first frame. r058's diagnosis made the same call, and r059 did not act on it. That is the whole gap to ~0.8.

This candidate is r058 (25/40) re-expressed, not a regression. Of the two tasks that dropped, R3 lost one episode to a scoring artifact, and Quest-Easy lost one to the known letter trap. The new `door` stage fired **0 times** in 40 episodes (no `door:*` rung in any CSV), so it is inert on this split.

## 1. What this harness does (mechanism)

`harness.py:GradientAgent.act` parses the frame (`tools/field.py:parse_tty_frame`). It then runs `STAGES = (prompt, orient, stairs, swing, door, cross, hold, kit, fetch, descend, consult)`, and the first stage that answers is emitted with its rung in `reasoning`.

- **`orient`**:
  - `Chart.fold` records drawn glyphs, door squares (from `+`, prose `doorway`/`open door`, or "The door opens."), bumps into `blocked`, and locked doors into `locked`.
  - Lethal directions are vetoed: prose lava/water and adjacent `}`.
  - `Chart.field(pos)` does a BFS potential over the **reachable** region. It is seeded on a reachable `>` if one exists, otherwise on reachable frontier squares (unvisited, with an undrawn neighbour).
- **`stairs` / `swing`**: reflexes (`down`; attack an adjacent letter that isn't known peaceful).
- **`door`**: kicks a locked door booked by "This door is locked." (`KICK_CAP=8`). It is armed only if `kick` is legal.
- **`cross` / `hold` / `kit` / `fetch`**: `tools/quest.py`, armed only when `{pickup,kick,zap} ⊆ legal` (the Quest pair).
- **`descend`**: takes the argmin of `field[cell] + 0.25*visits`, **ties broken by `STEP` dict order** (N, E, S, W, NE, SE, SW, NW).
- **`consult`**: on a flat field, asks the VLM (priors + a least-trodden menu). On Quest it takes `menu[0]` without a model call.

## 2. What actually happened (causal chains)

### 2a. Boxoban 0/10: no owner for the task
- Rungs are `model:consult` on 96–99 of 100 steps in every episode. The field fires on 1–4 steps and then goes flat, because the level is fully drawn and lit, so no frontier exists.
- Medium run_00 message census:
  - "You cannot pass through the bars." 28×
  - "You try to move the boulder, but in vain." 6×
  - "With great effort you move the boulder." 4×
- Negative-reward (frozen, no-time) steps number 96–99 of 100 per episode: the model is bumping walls and bars almost continuously.
- The frame is exact Sokoban: `#` is bars, `` ` `` is a boulder, `{` is a fountain, and the hero is an Archeologist on a 4-cardinal action set. `sokoban.solve(cells, pos)` returns plans of **79/73/56/50/48** (Hard) and **38/54/61/46/36** (Medium). All fit the 99 steps left after step 0. The VLM has placed 0 boxes across r041–r059.

### 2b. Corridor-R3 run_03 (timeout, 100 steps, all `field:frontier`): drawn `>` never aimed at
- Step 21: the frame first draws `>` at (8,7) in the row `|....>.    #@#`. The hero is at (15,7). The `>` is 7 columns west, and the only gap is the undrawn dark corridor x=10..13.
- Because that gap is undrawn, `Chart.reach` does not contain the `>`, so `field()` falls back to **frontier** seeding. Frontier squares (14,7) to the west and (16,7) to the east are both 1 hop away.
- The tie is broken by `STEP` order, **east before west**, so step 22 goes east.
- The walker then explores the whole eastern complex: 60 steps south and east, then "The door resists!" at (35,14) on step 79. It turns back and times out at (21,9), still ~13 squares from `>`.
- A frontier ranking that knows where the drawn-but-unreachable `>` is would have taken the west frontier at step 22 and arrived in ~8 steps. This is exactly the memory's "goal-attracted seek phase" (R3 4/5→5/5, +2/−0 over 125 levels).

### 2c. Corridor-R3 run_01: reached the goal, scored 0 (env artifact, not a harness bug)
- Steps 36–42 go `west` through the opened door onto `>`.
- The final step's reward is **0.99**, not 1.0. MiniHack `_reward_fn` returns `reward + time_penalty`, and `penalty_step=-0.01` applies when the game clock does not advance.
- The Samurai is intrinsically Fast. In this trace, 10 of 43 steps are frozen (steps 38 and 40 alternate −0.01).
- BALROG `BaseProgress.update` sets `progression = 1.0 iff reward >= 1.0` on the last step, so a goal reached on a free move scores 0.
- The turn counter is not in the observation (no `T:` on the status line), so the harness cannot time the arrival. Treat this as an irreducible ~20–30% tax on Fast heroes' wins. Do not chase it.

### 2d. Quest-Easy run_04: letter trap, plus a peaceful-goblin bump loop
- The spawn pile is kicked (steps 2–3). The horn is picked up at step 8 as **`f`** and the wand at step 16 as **`g`**.
- Neither letter is typeable: `LETTER_VERB` has no `f` or `g`, and there is no `drop`. So `Quest.usable()` is None, `cross` never fires, and `consult` takes `quest:menu` for the remaining 83 steps.
- No filler object lay on the level before the pickups, so `kit`'s filler detour could not help. `next_letter()` was also None at pickup time, because the silent spawn pickup taught nothing about the starting kit (a–e).
- A secondary waste of 30 steps: `quest:menu` repeatedly steps north into a peaceful goblin. That triggers "Really attack the goblin? [yn]" → `more` (answers the prompt) → `more` again ("Unknown command '^M'"), a 3-step cycle ×9.
  - `self.peaceful` is only read by `st_swing`, and `st_consult`'s Quest menu ignores it.
  - Harmless here (the episode was already lost), but the same loop could eat a winnable episode's budget.
- Winning Quest episodes landed their items on `j`/`k`, `i`/`j`, `e`/`f` or `h`, so this is the r052 residue (~1/10).

### 2e. CB-Dark 1/5
- All 4 deaths are `reflex:swing` streaks at (34–38, 11–12), the mouth of the open room, against 3–5 giant rats. HP runs 16→1 over 8–14 swings, e.g. run_00: "You miss the giant rat" ×4 at HP 3.
- The single win (run_04) walked through at HP 3–4 without contact.
- Same pattern as r058. Per chain memory, this is dice at n=5.

## 3. Failure breakdown (16 losses)

| Category | Count | Class |
|---|---|---|
| Boxoban: no push planner, VLM random-walks into bars | 10 | HARNESS-BUG (missing mechanism) |
| CB-Dark melee deaths | 4 | dice / VLM-free, closed axis |
| R3 drawn-but-unreachable `>` not attracted | 1 | HARNESS-BUG (frontier tie-break) |
| R3 goal reached on a frozen step (0.99) | 1 | env artifact, unfixable from observation |
| Quest-Easy letter trap (`f`/`g`) | 1 | TASK-GAP residue |

## 4. Binding constraint

**HARNESS-BUG: Boxoban has no owner.** It is 10/16 losses and 0.25 of the scalar. The fix is a pre-computed plan (a scaffold), not prompting. The VLM is at its ceiling on multi-step push planning, and 19 iterations of consults have produced 0 boxes. The solver already exists in-chain and is verified offline on these exact frames. Expected gain is ~+7 wins (10 solvable × ~0.75 after the Fast-hero frozen-step tax), taking primary from **0.600 to ~0.78**.

## 5. Concrete changes (ranked by leverage)

1. **Port `prop_r008/cand_r008_0_spatial-grounding/tools/sokoban.py` as `tools/sokoban.py` and add a `push` stage.** This is a new `st_push` in `STAGES` right after `orient` (before `stairs`/`swing`).
   - **Gate:** `c["dirs"]` is exactly the 4 cardinals (Boxoban has no diagonals; every other task offers 8), and the current frame draws at least one `` ` `` and one `{`. This is observation-only; it reads no task name.
   - **Cells:** build them from **the current frame's rows**, not `Chart.terrain`. `terrain` keeps a stale `` ` `` on a square the hero stepped onto after a push, which is the "phantom boulder under `@`" trap: 0/10 with a correct solver.
     - For the hero's own square, use `{` only if the chart once saw a fountain there (the hero may spawn on one); otherwise use `.`.
   - **Replanning:** re-solve every step (≤0.3 s) and emit `plan[0]` with rung `push:<len>`. Re-solving makes a bump or a no-move self-correcting with no plan-invalidation bookkeeping.
   - **Fallback:** if `solve` returns `[]`, fall through to the existing stages.
   - **Predicted effect:** Boxoban 0/10 → ~7–8/10 (frozen-step tax on the final push), primary +0.17–0.20, and Boxoban tokens → 0. It touches no other task because the 4-cardinal gate is false elsewhere; verify that `c["dirs"]` has 8 entries on MazeWalk/R3/CB/Quest in one stub run.
   - **Cell note:** this lineage is placed by cvt_v3_k16, and one extra `tools/` module will move the descriptor. Accept it; +7 wins dominates any cell bonus.

2. **`tools/chart.py:Chart.field`: attract the frontier toward a drawn-but-unreachable `>`.** When no `>` is in `region` but some `>` is in `terrain`:
   - Seed the frontier squares with an initial potential of `k * cheb(frontier, nearest_drawn_>)` (k≈0.5), instead of 0.
   - Or, minimally, change `st_descend`'s tie-break from `STEP` order to distance-to-`>` in the level frame.
   - Gate it off when `}` is drawn, so Quest keeps its own owners.
   - **Predicted effect:** R3 run_03-type levels finish in ~10 steps instead of timing out (memory: +2/−0 over 125 R3 levels, 0 arms changed off R3). MazeWalk has no unreachable-`>` state, so it is unaffected.

3. **`harness.py:st_consult` (Quest menu) + `st_prompt`: stop the peaceful bump loop.**
   - On "Really attack … [yn]", book `(pos, dir)` into `chart.blocked` using `chart.pending`.
   - Answer the prompt with `southeast` (= `n`, a single keystroke) instead of `more`, which costs a second wasted `more` on "Unknown command '^M'".
   - Saves ~3 steps per bump. It only matters for efficiency and budget on Quest; low leverage.

4. **(Optional, ~1/10 Quest-Easy) Letter-trap pre-emption in `tools/quest.py:kit`.**
   - The first picked-up object's letter reveals the kit size. If that letter is dead (`f`/`g`) and the partner tool would also land dead, the only recovery with no `drop` is burning letters on corpses or fillers *before* the second tool.
   - Low expected value. Do this only after 1–2.

## 6. Exhausted — do NOT repeat
- **Consulting or prompting the VLM on Boxoban** (menus, priors, tie-break prose): 0 boxes placed across r041–r059.
- **CB-Dark melee / retreat / hold / HP-gated disengage:** priced as dice rerolls in chain memory (recoil +9/−10, +8/−11 over 100 eps). Leave `st_swing` alone; 5 episodes cannot price a CB change.
- **Stage-table rewiring alone:** r041/r046/r051/r056/r059 rewires reproduce the same vector ±1 episode. Only new mechanisms move the score.
- **Restructuring `tools/quest.py`:** it is at 9–10/10. Touch only the letter-trap residue.
- **Chasing the 0.99 frozen-step artifact:** the clock is not in the observation.
- **The `door`/kick stage as a score lever on this split:** it fired 0/40 episodes. Keep it (it is harmless), but don't expect R3 gains from it.
- **Boxoban "unwinnable from drawn count" pruning:** refuted. Prune dead corners on walls only (the r008 solver already does this).
