# Diagnosis — minihack iter r055 / cand_r055_0_spatial-grounding (middleware pipeline + chart/BFS RouteStage)

**Headline:** primary = **0.275** (11/40, se 0.071). Survival 0.825 (7 deaths), step_efficiency 0.234, cost **234k tok/ep**.

| Task | Result | Who decided |
|---|---|---|
| MazeWalk-9x9 | 5/5 | route 31/31 steps, 0 tokens |
| MazeWalk-15x15 | 4/5 | route 164/164 steps, 0 tokens (ep03 hit the 100-step cap while still exploring) |
| Corridor-R3 | 2/5 | the two wins are 100% route (0 tokens); **all 3 losses start right after `The door opens.`** |
| CorridorBattle-Dark | 0/5, **5/5 died** | route charges east into the rat pack, then the model swings `east` until death |
| Boxoban-M/H | 0/10 | model 820/1000 steps: a random walk, ~510k tok/ep |
| Quest-Easy | 0/5 | model 496/500 steps: orbit, `quaff` ×69 (ep00), illegal `y` ×69 (ep04) |
| Quest-Medium | 0/5, 2 died | route to step ~20, then the model walks to the lava edge / into rats |

**The live score equals the proposer's null-model stub rig exactly** (0.275; MW 5/5 + 4/5, R3 2/5, all else 0). The VLM contributed **zero** wins and 100% of the cost.

**Binding constraint:** a **HARNESS-BUG** in `tools/chart.py:Chart.fold`. When a move opens a door, the chart books that edge as a wall. The chart also loses track of the door once it is redrawn as `-`/`|`. That cuts the only exit, so the route stage stops answering and the model orbits until the cap. This costs R3 **3/5**. The chart lineage (r051 `tools/field.py`) had already fixed both bugs; this rewrite dropped the fixes. The rest of the gap (Boxoban, Quest, CB-Dark = 25/40 episodes at 0) is the known **TASK-GAP**: no push planner and no item/lava mechanism.

## 1. What this harness does (mechanism)

`harness.py:PipelineHarness.act` builds a `StepContext` (`tools/context.py:build`: cursor with row = y−1, Dlvl, map rows, message-line combat flag, `<noun> adjacent <dirs>` prose). It then runs `pre` hooks outer→inner over `[Legal, Modal, Hazard, Route, Bump]`:

- **ModalStage.pre** answers `[yn` with `y` **only if `y` is in the legal set**, and `--More--` with `more`.
- **RouteStage.pre** folds the observation into `Chart` (drawn glyphs, visits, bump-closed edges, prose-noun doors/walls/lava). If a letter glyph is adjacent, it returns None (the model fights). Otherwise it returns `Chart.route()`'s BFS first hop toward a drawn `>`, else toward the nearest unvisited drawn square bordering undrawn space. If there is no route, it returns None.
- If no stage short-circuits, there is **one model call** on the naive transcript + AGENT.md + "Checks on this frame". The reply is snapped to the legal set, then the `post` hooks run: Bump (a refused/walled target → least-visited open neighbour), Hazard (never onto `}`), Legal (audit only).

`Chart.passable` treats `|`, `-`, space, `}`, `` ` `` and `0` as solid **unless the cell is in `self.doors`**, and `self.doors` is filled **only from adjacency prose containing "door"**. In `Chart.fold`, `if self.last[0] == ctx.pos and not ctx.combat: self.blocked.add(self.last)` books any no-move without a combat message as a closed edge.

## 2. What actually happened (causal chain)

### 2a. Corridor-R3: `The door opens.` seals the exit (3/3 losses)
Opening a closed door `+` costs a turn and **does not move the hero**. `Chart.fold` sees `pos` unchanged and no combat message, so it books `(pos, dir)` into `blocked`. The door is also redrawn as `-`/`|` (an open door), and `passable()` reads that as WALL unless the prose happened to name it.

- **R3/03** step 12: `route:frontier:1->west` → `The door opens.` → from step 13 onward the route never fires again (**87/100 steps are `model:*`**). The map shows the hero in a closed room `-.@..|` whose west door is the only exit. The model wanders the room (`south`, `southeast`, `bump-veto south->west`, …) until the cap.
- **R3/00** step 64: `route:frontier:1->east` at (12,17) → `The door opens.` → steps 65–99 are all model: `It's a wall.` ×4, `You can't move diagonally into an intact doorway.`, `bump-veto east->north`. The `>` is drawn 7 columns east (`....>|`), behind that door.
- **R3/01** step 55: the door on the `>` room's east wall opens (drawn `-` at map row 2, `|>....-#######`). The route keeps exploring the remaining frontier until step 77, then has no target, because the `>` is behind a `-` it treats as wall. From step 78 on it is the model at a dead-end corridor (`It's solid stone.`, `It's a wall.`).

Census: `door opens` appears in exactly 3 R3 episodes, and they are exactly the 3 losses. In each, the first `model:` row comes after the door event (13 / 65 / 78). The 2 wins never meet a closed door and use 0 model tokens.

### 2b. Quest-Easy/04: the `[yn]` prompt freezes 69 steps
Step 30 `southwest` → `Really attack the goblin? [yn] (n)`. `ModalStage` skips it because `y` is not in the legal set. The model then answers `y` (trail `illegal 'y'`), and BALROG replaces it with `north`. The prompt stays open and the frame is frozen for **69 steps**. The r047 §6 note already says: *neither `y` nor `esc` is in the vocabulary; only `more` escapes both prompt types.*

### 2c. Quest: the route exhausts, then the model orbits or dies
The lava column `}` partitions both Quest maps (QM: `|...}>..|`, the `>` is drawn behind lava). `route()` never reaches `>`, and every reachable frontier is visited within ~20 steps. From then on every step is a model call:
- QE/00: `quaff` ×69, each answered with `You don't have anything to drink.`
- QE/01: N/E/S/W circuits in the start room.
- QM/00 and QM/03: walked east into giant rats and died at step 36.

Nothing picks up the `(` horn/wand drawn in the start room (QM: `|.(.|`), and nothing applies it toward the lava.

### 2d. CorridorBattle-Dark: 5/5 deaths
The route walks straight east (frontier distance 1 every step) into the dark corridor. On contact (ep00 step 14, ep02 step 15, ep04 step 14) the model takes over and emits `east`/`southeast` every step while 2+ rats bite (`The giant rat bites! The giant rat bites!`). HP goes 16→0 in ~10 steps. That is ordinary CB-Dark dice (chain memory: price CB mechanisms at ≥200–500 eps), but 0/5 is at the low end: r051 went 1/5 and earlier lineages up to 4/5.

### 2e. Boxoban: the VLM random-walks
The map is fully drawn, so the frontier is empty. The route fires only for a handful of early steps, and then the model emits N/E/S/W roughly uniformly (W 122, E 117, S 95, N 90) at ~510k tokens per episode. It is 0/10, exactly as with the null stub.

## 3. Failure breakdown (29 losses)

| Category | Episodes | Cause | Fixable in code? |
|---|---|---|---|
| Door-open booked as wall / open door `-`/`|` read as wall | **3** (R3/00, 01, 03) | `Chart.fold` + `Chart.passable` | **yes, a ~10-line fix; r051 has it** |
| No push planner | 10 (Boxoban) | TASK-GAP | yes (an exact Sokoban search; chain memory: 10/10 live when the plan is ≤99 moves) |
| No item/lava mechanism; model orbit | 10 (Quest), incl. 69 frozen `[yn]` steps and 2 QM deaths | TASK-GAP + a ModalStage bug | partly (modal: trivial; Quest: pickup → apply/zap toward the lava) |
| Melee deaths | 5 (CB-Dark) | dice at n=5 | not measurable at n=5 |
| Exploration exceeds the 100-step cap | 1 (MW15/03) | long BFS backtracks (`frontier:12`, `:9`) | marginal; r051's field walker went 5/5 on MW15 |

## 4. Binding constraint — classification

**HARNESS-BUG (primary, cheap):** the door-open and open-door-glyph handling in `tools/chart.py`. It is worth up to +3 R3 episodes (+0.075), and it recovers the chart lineage's 4–5/5 on R3.

**TASK-GAP (structural, 20 episodes):** Boxoban and Quest have no code mechanism, and the VLM is inert there (the live score equals the null-model stub). Every model call on those tasks is pure cost: 234k tok/ep here vs 16–65k on the chart lineage.

This candidate is a **regression relative to the chart lineage** (r051/r054 = 0.375 with the same task coverage). It rebuilt the r036/r046 chart inside r018's pipeline and dropped two fixes the lineage had already learned. Structure is not the lever (r054 §6: stage rewiring replicated neutral three times).

## 5. Concrete changes to try (ranked by leverage)

1. **Fix the door handling in `tools/chart.py:Chart.fold` / `passable`.** This is a port of r051 `tools/field.py` (`DOOR_OPENS`, "a known door = ever drawn `+`").
   - Parse the message line in `context.build` (for example `ctx.msg`). In `fold`, if `re.search(r"door opens", ctx.msg, re.I)` and `self.last` is set, **do not** add `self.last` to `blocked`. Instead add `target(self.last)` to `self.doors`.
   - Before overwriting `self.drawn`, record every cell that was ever drawn `+` into `self.doors`. When a door later shows `-`/`|` it stays passable, orthogonally only (the existing `_doorish` diagonal ban already covers it).
   - Also on `This door is locked.`: do not book the edge as a wall; emit `kick` + the bearing (r051's rung: up to 4 tries, through ModalStage's `In what direction`).
   - Predicted: R3 2/5 → 4–5/5 (+0.05 to +0.075 primary), fewer R3 model calls (R3/03 alone spent 336k tokens), and no change on other tasks.
2. **ModalStage.pre: answer any open prompt with `more` when `y` is not legal.** Also answer `In what direction?` with the pending kick bearing. This removes the 69-step freeze in QE/04. Primary is likely unchanged, but it takes 69 wasted model calls out of the Quest residue and is a prerequisite for #4's `apply`/`zap` direction prompts.
3. **A Boxoban push planner as a RouteStage-level stage (`tools/sokoban.py`).**
   - Decode the premapped board: `0`/`` ` `` = boulder, `^` = target hole, `#` = bars/wall.
   - Run push-space A*/BFS with dead-square pruning on walls only, and emit the move sequence as a committed tape. Re-plan if the observed boulder set diverges from the plan's prediction.
   - Chain memory says an exact solver solves 10/10 live when the optimal plan is ≤99 moves; watch the phantom boulder under `@` at the start square.
   - Predicted: up to +10 episodes in theory. Realistically +3–6 under the 100-step cap. It also removes ~500k tok/ep.
4. **Quest: a code item/lava stage.** When the route is exhausted and a `}` column separates the hero from a drawn `>`:
   - Route to the drawn `(` item square, `pickup` (handle the pile/menu via ModalStage), then walk to a square cardinally adjacent to the lava, facing the `>`.
   - `apply`/`zap` the item and answer the direction prompt with that bearing. After `solidifies`, stop and re-route (the `}` squares should now draw as floor).
   - Chain memory's exhausted list: a single kick only (a double kick re-piles the items), never a kick toward lava (it burns the horn), and never zap after a dry route exists.
   - Predicted: QM/QE move off 0 (prior lineages reached QM 3/5). It also removes most of Quest's ~440k tok/ep.
5. **Stop calling the model on no-route residue frames.** On Quest and Boxoban the model is net-zero or harmful: `quaff` ×69, a uniform random walk, walking into rat packs. Until #3/#4 exist, a code fallback is as good and costs 0 tokens. On Quest-Medium, prefer `wait`/`search` at a corridor square away from the lava edge, not a march toward rats. Keep the model only for the adjacent-creature case. Predicted: cost/ep 234k → <60k, primary unchanged, survival on QM possibly better (2 of the 7 deaths here are QM rat walks).

## 6. Exhausted — do NOT repeat
- **Stage/pipeline rewiring by itself** (r026/r046/r054 and now r055): structure is neutral; only the mechanisms inside the stages move the score.
- **Frontier re-ranking or visit constants on MazeWalk/R3.** They are saturated at 10/10 and 4/5 in the chart lineage. The R3 losses here are the door bug, not the ranking.
- **Prompt/AGENT.md/prior wording** to get the VLM to push boulders, use items or break orbits. Here the live score equals the null-model stub, so the model adds nothing.
- **Advisory fact tables to the VLM** (r029, r047).
- **CB-Dark melee posture** (retreat, hold, recoil, HP gates) priced on 5 eps. It is dice; price it on ≥200–500 eps or leave it alone. Never make a creature a refusal/veto.
- **`y`/`esc` as prompt answers.** They are not in the vocabulary; use `more`.
- **Quest:** pickup without handling the kick-split, a double kick, a kick toward lava, or zapping after a dry route exists.
