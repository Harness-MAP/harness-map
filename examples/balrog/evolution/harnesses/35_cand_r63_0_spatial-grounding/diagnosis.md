# Diagnosis — minihack iter r063 / cand_r063_0_spatial-grounding (r053 menu + r027-style potential field)

**Headline:** primary = **0.325 ± 0.074** (13/40). Survival 0.775, step_efficiency 0.257, cost **76k tok/ep** (parent r053 had 308k).
- Per task: MazeWalk-9x9 5/5, MazeWalk-15x15 4/5, Corridor-R3 4/5.
- Zero on Boxoban-H/M (0/10), CB-Dark (0/5, all deaths), Quest-Easy (0/5, time-outs) and Quest-Medium (0/5; 4 deaths, 1 time-out).
- Against the parent r053 (0.225): **+4 episodes** (R3 +3, MW15 +1), and cost fell 4×. The field graft worked where it was aimed.
- It is still **strictly dominated by r058** (0.625 at 8k tok/ep; Quest 10/10, R3 4/5, MW 10/10).

**Binding constraint:** 20 of the 27 losses come from two things:
- the Quest pair (10 episodes), which needs a verb plan this harness cannot emit;
- Boxoban (10 episodes), where the field treats bars `#` as floor and boulders as walls, so it ping-pongs the hero between two squares for 100 steps.

Inside what the field does cover, a **HARNESS-BUG** cost one R3 win: `The door opens.` is booked as a failed edge (R3 run_01). **Classification: TASK-GAP** (Quest verbs, Boxoban pushes) **+ HARNESS-BUG** (door-open booking, a field with no fallback when `>` is unreachable).

## 1. What this harness does (mechanism)
`orchestrator.py:MenuHarness.act` runs, in order:
1. **Reflexes:**
   - `Really attack? [yn]` → `y` (only if `y` is legal, which it is not on Quest; see §2E);
   - `--More--` → `more`;
   - standing on a charted `>` → `down`.
2. **`Ledger.observe`** parses the cursor (row = y−1), `Dlvl`, the `map:` rows and the adjacent-noun prose. `deadly` = lava/water; `blocked` = wall/bars/dark area.
   - **It books `(square, dir)` into `failed` whenever the cursor did not move, the target glyph was not a letter, and there was no combat text** (`orchestrator.py:61-65`).
3. **`Ledger.chart` + `tools/field.py:flood`** folds the map into `terrain`, then runs a reverse BFS. The sources are:
   - **every drawn `>`**;
   - **only if no `>` is drawn**, every unvisited passable cell that touches an undrawn cell.

   Edge cost is `1 + 0.5·min(visits,6)`, `failed` edges are cut, and `SOLID = " }`0"`. `#` counts as passable.
4. **`Ledger.menu`** ranks moves by `(tier >/letter/other, f is None, f + 0.25·visits, wallish, just_left, visits, index)`.
5. **Emission:**
   - If option 1 is `>`, a creature, or has a finite field value, the step is emitted by code (`field:descend` / `reflex:swing` / `reflex:stairs`, 0 tokens).
   - Otherwise it makes **one model call** with the menu, and `extract_action` projects the reply onto the legal set with the veto set.

No path emits any non-compass verb (`pickup`, `kick`, `apply`, `zap`, `open`, `search`).

Rung census over all 40 episodes: `field:descend` 1501 rows, `reflex:swing` 100, `reflex:stairs` 13, model 679. **603 of the 679 model rows (89%) are on the Quest pair.**

## 2. What actually happened (causal chain, with evidence)

**A. The field works on MazeWalk/R3 (the targeted gain).**
- MW9 5/5 in 2–18 steps and MW15 4/5, all fully code-driven at 0 tokens.
- R3 run_02: the `>` that r053 walked away from is now routed to in 31 steps.
- R3 runs 00, 02, 03 and 04 are wins (73, 31, 36 and 36 steps).

**B. HARNESS-BUG: an opened door is booked as a bump and deletes the route to a drawn `>` (R3 run_01, the only R3 loss).**
- Steps 0–53: the field walks the hero to (20,3), next to `vertical closed door adjacent west`.
- Step 53: it emits `west`.
- Step 54: the obs reads `The door opens.`, the cursor is still at (20,3), and **`>` is now drawn at (14,2)**: `|>....-@###`, 6 hops away, all floor.
- `Ledger.observe` sees `last[0]==pos`, `last_glyph='+'` (not a letter), and no COMBAT text, so it runs `failed.add(((20,2),'west'))`.
- That edge is the only way into the stairs room, so `flood` cuts it. The only seed is the `>`, so the hero's neighbours are no longer in the field.
- Every option now has `f is None`, so each step falls through to the model: **45 model calls, 220k tokens, 0 progress.** The model walks east back into the explored room and times out at 100 steps.
- The chain already knew this: r058's `observe` has `if "door opens" in low: self.doors.add(target)`, and the memory note says "`the door opens.` must CLEAR a record".
- Census of no-move messages on R3: `The door opens.` ×3, `You can't move diagonally out of an intact doorway` ×2, `The door resists!` ×1. All six are booked as permanent `failed` edges.

**C. When a `>` is drawn but unreachable, the field has NO frontier fallback.**
- `flood` seeds from the frontier *only if no `>` is drawn*. A drawn-but-unreachable `>` therefore makes the field empty around the hero, and control goes to the model.
- **Quest-Medium:** in all 5 episodes, `>` is drawn behind the lava column at step 6.
  - After 6 `field:descend` steps, every step is a model call.
  - The model marches `east` into the room with the rat pack. run_00: steps 6–14 go east, the first bite comes at step 15 (HP 16 → 10), and the hero dies at step 19 after 4 swings.
  - Runs 01, 02 and 04 die at 23–25 steps the same way.
- R3 run_01 after §B is the same mechanism.

**D. Boxoban: the field is wrong about both obstacles, so it produces a deterministic 2-cycle at 0 tokens.**
- Boxoban is premapped. `#` is **bars** (the prose says `bars adjacent …` on every frame), and boulders are drawn `` ` ``.
- `field.passable` treats `#` as floor and `` ` `` as SOLID, so:
  - the border `#` cells, which touch undrawn blanks, become frontier seeds;
  - **every boulder push has `f is None` and is ranked below every finite option**, so a push is never emitted.
- The prose `bars` veto stops the actual bump, so there are **0 no-moves**, only oscillation.
  - Boxoban-Hard run_03 and Medium run_01: 3 unique squares, `west`/`east` 50/50, **97/100 steps are A→B→A reversals**. In Hard run_03 the hero shuttles between (41,9) and the `<` at (42,9) all episode.
  - Across the 10 Boxoban episodes there are 17–97 field-driven reversals each, against ≤5 on every other task.
- Result: 0/10 at almost 0 cost. The model was not the issue (r053 spent 6M tok here for the same 0/10).

**E. Quest-Easy: sealed start room, flat field, 100 model calls per episode, no verb.**
- The hero spawns in a 5-wide room walled by a lava column at x=30, with `horn adjacent south` at step 0.
- No `>` is drawn. The only frontier cells are across the lava, so no neighbour is in the field and **every step (93–100/100) is a model call**: about 450k tok/ep, 2.24M total, **74% of the whole eval's token bill**.
- The model orbits the 26 room squares.
- `Really attack the goblin? [yn] (n)` appears 4 times. `y` is not in the legal set, so the reflex never fires. The prompt answers `n` and the frame freezes. r058 maps `y` to `northwest`.

**F. CB-Dark 0/5 (all deaths, 22–31 steps, 0 tokens).** The field walks east down the dark corridor, and the swing reflex fights 7–15 times per episode (kills 1–3). This matches the parent (0/5). The chain's census classifies CB-Dark as dice (r058 1/5, r061 2/5 at n=5).

**G. MazeWalk-15 run_03 (100 steps, 62 unique squares, `>` never drawn).**
- Because MazeWalk walls are undrawn blanks, the frontier "unvisited cell touching an undrawn cell" means *every unvisited floor cell*, including already-seen corridor interiors.
- Steps 76–87 walk the fully drawn x=32 column end to end. The visit-weighted cost also causes goal re-election reversals: steps 61–63 go (35,10)→(36,10)→(35,10), and steps 96–98 do the same.
- The unexplored east branches (x≥40, rows 8–11) are still pending at time-out.
- r058 uses the same frontier definition but pure BFS distance with break-on-nearest, and scores MW15 5/5. The visit weight is the only difference, and memory says a re-chosen-every-step objective must be a distance-ascending total order.

## 3. Failure breakdown (27 failed episodes)
| Category | Eps | Mechanism | Class |
|---|---|---|---|
| Boxoban: bars-as-floor field 2-cycle, pushes never ranked | 10 | §2D | TASK-GAP + field bug |
| Quest-Easy: sealed room, no verb, model orbit | 5 | §2E | TASK-GAP |
| Quest-Medium: `>` behind lava, no fallback, model walks into rats | 5 (4 deaths) | §2C | TASK-GAP + HARNESS-BUG |
| CB-Dark melee deaths | 5 | §2F | dice |
| R3 door-open booked as bump | 1 (run_01) | §2B | HARNESS-BUG |
| MW15 visit-weighted frontier thrash | 1 (run_03) | §2G | HARNESS-BUG (weak) |

## 4. Binding constraint
**TASK-GAP first:** the Quest verb plan and a Boxoban push plan. Both are absent, and together they are 20/40 episodes. The field cannot express either one, and the model, called on 603 Quest steps, never produces a verb.

**HARNESS-BUG second:** door-open booking and the missing frontier fallback. These are cheap code fixes, each worth about 1 episode, and they keep the model out of the loop.

None of this is VLM-ceiling: the model is only ever consulted where the code has nothing to offer, and it adds nothing there (0 wins from 679 calls).

## 5. Concrete changes to try (ranked by leverage)
1. **Graft the Quest regime from `prop_r058/cand_r058_0_spatial-grounding/tools/quest.py`** (288 lines: acquisition, a single kick of the pile, horn-before-wand pickup, apply/zap across the lava, direction and letter prompts, a hold on a ≤2-lane square while a foe is named).
   - Arm it on `{pickup, kick, zap} ⊆ legal`, which is Quest only, and run it in `MenuHarness.act` **before** the menu.
   - Verbs must call `self.ledger.commit(pos, verb, " ")`. `commit` already sets `last=None` for non-compass actions, so they will not be booked as bumps.
   - `Quest` expects r058's chart and frame API (`ChartView`, `fr["adjacent"]`, `fr["message"]`). Write a thin adapter over `Ledger.terrain/visits/doors`, and apply `The lava cools and solidifies.` to `Ledger.terrain` (set `}`→`.`) so the field then routes to the `>`.
   - Predicted: Quest 0/10 → 8–10/10 (+0.20–0.25 primary; replicated in r052 and r058), plus about 2.7M tokens of the 3.04M bill removed.
2. **`orchestrator.py:Ledger.observe` — classify the no-move message before booking `failed`.**
   - `door opens` → `self.doors.add(target)`; do not book.
   - `door resists` / `This door is locked` → keep the target a door; do not book permanently. Better, kick it (memory: kick opener on R3).
   - `can't move diagonally … doorway` → book only that diagonal.
   - Parse `msg` in `observe`; it is already extracted in `act`.
   - Predicted: R3 run_01 becomes a win (+1 episode, R3 4/5 → 5/5), and about 220k fewer tokens.
3. **`tools/field.py:flood` — frontier fallback when the `>` does not reach the hero.**
   - After the `>`-seeded pass, if none of the hero's passable neighbours got a value, re-flood from the frontier seeds and use that field.
   - Simpler alternative: always seed both sets, with `>` seeds at 0 and frontier seeds at a constant offset (e.g. +50).
   - Predicted: Quest-Medium and R3-after-door no longer drop to the model. Alone this does not win Quest-Medium (lava needs change 1), but it removes the model's walk into the rat pack and is what change 1's post-crossing routing needs.
4. **Boxoban push planner (the last +0.25).**
   - Detect the regime from the observation: a stairless frame with `{` fountains, `` ` `` boulders, and prose `bars adjacent`.
   - In that regime, `#` is SOLID and the moves are the 4 cardinals.
   - Run an A* over (hero, boulder set), with each boulder going onto a `{` and dead-square pruning on walls only. Replay the plan as code steps.
   - Chain memory: Boxoban decodes to an exact Sokoban. A push-space A* reached 19/20 on the stub rig at 0 calls, and wins iff the optimal plan is ≤99 steps.
   - Minimum fix even without a solver: make `#` SOLID when the prose names bars. This removes the fake frontier; the model gets called on a flat field, but that buys nothing, so the solver is the real lever.
5. **`tools/field.py:flood` — drop the visit weight on edges (`+0.5·min(visits,6)`) and `+0.25·visits` in `Ledger.potential`, and use pure BFS hop distance.** Break ties with a fixed order to make re-election a fixed point.
   - Predicted: MW15 run_03-type reversals go away, and MW15 goes 4/5 → 5/5 as in r058.
   - Low confidence at n=5; price it on a stub rig first.
6. **Minor fixes:**
   - Answer `[yn]` with `northwest` (the `y` key) when `y` is not legal (Quest-Easy ×4 frozen frames).
   - Write the model's reply text into `reasoning` on model rows; all 679 are blank, which hides what the model did.

**Strategic note for the proposer:** changes 1–3 turn this candidate into r058's mechanism set. Unless the goal is to test the menu-and-field structure itself, **rebasing on r058 (0.625, 8k tok) and adding change 4 (Boxoban solver) is the higher-EV move.** Boxoban is the only +0.25 left that no lineage has claimed live.

## 6. Exhausted — do NOT repeat
- **Calling the model on a flat field.** 679 calls and 3.04M tokens produced 0 wins. It orbits Quest-Easy and walks into rats on Quest-Medium.
- **Re-wording AGENT.md or the menu.** The model already followed option 1 86–99% of the time in r053; the ordering is not the problem.
- **CB-Dark melee tuning** (retreat, hold dose, streak caps, arrival time). Chain memory marks it as dice and says to price it only at ≥500 episodes.
- **Frontier re-rankings on R3/MW** beyond pure distance (gain rate, extremity, nub demotion). These were refuted repeatedly in chain memory; the goal is one hop away on 66% of steps.
- **One-hop menu ranking tweaks** (r029/r053 lineage). This candidate already superseded them with the field.
