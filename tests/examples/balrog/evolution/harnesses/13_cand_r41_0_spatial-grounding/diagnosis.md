# Diagnosis — minihack iter 41 / cand_r041_0_spatial-grounding (charted learner: BFS chart first, TD learner as fallback)

**Headline:** primary = **0.375** (15/40, se 0.077). Survival 0.80, step_efficiency 0.27, cost 21.9k tokens/ep. By task:

| Task | Success |
|---|---|
| MazeWalk-9x9 | 5/5 |
| MazeWalk-15x15 | 5/5 |
| Corridor-R3 | 4/5 |
| CorridorBattle-Dark | 1/5 |
| Boxoban-Hard | 0/5 |
| Boxoban-Medium | 0/5 |
| Quest-Easy | 0/5 |
| Quest-Medium | 0/5 |

This is +0.225 over the parent r035 (0.150). It ties the chain's navigation-only plateau of 0.325–0.375 (r1, r26, r36, r38).

**Binding constraint: TASK-GAP.** Navigation is effectively solved: MW 10/10, R3 4/5, bump rate ≤3%, and 0 model tokens on those tasks. **20 of the 25 losses** are on Boxoban and Quest. The harness has no lever there at all: it has no push planner, and it never emits `pickup`, `apply`, `kick` or `zap`. A smaller share (one R3 episode and the Quest-Medium deaths) comes from two routing gaps that are fixable in the harness.

## 1. What this harness does (mechanism)
Everything is in `agent.py:LearnerAgent.act`, one module with no tools. The order is:
1. **Prompt reflexes:** `[yn]` marks the last swung glyph as peaceful and answers `more` (at most 2 in a row). Other prompts get `esc` only if it is in the vocabulary.
2. **`observe()`:** books door cells, from prose `doorway`/`open door`/`broken door` or from "door opens". Any other no-move books the `(square, bearing)` edge as blocked.
3. **`credit()`:** a TD(λ) update, and folds the frame into `self.drawn`.
4. **Stairs reflex:** `down` on `>`. This is dead code, because MiniHack ends the episode when the hero steps onto `>`.
5. **`deadly()`:** vetoes lava and water, from the prose or from `}`.
6. **`foe_bearing()`:** swings at any adjacent letter.
7. **`route()`:** a BFS over `passable()` drawn cells, where `WALL = "|- "` and blank is impassable. The target is a drawn `>`, or else the nearest never-stood cell that has an undrawn neighbour. It emits the first hop (`chart:route`).
8. **Learner fallback:** used only when `route()` returns None. It takes the argmax of optimistic q over `admissible()` moves, and consults the model once per square on ties.

Rung census from the `Reasoning` column (all 40 episodes):

| Task | chart:route | learner (argmax+consult) | swing | more |
|---|---|---|---|---|
| MW9 / MW15 / R3 | 31 / 155 / 274 | 0 | 0 | 0 |
| Boxoban-H / M | 293 / 441 | 207 / 59 | 0 | 0 |
| CB-Dark | 108 | 0 | 59 | 0 |
| Quest-Easy | **5** | **471** | 9 | 15 |
| Quest-Medium | 70 | 88 | 47 | 0 |

On the navigation tasks the chart owns 100% of steps. There are 0 learner steps and 0 tokens there, and ≤3 no-moves per episode (MW: 0; R3: 1–3, all door opens or resists). The r035 failure (26–49% bumps, walking past an adjacent `>`) is **fully fixed**. The learner and the model now matter only where the chart has nothing to say, which is Boxoban and the Quest pair.

## 2. What actually happened (causal chains)
CSV rows are (action, obs AFTER it).

### A. Boxoban (0/10): the chart walks into bars and nothing pushes toward the targets
- In Boxoban, `#` is **iron bars**, not corridor, and the level is fully premapped. `WALL = set("|- ")`, so `passable()` treats `#` as floor, and `route()` plans first hops straight into it.
- Example, Boxoban-M/00 steps 0–3:
  - The hero is at (42,3), flanked by `#`. The route emits `north` and then `east`, and gets "You cannot pass through the bars." each time.
  - Each bump books a single edge in `self.blocked`, so the next plan picks a different `#` neighbour.
- Across the 10 episodes:

  | Message | Episodes | Range per episode |
  |---|---|---|
  | "cannot pass through the bars" | all 10 | 17–44 |
  | total no-moves | all 10 | 20–46 |
  | "boulder, but in vain" | all 10 | 2–9 |

- The "frontier" targets are meaningless on a premapped level: they are just unstood cells next to the undrawn outer rim. No step tries to move a `` ` `` boulder onto a `{` target on purpose.
- Once the bumps have closed the edges out of a square, `route()` returns None and the learner takes over. It consults the model 53 times, and those consults are the entire Boxoban token bill (≈340k): ties that cannot matter.
- **This is TASK-GAP. Fixing bars alone scores 0.** The level decodes to an exact Sokoban instance. It needs a push-space planner.

### B. Quest-Easy (0/5): the room is sealed by lava, so the chart has no route and the learner wanders for 95 steps
- The QE/01 spawn frame shows a room whose only exit is east across a full `}` column, with `horn adjacent south` (the `(` under/next to spawn).
- `}` is in `LETHAL`, so `route()` returns None after the room is charted. It fires 5 times in the whole task. The learner then owns 471 steps. Example, QE/01 steps 40–70: N/S/E/W shuffling inside the 5×5 room.
- Across the 5 episodes: 21–22 unique squares per episode and 18–26 A→B→A reversals per 100 steps.
- The horn is never picked up and nothing is applied.
- QE/04 adds a side loop. The learner walks into a peaceful goblin 8 times (steps 38, 49, 54, 65, 89, …). Each time it gets "Really attack the goblin? [yn]", then `more` → "Unknown command '^M'", which burns 2 steps. The `peaceful` set is consulted only by `foe_bearing`, never by `admissible()`.
- The learner consults the model on ties here, so Quest-Easy is **≈370k of the 874k total tokens**.

### C. Quest-Medium (0/5, 4 deaths): the route leads the hero into the open rat room, and the lava still blocks `>`
QM/01 walkthrough:
- **Step 13:** the hero reaches the room mouth at (33,12). The room holds 6 rats (`rrrr..rr`), with `}` and `>` behind them.
- **Steps 14–15:** the learner/consult steps the hero **two squares into the open room**, to (35,12).
- **Steps 16–29:** `reflex:swing` fires 14 times while rats hit from east **and** northeast. HP goes 16→12→5→3→1, and the hero dies at step 30.

The same thing happens in QM/02–04, with deaths at steps 23–26. In QM/00 the hero survived the rats but timed out behind the lava with no horn.

Two gaps compound here:
1. **Nothing chooses the fight square.** The corridor square west of the mouth, (31,13), has exactly one approach lane. The room square has 3 or more.
2. **Surviving still loses.** The `}` column separates the hero from `>`, and the same missing horn recipe as in QE applies.

### D. Corridor-R3/03 (the one R3 loss): a drawn `>` the chart cannot reach is ignored
- `>` is drawn from step 23 at (8,10), in a west room whose connection is not charted yet.
- `route()` finds no path to it. Because `stairs` is non-empty the BFS does not break early, but `best` keeps the **first** (nearest) frontier and never re-ranks toward `>`.
- The hero therefore explores east and south-east corridors. By step 98 it has charted a whole new room at x=35–43, and it times out at 71 unique squares with `>` still 16+ columns west.
- This is the known "drawn-but-unreachable `>` must attract the frontier" gap.

### E. CorridorBattle-Dark (1/5): dice
- 4 deaths at steps 27–34, all in swing exchanges ("The giant rat bites!"). There are 59 swings and 6–18 no-moves per episode.
- This matches the chain's settled finding that CB-Dark outcomes are dominated by dice at n=5.

## 3. Failure breakdown (25 losses)
| Failure | Eps | Class |
|---|---|---|
| Boxoban: no push planner; `#` bars treated as corridor (20–46 no-moves/ep) | 10 | TASK-GAP (+ bars HARNESS-BUG) |
| Quest-Easy: lava-sealed room, no horn pickup/apply, learner wanders | 5 | TASK-GAP |
| Quest-Medium: the route enters the open rat room → 4 deaths; the 5th is lava-blocked | 5 | TASK-GAP + HARNESS (fight square) |
| CB-Dark melee deaths | 4 | dice (known) |
| R3/03: drawn-unreachable `>` not used as a frontier attractor | 1 | HARNESS-BUG |

## 4. Binding constraint
**TASK-GAP.** The chart-first wiring works: it scores 100% on reachable-`>` navigation at 0 tokens. The remaining score is locked behind two *non-navigation* competences that nothing in `act()` can produce:
- **Sokoban push planning** (10 eps).
- **The item → lava recipe** (10 eps).

Rewording prompts or retuning the learner cannot reach these. The learner is only a fallback, and on these tasks it is pure wandering. The chain's elite band (≥0.6) is exactly this candidate's navigation profile plus Quest 9–10/10, and 0.875 adds Boxoban 10/10. That is the gradient.

## 5. Concrete changes to try (ranked by leverage)

1. **Quest recipe rung, before `route()` in `act()`** (Quest-E 0→~4/5, Quest-M 0→~3–5/5, i.e. up to +0.25 primary). Build it as a small state machine in the hub:
   - **(a) Detect a lava seal.** `route()` returns None (or has only frontier targets) **and** drawn `}` cells exist **and** a prose item noun (`horn`, `wand`) or a `(`/`/` glyph is charted.
   - **(b) Get the item.** Walk to it with the same BFS, targeting the item cell. When standing on it, emit `pickup`. Autopickup is off. If "There are several objects here" or a pile blocks the pickup, `kick` the pile **once** from an adjacent square; the chain found that a double kick re-piles the items.
   - **(c) Use it.** Route to a square adjacent to the nearest `}`, emit `apply` (or `zap` for a wand), answer the item-letter prompt, then answer the direction prompt with the bearing of the `}`. Stop re-firing the rung once the message contains `solidifies`/`cools`.
   - **(d)** After that the ordinary `route()` takes over, because a solidified `}` redraws as floor.
   - All of these steps read only the frame and prose, so there is no oracle use.
   - **Verify first:** that `pickup`, `apply`, `kick` and `zap` are in `legal_actions()` for Quest, and how the item-letter prompt is answered with the frozen vocabulary. The previous `y`/`esc` reflexes were dead because those tokens were not in the vocabulary.

2. **Boxoban push-space planner** (Boxoban 0→up to 10/10 when the optimal plan is ≤99 steps, i.e. +0.25 primary).
   - **Detection:** the prose contains `bars` and `boulder` and the level has no `>`. Decode the map as walls = `#`/`-`/`|`, boulders = `` ` ``, targets = `{`, floor = `.`. The hero's own square is floor.
   - **Planning:** run A* over (hero-reachable region, boulder set) with push moves only in the 4 cardinals. Boxoban has no diagonal moves. Prune only on walls: pruning on "unwinnable" boulder counts was refuted.
   - **Execution:** compile the plan to a move tape and replay it with **0 model calls**. Re-plan if a step produces no position change.
   - **Minimum fix even without a planner:** add `#` to the impassable set whenever the prose names `bars`, via `passable()` / `WALL`. This removes the 17–44 "cannot pass through the bars" bumps per episode, and with them most of the learner fallbacks that produce the Boxoban consults.

3. **Frontier attractor toward a drawn-but-unreachable `>` (`route()`)** (R3 4/5→5/5; MW15 unchanged).
   - When `stairs` is non-empty and unreachable, collect **all** frontier cells found by the BFS instead of keeping only the first.
   - Pick the one that minimises `bfs_dist + cheb(frontier, nearest >)`, with ties broken by `(y, x)` in the level frame. This is still a total order, so re-electing every step cannot orbit.
   - Gate the attractor on no `}` being charted, so it does not pull the hero into Quest lava before the recipe fires.
   - Target: R3/03, where `>` was drawn from step 23 and the timeout came at step 100 while exploring east.

4. **Committed fight square for rooms with visible creatures (`act()`, before `route()`)** (Quest-M deaths 4→~1; needed for rung 1 to pay on QM).
   - If ≥2 creature letters are drawn within 6 squares and the next route hop would enter a cell with ≥2 open approach lanes (count passable neighbours not in `WALL`), do not advance.
   - Hold on the current corridor square instead, using the null move the vocabulary provides (`search`, or `wait` if present), until a creature is adjacent. Then swing.
   - Commit the post, meaning the square you stepped **to**, rather than re-deriving it every step. This is the chain's "exposure(cell) is the fight square" lesson (one-way anchor at 2 lanes).
   - Evidence: at QM/01 step 12 the hero stood on the corridor-mouth square with a single lane. The two extra steps into the room (steps 14–15) are what let two rats bite per turn.

5. **Small hygiene (cost and step-efficiency, not primary):**
   - `admissible()` should reject a target cell whose glyph is in `self.peaceful`. This avoids the QE/04 `[yn]` → `^M` 2-step loops.
   - Drop the learner's model consult on squares where the tie is between moves that all stay inside a fully charted, sealed room. It changes nothing, and the Quest-Easy consults are ≈40% of the token bill.
   - The `down` reflex on `>` is dead code. It is harmless and can be removed.

## 6. Exhausted — do NOT repeat
- Optimism- or value-only exploration without the chart (r035, 0.150: 26–49% bumps). The chart must stay first; this candidate proves it.
- A prose foe blacklist (r040): wall and item nouns became swing targets. Keep `foe_bearing` on map glyphs only, as here.
- Signed `>` offset vectors in the prompt (r033). The model does not follow them; emit the BFS hop.
- Stair reward shaping as a substitute for a route (r035 §2B).
- Melee retreat / disengagement / HP-gated retreat on CB-Dark. The chain found them dose-response negative and CB-Dark is dice. Do not spend a round on CB-Dark at n=5.
- Re-tuning learner constants (ALPHA/LAMBDA/BONUS/TIE). The learner owns 0 steps on the tasks it could affect.
