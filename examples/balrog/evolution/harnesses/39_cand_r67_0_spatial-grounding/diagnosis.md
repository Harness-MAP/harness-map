# Diagnosis — minihack iter r067 / cand_r067_0_spatial-grounding (middleware pipeline + ledger + binding chart route)

**Headline:** primary = **0.275** (11/40, se 0.071). Progression 0.275, survival 0.800 (8 deaths: 5 CB-Dark, 2 Quest-Medium, 1 Quest-Easy), step_eff 0.223, cost **301k tok/ep**.
By task: MazeWalk-9x9 5/5, MazeWalk-15x15 5/5, Corridor-R3 1/5, and 0/5 on the other five tasks.
**Binding constraint:** a HARNESS-BUG in `tools/route.py`. `IMPASSABLE = set("|- }`0")` treats an **open door**, which is drawn `-`/`|`, as a wall. The ledger also books `The door opens.` as a failed edge. So every R3 door the agent opens deletes the path behind it. In 2 of the 4 R3 losses the `>` was already drawn behind that door, once only 3 squares away.
The second constraint is a missing owner. No stage ever elects a swing, and the model parrots the route note, so the agent walks away from adjacent rats until it dies (CB-Dark 0/5). The third is that Boxoban and Quest have no mechanism at all. The chain already has donors for both.

## 1. What this harness does (mechanism)
`orchestrator.py:PipelineHarness.act` does the following each step:
1. `context.build` parses the cursor (row = y−1), the `map:` grid, the message and the `adjacent` prose into a `StepContext`.
2. `Ledger.observe` records visits, failed (square, dir) edges when there is no displacement and no combat message, and cells whose prose noun is wall/stone/bars or lava/water.
3. The stages' `pre` hooks run: Modal (`--More--`), Hazard (adds a "Blocked or deadly" note), and Route (BFS: a drawn `>`, else the frontier minimising `(dist−2·gain, …)`; adds a "Route: move X" note).
4. There is ONE model call.
5. The `post` hooks run: Route (keeps the model's move only if it equals the hop or is a swing at an adjacent letter glyph, otherwise substitutes the hop), then Hazard (last-word veto), then Legal.

When the route is `None` (Boxoban, a sealed Quest room, a severed R3 chart), the model decides alone.

## 2. What actually happened (causal chain, with evidence)

### 2A. Corridor-R3: opening a door severs the chart (4 of the 4 R3 losses are shaped by it)
The underlying failure is that an open door is drawn `-`/`|`, and `route.py:IMPASSABLE` contains both. After `The door opens.` the doorway becomes impassable. `ledger.observe` also books `(square, dir)` as failed, because the cursor did not move and the message matches no `COMBAT` pattern.

- **run_03** (replayed offline with `Ledger` + `plan` on the CSV frames):
  - Steps 0–8 route to the west door at (19,2): `route:explore@(19, 2)`.
  - Step 9 is `west`, with `message: The door opens.` The grid now shows `-@...|` and the prose says `vertical open door adjacent west`. The ledger has `failed = {((20,2),'west')}`, and `plan()` returns **(None, None, None)** from step 9 on.
  - The east `#` stub was already "probed" (stood on at (25,5)/(25,6)), so no frontier remains.
  - The model then runs a **3-cycle `east → southwest → north` for 90 steps** (steps 10–99, positions (22,6)→(21,7)→(21,6)).
- **run_01, step 57**:
  - The agent stands at (20,2), and the prose says `vertical open door adjacent west`.
  - The row reads `|>....-@######`, so the **`>` at (17,2) is 3 squares away**, behind the open door `-` at (19,2).
  - Because `-` is impassable, `route:stairs` is never elected. Step 58 turns **east** (`route:explore@(26, 2)`), and the episode times out at step 99.
- **run_00**:
  - The `>` is drawn from step 33, at (22,15) in a room whose west entrance `#-.......` is an open door.
  - Stairs are never elected. From step 62 on `plan()` is None and the model orbits `north/southwest/southeast` to the cap.
- **run_04**: one `The door opens.` at step 60, (33,5)→east, and the agent never passes it. The goal flips between far frontiers ((27,5)→(40,6)→(54,7)), and no `>` is ever drawn (68 distinct squares in 100 steps).
- **run_02 (the win)**: the `>` was drawn at step 6 in the same lit room, with no door on the path, and was reached in 22 steps.

This matches a known chain law: "an OPEN DOOR is drawn `-`/`|`… `the door opens.` must CLEAR a record" (see the memory notes on door-passable plus one-shot refusal).

### 2B. CorridorBattle-Dark: the agent never fights (0/5, all 5 dead in 22–43 steps)
Across the 5 episodes there are **51 frames with a creature glyph on an adjacent square. The next action attacked it in 3 of them (6%)**, and there is 1 `You hit/miss` message in total.

The model's move equals the route hop on 100% of routed steps (`model=route` on 137/137 CB steps). The "Route: move X" note is in the prompt, the model copies it, and `RouteStage.post` only *allows* a swing without ever *electing* one.

The frontier BFS skips creature squares as goals but happily routes the agent **past and away from** them. In **run_02**, steps 15–21: `The giant rat bites!` at HP 16→13→10→6→1→dead, while the actions are `north, northeast, west, southwest, southeast, west`. The goal re-elects around the rats every step ((37,10)→(35,12)→(39,11)→(35,12)). **run_03** is the same: HP 16→14→8→5→4→dead while it walks. The failure mode is fleeing while surrounded, which is strictly worse than holding in the corridor and swinging.

### 2C. Boxoban (0/10) and Quest (0/10): no owner, the model orbits
- **Boxoban.** `route=None` on 1000/1000 steps (no `>`, and `hash_is_bars` turns every `#` into bars). The model is a pure 2-cycle: Medium run_01 is `west` ×50 / `east` ×50, and run_00 is W30/E30/N20/S20. That costs ~600k tokens per episode for zero progress, about 60% of the whole bill.
- **Quest-Easy.** `route=None` on all 423 steps. The agent is sealed behind the `}` column: `|.(.@.}:.d...`. The model orbits (run_00: E33/NW32/S32). The `(` item on the near side is never picked up.
- **Quest-Medium.** The route runs 20 steps, then `route=None` for the rest (80 steps), and the rat fights are left to the model: 2 deaths (run_00, run_04, both at 39 steps).

### 2D. The model call is pure cost when a route exists
On every routed step of every task (MazeWalk, R3, CB), the model's choice equals the hop, apart from 0 useful overrides. Only 8 `hazard:` rewrites occur in the whole run. The one-call-per-step contract is spending ~300k tok/ep to transcribe the note back.

## 3. Failure breakdown (40 episodes)
| category | episodes | evidence |
|---|---|---|
| MazeWalk solved by the route | 10 won | `route:stairs` elected in every one |
| R3: open door impassable plus a door-open booked as failed | 4 lost (r00, r01, r03 directly; r04 wandered past its opened door) | §2A |
| CB-Dark: no fight owner, walks while bitten | 5 dead | 3/51 adjacent-creature frames swung |
| Boxoban: no mechanism, 2-cycle | 10 lost at cap | route None 100% |
| Quest-Easy: no lava/item mechanism | 5 lost (1 death) | route None 100% |
| Quest-Medium: no mechanism past the lava plus model melee | 5 lost (2 deaths) | route None after step 20 |

## 4. Binding constraint — classification
- **R3 → HARNESS-BUG** (glyph table plus the ledger's no-move classification). This is a code fix, and the `>` was already on screen in 2 of the losses.
- **CB-Dark → HARNESS-BUG / missing owner.** The model *can* swing (it does in Quest-Medium), but the binding route note crowds it out. A code-elected swing is needed. After that, CB-Dark is dice (chain memory: 0.38–0.55 win rate once swinging).
- **Boxoban / Quest → TASK-GAP.** The agent acts legally but has no planner. The chain already solved both: r064 scored 0.875 with Boxoban 10/10 via `tools/sokoban.py`, and r052/r058 took QM 0→5/5 and QE 0→4/5 via `tools/quest.py`.

## 5. Concrete changes to try (ranked by leverage)

1. **Graft the Sokoban solver** from `prop_r064/cand_r064_0_spatial-grounding/tools/sokoban.py`.
   - How: add it as a pipeline stage `SokobanStage.pre` that runs when `ledger.hash_is_bars` is set and the grid contains `` ` `` and `{`. Read the floor, boxes, goals and position from the drawn grid, `solve()` once, and replay the push plan as a committed tape, re-solving if `simulate` diverges from the next frame.
   - Predicted: Boxoban 0/10 → up to 10/10 (**+0.25**), with that task's ~600k tok/ep dropping to ~0.
   - Why it should work: the chain has shown the map decodes exactly, and r064 got +10 episodes from exactly this graft.

2. **Make open doors passable and stop booking `The door opens.` as a failure.** This is the R3 fix.
   - `route.py:walkable`: a `-`/`|` cell is passable if the prose noun on that cell contains `open door`, `doorway` or `broken door`. Credit the noun to the cell via `ctx.nouns` and keep a ledger set `ledger.doors`, so the cell stays passable after the agent walks away. Keep the existing no-diagonal rule for door cells, and extend it to cells in `ledger.doors`, since open doors also refuse diagonals.
   - `ledger.observe`: if the message matches `door opens|door is locked|This door`, do NOT add `self.last` to `failed`. Opening is progress, and it takes a second move to step through.
   - Optional, per chain memory: on `This door is locked.`, emit `kick` toward it when `kick` is legal. This needs a 2-step verb (`kick` then the direction).
   - Predicted: R3 1/5 → 3–4/5 (**+0.05 to +0.075**). run_01 and run_00 had a drawn `>` behind an opened door, and run_03's only exit was the opened door.

3. **Add a FightStage that elects the swing.** It goes in `stages.py`, inside RouteStage, so its `post` runs first.
   - Behaviour: if any adjacent cell holds a letter glyph (`is_creature`) that is not a peaceful/shopkeeper noun, return the move into it. With several creatures, pick the one on the same row/column first. Remove the "Route: move X" note on those frames so the model is not contradicted.
   - Never retreat: chain memory says every degree of disengagement kills more (9→13→17→42 deaths), and HP retreat is refuted.
   - Also exclude creature-adjacent squares from explore goals only when HP is low? **No.** Just swing; CB-Dark's threat horizon is one square.
   - Predicted: CB-Dark 0/5 → ~2/5 (dice-limited), Quest-Medium deaths 2 → ~0, survival 0.80 → ~0.90.

4. **Graft the Quest module** from `prop_r064/.../tools/quest.py` (it originated in r052).
   - Trigger: `}` is drawn and the route is None or sealed.
   - Behaviour: pick up the near-side item, then `apply`/`zap` toward the lava (the chain found that `kick` splits the pile and that the wand solidifies the lava). This needs the legal-verb set to include `pickup`/`zap`/`apply`; read it from `self.legal`.
   - Predicted: Quest 0/10 → 5–9/10 (**+0.125 to +0.225**).
   - Larger code; do it after 1–3 if one iteration can only carry two changes.

5. **Skip the model call when a non-modal stage has already decided** (a route hop exists, or a fight or solver move is elected).
   - How: move RouteStage's decision into `pre` as a short-circuit, the way ModalStage does it.
   - Predicted: cost down 50–70% (routed steps are ~40% of steps, and Boxoban goes away with change 1), with no change to primary, since `model=route` holds on 100% of routed steps (§2D).

6. **Anti-orbit fallback when `plan()` is None.**
   - How: in `RouteStage.post`, when there is no hop, replace the model's move by the legal non-hazard move to the least-visited neighbour, and emit `search` once a square has been visited 3 times or more with no frontier. R3 hides passages in drawn walls, and the probed cell should be budgeted.
   - This converts the 90-step 3-cycle in R3 run_03 into a search, and removes the Boxoban/Quest ping-pong if 1 and 4 are not grafted.

## 6. Exhausted — do NOT repeat
- **Prose or table grounding of visit counts** (r029): the model ignores it. The binding route in this candidate is the right fix and works: MazeWalk is 10/10.
- **Advisory-only route notes for melee:** the model copies the route hop even while being bitten (§2B). A swing must be code-elected, not suggested.
- **HP-gated retreat or disengagement on CB-Dark:** chain-wide refuted (dose-response 9→42 deaths).
- **Treating a DARK blank as walkable** (r047's `dark area` noun as open): refuted. Keep blanks impassable, and fix doors instead.
- **Frontier-ranking reformulations** (gain rate, extremity, nub demotion): flat per chain memory. The goal is the adjacent square on ~66% of steps. R3's losses here are topological (the door), not ranking.
