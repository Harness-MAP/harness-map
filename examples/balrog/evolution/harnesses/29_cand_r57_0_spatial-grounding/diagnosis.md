# Diagnosis — minihack iter r057 / cand_r057_0_spatial-grounding (stage pipeline + code-owned BFS RouteStage)

**Headline:** primary = **0.300** (12/40).

| Task | Result | Who decided |
|---|---|---|
| MazeWalk-9x9 / 15x15 | **5/5, 5/5** | `route:frontier`/`route:goal`, **0 tokens** |
| Corridor-R3 | 1/5: run_02 won in 22 steps; 4 timed out at 100 | route, then `route:none->model` random walk in run_00 and run_03 |
| CorridorBattle-Dark | 1/5 (4 died) | `route:frontier` east, then `melee->model` swings "east" until dead |
| Quest-Medium | 0/5 (**5/5 died** at steps 19–24) | route walks into the lit rat room, then the model swings to death |
| Quest-Easy | 0/5 (all timed out) | `route:none->model` on 91–100% of steps |
| Boxoban-Hard / Medium | 0/5, 0/5 | `route:none->model` on 59–89% of steps (random cardinals) |

**Binding constraint:** the RouteStage is correct on MazeWalk, but it has no verbs besides moving. It cannot see an open door (drawn `-`/`|`). It books a one-time door failure (`The door resists!`) as permanent. It cannot answer a `[yn]` prompt, because `y` is not in the vocabulary. And nothing in the harness uses items (Quest) or plans pushes (Boxoban). Whenever the route returns `None`, it hands control to a VLM that contributes 0 wins in this lineage.

This is mostly **HARNESS-BUG** (door glyphs, door-failure booking, dead ModalStage) plus **TASK-GAP** (no Quest item mechanism: r052's `tools/quest.py` scored QM 5/5 and QE 4/5 in this same chain and was dropped).

## 1. What this harness does (mechanism)
`agent.py:PipelineHarness.act` works in five steps:
1. `tools/ledger.Ledger.observe(text)` parses the cursor `(x, y-1)`, the `map:` rows and the adjacent prose. It books a `failed{(sq,dir)}` edge whenever the cursor did not move and no COMBAT regex matched (`ledger.py:71-72`).
2. The stages run in order: `ModalStage` → `MeleeStage` → `RouteStage` → `ArbitrateStage` (`tools/stages.py`).
   - `ModalStage.pre` answers `[yn` with `"y"` **only if `"y" in legal`**, and `--More--` with `more`.
   - `MeleeStage.pre` sets `ctx["fight"]` when any 8-neighbour row has a letter glyph. The route then stands down, and the model decides the step.
   - `RouteStage.pre` → `tools/route.plan`: a BFS to a charted `>`, else to the nearest never-stood walkable square with a `' '` neighbour.
     - `BLOCK = "|- }`0@^"` plus space, so **every `-`/`|` is a wall**.
     - `+` doors are walkable but not diagonally.
     - Failed edges and prose-blocked squares are pruned.
     - The route returns `None` when neither a goal nor a frontier is found.
3. On `None`, the VLM is called with the r047 neighbour table, and `arbitrate` projects, vetoes and breaks orbits.
4. AGENT.md and memory.md are r047 priors. They matter only on the model path, and that path won nothing.

## 2. What actually happened (causal chain, with evidence)

### A. Corridor-R3: an open door is drawn `-`/`|`, and the BFS treats it as a wall (run_00, run_03)
- **run_00, step 57.** `The door opens.` The prose says `vertical open door adjacent east`. The map row at y=17 reads `@-.......`, and the `>` is drawn at (19,16), **7 squares away**, with the prose confirming `stairs down far eastnortheast`.
  - The `-` at (13,17) is the open door. `route.walkable('-')` is False, so the `>` is unreachable and there is no frontier.
  - From step 58, every step is `route:none->model`: 42 steps of model random walk, including `It's solid stone.` ×17 and `It's a wall.` ×9. Timeout.
- **run_03, step 9.** The route opened the room's NW door (`The door opens.`). The door is drawn `-` at (19,2), in row `-.@..|`. Everything west of it is unknown.
  - At step 10 the route returns `None`, because the lit room has no frontier and the only exit is the `-` door.
  - The model then orbits inside the room for **90 steps**, including an `open` answered `You see no door there.`
- This is a known result in this chain: an OPEN DOOR is drawn `-`/`|`, and door-passable plus a one-shot refusal gave +89/−6 over 320 R3 levels. The candidate ported r047's ledger but not r056's door-aware chart.

### B. Corridor-R3: a transient door message is booked as a permanent failed edge (run_01)
- **run_01, step ~50.** The hero stands at (20,2) with a `+` door adjacent west, and the only message in the episode is `The door resists!`.
  - A stuck door yields to repeated autoopen, but `ledger.py:71` books `((20,2),'west')` as failed forever, and the BFS prunes it.
  - Step 51 turns around (`east` ×6), and the rest of the episode re-explores the eastern corridors until the 100-step cap. The `>` is never charted.
- **run_04** also times out as pure route (0 tokens, `>` never charted). The trail shows long back-tracks: step 35 goes 12 hops back west, step 73 goes 17 hops back east. Those are nearest-frontier re-elections across one long corridor system. This is the R3 100-step cap, not a bug; this chain has refuted frontier re-rankings (see §6).

### C. Quest-Easy: the `[yn]` ModalStage is dead code (run_04), and nothing uses items (all runs)
- **run_04, step 5.** `southeast` bumps a peaceful goblin, and the game asks `Really attack the goblin? [yn] (n)`.
  - The cursor then parses as (35,0), the top-line prompt, so the ledger pos is garbage.
  - `ModalStage` needs `"y" in legal`, but **`y` is not a BALROG MiniHack verb** (`northwest` is the `y` key).
  - The model answers `east` (=`n`) and re-bumps. The prompt is on screen in **93/100 frames**, and `east` is emitted 81×. Timeout.
- **All 5 QE runs.** The level is split by a `}` lava column, so `route:none` fires on 91–100% of steps. In run_04 the prose says `horn adjacent west` at step 3, but no stage ever picks the horn up or applies it across the lava.
- r052's `tools/quest.py` did exactly that: kick the pile once, pick up the horn then the wand, fire across the lava, hold at the corridor mouth. It scored **QE 4/5 and QM 5/5**.

### D. Quest-Medium: the route marches into the rat room, and the model swings to death (5/5)
- **run_01.** The route owns steps 0–12 and walks east from the start room, past the horn `(` at (22,11) (`horn far west` in the prose), to the corridor mouth at (33,12).
  - The frame at step 11 shows a lit room with **6+ `r`** between the hero and the `}` column in front of `>`.
  - `MeleeStage` only fires on adjacent letters, and `route.walkable` only excludes the creature's own square. So the route and then the model (`route:none->model`, steps 13–14) step **out of the chokepoint into the open room**.
  - From step 15 it is `melee->model` "east" ×8 while HP drops 16→13→11→3, and the hero dies at step 22.
- The same holds in all 5 runs: `route:frontier` ×13, then death at steps 19–24. Even with perfect melee, the level cannot be won without the horn or wand across the lava.

### E. CorridorBattle-Dark: 4/5 died
- The pattern is `route:frontier` east, then 6–15 `melee->model` swings while HP drains (run_02: 16→11→7→2→1→dead).
- The chain has repeatedly priced this as dice at n=5 (see §6). It is not diagnosable here.

### F. Boxoban: 0/10
- After a few `route:frontier` steps (the premapped room leaves no frontier), `route:none->model` random-walks the four cardinals.
- It moved a boulder only 3–7 times per episode, with `in vain` pushes 3–9 times. This is the same result as r052/r054: no push planner.

## 3. Failure breakdown (28 losses)
| Category | Episodes | Class |
|---|---|---|
| Boxoban, no push planner, VLM random walk | 10 | TASK-GAP (VLM-CEILING on the fallback) |
| Quest-Easy/Medium, no item mechanism (plus QM death by walking into the rat room, plus a QE `[yn]` freeze) | 10 | TASK-GAP + HARNESS-BUG |
| CB-Dark melee deaths | 4 | dice (closed axis) |
| R3: `-`/`|` open door read as a wall (run_00, run_03) | 2 | HARNESS-BUG |
| R3: `The door resists!` booked as a permanent failure (run_01) | 1 | HARNESS-BUG |
| R3: 100-step cap on a long corridor system (run_04) | 1 | cap |

Token bill: MazeWalk 0; the R3 route runs 0. About 95% of all tokens go to Boxoban and Quest-Easy `route:none->model` steps (roughly 450–600k per episode) and win nothing.

## 4. Binding constraint
**HARNESS-BUG + TASK-GAP.** The BFS navigator is the right core; MazeWalk is 10/10 at zero cost. But every level that needs a non-move verb (a door, a `[yn]` prompt, an item, a push) falls to the VLM, and the VLM wins nothing. The fixes are code-level and mostly already proven in this chain (r052 quest.py, r056 door-aware chart).

## 5. Concrete changes to try (ranked by leverage)

1. **Port r052's `tools/quest.py` as a `QuestStage` placed before `RouteStage`.** Take it from `prop_r052/cand_r052_0_spatial-grounding/tools/quest.py` along with its `cross`/`hold`/`kit`/`fetch` stage wiring.
   - Arm it only when `{pickup, kick, zap} ⊆ legal` and a `}` is charted.
   - Order: kick the pile ONCE, not toward lava; pick up the horn, then the wand; hold at the corridor mouth while rats come; apply/zap toward the lava bearing.
   - The hold rung must sit **above** the errand rung (see §6).
   - Predicted: **QM 0→4–5/5, QE 0→3–4/5 (+7–9 episodes, about +0.2 primary)**. It also removes about 2.3M Quest-Easy tokens.

2. **Door-aware walkability in `tools/route.py`.**
   - Keep a per-level `doors` set in `Ledger`: any square ever seen as `+`, named `open door`/`doorway`/`broken door` in the adjacent prose, or the target square of the move that produced `The door opens.`
   - In `route.walkable`, treat `-`/`|` on a `doors` square as passable (orthogonal only; keep the diagonal ban for doors).
   - Also treat an **undrawn gap** next to a known door as a frontier.
   - Predicted: R3 run_00 and run_03 become wins (the `>` is 7 hops away in run_00). About **+1–2 R3 episodes**.

3. **Classify no-move messages in `ledger.py:observe` before booking `failed`.**
   - `The door resists!`, `This door is locked.`, `The door is closed.`, `The door opens.`: **do not** book a permanent failed edge.
   - A locked door gets a `kick` in that direction (legal on R3). Memory records locked-door kicks as +15/−3 over 360 R3 levels.
   - Book walls/stone (`It's a wall.`, `It's solid stone.`) as permanent. Book `in vain` (boulder) as one-shot per boulder position.
   - Predicted: fixes R3 run_01 (+1). It also stops the route from permanently pruning good doors.

4. **Make `ModalStage` vocabulary-correct.**
   - `[yn]`: answer `northwest` (the `y` key) for `Really attack`, where the peaceful blocker sits on the route, or `more` to decline (esc-equivalent).
   - Never let a direction answer a `[yn]` prompt: while `[yn` or `--More--` is in the message, the cursor is (x,0), so the ledger pos is garbage, and route/melee must not run.
   - Also skip `ledger.observe` booking on prompt frames.
   - Predicted: removes the 93-frame QE run_04 freeze. Necessary but not sufficient for QE; pair it with #1.

5. **Chokepoint-aware routing when creatures are drawn (QM/CB).** In `RouteStage.pre`, when ≥2 letter glyphs are charted within 6 squares and the hero is on a corridor/doorway square, emit `search` (a null turn) instead of stepping into the open room. The rats then come to the corridor one at a time.
   - This is the "hold at the corridor mouth" rung from r052. Only price it together with #1 on Quest-Medium. On CB-Dark it is dice (see §6).

6. **(Big, optional) A Boxoban push planner as a `SokobanStage`**, armed when the map is premapped, has no `>`, and has `` ` `` and `{`/target glyphs.
   - `#` there is BARS, not corridor. Use a push-space A* over boxes and targets and emit the plan as cardinals; re-plan on an `in vain` message.
   - Memory records a 19/20 stub-rig result, and wins happen iff the optimal plan is ≤99 steps.
   - Up to **+10 episodes**, and it removes about 5M tokens. Highest ceiling, highest implementation cost.

Combined expectation for #1–#4: roughly 12 → 21–24/40 (0.52–0.60), at much lower cost.

## 6. Exhausted — do NOT repeat
- **Stage/pipeline rewiring by itself** (r026/r046/r054/r055/r056/r057). This rewire is neutral-to-negative against r052's 0.600, and its losses are missing mechanisms, not wiring.
- **Frontier re-ranking on MazeWalk** is saturated at 10/10. R3 frontier reformulations (extremity, gain-rate, nub demotion) are refuted or task-split. Don't try to fix run_04 that way.
- **Prompt/AGENT.md wording** to make the VLM avoid rats, push boulders or use items. The VLM has 0 wins across this lineage.
- **`y`/`esc` as literal answers.** They are not in the vocabulary; use `northwest` and `more`.
- **CB-Dark melee posture** (retreat, hold, recoil, HP gates) priced on 5 episodes. It is dice. Never make a creature a refusal/veto.
- **Quest:** a pickup on the spawn pile without the kick-split, a double kick (re-piles), a kick toward lava (burns the horn), zapping after a dry route exists, an errand rung above the fight rung.
- **`#`-as-bars on Boxoban as a primary lever.** It only cleans the fallback (r054: 0/10).
