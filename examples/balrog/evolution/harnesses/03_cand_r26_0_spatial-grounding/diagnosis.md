# Diagnosis — minihack evolve r026 / cand_r026_0_spatial-grounding (one-module chart pilot)

**Headline:** primary = **0.325 ± 0.074** (13/40; anchor 0.100), progression 0.300, survival 0.775,
step_efficiency 0.249, cost 115k tok/ep (anchor 309k). Per task: MazeWalk-9x9 5/5, MazeWalk-15x15 5/5,
Corridor-R3 2/5, CB-Dark 1/5, Boxoban-M/H 0/10, Quest-E/M 0/10.
**Binding constraint:** the chart has a correct spine. Every remaining loss traces to one of four
**HARNESS-BUGs** in how the chart reads glyphs and prompts, plus one missing **TASK-GAP** rung (items):
1. an opened door is drawn `-`/`|` and charted as WALL. This causes 3/3 R3 losses.
2. the prompt reflex answers with `esc`/`y`, and neither exists in BALROG's vocabulary. This causes 2 Quest-Easy lockups of 88 and 95 steps.
3. the chart treats Boxoban `#` (bars) and `` ` `` (boulders) as floor. This causes 10/10 Boxoban losses.
4. melee is fought on the most exposed square and routing walks into the pack. This causes 5/5 Quest-Medium deaths and 3–4 of the 4 CB-Dark deaths.
5. no item/lava rung exists, so the Quest pair is unwinnable by construction.

## 1. What this harness does (mechanism)
`orchestrator.py` is one module. `GroundedAgent.act` runs this order:
modal reflex → stairs `down` → hazard veto (`lethal_moves`: prose lava/water + `}` glyph) → melee
(`adjacent_foe`: any letter glyph adjacent → move into it) → `route()` (BFS over `self.terrain` to a
charted `>`, else the nearest *unstood charted* square that touches undrawn space, with a per-(square,
bearing) bump ledger) → `ask_model()` only when `route` returns None.
AGENT.md, rules.md, skills and memory.md are all empty, so the model sees the stock BALROG
history plus a list of closed moves. `passable()` = charted and glyph ∉ `WALL="|- "` ∪ `LETHAL="}"`.

The route answers 100% of steps on MazeWalk and the winning R3 episodes, which is why those tasks
went from 4/10 to 12/15. The model is reached only on Boxoban, Quest and exhausted R3 charts. There it is the
anchor again: it orbits.

## 2. What actually happened (causal chains)
CSV rows are (action, observation AFTER it). Positions are the cursor `(x,y)`.

**A. Corridor-R3: an open door becomes a wall, so the charted `>` is unreachable (all 3 losses).**
- R3/01: the route walks to the west door. At step 53 `west` gives "The door opens." The position is unchanged, so
  `observe()` books a bump on ((20,3),west). The door is now drawn `-`, at map row 2 col 20: `|>....-#######`,
  and `-` ∈ `WALL`. The `>` at col 15 is charted **but unreachable** for the remaining 46 steps. `route`
  keeps chasing dark-corridor frontiers, and the episode times out at the corridor at (30,10).
- R3/03: the same thing happens at step 9 (`northwest` → "The door opens." at (20,4)). A diagonal entry into a door
  square is illegal anyway. The chart then has no route, and the model orbits (22,6)↔(23,6)↔(22,7) for **90 steps**.
- R3/00: the same thing happens at step 38 ("The door opens." at (12,17)). The `>` at row 14 col 22 stays charted, and the room's
  west door `-` at (16,15) walls it off. The episode times out.
- Both wins (R3/02, R3/04) opened **zero** doors. The separator is 3/3 vs 0/2.

**B. Prompt lockups: the escape verbs do not exist.** BALROG's MiniHack vocabulary
(`BALROG/balrog/environments/minihack/__init__.py:ACTIONS`) has **no `esc` and no `y`**. The only prompt
answers are `more` (cancel → "Never mind.") and the letter-keystroke verbs (`y`=northwest, `n`=southeast).
- QE/00 step 11: the model emits `pickup` then `quaff` → "What do you want to drink? [f or ?*]". `act()`
  takes the `fr["prompt"]` branch, finds no `esc`, and falls through. `pos` parses as (37,-1) from the top-line
  cursor, `route` returns None, and the model keeps emitting `quaff`. The frame is frozen for steps 12–99 (88 steps).
- QE/04 step 5: `reflex:swing` southeast into a **peaceful goblin** → "Really attack the goblin? [yn] (n)".
  `"y" in legal` is False, so the reflex dies. The model emits `n`/`south`/`yes`/`attack` for 95 steps
  (the cursor stays at (35,0), HP 15/15, no clock tick).
- Contributing cause: `ask_model` passes ANY legal verb through (`quaff`, `pickup`). On Quest that opens
  item prompts that nothing in the table can answer.

**C. Boxoban: bars and boulders are charted as floor.** On Boxoban `#` is iron BARS and `` ` `` is a boulder.
Neither is in `WALL`. The frontier test (charted, unstood, touches undrawn space) is true for every
square along the bar ring, because outside the ring is undrawn. So `route` walks the perimeter pushing
boulders. For example, Boxoban-M/00 steps 39–54 alternate a `north` into "You cannot pass through the bars." with a
`west` into "You try to move the boulder, but in vain." That episode made 100/100 route steps with 0 model calls.
Each bump ledger entry fixes one edge, and the next one is tried. Boulders are shoved into corners, which makes the
level irrecoverable.

**D. Melee is fought on the exposed square.**
- Quest-Medium, all 5 episodes, same trajectory: the route reaches the lit rat room at step 12 (33,12). The room's `>` sits
  behind the `}` column, so `route` has no target and the model is called (steps 13–15, empty reasoning).
  The model walks **east three squares into the pack**. At (36,12) the agent has rats NE+E+SE and takes 2–3 bites per turn.
  `reflex:swing east` never moves. The agent is dead by step 19–24 at 16 HP (e.g. QM/00 HP 16→13→8→2→dead).
  The corridor mouth two squares west has exactly one open neighbour.
- CB-Dark/01: at the mouth (34,12) the agent kills 3 rats with HP 16→3. At step 23 `chart:route` then steps **east
  into the open room at 3 HP**, where rats are NE+E+SE, and it dies at step 26. CB-Dark/00, 02 and 03 have the same shape: 7–14 swings, then death.

**E. Quest-Easy without a lockup (QE/01–03): the model orbits the lava bank.** The `>` is charted behind `}`, and
the lit room has no frontier, so the model decides 100/100 steps. In QE/01 it cycles
(29,10)→(29,9)→(28,9)→(28,10)→(28,11) with period 8 and never touches the `horn adjacent west` that
the prose prints at step 0.

## 3. Failure breakdown (27 losses)
| Category | Episodes | Mechanism |
|---|---|---|
| Open door charted as wall | R3/00, 01, 03 (3) | `WALL` contains `-`/`|`; "The door opens." booked as a bump |
| Boxoban bars/boulders charted as floor | Boxoban 10/10 | `#` and `` ` `` pass `passable()` |
| Exposed-square melee / walking into the pack | QM 5/5 (deaths), CB-Dark 4/5 | swing never repositions; route/model advances at low HP |
| Unanswerable prompt | QE/00, QE/04 (2) | `esc`/`y` not in the vocabulary; verbs passed through |
| No item rung, so lava is uncrossable | QE 5/5, QM 5/5 (ceiling) | the harness never kicks, picks up or applies |

Survival 0.775 = 9 deaths: 5 QM + 4 CB-Dark, all melee. There are no lava deaths, so the hazard veto works.

## 4. Binding constraint — classification
- A, B, C: **HARNESS-BUG.** Each is a one-function fix in `orchestrator.py`, and each maps to named episodes.
- D: **HARNESS-BUG/TASK-GAP.** The fight position is a harness decision, and the model is never asked about it.
  CB-Dark is partly dice (prior campaign evidence), so price any CB change on many seeds. QM is deterministic.
- E/Quest: **TASK-GAP**, and it is NOT a ceiling. Prior probes in this campaign measured Quest as winnable
  from observation alone. The spawn pile of horn and wand does not answer `pickup` until it is split with **one** `kick`
  (east | kick | west). After that, `pickup` works, and a prompt `[ij or ?*]` is answered by the verb whose keystroke is the
  letter (`j`=south, `k`=north, `l`=east, `h`=west, `y`=northwest, `u`=northeast, `b`=southwest, `n`=southeast, `a`=apply,
  `e`=eat, `z`=zap, …). `zap <letter-verb> <bearing>` toward the lava gives "The lava cools and solidifies.".
  A fixed policy scored 5/5 QM + 2/5 QE that way. Kicking twice re-piles the items.

## 5. Concrete changes (ranked by leverage)
1. **Doors are passable; "The door opens." is not a bump** (`observe`, `passable`, `route`). Predicted R3 +2 to +3.
   - In `observe()`, when the message contains `door opens` (or `opens.`), do NOT add `(prev, move)` to `blocked`.
     Mark the target square `terrain[target] = "+"` as a door.
   - Credit the prose noun to the square. For a `(vertical|horizontal) (open|closed) door adjacent <dir>` or `doorway adjacent <dir>`
     clause, set that neighbour's terrain to a door class (e.g. `"D"`) that overrides the glyph `-`/`|`.
     `passable()` accepts `D`.
   - In `route()`, forbid diagonal edges into or out of a door square (NetHack rule). This also removes the
     R3/03 `northwest`-into-door bump.
   - Closed door: routing into it opens it (a move into a closed door auto-opens). "This door is locked." →
     `kick` toward it, which is a two-step verb (`kick`, then the bearing).
2. **Universal prompt escape** (`act`, prompt branch). Predicted QE −2 lockups, which is a prerequisite for #5.
   - `[yn]` + `attack` → emit `northwest` (the `y` keystroke) to confirm, or `more` to cancel. If cancelling, add the
     creature's square to a `peaceful` set that `adjacent_foe` skips for the level. This stops the goblin loop.
   - Every other prompt (`What do you want…`, `In what direction?`, `--More--`) → `more`.
   - Detect the prompt from `cursor y==0` as well as from message text. Never feed a (x,-1) pos to `route`.
   - In `ask_model`, project the reply onto compass moves + `search` only. Map `quaff`/`pickup`/`apply`/… to the
     least-visited safe bearing, so the model can no longer open an item prompt.
3. **Boxoban: stop walking, then solve** (`passable`, new solver method). Predicted Boxoban 0 → several.
   - Quick fix: when the prose names `bars adjacent <dir>`, mark that neighbour impassable. When the map has
     `` ` `` + `{` and no `>`, treat `#` as wall and boulders as non-routable, so the explorer stops pushing.
   - Real fix: Boxoban maps are premapped and lit, so they are an exact Sokoban. Run a push-space A*/BFS over
     (player region, boulder set) with walls = `#`, goals = `{`, pruning only on wall corners. Then execute the move tape.
     This was measured at 19/20 in a sibling lineage. The start square under `@` is floor; do not insert a phantom boulder.
4. **Fight from the mouth, never advance at low HP** (`adjacent_foe`, `route`). Predicted QM survival +, CB-Dark
   small (dice), QM wins need #5 too.
   - Define `exposure(c)` = number of passable 8-neighbours of `c`. When ≥2 creature glyphs are within Chebyshev
     distance 3, restrict `route` hops to squares whose exposure is ≤ the current square's. If the current square
     has more than one adjacent foe and an adjacent lower-exposure square exists (the corridor mouth behind you), step
     back ONCE and then commit: swing, or `search` to spend the turn when no foe is adjacent. Do not re-derive the post
     each turn.
   - Do not call the model while creatures are drawn in view. Hold the post (`search`) instead. In QM/00–04 the model's three
     `east` steps (13–15) are what delivered the agent into the pack.
5. **Quest item rung** (new method, gated on `horn`/`wand` in the prose and a charted `>` behind `}`).
   Predicted up to QM 5/5, QE 2/5, the largest single lever.
   - Opener: one `kick` at the spawn pile (east | kick | west), then walk onto each item and `pickup`. Read the letter
     from the pickup message (`j - a zinc wand.`) or from the prompt `[ij or ?*]`.
   - Pick up the wand first, so its letter lands on a reachable keystroke.
   - At the lava edge, run `zap` → letter-verb → bearing toward `}`, and re-run until the prose no longer names lava on that
     bearing. Then `route` to `>`.
   - QM needs #4 first, because the rats guard the lava edge.

## 6. Exhausted / do NOT repeat
- Model-decides-every-step (anchor): 0.100. The chart spine is already the win, so do not hand more steps back to the model.
- Convicting a GLYPH class from a bump: one booked `#` shuts every corridor. Keep the (square, bearing) ledger and
  let prose nouns override glyphs.
- `pray` for HP: refuted in prior rounds.
- HP-threshold retreat that walks away from the fight: every extra degree of disengagement killed more in prior
  rounds. The lever is WHERE you stand when the fight starts (#4), not fleeing mid-fight.
- Kicking the Quest pile twice re-piles it. Kick exactly once.
