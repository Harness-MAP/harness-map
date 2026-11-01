# Diagnosis — minihack iter r050 / cand_r050_0_spatial-grounding (chart-routed learner)

**Headline:** primary = 0.325 (13/40 on the env success flag; the parent r035 scored 0.150).
Per task: MazeWalk-9x9 5/5, MazeWalk-15x15 5/5, Corridor-R3 2/5, CorridorBattle-Dark 1/5, Boxoban-Hard 0/5,
Boxoban-Medium 0/5, Quest-Easy 0/5 (all timed out), Quest-Medium 0/5 (all **died** by step 26).

**Binding constraint:** the chart writes the glyph of an opened door (`-`/`|`) into `WALL`. On R3, `route()`
then loses every frontier behind that door, and the random-walking learner takes over for the rest of the
episode. This happened in all 3 R3 losses, and in one of them the `>` was 6 squares past the door.
It is a HARNESS-BUG. The largest remaining pool is Boxoban (10/40), where no route exists and the learner
wanders. A push-space solver fixes that (see TASK-GAP below).

Note: R3 run_04 shows `progression=0.0` but `end_reason=2` (return 0.99 on its last step). It **is** a
success under the evaluator's `success` flag (BALROG's `progression` requires reward ≥ 1.0). R3 is therefore 2/5, not 1/5.

## 1. What this harness does (mechanism)
`orchestrator.py:LearnerAgent.act` runs these steps in order, one module, with no tools/skills:
1. Prompt reflexes: an attack `[yn]` gets `y`, `--More--` gets `more`, any other prompt gets `esc`.
2. `chart(fr)` copies every non-blank glyph into `self.terrain`. A no-move after a non-swing move adds
   `(square, move)` to `self.blocked`.
3. `credit()` runs the TD(λ) update. The stairs reflex emits `down` on a remembered `>`.
4. `foe_bearing`: swing at any adjacent letter (`reflex:swing`).
5. `route()` runs BFS over `passable` terrain (`WALL = "|- "`, `}` lethal) to a drawn `>`, or else to the nearest
   unstood square that touches undrawn space. It returns the first hop (`chart:route`).
6. If there is no route, the value learner takes an argmax with an optimism bonus (`learner:argmax`). On ties it
   consults the VLM once per square (`learner:consult`). AGENT.md is one sentence used only for that consult.

Reasoning census (judge.json): argmax 1300, route 589, consult 347, swing 101. **The learner owns 1647 of
2337 steps (70%), and every one of those steps is in a losing episode.** Every win is route-driven and
model-free (the MazeWalk and R3 wins use 0 tokens).

## 2. What actually happened (causal chains)

### A. Corridor-R3: an opened door becomes a wall and routing dies (3/3 losses)
- **run_01**, steps 52–54: the hero is at (20,3) next to a `+` door. Step 53's message is `The door opens.`, and the frame
  now shows `|>....-@######`: the door is drawn `-` and **`>` is drawn 6 squares west**. `chart()` stores `-`,
  `passable()` rejects it (`-` ∈ `WALL`), and BFS cannot reach `>`. Step 54 turns **east** (`chart:route` to a
  different frontier). From step 76 the route is gone and the learner bumps (`It's solid stone.` / `It's a wall.`
  ×10) until the step cap.
- **run_03**, steps 8–10: the route heads for the room's only exit, a `+` at the NW corner, and it approaches
  **diagonally** (`northwest`). Step 9 shows `The door opens.` with the position unchanged at (20,4). `chart()` then
  (a) convicts `((20,4),'northwest')` in `blocked` because of the no-move, and (b) the door becomes `-`, so it is a wall.
  The other exit, a corridor from the east doorway, is a dead end at (25,5), where a hidden passage needs a search.
  So no frontier is left. Steps 10–99 are learner steps (73 argmax + 17 consult) inside a 5×5 room.
- **run_00**, step 38: `The door opens.` at (12,17). From step ~55 the route is gone, and 30+ learner steps later
  the hero is still bumping at (12,17).
- The one `The door resists!` (a stuck door) also convicts its edge through the same no-move rule, even though
  retrying would open it.

Both R3 wins (run_02 in 30 steps, run_04 in 40) were 100% `chart:route` and went through no door.

### B. Boxoban: no route exists, so the learner walks at random (10/10 timeouts)
Boxoban is fully lit and walled, so `frontier()` is empty from step 0 and `route()` returns None on 482 of 500
steps. The learner's reward (new cells, new squares, `>` gap) has nothing to do with sokoban: it never reads
boulders (`` ` ``) or goals (`{`). The result is 14–30 unique squares per 100 steps, 55 `boulder in vain` bumps, and
40 random pushes (`With great effort you move the boulder.`), which usually create deadlocks. Example:
Boxoban-Medium run_00 is `east south north south south west east west north …`.

### C. Quest-Easy: lava splits the level, so the learner orbits (5/5 timeouts)
The run_00 opening frame shows the hero's side walled in, with a `}` column across the room. No `>` is reachable
and there is no frontier on the hero's side, so the route never fires (0 `chart:route`). There are 100 learner steps
on about 22 squares. Known: winning requires the item / lava-crossing chain (kick the pile, pick up, apply/zap).
This is a TASK-GAP.

### D. Quest-Medium: the learner walks into a rat swarm in an open room (5/5 deaths, steps 19–26)
In all 5 runs the same route opening (13 hops) brings the hero through the corridor into the lit east room. There
the frontier is empty, because the room is fully drawn and the `>` is beyond the `}` column. The learner then
emits `east` toward the rats. For example, in run_00 at step 15 (hero at (36,12), `|...@rrr`) there are 3 rats
adjacent (NE/E/SE). `reflex:swing` then trades blows at 3:1 exposure until `Killed by a giant rat` at step 18.
The doorway at (32,12), one move back west, has exposure 1. No win was available (`>` beyond lava), but
**survival 0/5 on this task drags down the survival key**.

### E. CorridorBattle-Dark: 4 deaths at steps 27–34
This is route plus swing. The episodes look like the documented dice-roll melee; there is no new failure mode.

## 3. Failure breakdown (40 episodes)
| Category | Episodes | Owner rung | Class |
|---|---|---|---|
| Opened door is charted as a wall, so the route dies | 3 (R3 00/01/03) | `chart`/`passable` | HARNESS-BUG |
| No sokoban planner, random walk | 10 (Boxoban) | learner | TASK-GAP (scaffold) |
| Lava-partitioned level, no acquisition chain | 5 (QE) + 5 (QM) | learner | TASK-GAP |
| Advancing into a swarm in the open | 5 deaths (QM) | learner + swing | HARNESS-BUG (survival) |
| Melee dice | 4 deaths (CB-Dark) | swing | VLM-free dice, closed |

## 4. Binding constraint
**HARNESS-BUG in `chart()`/`passable()`/`route()`: door handling.** It is cheap and deterministic, and it recovers the
R3 losses. After that fix, the next binding constraint is the **Boxoban TASK-GAP**: the frozen VLM plays no part in it (it is
consulted only on ties), so the gain has to come from a planner in the harness.

## 5. Concrete changes (ranked by leverage and certainty)

1. **Door-aware chart (R3; predicted +1 to +3 wins, i.e. +0.025 to +0.075 primary).** In `orchestrator.py`:
   - `chart()`: keep `self.doors`, a set of cells whose terrain was ever `+` or that are named `door` in the adjacent
     prose. When a cell in `self.doors` is now drawn `-`/`|`, store the marker `"'"` for an open door
     instead of the wall glyph. Also add the pending target square to `self.doors` when the message contains
     `door opens`.
   - `chart()`: do **not** add `(s, a)` to `blocked` when the message contains `door opens`, `door resists`,
     `door is stuck` or `door is locked`. The turn was spent on the door, and no wall was hit.
     Classify every no-move message rather than treating every no-move as a wall.
   - `route()`: forbid a diagonal hop whose source or destination is in `self.doors` (NetHack: no diagonal moves
     through a real door). Doorless `.` doorways stay diagonal-legal. This also fixes run_03's diagonal
     approach to the `+`.
   - `This door is locked.`: if `kick` is in `legal`, emit `kick`, then the door's bearing (two steps; answer the
     direction prompt instead of sending `esc` from the prompt reflex).
   Prior evidence on sibling lineages: door-passable plus a one-shot refusal gave +89/−6 over 320 R3 levels.

2. **Sokoban planner for the no-route, boulder-plus-fountain regime (Boxoban; predicted up to +0.25, the largest pool).**
   Add `LearnerAgent.sokoban(fr)`, called in `act()` before the learner fallback and gated on drawn `` ` ``
   boulders, `{` goals and no `>`. Parse walls (`#` here is **bars**, plus `-`/`|`), boulders and goals from the
   premapped frame. Run BFS or A* in push space: the state is (boulder set, hero reachability region), and moves are
   pushes; prune simple corner deadlocks (a boulder in a non-goal corner). Compile the solution into a tape of
   compass moves and commit to it. Re-plan whenever the observed boulder set differs from the predicted one. The
   learner must never push a boulder: until a plan exists, give pushes a value of `BUMP_R`. Prior evidence: a push-space
   A* scored 19/20 on a stub rig, and Boxoban wins exactly when the optimal plan fits in about 99 steps.

3. **Don't advance into the open when the frontier is empty and creatures are drawn (QM survival; primary ±0).**
   In `act()`, when `route()` is None and at least 2 creature glyphs are drawn within 3 squares, prefer the move
   that reduces exposure (the number of passable neighbours reachable by a creature). Here that is back into the doorway at (32,12), and
   then `search`/`wait` as the null move, swinging only at an adjacent foe. Gate it on the no-route regime so that
   CB-Dark (always routed) is untouched. Earlier lineages found that disengaging on CB-Dark kills more, so keep it
   off the routed path.

4. **Learner fallback on an exhausted level (cost/efficiency).** When there is no route, no `>` and no boulder, the learner's
   bumps (`It's solid stone.` ×7 on R3) are wasted. Emit `search` at the dead-end corridor square nearest to
   undrawn space (R3 run_03's (25,5) is a hidden-corridor dead end) instead of the argmax walk. `search` probes
   all 8 neighbours, so budget it per probed cell.

5. **Quest pair (lower priority, heavy).** Both Quest tasks are lava-partitioned. The known winning chain is to kick the
   spawn pile, pick up the item, `apply`/`zap` it toward the lava, and then walk the solidified bridge. This is only
   worth building after items 1 and 2.

## 6. Exhausted — do NOT repeat
- Prompt or prose edits (AGENT.md/rules/skills): the model is consulted only on learner ties, and every win is
  model-free. Prose cannot move primary.
- Re-tuning the learner's reward constants (optimism, bonus, λ): the learner owns only losing, no-route regimes, and
  the fix belongs to the regime (doors, sokoban), not to its constants.
- Frontier ranking reformulations (gain rate, extremity, nub demotion): documented as flat, and MazeWalk is already 10/10.
- CB-Dark melee posture (retreat, hold, recoil): documented as dice. Price any change on ≥ 200 episodes, never on 5.
