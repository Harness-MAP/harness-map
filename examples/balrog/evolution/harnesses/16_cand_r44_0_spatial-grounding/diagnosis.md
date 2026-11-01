# Diagnosis — minihack evolve r044 / cand_r044_0_spatial-grounding

**Headline:** primary = **0.375** (15/40, se 0.077). This is the same as r036 and r041, and the same per-task vector:
- **Wins:** MazeWalk-9x9 5/5, MazeWalk-15x15 5/5, Corridor-R3 4/5, CorridorBattle-Dark 1/5.
- **Losses:** Boxoban-M and Boxoban-H 0/10, Quest-Easy 0/5, Quest-Medium 0/5.
- **Survival 0.775.** 9 deaths: CB-Dark 4, Quest-Medium **5/5**.
- **step_efficiency 0.272.**
- **Cost: 194k tokens/episode, about 9x r041's 21.9k.** Boxoban is 77.6% of the bill and Quest-Easy is 21.6%.

**Binding constraint:** the terrain module works. R3 went 2/5→4/5 against r026, and the bars rule removed about 90% of Boxoban bar bumps. But navigation is still saturated, and everything left over goes to the raw model fallback, which scores 0. On Boxoban and Quest-Easy that fallback owns **893 of 1 000 steps**, orbits, and sometimes emits out-of-vocabulary actions. The three task-gaps named by r041 are still unaddressed: no Sokoban planner, no horn/lava rung, and the fight is taken on the room-mouth square. There are also four cheap HARNESS-BUGs in the fallback and prompt code.

## 1. What this harness does (mechanism)
`harness.py:GroundedAgent.act` tries these rungs in order, and the first one that fires decides the step:
1. **Prompt reflex.** If `fr["prompt"]`, emit `more`, capped at 3 in a row.
   - On `[yn]` + "attack", the last swung glyph is added to `self.peaceful`.
   - A cursor on the message line sets `pos=None`.
2. **`observe()`.** Charts the glyphs and fills `self.facts` from `tools/terrain.py`:
   - adjacent prose nouns become door, doorway, solid or liquid;
   - every charted `+` is a DOOR;
   - "The door opens." marks a DOOR;
   - any other no-move books a `(square, move)` edge in `self.blocked`;
   - if the prose names `bars`, it sets `self.bars`, and then `#` is impassable.
3. **Stairs.** On `>`, emit `down`.
4. **`adjacent_foe`.** Swing at an adjacent creature letter that is not tame and not marked peaceful.
5. **`route()`.** BFS to a drawn `>`, or else to the first frontier dequeued. The frontier is an unvisited cell with an undrawn neighbour.
6. **`ask_model()`.** The raw model is called with the naive instruction plus a banned list, and the reply is projected onto the legal vocabulary.
   - If the model named a banned move, the least-visited safe move is used instead.
   - Otherwise it falls back to `re.sub(...)` of the **raw completion**, so an out-of-vocabulary string is possible.

`AGENT.md`, `rules.md` and `skills/` are empty, so the model gets no priors.

Rung census over this eval's CSVs (`Reasoning` column; empty = model):

| task | chart:route | reflex:swing | reflex:more | model |
|---|---|---|---|---|
| MW9 / MW15 / R3 | 31 / 155 / 276 | 0 | 0 | 0 |
| CB-Dark | 108 | 59 | 0 | 0 |
| Boxoban M + H | **0** | 0 | 0 | **1000 (500+500)** |
| Quest-Easy | 0 | 9 | 98 | **393** |
| Quest-Medium | 65 | 31 | 0 | 16 |

## 2. What actually happened (causal chains)
A CSV row is (action, the observation AFTER that action). Map row = cursor.y − 1.

**A. The terrain module fixed what it targeted.**
- R3 is 4/5, with 0 model steps. The door that `+`/"The door opens." now handles works: R3/03 step 9 "The door opens.", then step 10 walks west through it.
- Boxoban bar bumps fell from 292 in r041 to **about 35 over 800 non-degenerate steps**, because `#` is now impassable once the prose says `bars`.
- MazeWalk is 10/10, unchanged.

**B. Boxoban is now 100% model, and the model orbits (HARNESS-BUG in the fallback, on top of the known TASK-GAP).**
- Once `#` is correctly a wall, a premapped, lit Boxoban room has **no frontier** (every `.` has all neighbours drawn) and no `>`. So `route()` returns None from step 0, and every step goes to `ask_model`.
- **Box-M/00:** the hero walks a strict period-4 loop W→N→E→S over the 2×2 block (41..42, 11..12) for 100 steps, visiting only **5 unique squares**, with 1 boulder push in the whole episode.
- This is the `ask_model` visit-count redirect plus the model's reply producing a rotation. The redirect's `min(visits)` over a 2×2 of equally visited cells is a cycle generator.
- **Box-M/01 and Box-H/01 (HARNESS-BUG):** the model replies `southwest`, but Boxoban's vocabulary is 4 cardinals only, so it is not legal.
  - `decide_action` returns None, and the redirect is skipped because `decide_action(reply, legal)` is also None.
  - The last line then emits the **raw string** `southwest`.
  - BALROG answers "Your previous output did not contain a valid action. Defaulted to action: north".
  - The next prompt contains that complaint, the model repeats `southwest`, and this runs **98/100 and 97/100 steps**, with `failed_candidates` of 97–98 entries.
- Each Boxoban episode costs about 600k tokens for zero pushes planned.

**C. Quest-Easy: the prompt reflex and the fallback create two new 2-cycles (HARNESS-BUG), on top of the known horn/lava TASK-GAP.**
- **QE/00 (quaff loop):**
  - Step 15: `pickup` → "f - a sky blue potion."
  - Step 16: the model emits `quaff` → "What do you want to drink? [f or ?*]". `pos=None`, so the prompt reflex fires.
  - Step 17: `more` → "Never mind."
  - Step 18: the model emits `quaff` again. The **quaff/more 2-cycle repeats 42 times**, burning 84 of 100 steps.
  - Nothing remembers that a verb's prompt was cancelled.
- **QE/04 (peaceful goblin):**
  - Step 5: the swing reflex fires `southeast` → "Really attack the goblin? [yn]". The goblin is correctly added to `peaceful`.
  - The **model**, not the swing rung, then keeps stepping into the goblin with `south`/`southwest` at steps 10, 14, 18, 22, 26, 29, … → 14 more `[yn]` prompts.
  - Each prompt costs 3 steps: move → `more` → `more`. The second `more` yields "Unknown command '^M'.", because the prompt text lingers in `message` after the first Enter has already answered `n`.
  - That is **56 `more` + 28 "Unknown command"**.
  - Root cause: `peaceful` only filters `adjacent_foe`. The peaceful creature's bearing is never added to `closed`/`bad` for `ask_model`. `observe()` also skips edge booking on prompt frames (`not fr["prompt"]`), so the move that triggered the prompt is never blocked.
- **QE/01–03:** the hero is sealed behind the `}` column; `route()` is None, and the model orbits 7–10 squares for 100 steps.
  - QE/02 and QE/03 stand on a dart or dagger pile 23–24 times.
  - No rung ever emits `kick`/`pickup`/`apply` toward the lava.

**D. R3/03 has the same unreachable-`>` loss r041 diagnosed; it is still unfixed (HARNESS-BUG).**
- Step 20: the hero is at (16,7) in the corridor `##@#`. The `>` is drawn at (8,7) in `|....>.`, which is the same row, 8 columns west, across undrawn dark corridor.
- BFS takes the first frontier dequeued, and `ORDER` puts east before west. So the agent tours the east corridor network for 80 steps and times out at step 99, with `>` on screen the whole time.

**E. Every death is a fight on the room-mouth square after stepping out of the corridor (TASK-GAP, posture). 9/9.**
- **CB/00, 01, 02:** die at (35,12). **CB/03:** dies at (38,11). All take the first swing at step 17–19 and reach minimum HP 1–2.
- **Quest-Medium is now 5/5 deaths**; r041 had 4/5. Every episode dies at (36,12) or (37,12) by step 19–24.
- **The QM pattern is visible in QM/00:**
  - `chart:route` reaches (33,12) at step 12.
  - `route()` returns None, the model takes over and steps `east, east, east` into the room: (34,12)→(35,12)→(36,12).
  - "The giant rat bites!" follows, and the hero dies 3 swings later.
- The model fallback walks the hero *out* of the corridor into the rat room on 3 of the 16 QM model steps per episode. That is the whole QM death path.

## 3. Failure breakdown (25 losses)
| category | eps | class | note |
|---|---|---|---|
| Boxoban: no push planner; the fallback orbits or emits illegal `southwest` | 10 | TASK-GAP + HARNESS-BUG | the planner is the only lever that scores; the illegal emit is a cheap cost/validity fix |
| Quest-Easy: sealed behind `}`, no horn rung | 5 | TASK-GAP | QE/00 and QE/04 also lose 84 and 84 steps to the quaff/more and peaceful/more cycles |
| Quest-Medium: dies at the room mouth after the model steps east | 5 | TASK-GAP (posture) | a regression from 4/5 to 5/5 deaths vs r041; noisy at n=5 |
| CB-Dark: dies at the room mouth | 4 | TASK-GAP (posture), dice | price at ≥200 eps |
| R3/03: drawn `>` unreachable, frontier tie goes east | 1 | HARNESS-BUG | the same episode r041 flagged |

## 4. Binding constraint
**TASK-GAP, with one fixable HARNESS-BUG on R3 and several cost bugs.**
- The frozen model decides 409 steps, every one of them on tasks that score 0. It is harmful: illegal emits, 2-cycles, and walking into the QM rat room. It is also 100% of the cost.
- The terrain layer did its job. Re-wiring glyph semantics again cannot add points, because MazeWalk and R3 are already won on every episode the chart can reach.
- Primary moves only through new goal mechanisms: a Sokoban planner, a horn/lava tape, and a corridor fight post. The R3 `>`-attractor is the one cheap point.

## 5. Concrete changes to try (ranked by leverage)
1. **A `sokoban_plan()` rung in `act()` before `route()`. Up to +10 eps, about +0.25 primary.**
   - **Trigger:** the frame has `{` and `` ` `` and `self.bars` is set (observation-only).
   - **Decode:**
     - The whole map block: `#` = wall, `` ` `` = boulder, `{` = target.
     - The hero's start square may be a masked target: `#{` may be `#boulders − 1`. Allow one unknown and place it under `@`.
   - **Search:** push-space A*.
     - State: the canonical reachable region plus the boulder set.
     - Heuristic: the sum of each boulder's minimum Manhattan distance to a target.
     - Dead-square prune on walls only.
     - Cap at about 200k nodes.
   - **Execution:** cardinal moves only, from a committed tape. Re-plan only when a step no-moves or the boulder set deviates.
   - **Predicted:** chain memory reports 19/20 stub and 10/10 live, winning iff the optimal plan is ≤99 steps. Boxoban cost drops to 0 tokens, which is **−78% of the whole bill**.
2. **Make the fallback vocabulary-safe (`harness.py:ask_model`, last lines). Cost and validity; also removes the Box-*/01 degenerate episodes.**
   - Never emit a raw completion. If `pick` is None, use the least-visited legal, non-banned move whenever `pos` is known. When `pos` is None, emit `more` if legal, else `wait`.
   - Also break the visit-count rotation: tie-break by `(visits, turns-since-last-visit, ORDER)` instead of `ORDER`, or better, make Boxoban unreachable for this path via (1).
3. **Goal-attracted frontier in `route()` (fixes §2D). R3 4/5→5/5; chain memory says +2/−0 of 125.**
   - When `stairs` is non-empty and BFS exhausts without reaching one, collect **all** frontier cells with their BFS depth.
   - Pick the one minimising `depth + cheb(cell, nearest >)`, with ties broken by level coordinates, not `ORDER`.
   - Remove the `break` on the first frontier when stairs exist; currently it only breaks when `not stairs`, but `best` is still the first-dequeued frontier.
   - Gate it so it does not aim across a `}` column.
4. **Peaceful and cancelled-prompt memory (`observe()` and `act()`). Frees about 80 steps each in QE/00 and QE/04; a prerequisite for any Quest rung.**
   - On a `[yn]` "Really attack" frame, book `self.blocked.add((prev_pos, move))` for the move that caused it. Keep `self.last` across the prompt frame instead of resetting it.
   - Also add the bearings of every adjacent glyph in `self.peaceful` to `closed` before `ask_model`.
   - Answer `[yn]` once. When the previous action was `more` and the same `[yn]` text is still in `message` while the cursor is back on the map (`pos` not None), treat the prompt as closed and do not emit a second `more`.
   - On "Never mind." after a verb prompt (`quaff`, `apply`, `read`, …), ban that verb in `ask_model` for the next K=10 steps.
5. **A `quest_tape()` rung. Up to +5 QE, and QM once (6) holds.** Follow the chain-memory recipe:
   - Walk to the spawn pile.
   - `kick` it once, never toward `}`.
   - `pickup` the horn.
   - `apply` it, and answer "In what direction?" with the lava bearing. This needs a direction-prompt branch in the prompt reflex, which today answers every prompt with `more`.
6. **A corridor fight post, and never let the model step out of a corridor while a creature is drawn (§2E).**
   - When the current square has ≤2 passable neighbours, a creature letter is within Chebyshev 3, and the next hop (route **or model**) enters a square with ≥3 passable neighbours, emit `wait`.
   - Release after 2 steps with no letter within 3, and don't count the walk to the post in that clock.
   - At minimum, when `route()` is None on QM/CB, prefer `wait` over asking the model. The model's `east×3` is the whole QM death path.
   - No HP retreat.
   - Price CB on ≥200 eps.

## 6. Exhausted — do NOT repeat
- **Glyph→terrain rewiring.** This candidate is exactly that axis. It fixed doors and bars, and primary is unchanged at the 0.325–0.375 plateau (r026/r036–r038/r041/r043/r044).
- **Re-plumbing the chart/learner/model order.** Same plateau.
- **Prompt, AGENT.md or skills wording.** The model owns only 0-score tasks.
- **Glyph-level `#` = wall without a bars gate.** It breaks R3 and MazeWalk; the prose `bars` gate used here is the safe form.
- **HP-triggered retreat.** Refuted by the dose-response 9→13→17→42.
- **Pricing CB-Dark on 5 episodes.**
- **Answering `[yn]` with `esc`/`y`.** Not in the vocabulary. `more` works, but only once per prompt (§5.4).
