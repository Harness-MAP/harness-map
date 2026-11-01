# Diagnosis — minihack iter r056 / cand_r056_0_spatial-grounding (one-module stage table + door-aware chart)

**Headline:** primary = **0.375** (15/40, se 0.077). Survival 0.775 (9 deaths), step_efficiency 0.273, cost **97k tok/ep** (parent r055: 234k).

| Task | Result | Who decided |
|---|---|---|
| MazeWalk-9x9 | 5/5 | `chart:route` 31/31 steps, 0 tokens |
| MazeWalk-15x15 | 5/5 | `chart:route` 155/155 steps, 0 tokens |
| Corridor-R3 | **4/5** (parent 2/5) | route 274/274 steps, 0 tokens; the one loss hit the 100-step cap |
| CorridorBattle-Dark | 1/5, 4 died | route 108 steps + `reflex:swing` 59, 0 tokens |
| Boxoban-M/H | 0/10 | route 722, model 278 (Hard ≈ 277k tok/ep) |
| Quest-Easy | 0/5 | model 488/500 steps (~415k tok/ep) |
| Quest-Medium | 0/5, **5/5 died** (parent 2 died) | route 65, model 16, swing 31 |

**The door fix worked.** The r055 HARNESS-BUG (an opened door charted as wall) is fixed. R3 went 2/5 → 4/5, and there is no `model` row on R3 at all. The whole +0.100 over r055 is R3 +2, MW15 +1 and CB-Dark +1. The CB-Dark gain is dice at n=5. This candidate now sits exactly at the chart lineage's plateau (r041/r046/r051/r054 = 0.375).

**Binding constraint:** a **TASK-GAP**. The 25 lost episodes fall into two groups:
- **20 episodes (Boxoban, Quest)** have no code mechanism, so the frozen VLM random-walks.
- **5 episodes (CB-Dark ×4, R3/03)** are dice or a single frontier-ordering miss.

One cheap HARNESS-BUG sits on top: the route has no "drawn-but-unreachable `>`" phase (R3/03). The QM survival regression is also self-inflicted: after the route runs out of targets, the model walks the hero into the rat room.

## 1. What this harness does (mechanism)
`orchestrator.py:GroundedAgent.act` parses the frame (`parse_glyph_map`: map row = cursor.y−1) and walks `STAGES = (prompt, orient, stairs, fight, route, model)`:
- **`st_prompt`**
  - Answers `Really attack? [yn]` with `northwest` (the `y` key).
  - Answers every other prompt, **including every `What do you want to …`**, with `more` (cancel).
- **`st_orient` / `observe`**
  - Folds glyphs into `terrain`.
  - Adds a cell to `doors` if it is ever drawn `+`, named as a door in the prose, or is the bump target of `The door opens.` (and that bump is not booked as `blocked`).
  - Books every other no-move as a blocked edge.
- **`st_stairs`**: `down` on a `>`.
- **`st_fight`**: swings at any adjacent letter glyph.
- **`st_route`**: BFS over charted floor. It targets a **reachable** `>` first, else the nearest unvisited cell that borders undrawn space. There is no diagonal into or out of a door.
- **`st_model`**: only when the route returns None. It makes one naive-prompt call, and the reply is projected onto moves + `down`.

## 2. What actually happened (causal chain)

### 2a. Quest-Medium: the route exhausts in the rat room, then the model marches east (5/5 deaths, identical trajectories)
The trajectory is byte-identical across all 5 episodes up to step 12 (`ea no ea ea so so ea ea ea ea ea ea no`):
1. The route walks the corridor and, at step 12, enters the lit room at (33,12) through its west mouth.
2. On that frame the whole room is drawn: `|@.....r.rr.........}...|`, with `>` behind the `}` column.
   - Lava is `LETHAL`, so `>` is unreachable.
   - The room has no undrawn border cell, so there is no frontier.
3. **`st_route` returns None and `st_model` fires.** It answers `east` (toward the drawn stairs) 3–4 times (ep00 steps 13–15), into a room holding 6–8 giant rats.
4. At (36,12) the hero has rats **NE, E and SE** (`giant rat adjacent northeast, east, and southeast`). `st_fight` swings `east` every step, HP falls 16→2 in about 6 steps, and the hero dies at step 18–23.

The same frame in chain memory: *"advancing into the room dies at step 39, holding the mouth wins at step 68"*. The corridor mouth (x≈31–32, row 12) has exposure 1. The room interior has exposure 3.

### 2b. Quest-Easy: the route never fires (488/500 steps are the model)
The start room is `|.@...}` with the lava column at x=30, 4 squares east. Every reachable square is already drawn, and every border square is across the lava, so **`st_route` has no target from step 0**. The model then orbits the 4×5 room for 100 steps at ~415k tok/ep. It passes `horn adjacent south` on step 40 (frame shows `|.(...}`) and never picks the horn up. Nothing in the harness ever emits `kick`/`pickup`/`apply`/`zap`. `ask_model` projects the reply onto moves + `down`, so the model cannot either.

**`st_prompt` would also actively break an item mechanism.** It answers every `What do you want to use or apply? [ij or ?*]` with `more` (cancel). The item-letter answer (a letter-equal verb, see §5.1) has to come before that branch.

### 2c. Corridor-R3/03: `>` drawn, unreachable, and the frontier walks away (1 loss)
At step 30 the map shows `|....>.` at the far west (x≈8, row 6). The corridor `#####------` stops at x≈14, so the gap x=10–13 is undrawn and the `>` is **not BFS-reachable**. `route()` then takes the *nearest* frontier (`south`, distance 1). Over the remaining 70 steps it explores the east and south corridors: the most-visited cell is (34,14), 26 columns from the `>`. The episode hits the cap at step 100. There is 0 model involvement; it is a pure frontier-ordering loss. This is exactly the case in chain memory *"a DRAWN but unreachable `>` is an attractor"*: R3 4/5→5/5, +2/−0 over 125 levels, 0 arms off R3 once gated off when `}` is drawn.

### 2d. CorridorBattle-Dark: 4/5 deaths
The route walks east along the corridor to the room mouth (x=34). In ep00 it kills the first rat at the mouth, then **steps into the room at x=35** (step 20, `chart:route` on a frame with no adjacent foe). Two rats close diagonally (`giant rat adjacent northeast and southeast`), and HP goes 13→1. Ep01 and ep02 follow the same pattern: the fight moves one square past the mouth, where it is 2-exposed. The win (ep04) fought 11 steps at x=38 with **one** adjacent rat. This is consistent with the chain's "exposure(cell) is the fight square" result, but CB-Dark is dice at n=5 (chain: price at ≥200–500 eps).

### 2e. Boxoban: 0/10, as expected
The board is premapped. `route` fires on 72% of steps because `0` (boulder) and `#` (bars) are not in `WALL`, so the BFS treats them as passable and shoves boulders at random frontier cells. Boxoban has no push planner.

## 3. Failure breakdown (25 losses)
| Category | Eps | Cause | Fixable in code? |
|---|---|---|---|
| Boxoban: no push planner | 10 | TASK-GAP | yes: an exact push-space search (chain: 10/10 when the plan is ≤99) |
| Quest: no item/lava mechanism | 10 (5 QM deaths, 5 QE orbits) | TASK-GAP; `st_prompt` cancels item prompts | yes: the kick-split opener, then apply/zap (chain: QM 3–5/5, QE 2/5) |
| CB-Dark melee deaths | 4 | dice, plus a 2-exposed fight square | survival only; not measurable at n=5 |
| R3 drawn-unreachable `>` | 1 | `route()` frontier ignores the `>` | **yes, ~10 lines, known +** |

## 4. Binding constraint
**TASK-GAP (20/25 losses).** The harness is at the chart lineage's ceiling on the 20 "chart" episodes: MW 10/10, R3 4/5, CB-Dark dice. Every remaining point is in Boxoban or Quest, where the VLM is inert: the live score equals the null-model score on those 20 episodes, and the model contributes 0 wins and ~100% of the tokens. Neither prompt wording nor stage restructuring will move them. Only a code mechanism will.

**Secondary HARNESS-BUG:** there is no seek phase toward a drawn-unreachable `>` (R3/03), and the no-target fallback hands QM to a model that walks into the rat pack.

## 5. Concrete changes (ranked by leverage)

1. **Quest item stage `st_quest`, placed after `st_fight` and before `st_route`, armed only when `}` is drawn.** This is the replicated opener from chain memory (QM 0/5→3/5 on fixed_k8_c1, 5/5 QM + 2/5 QE on adaptive_k4_questfix).
   - **Acquisition.** At episode start the hero stands on a 2-object pile. `pickup` there is silent, so do this instead:
     - step off, `kick`, and answer the direction prompt with the bearing back to the pile, which splits it;
     - walk to the drawn `(`, `pickup` (→ `i - a horn.`), then walk to the `/` and `pickup`.
     - Kick **once** (a double kick re-piles the items), and never kick toward lava (it burns the horn).
   - **Use.**
     - Walk to a square cardinally adjacent to the `}` column, on the `>` side.
     - Then either `zap` → the letter-verb → the bearing, or `apply` → the letter-verb → `northwest` (answers `Improvise? [ynq]`) → the bearing.
     - Read the item letter from the prompt text `[ij or ?*]` and map it to the verb with the same key: h/j/k/l/y/u/b/n = directions, a=apply, e=eat, o=open, q=quaff, s=search, z=zap, c=close.
     - Pick up the wand first if the horn's letter would be unreachable.
     - After `The lava cools and solidifies.`, hand back to `st_route` (`}` squares redraw as floor).
   - **`st_prompt` must yield to this stage.** Answer `What do you want to …` / `In what direction?` from the pending item plan first, and fall back to `more` only when no plan is pending. As written, `st_prompt` sits first in `STAGES` and cancels every item prompt, so the mechanism would be dead on arrival.
   - **QM fight placement.** While a drawn rat is in the room and the hero is at the corridor mouth (x≈31, exposure 1), **hold there**: emit `search`, and swing only on adjacency. Do not do the lava errand until the rats are dead. Chain memory: an errand rung above the fight rung gets the hero eaten.
   - **Predicted:** QM 0→3–5/5 and QE 0→1–2/5 (+0.10 to +0.175 primary), QM survival +3–5 episodes, and Quest cost down ~400k tok/ep.

2. **`route()`: a seek phase for a drawn-unreachable `>`, gated OFF when `}` is drawn.** If `stairs` is non-empty and the BFS never reached it, choose the reached unvisited cell (plus frontier cells) that minimises `max(|dx|,|dy|)` to the nearest `>`. Break ties in the level frame `(cheb, y, x)`, not by BFS order, so the target does not flip as the body moves. This re-orders the existing candidate set and adds no veto.
   - Predicted: R3 4/5 → 5/5 (+0.025). Chain replay armed it on R3 only (106 arms, 0 on other tasks).
   - Implementation: keep the BFS running to completion when `stairs` is drawn, collect `seen`, and pick the target after the BFS instead of breaking at the first frontier.

3. **Boxoban push planner stage `st_sokoban`**, armed when `^` targets and boulders are drawn with no `>`:
   - decode the premapped board (`0`/`` ` `` = boulder, `#` = bars = wall on Boxoban, `^` = target);
   - search push space with dead-square pruning on walls only;
   - emit the plan as a committed tape, and re-plan if the observed boulder set diverges from the prediction.
   - Watch the phantom boulder: the square under `@` at the start is floor.
   - Predicted: +3–6/10 realistically under the 100-step cap (chain: 10/10 live where the optimal plan is ≤99; exact solver 19/20). It also removes ~180k tok/ep.

4. **Never call the model on a no-target frame when creatures are drawn** (`st_model` pre-empt). On QM the model's `east` ×3 after route exhaustion is what walked the hero from the mouth into a 3-exposed square. Until #1 lands, emit `search` at the current square if it has ≤1 open neighbour toward the drawn creatures; otherwise take the BFS hop back toward the last corridor (`#`) square. Predicted: QM deaths 5 → ~2–3 (survival +0.05 to +0.075), primary unchanged, fewer tokens. Chain memory: *a creature is a RANKING term, never a REFUSAL*. Keep the reflex swing on adjacency.

5. **(Survival only; price at ≥200 eps before shipping.)** CB-Dark: after a kill at the mouth, do not let `st_route` advance into a square with more open neighbours while creatures are drawn within 2. This is the "committed fight post" (chain: +9/200 CB-Dark, p≈0.14). Do not expect it to move primary at n=5.

6. **Cost:** on QE and Boxoban, until #1 and #3 exist, replace `st_model` with a zero-token least-visited walk. Primary is unchanged (the VLM is inert there) and cost falls 97k → <20k tok/ep.

## 6. Exhausted — do NOT repeat
- **Stage/pipeline rewiring by itself.** r026/r046/r054/r055 and now r056: the rewire itself was neutral, and this candidate's gain is entirely the door mechanism inside it.
- **Frontier re-ranking or visit constants on MazeWalk.** Saturated at 10/10. Change #2 is the only R3 frontier reformulation not already refuted.
- **Prompt/AGENT.md/prior wording** to make the VLM push boulders, use items or avoid rats. The VLM contributes 0 wins in this lineage.
- **`y` / `esc` as prompt answers.** They are not in the vocabulary; use `northwest` for `y` and `more` to cancel. (Already fixed here.)
- **CB-Dark melee posture priced on 5 eps** (retreat, hold, recoil, HP gates). It is dice. Never make a creature a refusal/veto.
- **Quest:** `pickup` on the spawn pile without the kick-split (silent no-op), a double kick (re-piles), a kick toward lava (burns the horn), zapping after a dry route exists, and an errand rung above the fight rung.
- **Treating `#` as bars on Boxoban as a primary lever** (r054: bumps 297→27, score still 0/10). It only cleans the fallback.
