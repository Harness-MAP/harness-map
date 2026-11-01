# Diagnosis — minihack iter r066 / cand_r066_0_spatial-grounding (stage pipeline + r046 Quest port)

**Headline:** primary = **0.375 (15/40)**, survival 0.800, step_eff 0.273, cost **162.8k tok/ep**.

| Task | Wins |
|---|---|
| MazeWalk-9x9 | 5/5 |
| MazeWalk-15x15 | 5/5 |
| Corridor-R3 | 4/5 |
| CB-Dark | 1/5 |
| Boxoban-Medium + Boxoban-Hard | 0/10 |
| Quest-Easy | 0/5 |
| Quest-Medium | 0/5 (4 deaths) |

This is the same per-task vector as r041, r044, r046, r048, r051, r054 and r056, which is the lineage's 0.375 floor.

**Binding constraint:** the new `quest` stage is a port of the wrong module. `tools/quest.py` is the **r046** Quest module, which never scored a Quest win. It has no pile **kick**, no mouth **hold**, and no letter filler. The two modules that already solve these tasks were left behind:
- r058/r065 `tools/quest.py`, which takes Quest 0/10 → 10/10.
- r064 `tools/sokoban.py`, which takes Boxoban 0/10 → 10/10.

Both are validated modules, and both already fit this exact `GroundedAgent`/`tools/chart.Chart` layout (r065 *is* this lineage).

This is a **TASK-GAP with a known, verified fix**: 20 of the 25 losses come from two missing stage owners. The one fixable remainder is R3/03, where the `>` is drawn but unreachable.

## 1. What this harness does (mechanism)
`harness.py:GroundedAgent.act` builds a context `c` and walks `STAGES = (prompt, orient, stairs, fight, quest, fetch, route, model)`. The first answer wins, and its rung goes to `reasoning`. A blank `Reasoning` in the CSV means the model's template was emitted, i.e. a VLM step.
- `st_orient`:
  - `chart.book()` credits the last move. "The door opens." marks a door; any other no-move closes the (square, bearing) edge.
  - `chart.fold()` charts glyphs plus prose door/bar nouns.
  - `deadly()` covers prose lava/water and `}`.
  - `chart.route()` returns a BFS hop to a drawn `>`, or else to the nearest unvisited frontier.
- `st_fight` moves into any adjacent letter that is not `@`/peaceful.
- `st_quest`/`st_fetch` (tools/quest.py) arm when `pickup ∈ legal` and a `}` is charted:
  - one spawn `pickup`, then a `pickup` on any drawn item glyph;
  - once a horn or wand is in `inv`, walk to a bank and `apply`/`zap` across the lava.
- `st_model` projects the reply onto compass moves plus `search`.

AGENT.md, rules.md, skills and memory.md are all empty.

## 2. What actually happened (causal chains; CSV row = action, obs AFTER it)

**A. Quest-Easy 0/5: the horn is never acquired, so the model orbits for 90–99 steps.**
- QE/00 steps 7–9: `quest:pickup:spawn` returns a blank message. That is the **silent pile** signature: the spawn square holds a pile that `pickup` cannot open.
- It then walks one square east, picks up "f - a sky blue potion" and nothing else.
- `usable()` finds no horn or wand. `targets()` has no untried item glyphs left. `route` has no frontier, because the lit room is fully charted and the far side lies across `}`.
- Every step after that falls to `st_model`. Steps 10–99 are a strict 4-cycle (`east, east, southwest, northwest`) with blank reasoning.
- The same thing happens in QE/01–03: 97–99 model steps, and the only items picked up are a potion, darts or a dagger. **0 `kick`, 0 `apply`, 0 `zap` in 500 QE steps.**
- Chain memory has this recorded as REFUTED-unwinnable: a single `kick` splits the spawn pile, and the horn then lands on a drawn square. r052, r058 and r065 carry that `kit` logic in `tools/quest.py:kit/kick_dir` ("kick ONCE, never toward lava"). r046's module has no `kick` word anywhere.

**B. Quest-Easy/04: the model keeps attacking a peaceful goblin (59 `reflex:more` steps).**
- At step 9 the swing reflex hits "Really attack the goblin? [yn]". `st_prompt` then correctly adds the goblin to `self.peaceful` and answers `more`.
- The goblin stays east of the corridor. From step 14 on, the **model** (blank reasoning) emits `east` into it again.
- Each attempt costs a 3-step cycle (`east` → `[yn]` → `more` → "Unknown command '^M'"), repeated about 20 times.
- `peaceful` is only consulted in `st_fight`. `st_model`'s `banned` set does not include the square of a known peaceful. The `if ... last[2] == "swing"` guard in `st_prompt` also means a model-caused prompt never marks it.
- The −0.90 return, versus about −0.03 on the other QE runs, is this loop.

**C. Quest-Medium: 4/5 deaths in a rat corridor, with no hold.**
- All five runs follow the same script:
  - steps 0–5 are `chart:route:frontier`;
  - step 6 `pickup:spawn` gives "There is nothing here to pick up.";
  - steps 7–11 are `quest:walk`;
  - step 12 `pickup:item` returns a blank message (another silent pile, never kicked).
- After that, `route` walks **east down the corridor into the giant-rat pack** (steps 13–21). `st_fight` then swings on whatever square it stands on. QM/00 dies at step 31 after 9 swings; QM/02, 03 and 04 die at steps 28, 33 and 32 the same way.
- r058/r065 solve QM 5/5 with a `hold` stage (fight at the mouth, spend turns rather than advance). That stage is exactly what `tools/quest.py` here lacks.
- QM/01 survived only because the model took over (66 blank steps).

**D. Boxoban 0/10: nothing owns the push puzzle.**
- `route` spends 18–46 steps per episode pushing frontier squares along the bar ring. The rest (54–82%) are VLM steps.
- Returns are all ≈0 or −0.1. The VLM steps are **≈100% of the 162.8k tok/ep bill**: 4.3M Boxoban tokens plus 1.8M QE tokens, against 0 tokens on all 20 MazeWalk/R3/CB-Dark episodes.
- r064's `tools/sokoban.py` (push-level A*, armed when the legal moves are exactly the 4 cardinals and a boulder plus `{` are drawn) solved all 10 of these seeds' frames offline and 10/10 live.

**E. Corridor-R3/03 loss: the `>` is drawn but unreachable, and the frontier ignores it.**
- By step 50 the map shows `|....>.` (row 6, `>` at col 8), with a 4-column undrawn gap before the corridor `#####` at col 14.
- The room's walls and doorway are undrawn, so BFS has no path to `>` and `route` returns `kind=frontier` for **all 100 steps**.
- The nearest-unvisited frontier then walks the agent to the east rooms (step 97 is at (24,9), exploring the SE room at cols 35–43). The walls around `>` never get charted.
- Memory `balrog-goal-attracted-seek-phase-converts-drawn-unreachable-stairs`: aiming the frontier at a drawn-but-unreachable `>` converted R3 4/5 → 5/5 (+2/−0 over 125 levels).

**F. CB-Dark 1/5 (4 deaths after 7–14 swings).** This is at the lineage's ceiling. Memory says CB-Dark is dice and the melee axis is closed (see §6).

## 3. Failure breakdown (25 losses)
| Cause | Episodes | Class |
|---|---|---|
| Boxoban: no push planner; route and VLM random-walk | 10 | TASK-GAP (fix exists: r064 sokoban.py) |
| Quest-Easy: silent pile never kicked, so no horn; model orbit | 5 (1 also has the peaceful-goblin loop) | TASK-GAP / port bug (fix exists: r065 quest.py) |
| Quest-Medium: pile never kicked, and the route walks into the rat pack with no hold | 5 (4 deaths) | TASK-GAP (fix exists: r065 quest.py hold) |
| CB-Dark melee deaths | 4 | dice (closed axis) |
| R3 drawn-but-unreachable `>` | 1 | HARNESS/TASK-GAP (goal-attracted frontier) |

The restructure's claimed repairs did work:
- R3 went 2/5 → 4/5 versus parent r026, with no open-door walling. R3/00, 01 and 04 route through doors after "The door opens.".
- The `quaff` lockup is gone: 0 item verbs were emitted by the model.

## 4. Binding constraint
**TASK-GAP, with the fix already built and validated in sibling candidates.** The stage table is right, and the mechanism modules plugged into it are the weak ones:
- Quest uses the r046 version, where it should use r058/r065.
- The Boxoban owner is missing (r064).

It is not VLM-CEILING: on every lost Quest and Boxoban episode, the VLM is reached only because no stage owns the step.

## 5. Concrete changes (ranked by leverage)

1. **Replace `tools/quest.py` with r065's module and wire its four stages.**
   - Source: `prop_r065/cand_r065_0_spatial-grounding/tools/quest.py`, which is the same lineage and the same `tools/chart.Chart`.
   - Copy r065's `Chart.terrain` / `Chart.edge(a,d)` into `tools/chart.py`, and `lang`/`modal` into `tools/frame.parse_glyph_map`.
   - Replace `quest, fetch` in `STAGES` with r065's `cross, hold, kit, fetch`, armed on `{pickup, kick, zap} <= legal`, not on "a `}` is charted".
   - In `st_prompt`, call `quest.answer` before `more`.
   - On an armed Quest level, `st_model` takes the least-visited open edge (`quest:wander`) and makes no VLM call.
   - Predicted: QE 0→5, QM 0→5 (r065 was 5/5 and 5/5 on these seeds), survival 0.80 → 0.90, and QE tokens (~1.8M) → ~0. **Primary +0.25 → ~0.625.**

2. **Add r064's `tools/sokoban.py` as an `st_sokoban` stage right after `orient`.**
   - Arm it when the legal moves are exactly the 4 cardinals and a `` ` `` and a `{` are drawn.
   - Cache the plan; if the observed (hero, boulders) differs from the expected state, replan; pop one move per step via the `pending` move book.
   - Once `bars` appears in the prose, treat `#` as a wall in `Chart.passable` (the fallback hygiene r064 uses).
   - Predicted: Boxoban 0→10 at 0 tokens (r064's result). **Primary +0.25**, and cost drops by roughly 70%.
   - With (1), this reproduces r064's 0.875 vector on this lineage.

3. **Aim the frontier at a drawn but unreachable `>` (tools/chart.py:route).**
   - If a `>` is drawn but BFS cannot reach it, rank frontier squares by (L∞ distance to `>`, then BFS distance) instead of BFS distance alone.
   - Tie-break in the level frame so the order stays total. It is a total order, so re-electing the goal every step remains a fixed point.
   - Predicted: R3/03 converts (→5/5, +0.025). It is inert on MazeWalk, where `>` is either reachable or not drawn.

4. **Make `peaceful` a model-level veto (harness.py).**
   - In `st_prompt`, mark the foe peaceful on any `Really attack` prompt, whatever `last[2]` was: take the square from `pending` (prev + move) and the glyph from the chart.
   - In `st_orient`, add to `c["bad"]` every direction whose target glyph is in `self.peaceful`.
   - This removes QE/04's roughly 60-step `east`→`[yn]`→`more` loop.
   - It is also mostly subsumed by (1), since on Quest the model is no longer consulted.

## 6. Exhausted — do NOT repeat
- **CB-Dark melee**: retreat, hold, recoil, exposure and threat-clock variants all sit inside dice noise (memories `balrog-melee-axis-is-closed-every-mechanism-is-a-dice-reroll`, `balrog-cbdark-is-dice-*`). Don't spend a round there on 5 episodes.
- **Porting r046's quest.py**, which this round did: 0/10 Quest. The r046 module is not the winning one. Port r058/r065.
- **Prose-only edits.** AGENT.md, rules.md and skills are empty here and on r064/r065; every Quest and Boxoban gain in the chain came from a stage owner.
- **Two-kick pile opening** (it re-piles), and **kicking toward lava** (it burns the horn). Both are already encoded in r065's `kick_dir`.
