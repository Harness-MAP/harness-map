# Diagnosis — minihack iter 39 / cand_r039_0_spatial-grounding
**Headline:** primary = **0.125** (5/40). The chain best is r036 at 0.375, and r038 scored 0.325.

| Task | Wins |
|---|---|
| MazeWalk-9x9 | 4/5 |
| MazeWalk-15x15 | 1/5 |
| Every other task | 0/5 |

All 5 wins are episodes where `>` was reached in ≤4 steps. There are 7 deaths (Quest-Medium 5/5, CB-Dark 2/5). All other losses hit the 100-step cap.

**Binding constraint (HARNESS-BUG):**
- **The model is not the problem.** It copies the ROUTE hop on **100%** of the steps where a hop exists, on 6 of 8 tasks.
- **The route itself is a deterministic 2-cycle.** `tools/chart.py:Chart._frontier` counts a square as unexplored if it was never stood on **OR** it borders a blank glyph.
  - In a dark maze or a corridor, every square borders undrawn rock, so every square is a "frontier".
  - The BFS therefore always elects the adjacent square the agent just came from.
- **The chain already fixed this.** r036's `chart.frontier` uses AND ("never stood on" and "borders uncharted"). r039 rebuilt the chart from scratch and brought the orbit back.

## 1. What this harness does (mechanism)
`orchestrator.py:ChartHarness.act` runs once per step:
1. `Chart.observe(long_term_context)` updates the chart. It parses the cursor `(x,y-1)`, `Dlvl`, the `map:` rows (row = y−1, verified on frames) and the lava/water "adjacent" directions. A no-move after the last action books a `(square,dir)` blocked edge.
2. `Chart.plan(legal)` runs a BFS in `STEP` order (N, NE, E, SE, S, SW, W, NW). The target is a charted `>`, else the first square for which `_frontier()` is true.
3. `render_chart` adds `ROUTE: nearest <kind> is N step(s) away; first move toward it: <dir>` and an 8-row neighbour table to the naive prompt. `AGENT.md` says "follow it unless something you can see says otherwise".
4. The harness makes one VLM call, then `action_io.extract_action` handles the reply:
   - It takes the earliest whole-word legal mention that is not vetoed (blocked or deadly).
   - If none survives, it takes the BFS hop.
   - If there is no hop, it takes the least-visited open neighbour.

There is no creature logic, no item logic (Quest) and no push planner (Boxoban). A creature letter is simply impassable to the BFS.

## 2. What actually happened (causal chain, with evidence)
CSV row = (action, observation AFTER it). I replayed `tools/chart.py` on the logged observations, and it reproduces the logged `route=` values exactly.

**MazeWalk-15x15 run_00 (100 steps, loss):**
- **Where it starts.** The agent starts at (44,5), with `<` at (44,6) directly south. The prose says "dark area adjacent north, northeast, east, ...".
- **The orbit.**
  - At (44,5), BFS neighbour order puts `south` before `west`. The `<` square counts as a frontier because it borders blank, so the ROUTE says `south@1`.
  - At (44,6), (44,5) borders blank (north), so the ROUTE says `north@1`.
  - Steps 0, 1, 2, 30, 31, 60 are byte-identical frames. The agent alternates N/S for all 100 steps.
- **The explored squares were ignored.** (43,5) west and (43,7)/(44,7) south were charted `.` with 0 visits. They were never chosen, because a dist-1 "borders blank" square always wins first in BFS order.

**Orbit census (all runs):**

| Task | Route followed | Route points back to the previous square | Revisit |
|---|---|---|---|
| MazeWalk-15x15 | 1.00 | **0.97** | 0.95 |
| Corridor-R3 | 1.00 | **0.86** | 0.85 (ABA 0.84) |
| Quest-Medium | 1.00 | **0.86** | 0.84 (ABA 0.81) |
| MazeWalk-9x9 | 1.00 | **0.86** | 0.81 |

The only wins are the 4 MazeWalk-9x9 and 1 MazeWalk-15x15 episodes where the first 2–4 hops already reached `>`. That is the "stairs down" route kind, 9% of 9x9 steps.

**Offline counterfactual:** I replayed the same frames with frontier = `visits==0`, or with a blank neighbour that is not adjacent to any stood-on square. "Route points back" drops as follows:

| Task | Before | After |
|---|---|---|
| MazeWalk-15x15 | 0.97 | 0.12 |
| Corridor-R3 | 0.86 | 0.24 |
| Quest-Medium | 0.86 | 0.41 |
| MazeWalk-9x9 | 0.86 | 0.44 |

These frames come from orbiting trajectories, so this under-states the fix.

**Deaths.**
- **Quest-Medium run_00, step 36→37:**
  - The frame shows `giant rat adjacent south`. The ROUTE says `east`, then `west`, walking away through the corridor mouth, and the agent is killed by the giant rat.
  - All 5 Quest-Medium episodes die at step 38–47 in this same corridor orbit.
- **CB-Dark run_01, step 19→20:** `giant rat adjacent northeast and east`, the ROUTE says `west` then `north`, and the agent dies.

Creatures are impassable to the BFS, so the route always walks away and the agent is struck while retreating. The model never overrides the route, despite the AGENT.md "unless a monster" clause.

**Quest-Easy:**
- 73% of steps have no route (`nohop`). Once the lit room is charted, the only way on is across the `}` lava column, which the chart excludes, so no frontier is reachable.
- The fallback's least-visited neighbour then wanders for the remaining 100 steps (e.g. run_01 steps 0–11 finish the room).
- This is the known TASK-GAP: the item acquisition → `zap`/`apply` opener.

**Boxoban:**
- The route kind is always "unexplored", and the level is fully lit, so the route carries no signal.
- No-move steps are 35–42% (pushing boulders into walls). The chart books those edges, but there is no push planner.

## 3. Failure breakdown (40 episodes)

| Failure mode | Episodes | Tasks |
|---|---|---|
| Route 2-cycle orbit to the 100-step cap | ~12 | MW15 ×4, MW9 ×1, R3 ×5, plus the orbit that feeds the Quest-M deaths |
| Death by an adjacent creature while the route walks away | 7 | Quest-M 5, CB-Dark 2 |
| No reachable frontier after lava, then aimless fallback | 5 | Quest-Easy |
| No push planner, so walls get bumped | 10 | Boxoban H+M |
| Cheap wins (`>` within 4 steps) | 5 | |

Parse and illegal actions: none. The whole-word projection works, and the judge found no cheat.

## 4. Binding constraint
**HARNESS-BUG**, fully fixable in code:
- The frontier predicate in `tools/chart.py:Chart._frontier` is wrong: OR where it must be AND.
- The model follows the hint 100% of the time, so the route **is** the policy.
- MazeWalk (10/10 in r036/r038) and R3 (3–4/5) are lost purely to this predicate. That accounts for 0.125 → ~0.325 on its own.

The remaining 0/5 tasks are the chain's standing TASK-GAPs (Boxoban planner, Quest opener, CB-Dark fight post).

## 5. Concrete changes to try (ranked by leverage)
1. **Fix `Chart._frontier` (tools/chart.py).**
   - Change it to `visits==0 AND any neighbour glyph == " "`, exactly as in r036's `chart.frontier`.
   - A stricter variant: a blank neighbour counts as unexplored only if no stood-on square is adjacent to it, because standing next to a blank square in NetHack shows it, so a blank that stays blank there is rock.
   - Keep the fallback frontier "any visits==0 passable square" so the route does not die early.
   - *Predicted:* MazeWalk-15x15 1/5 → 4–5/5, MazeWalk-9x9 → 5/5, R3 0/5 → 2–4/5, primary ~0.30–0.35.
2. **Make the BFS tie-break a total order (tools/chart.py:plan).**
   - Rank equal-distance frontiers by `(visits, dist, index)`, not by `STEP` iteration order.
   - Re-elect the target every step. Distance-ascending is a fixed point, per chain memory.
   - This kills the residual back-pointing, which is still 0.24–0.44 on R3/Quest-M in the counterfactual.
3. **Treat open doors as passable (tools/chart.py:passable).**
   - r038's lesson: an open door is drawn `-`/`|`. Use the prose noun `door`/`doorway` or a successful move through it to override `WALL` for that square.
   - This targets R3 losses.
4. **An adjacent creature is a fight, not a detour (orchestrator.py:act).**
   - When the prose says `<creature> adjacent <dir>` and the square is not a corpse, emit that direction (attack) instead of the route hop.
   - Don't flee: the chain shows every degree of disengagement kills more.
   - *Predicted:* removes most of the 7 deaths. Price the CB-Dark part at ≥200 episodes; at n=5 CB-Dark is dice.
5. **Quest-Easy/Medium opener (standing TASK-GAP).**
   - Route to the item glyph (`(`, `/`), `pickup`, then `zap`/`apply` toward the `}` bearing.
   - `kick` the spawn pile once if the items are stacked.
   - This is the only lever that makes the Quest pair non-zero.
6. **Boxoban:** a push-space A* over the lit map. Chain memory: 19/20 model-free. Out of scope for the chart, but it is +10 episodes of headroom.

## 6. Exhausted — do NOT repeat
- **Prompt/AGENT.md wording:** the model copies the hint ~100%, so prose is inert. Fix the hint, not the text.
- **Re-plumbing model-first vs chart-first without new mechanism:** r026/r027/r036/r037/r038 all land at 0.325–0.375.
- **Re-deriving the frontier orbit:** it was solved in r036 ("stood cells retired"). **Port r036's `chart.frontier`/`_bfs`/`edge_ok`** rather than rewriting the chart from scratch. This candidate is a regression caused by a rewrite.
- **CB-Dark tweaks priced at n=5.**
