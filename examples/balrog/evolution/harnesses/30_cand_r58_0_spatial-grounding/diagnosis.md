# Diagnosis — minihack iter 58 / cand_r058_0_spatial-grounding (charted learner as a stage table + tools/quest.py)

**Headline:** primary = **25/40 = 0.625**.

| Task | Wins |
|---|---|
| Quest-Easy | **5/5** |
| Quest-Medium | **5/5** |
| MazeWalk-9x9 | 5/5 |
| MazeWalk-15x15 | 5/5 |
| Corridor-R3 | 4/5 |
| CorridorBattle-Dark | 1/5 (4 deaths) |
| Boxoban-Hard | **0/5** |
| Boxoban-Medium | **0/5** |

- Tokens are near zero. Only Boxoban calls the model: 21 learner consults for 13k–70k input tokens per episode. The other 30 episodes use 0 tokens.
- The Quest transplant worked as designed: the Quest pair went from 0/10 in the parent to 10/10.

**Binding constraint:** Boxoban is 10/15 of the remaining losses. This harness has **no push planner**, and its chart treats `#` (iron bars in Boxoban) as floor. As a result 20–46% of Boxoban steps are bumps into bars, and no episode places all boulders. This is a **TASK-GAP plus a HARNESS-BUG**. It is fixable in code: the chain already has a solver in `prop_r008/.../tools/sokoban.py`.

## 1. What this harness does (mechanism)

`agent.py:LearnerAgent.act()` runs `STAGES = (prompt, orient, stairs, swing, cross, hold, kit, fetch, route, learner)`. The first stage that returns an action wins, and its rung name goes to `reasoning`.

- **`orient`** does the bookkeeping:
  - `observe()` is the edge ledger: bumps become `blocked`, and doors come from prose or `door opens`.
  - `credit()` runs the TD(λ) value table and adds every drawn glyph to `self.drawn`.
  - `deadly()` vetoes lava or water named in the prose, and `}`.
- **`cross` / `hold` / `kit` / `fetch`** are `tools/quest.py`, armed only when `{pickup, kick, zap} ⊆ legal` (the Quest pair).
- **`route()`** is a BFS over `passable()` cells. It targets a reachable drawn `>`. If there is none, it takes the **nearest** unvisited frontier. `passable()` treats every drawn glyph except `WALL = "|- "` and `}` as floor, so **`#` and `` ` `` count as floor**.
- **`learner`** takes the argmax over the value table and consults the VLM on ties, except on Quest levels.

## 2. What actually happened (causal chain, with evidence)

**A. Boxoban (0/10): the chart walks into bars and never plans a push.**

- Rung census over the 10 episodes (1000 steps):
  - `chart:route` 734
  - `learner:argmax` 215
  - `learner:consult` 51
  - No stage ever reasons about boulders or goals.
- **Hard run_00, step 0.** The hero is at (35,15) and `bars adjacent north, …, west` is in the prose. `route()` sends `north` into a `#`, and the reply is "You cannot pass through the bars."
  - `#` is in `self.drawn` and not in `WALL`, so the BFS both routes through bars and counts bars cells as "frontier": their blank neighbours read as undrawn, `frontier()` returns true, and the BFS aims at them.
  - Each bump closes only one (square, bearing) edge in `blocked`, so the next bars cell gets tried.
- Bump rates, counting a step with an unchanged cursor as no-move:
  - Hard: 20, 29, 39, 23, 36 per 100 steps.
  - Medium: 46, 44, 27, 45, 42 per 100 steps.
  - The messages are dominated by `You cannot pass through the bars.` (17–44 per episode) and `You try to move the boulder, but in vain` (2–9).
- Boulders move only by accident: "With great effort you move the boulder" happens 1–6 times per episode. There are 0–3 positive-reward steps per episode, and some negative ones (boulders pushed off goals). Unique squares visited: 13–40.
- Every Boxoban frame is fully premapped and lit. The level has 4 boulders `` ` `` and 4 fountains `{`, the goals, with one exception: **Medium run_04 draws only 3 `{`**. The fourth goal is hidden, either under the hero's start square or under a boulder. A solver must handle this (see §5.1).
- Chain memory: Boxoban decodes to an exact Sokoban, and a push-space search won 19/20 on the stub rig at 0 calls. A win happens iff the optimal plan fits in about 99 steps.

**B. Corridor-R3 run_03 (the one R3 loss): a drawn `>` the chart cannot reach is ignored.**

- **By step 40** the frame already shows the goal room `|....>.` with `>` at (8,7). The hero is at (27,12).
- The `>` is not BFS-reachable because the cells between the room and the `#####` corridor at x=14 are still blank. So `route()` falls through to `best = the first frontier in BFS order`, which is the corridor heading **east**.
- The hero spends the remaining 60 steps east and south, and ends at (24,10) on step 100, still about 16 columns from the `>`.
- `route()` keeps no sense of where the known `>` is when choosing a frontier. This is the known "drawn-but-unreachable `>` is an attractor" gap, and it is a HARNESS-BUG.

**C. CorridorBattle-Dark (4 deaths, 1 win).** There are 59 `reflex:swing` against 108 `chart:route`, and deaths come at steps 27–34 to giant rats in the dark corridor. The fight exchanges ("You miss the giant rat. The giant rat bites!") repeat the pattern that chain memory has priced repeatedly: the melee axis is dice at n=5. This is not a harness defect here.

**D. What works (do not regress).**

- **Quest 10/10.** On an armed level the sequence is: `pickup` at spawn, one `kick` to split the pile, horn-first `pickup`, then `zap`/`apply` across the lava. `quest:hold` (`wait`) covers the corridor mouth: 91 holds on Quest-Medium.
- **MazeWalk 10/10** and **R3 4/5**, all `chart:route` at 0 tokens.

## 3. Failure breakdown

| Failure | Episodes | Cause | Class |
|---|---|---|---|
| Boxoban: no push planner, and `#` routed as floor | 10 | §2A: `route()` / `passable()` / `WALL`, no Sokoban stage | TASK-GAP + HARNESS-BUG |
| CB-Dark melee deaths | 4 | §2C, dice | VLM/luck (closed axis) |
| R3: drawn `>` unreachable, frontier chosen by BFS order | 1 | §2B: `route()` frontier choice | HARNESS-BUG |

## 4. Binding constraint

**TASK-GAP (Boxoban) with an embedded HARNESS-BUG.**

- Boxoban is exactly solvable from the observation: the map is fully drawn, the moves are 4 cardinals, and the dynamics are deterministic.
- Neither the chart, nor the TD learner, nor the one-word VLM tie-break can produce a 30–90-step push sequence.
- Every other task is at or near its ceiling for this lineage.
- The ceiling is roughly 0.625 + 10/40 (Boxoban) + 1/40 (R3) ≈ **0.875 or more**.

## 5. Concrete changes to try (ranked by leverage)

### 5.1 Add a `sokoban` stage using an exact push-space solver (expected +6 to +10 Boxoban wins, +0.15 to +0.25 primary)

- **Port the solver.** Copy `prop_r008/cand_r008_0_spatial-grounding/tools/sokoban.py` into `tools/sokoban.py`. Load it the way `tools/quest.py` is loaded, under a path-unique `sys.modules` key.
- **Arming (observation only).** Arm the stage in `act()` when `c["moves"]` is exactly the 4 cardinals (Boxoban offers no diagonals), at least one `` ` `` is drawn, and at least one `{` is drawn.
  - Insert `"sokoban"` into `STAGES` **right after `orient`**, before `stairs`. There is no `>` on Boxoban.
  - Leave every other task untouched: MazeWalk, R3, CB-Dark and Quest all offer diagonals.
- **Parsing.**
  - Walls are `#`, `-`, `|` and blank.
  - The hero's square is floor.
  - Boulders are `` ` ``.
  - Goals are `{`. **Accumulate goals across frames** in a `self.soko_goals` set, because a goal hidden under the hero appears once the hero steps off.
  - If `len(goals) < len(boulders)` (Medium run_04), a boulder is already on a goal or the start square hides one. Add the hero's start square as a candidate goal. The success test is then `number of boulders on goals == len(boulders)`. Do not use r008's `goals <= bx` shortcut: it declares victory with 3/4 placed.
  - Chain memory: prune dead squares on **walls only**, never on boulders. Also never treat the hero's square as a boulder (the "phantom boulder" trap scored 0/10).
- **Search.** r008's search is a plain BFS over (player, boxes) with a 300k-node cap. That may be too slow or too capped for Hard. Upgrade it to **push-level A\***:
  - State: (normalized player-reachable region, frozenset of boxes).
  - Edges: pushes. Expand each push into walk-plus-push moves with a BFS.
  - Heuristic: the sum over boxes of the push distance to the nearest goal.
  - Add a time budget of a few seconds.
  - Solve **once** and cache the move list.
- **Execution.**
  - Pop one move per step, and record the expected position after it.
  - If the next frame's cursor differs from the expectation (a bump, or a boulder that failed to move), **replan from the current frame**.
  - Emit through `commit()` so the ledger stays consistent, with rung `sokoban:plan:<left>` so the trace shows it.
  - Clear the plan and goal cache in `forget()`, which `reset()` calls. Chain memory: a stale cached plan cost 8/10 on an adversarial order.
- **Fallback.** If the solver returns `[]`, fall through to `route`. With 5.2 in place that at least stops the bar-bumping.
- **Prediction.**
  - Boxoban goes from 0/10 to roughly 6–10/10, bounded by Hard plans longer than about 99 steps.
  - Boxoban tokens go to 0, because the learner consult disappears there.
  - The other 30 episodes stay byte-identical, because the stage is never armed on them.

### 5.2 Make `#` impassable when the prose names bars (small; Boxoban fallback hygiene)

- In `passable()` / `frontier()` (agent.py), treat `#` as `WALL` when the current frame's `fr["adjacent"]` contains a `bars` noun, or when the level is Boxoban-armed.
- Chain memory: prose `bars` ⟺ `#` is iron bars, and on R3 and MazeWalk `#` is corridor. Key it on the prose, never globally.
- This removes 17–44 bumps per Boxoban episode even when the solver bails.
- `frontier()` must not count a bars cell as a frontier either.

### 5.3 Aim frontiers at a drawn but unreachable `>` (expected +1 R3)

- In `route()`, record `stairs` (already computed). If the BFS completes without reaching any stairs cell, choose the frontier minimizing `(chebyshev(frontier, nearest_stairs), bfs_depth)` instead of the first one popped.
- To do this, collect all frontiers with their depths instead of breaking at the first.
- Keep it a **total order**, and re-elect it every step. Chain memory: two-class rules orbit.
- Gate it on `stairs` being non-empty, so MazeWalk and R3 without a drawn `>` stay byte-identical.
- In R3 run_03 this turns the east wander at step 40 into a westward walk toward (8,7).
- Chain memory: +2/−0 over 125 levels, and inert off R3.

### 5.4 (Optional, low EV) CB-Dark

Four deaths out of 5 is within the dice band the chain has measured. If you touch it, the only replicated lever is the `wait` hold at a corridor square with a creature in contact, which gave a survival gain on other lineages. Do not add retreat or HP-gated disengagement (see §6).

## 6. Exhausted — do NOT repeat

- **Boxoban via frontier walking, value learning or VLM tie-breaks:** 0/10 across every lineage, this one included. Only an exact push planner moves it.
- **Melee retreat, HP-crit disengage and lookahead recoil on CB-Dark:** refuted repeatedly (dose-response: more disengagement means more deaths).
- **Re-deriving the Quest plan:** `tools/quest.py` is 10/10. Keep it byte-identical and keep its arming gate `{pickup, kick, zap} ⊆ legal`.
- **Prompt re-wording for the tie-break consult:** the VLM is effectively inert. It is consulted only on Boxoban, and Boxoban is fixed by 5.1, not by prose.
