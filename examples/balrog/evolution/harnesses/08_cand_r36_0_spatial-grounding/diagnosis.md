# Diagnosis — minihack evolve r036 / cand_r036_0 (perception → chart → reflex → planner → arbiter pipeline)

**Headline:** primary = **0.375** (15/40), up from the parent r026's 0.325 (13/40). Cost is ~89k
tokens/ep (parent: 115k).

| Task | Result | Change vs r026 |
|---|---|---|
| MazeWalk-9x9 | 5/5 | same |
| MazeWalk-15x15 | 5/5 | same |
| Corridor-R3 | **4/5** | 2/5 → 4/5 |
| CB-Dark | 1/5 | same |
| Quest-Easy | 0/5 | same |
| Quest-Medium | 0/5, all 5 died at steps 19–24 | same, byte-identical action prefix |
| Boxoban-Medium | 0/5 | same |
| Boxoban-Hard | 0/5 | same |

The door-identity fix did what it predicted: R3/00 and R3/01 now win.

**Binding constraint:** there are 25 losses. **20 of them are TASK-GAPs that this candidate did not
touch**: Boxoban has no push planner (10), Quest has no item acquisition and lava crossing (10), and
Quest-Medium also has no corridor-mouth hold. **Two cheap HARNESS-BUGs** remain on top:
1. **R3/03:** the frontier ignores a *drawn but unreachable* `>`.
2. **Quest-Easy:** the new `more` reflex turned the old prompt-lock into a **`quaff`/`more`
   2-cycle** that eats 83 steps.

## 1. What this harness does (mechanism)
`agent.py:GroundedAgent.act` runs a rung ladder. The first rung that fires decides the step, and
its name is written to `reasoning`:
1. `tools/reflex.modal`: if a prompt is open, answer `northwest` for a `[yn]` attack prompt,
   otherwise `more`.
2. Stairs: standing on a `>`, emit `down`.
3. `tools/reflex.lethal_moves` closes bearings toward lava or water. Then `adjacent_foe` makes
   `reflex:swing`.
4. `tools/chart.Chart.route`: BFS to a charted `>`, else to the **nearest** unstood frontier. The
   BFS scans neighbours in `ORDER` = N, E, S, W, NE, SE, SW, NW. Door identity is kept, doors are
   passable, and diagonals through doors are pruned.
5. `Chart.search_plan`: runs only if the route is None **and no `>` is charted**.
6. `ask_model` builds the stock BALROG history plus the banned bearings, and `action_io.arbitrate`
   maps the reply onto a legal action. The prose files (AGENT.md, rules.md, skills/, memory.md) are
   all empty.

Rung census over 40 episodes:

| Rung | Steps |
|---|---|
| `chart:route` | 1357 |
| `model:*` | ~694 |
| `reflex:swing` | 99 |
| `reflex:close-prompt` | 82 |
| `arbiter:least-visited` | 8 |
| `chart:search` | **0** |
| `chart:to-dead-end` | **0** |

**The search rung never fired on any episode.**

## 2. What actually happened (causal chains)

**A. R3 wins (R3/00, R3/01, R3/02, R3/04): 100% `chart:route`, 0 model calls.** R3/00 and R3/01
are the parent's two open-door losses. They now route through the `-` door to the `>` in 46 and 60
steps. Door memory works. **Keep it.**

**B. R3/03 loss: the `>` is DRAWN and the route walks away from it (HARNESS-BUG).**
- **Step 19**, at (15,8): the frame draws `   |....>.    #@#` with the `>` at (8,8), 7 squares
  west. The prose says `stairs down far west`. The corridor `#` at (14,8) touches undrawn cells
  (x = 10..13), so the `>` is charted but not BFS-reachable.
- `route` falls through to "nearest frontier". Several frontiers are tied at distance 1, and `ORDER`
  puts `east` before `west`, so it steps **east** (steps 20–21: `###@#------`).
- It then explores the eastern corridor network for 80 steps (positions (23,11) → (34,15) →
  (31,8) → (27,5) → (35,16) → …) and never returns. Every one of the 100 steps is `chart:route`.
- `search_plan` is correctly skipped (`stairs_charted()` is True), but nothing *uses* the charted
  `>` either.
- The parent's diagnosis attributed R3/03 to "open door + hidden corridor". This trace refutes that:
  the real blocker is **goal-blind frontier selection**. A west walk of about 6 squares from step 19
  draws the corridor to the `>`.

**C. Quest-Easy: the prompt-lock became a 2-cycle (QE/00 and QE/04).**
- **QE/00 step 8:** `model:quaff` → `You don't have anything to drink.`
- **Step 9:** `model:pickup` → `f - a sky blue potion.`
- **Step 10:** `model:quaff` → `What do you want to drink? [f or ?*]` (cursor y=0).
- **Step 11:** `reflex:close-prompt` → `more` → `Never mind.`
- **Step 12:** `model:quaff` again. This repeats for 89 steps: 46 `quaff` and 45 `more`.
- QE/04 is the same: 39 `quaff` and 37 `more`.
- Closing the prompt is correct, but nothing tells the model that the verb just failed. Its history
  shows `quaff` as the last action, so it repeats it. The r026 lock lasted 87 and 94 steps; this one
  lasts 89 and 76. **Net gain: zero.**
- Meanwhile every QE/00 frame from step 7 on says **`horn adjacent west`**. That is the frost horn
  that crosses the lava, and it is never picked up.

**D. Quest-Easy QE/01–03: model orbit (VLM-inert).**
- 99/100 steps are `model:*`: period-4 cycles of `east north west south`.
- The room is lit, and the `}` column seals the east half with the `>` drawn beyond it. So `route`
  is None, and `search_plan` is skipped because the `>` is charted.
- The model has no goal it can act on.

**E. Quest-Medium: 5/5 deaths, same shape as r026.**
- Steps 0–12 are `chart:route` from the start room along the corridor. At step 12 the agent enters
  the rat room at `|@....r..rrr....}>..|`.
- It emits 3–4 `model:east`, then `reflex:swing` east into the rat pack.
- **QM/01 step 20:** `giant rat adjacent northeast, east, and southeast`, three attackers at once.
  Death comes at step 23.
- The horn `(` sits in the west start room (`|.(.|`, prose `horn far west`) and is never visited:
  it is a lit room with no frontier.

**F. Boxoban (0/10): the chart still runs on bars.**
- `You cannot pass through the bars.` appears on 17–44 frames per episode.
- BM/00 and BM/03 are 100/100 `chart:route`. BH hands 21–58 steps to the model once the bar ring is
  exhausted, and the model orbits `N/E/S/W`.
- Boulders were pushed 4–12 times per episode, all by accident. No push plan exists.

**G. CB-Dark (1/5):** 4 melee deaths at steps 27–34. This is dice at n=5, per the prior chain.

## 3. Failure breakdown (25 losses)
| Mode | Episodes | Class | Fix |
|---|---|---|---|
| Boxoban: no push planner; chart bumps bars | 10 | TASK-GAP | §5.1 |
| Quest-Easy: lava partition; horn adjacent but never acquired | 5 (2 of them in the `quaff`/`more` 2-cycle) | TASK-GAP + HARNESS-BUG | §5.3, §5.2 |
| Quest-Medium: advances into the open rat room and dies | 5 | TASK-GAP | §5.3 |
| CB-Dark: melee death | 4 | dice | leave it |
| R3/03: frontier walks away from a drawn, unreachable `>` | 1 | HARNESS-BUG | §5.2 |

Other counts:
- Illegal-verb coercions: **0**. The arbiter fix works.
- Model calls on MazeWalk and R3: 0.

## 4. Binding constraint
- **TASK-GAP (20/25 losses, dominant):** Boxoban needs a Sokoban solver, and Quest needs an item
  opener plus a lava crossing (and a corridor-mouth hold on QM).
  - Sibling chains measured these: an exact push-space A* solved ~10/10 on Boxoban, and a
    kick-split / pickup / `apply`/`zap` opener won 5/5 on QM and 2/5 on QE.
  - This is the only place with more than 2 episodes of upside.
  - The proposer chose re-architecture this round instead. That was correct for R3, but **the next
    round must build these planners.**
- **HARNESS-BUG (2 cheap fixes):** the goal-blind frontier (R3/03) and the unvetoed re-emission of a
  verb whose prompt was just cancelled (QE/00, QE/04).
- **Not VLM-CEILING:** every loss is on a step where the harness hands the model a decision it has
  no information to make.

## 5. Concrete changes (ranked by leverage)

### 5.1 Boxoban push planner — new module `tools/sokoban.py` plus a rung ahead of `route` (target: up to +10 eps, +0.25 primary)
- **Detect** from the observation: the system prompt or task text mentions Boxoban, or the frame
  contains `` ` `` boulders and `{` targets inside a `#` ring.
- **When detected, skip `Chart.route` entirely.** Here `#` means bars.
- **Decode:** `#` = wall, `` ` `` = box, `{` = target, `.` = floor, `@` = player (map row = cursor.y
  − 1). The hero's masked start square is floor. Remember the targets from the first frame, because
  a box sitting on a target hides the target.
- **Search:** A* over (the player's canonical reachable region, sorted tuple of boxes).
  - Moves are cardinal pushes only. Prune non-target corners and wall-line deadlocks.
  - Heuristic: sum of the minimum Manhattan distances from boxes to targets.
  - Wrap the heap entries so that a `frozenset` is never compared: frozenset `<` is a subset test,
    not an ordering.
- **Execute:** turn the plan into a queue of cardinal moves. Re-plan whenever the observed box
  positions diverge from the expected ones, e.g. after `in vain`.
- **Prediction:** Boxoban goes from 0/10 to 7–10/10 at 0 model calls. That also cuts ~30k tokens/ep
  on BH.

### 5.2 Two cheap harness bugs (target: +1 R3 episode, and the QE steps freed for §5.3)
1. **Frontier attracted to a charted `>`** (`tools/chart.py:Chart.route`).
   - When `stairs` is empty but `self.stairs_charted()` is True, rank the reachable frontier cells
     by `(chebyshev(c, stairs_cell) , bfs_dist)` instead of `bfs_dist` alone.
   - Gate this on no `}` being charted. On the Quest pair, lava is what separates the `>`, and
     aiming at it there is useless.
   - Break ties in the level frame, not by `ORDER`.
   - On R3/03 step 19 this picks the west corridor end (14,8), which is 6 from the `>`, over the
     east one, which is ≥ 8.
   - The prior sibling chain measured this lever at **R3 4/5 → 5/5, +2/−0 over 125 levels, with
     zero arms firing off R3**.
   - It is inert on MazeWalk: there the `>` is reachable as soon as it is drawn.
2. **Veto a cancelled verb** (`agent.py:act` + `tools/reflex.modal`).
   - When `modal` answers an item prompt (`What do you want to <verb>?`) with `more`, record
     `(pos, verb)` in `chart.blocked_verbs`. The verb is the previous action.
   - On the next `ask_model`, add that verb to `banned`, which `arbitrate` already honours. Also
     show it in the "closed" tail.
   - Clear the record when the position changes.
   - **Prediction:** QE/00 and QE/04 release about 80 steps each back to exploration or the opener,
     with no effect elsewhere. This does not win QE by itself; it unblocks §5.3.

### 5.3 Quest opener + corridor-mouth hold — new module `tools/quest.py`, a rung fired when a `}` is charted and `route` is None or leads into ≥2 drawn creatures (target: up to +5 QM, +2 QE)
1. **Acquire the horn/wand.**
   - The prose names it directly: `horn adjacent west` (QE/00 from step 7), `horn far west` (QM,
     the start room at `|.(.|`). Walk to the `(`/`/` glyph and `pickup`.
   - If a pickup is silent (a 2-item pile), `kick` the pile **once**, then pick up the pieces
     separately.
2. **Cross the lava.**
   - Use `apply` (horn) or `zap` (wand), answering item prompts by letter-as-verb: `h`=west,
     `j`=south, `k`=north, `l`=east, `y`=northwest, `u`=northeast, `b`=southwest, `n`=southeast.
   - Answer `Improvise?` with `northwest` (`y`).
   - Answer `In what direction?` with the bearing to the nearest `}`.
   - Stop firing on `The lava cools and solidifies.`, then hand back to `route`.
   - This rung must **own** its prompts ahead of `reflex.modal`. Otherwise `more` cancels them,
     exactly as in 2C.
3. **QM: hold the corridor mouth.**
   - At step 12 the agent is at the room entrance with 6+ `r` drawn within 10 squares.
   - When ≥ 2 creatures are drawn within 6 squares and the current cell has ≤ 2 passable
     neighbours, do not advance. Emit `search` to spend the turn, and let `adjacent_foe` swing at
     arrivals.
   - Commit the post to the cell you stepped to.
   - **Prediction:** at most one attacker at a time instead of three (QM/01 step 20). The parent
     diagnosis cites a prior chain in which the corridor hold survived to step 68 and won, where
     advancing died at step 39.

### 5.4 Keep
- Keep the MazeWalk, R3 door-identity and arbiter code paths byte-identical. They are 10/10, 4/5
  and 0 coercions respectively.
- The five-module layout is a good base: add `sokoban.py` and `quest.py` as siblings and keep
  `agent.py` as the orchestrator.

## 6. Exhausted — do NOT repeat
- Prose priors: the prose files are empty, and the lever has proven to be code every time.
- Convicting a glyph class after one bump: convict `(square, bearing)` only.
- Committed multi-step routes that are not re-elected every step, and ranking the frontier by raw
  `information`.
- The dead `esc`/`y` reflex has been replaced. Do not reintroduce non-vocabulary verbs.
- A bare `more` on every item prompt, **without** a verb veto. It only converts a lock into a
  2-cycle (2C).
- Double-kicking the Quest pile: it re-piles the items.
- CB-Dark melee tweaks at n=5: dice.
- The search rung as the fix for R3/03: it is correctly skipped there because the `>` is charted,
  and it fired 0 times in 40 episodes. Treat it as harmless, not as a lever.
