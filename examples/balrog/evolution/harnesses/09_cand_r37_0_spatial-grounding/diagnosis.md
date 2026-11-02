# Diagnosis — minihack evolve r037 / cand_r037_0 (middleware stack, NavStage BFS owns movement)

**Headline:** primary = **0.325** (13/40). The parent r029 scored 0.150; the chain best is r036 at 0.375.

| Task | Result | Stage that drove the episode |
|---|---|---|
| MazeWalk-9x9 | 5/5 | Nav |
| MazeWalk-15x15 | 5/5 | Nav |
| Corridor-R3 | 2/5 | Nav |
| CB-Dark | 1/5 | Nav + Foe |
| Boxoban-Hard | 0/5 | model |
| Boxoban-Medium | 0/5 | model |
| Quest-Easy | 0/5 | Nav exhausted/search; `[yn]` loop |
| Quest-Medium | 0/5 | Foe |

Deaths: 8/40 (CB-Dark 4, Quest-Medium 4), so survival is about 0.80. The model was called on 809 of 2 446 steps.
About 96% of all tokens are the model fallback on Boxoban, at 320–585k tokens per episode. Every other task
runs at 0–58k tokens.

**Binding constraint:** a **HARNESS-BUG that regresses r036**. NavStage's `WALKABLE` excludes `-`/`|`, so a door
that the agent itself opens (drawn `-`/`|` after `The door opens.`) becomes a wall. That seals the room holding a
drawn `>` in both stuck R3 runs. It is compounded by a ModalStage/FoeStage `[yn]` 3-cycle and by the known
TASK-GAPs: no Sokoban planner and no Quest item opener.

## 1. What this harness does (mechanism)
`agent.py:PipelineHarness.act` builds a `StepContext` (`tools/context.py`) and runs every stage's `pre()` in
this order:

Legal → Modal → Door → Bump → Hazard → Foe → Nav → Orbit.

The first legal answer wins. Only if no stage answers does it call the model, snapping the reply to the legal
set. The `post()` hooks then run inner to outer, and the provenance goes to `reasoning`.

- **NavStage** (`tools/stages.py`) is the decision locus. It keeps a per-level glyph chart and a set of stood-on
  squares, and runs a BFS over `WALKABLE` glyphs that respects `BumpStage.blocked` edges. The goal, in order, is
  the drawn `>`, then the nearest unstood square bordering blank, then the nearest unstood square. When nothing
  is left, it emits `search` ×10 per square, then falls back to the model.
- **FoeStage** swings at any adjacent letter glyph and never retreats.
- **ModalStage** answers `[yn]`/`--More--` with the first of `y`, `more`, `esc` that is in the legal set.
- The prose (AGENT.md, memory.md) only matters on model-fallback frames. Those are almost all Boxoban.

## 2. What actually happened (causal chains, with evidence)

**A. Corridor-R3 run_00 and run_01: an opened door becomes a wall (HARNESS-BUG, 2 lost episodes)**
- **run_01, step 50:** `The door resists!`.
- **run_01, step 51:** `The door opens.`, and the prose says `vertical open door adjacent west`.
- **run_01, step 52:** Nav turns **east**. The opened door is now drawn `-` in the room's east wall
  (`|>....-#######`), and `-` is not in `WALKABLE`. The `>` is drawn at step 51, directly behind that door,
  yet Nav explores elsewhere for the remaining 48 steps. It never emits a `nav:stairs` hop.
- **run_00, step 57:** the same thing. `The door opens.` / `vertical open door adjacent east` at (13,17). Nav
  turns north at step 58.
  - The `>` is drawn from step 57 (`#-.......` / `....>|`).
  - By step 78 every reachable square is stood on, so Nav says `nav:exhausted -> search`. The run spends
    **20 searches** (steps 78–87 and 89–98) plus 2 model calls, and times out.
- **The regression:** r036 (`tools/chart.py:61–74`) and r001 (`spatial_memory.py:47`) both handled
  `door opens` and treated the perpendicular `-`/`|` as an open door. This port dropped that handling. It is the
  whole difference between r036's R3 4/5 and this candidate's 2/5.
- R3 run_04 is plain step-budget exhaustion: 64 distinct squares, 37 backtrack hops with d≥3, and no `>` drawn.

**B. Quest-Easy run_04: the modal/foe 3-cycle (HARNESS-BUG, 86 of 100 steps)**
- **Step 14:** Foe swings southeast at a peaceful goblin, and the game asks `Really attack the goblin? [yn] (n)`.
- **ModalStage:** `y` is not in the legal vocabulary, so it answers `more`. The game replies
  `Unknown command '^M'.`, and the prompt is gone.
- **Next frame:** Foe swings again at the same `o`.
- **Result:** the cycle `southeast → more → more` repeats from step 14 to step 99, which is 29 cycles.
- **Cause:** nothing records that the creature is peaceful, and ModalStage has no "decline" answer.

**C. Melee deaths: CB-Dark run_01–04 and Quest-Medium run_01–04 (8 deaths)**
- Every one has the same shape. Nav walks the corridor at full speed and steps onto the corridor mouth
  (CB-Dark (38,11), Quest-Medium (35,11)). From there Foe swings east, then southeast, then south, at rats
  arriving from 2–3 bearings.
- HP goes 16→14→9→5→3→1, and the agent dies at steps 22–28.
- CB-Dark run_00, the one win, had the same shape: 7 swings, then stairs.
- Prior chains call CB-Dark dice at n=5. What is new here is Quest-Medium, which now dies in 4/5 at step ~24
  **because** Nav reaches the rat room quickly. r036 also lost Quest-Medium, by "advancing into the open rat
  room".

**D. Quest-Easy run_00–03: lava partition (TASK-GAP)**
- Nav charts the west half of the level. Then 60–69 of the 100 steps are `nav:exhausted -> search`, rotated
  over squares, because the `}` column cuts the level in two.
- A horn `(` is drawn in the start room (`horn very near westnorthwest`), but nothing ever picks it up.

**E. Boxoban, 10/10 lost (TASK-GAP, plus the token bill)**
- Nav explores the pre-mapped room in about 5–50 steps (`nav:unstood`). Then goal is None, and `search` is not
  in the 4-cardinal vocabulary, so the model is called on 49–98 steps per episode.
- Boulders are pushed only by accident, and returns swing between −0.099 and 0.001 as a box leaves or reaches a
  `{` target.
- `bars` is only learned after the first `You cannot pass through the bars.` Step 0 of Boxoban-Medium run_00
  is a wasted bump.

## 3. Failure breakdown (27 losses)
| Mode | Episodes | Class |
|---|---|---|
| Boxoban: no push planner; model fallback | 10 | TASK-GAP (and ~96% of tokens) |
| Melee death at a corridor mouth (CB-Dark 4, QM 4) | 8 | TASK-GAP; CB-Dark is dice |
| Quest-Easy: lava partition; horn never acquired | 4 | TASK-GAP |
| Quest-Easy: peaceful `[yn]` → `more` → re-swing 3-cycle | 1 | HARNESS-BUG |
| R3: opened door drawn `-`/`\|` is charted as a wall, sealing a drawn `>` | 2 | HARNESS-BUG (regression vs r036) |
| R3: 100-step budget with `>` never drawn | 1 | TASK-GAP |

## 4. Binding constraint
**HARNESS-BUG first.** Two code bugs cost 3 episodes outright: the open door (2) and the peaceful loop (1).
They can be fixed without touching the architecture, and they restore r036's R3 level.

After that, the score is capped by **TASK-GAPs** that the model cannot close:
- Boxoban (10 episodes) needs a planner.
- The Quest pair (10 episodes) needs an item-acquisition and lava-crossing opener, plus a melee posture.

This is not a VLM-CEILING. The model now touches only Boxoban frames, and there it only burns tokens.

## 5. Concrete changes to try (ranked by leverage)

**5.1 Open doors are walkable (`tools/stages.py:NavStage`, plus `context.py`), for +2 R3 episodes.**
1. In `NavStage._absorb`, record a door square in any of these cases:
   - After a move `last_move=(p,d)`, the message contains `door opens`, `crashes open` or `WHAMM`. The door is
     `target(p,d)`.
   - The prose contains `open door adjacent <dir>` or `broken door adjacent <dir>`. The door is
     `ctx.target(dir)`.
   - Structural rule: a `-` whose two horizontal neighbours are both non-`-`, one of them `#`/`.`; or a `|`
     whose two vertical neighbours are both non-`|`, one of them `#`/`.`. (r036 `chart.py:74` has this.)
2. Keep a `self.doors` set. `_walkable(p)` returns True for `p in self.doors`.
3. In `_bfs`, forbid **diagonal** steps into or out of a door square. NetHack refuses them, and doing so
   avoids a doorway-diagonal bump.
4. Predicted effect: R3 run_00 and run_01 go straight to the `>` once it is drawn (51/57 steps in, with ≤20
   hops needed). R3 goes from 2/5 to 4/5, and primary gains about +0.05. It is inert on MazeWalk and Boxoban
   (no doors).
5. Also add r036's §5.2(1) rule: when a `>` is charted but unreachable and no `}` is charted, rank frontier
   cells by `(chebyshev(cell, stairs), dist)`. That turns the 20 wasted searches into search/probe on the wall
   facing the `>`.

**5.2 Refuse the peaceful fight (`ModalStage.pre`, `FoeStage.pre`), fixing QE/04's 86 lost steps.**
1. On `Really attack`, answer `esc` if it is legal; otherwise answer any legal non-`y` key. Never answer
   `more`, because `^M` just leaves the prompt and re-arms Foe.
2. Record `ctx.shared['peaceful'].add(target square)` for the last swing bearing. FoeStage skips peaceful
   squares.
3. Nav treats a peaceful creature square as blocked for that step only (a ranking term, not a lasting veto).
4. Predicted effect: frees the QE/04 episode for exploration. Survival and primary are unchanged on its own,
   but this is a prerequisite for 5.4.

**5.3 A Boxoban push planner (`tools/sokoban.py` plus a `SokobanStage` ahead of Nav), up to +10 episodes, the
largest lever.**
1. **Detect:** `` ` `` boulders plus `{` targets inside a `#` ring. On detection, set `bars=True` from frame 0
   and skip Nav.
2. **Decode:** `#` = wall, `` ` `` = box, `{` = target, `.` = floor. The hero's masked start square is floor.
   Remember targets from the first frame, because a box on a target hides the `{`.
3. **Search:** A* over (the canonical player-reachable region, the sorted box tuple).
   - Moves are cardinal pushes only.
   - Prune dead corners that are not targets.
   - Heuristic: the sum of minimum Manhattan distances from boxes to targets.
   - Wrap `frozenset`s in heap entries (`<` on a frozenset is a subset test).
4. **Execute:** emit the plan as a move queue, and re-plan on any divergence (`in vain`, a box moved
   unexpectedly).
5. Prior chains measured this at 10/10 live with 0 model calls (winnable iff the optimal plan is ≤99 moves).
6. Predicted effect: primary up to about +0.20–0.25, and roughly −96% of this candidate's tokens.

**5.4 Quest opener (a `QuestStage`, gated on a charted `}` plus Nav reporting exhausted), up to +2 QE and some
QM.**
1. Walk to the prose-named `horn`/`wand` glyph (`(` or `/`) and `pickup`.
2. On a silent pile, `kick` it **once**.
3. `apply` or `zap` the item. Answer the item-letter prompt, and answer `In what direction?` with the bearing
   of the nearest `}`.
4. Stop when you see `The lava cools and solidifies.`, then hand back to Nav.
5. This stage must own its prompts ahead of ModalStage.

**5.5 A corridor-mouth hold (`FoeStage`/`NavStage`), aimed at Quest-Medium's 4/5 deaths. Price it at ≥200
episodes before trusting it.**
1. When ≥2 creatures are drawn within 6 squares and the next Nav hop leads to a square with >2 passable
   neighbours, do not advance. Hold on the current 1–2-lane corridor square with `search`, where it is legal,
   and swing at arrivals.
2. Commit the post to the square you stepped *to*.
3. Leave CB-Dark ranking alone; it is dice at n=5.

## 6. Exhausted — do NOT repeat
- Grounding the model's prompt (ledger tables, visit counts, prose priors). The r029 lineage showed that 63.6% of
  the model's picks contradict it.
- A bump veto that books every no-move. This candidate's message-classified Bump works; keep it.
- A bare `more` answer on `[yn]`/item prompts. It turns a lock into a cycle (again, see §2B).
- Kicking the Quest pile twice (it re-piles the items).
- Melee retreat keyed on HP, or any disengagement dose. It kills more.
- Ranking the frontier by raw `information`.
- Committed routes that are not re-elected every step.
- CB-Dark melee tweaks judged at n=5.
- Using `search` as the fix when a charted `>` is unreachable. Here the real cause was the open-door glyph (§5.1).
