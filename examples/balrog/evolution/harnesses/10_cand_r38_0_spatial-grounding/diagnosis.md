# Diagnosis — minihack iter 38 / cand_r038_0_spatial-grounding (parent cand_r001_0)
**Headline:** primary = **0.325 ± 0.074** (13/40). That is below the parent r001's 0.350 and the chain best
r036's 0.375.
- Progression 0.300, survival 0.775, step_efficiency 0.248.
- Cost is 188k tokens/ep. The parent was 301k, but r036 is 89k.
- Per task: MazeWalk-9x9 5/5, MazeWalk-15x15 5/5, Corridor-R3 **3/5** (parent 4/5). The other five tasks
  (Boxoban-H, Boxoban-M, CB-Dark, Quest-Easy, Quest-Medium) are all 0/5.

**Binding constraint:** the route-first rewiring works: MazeWalk and the R3 wins use 0 model calls. But
`plan_route` treats an **opened door as a wall**, because an open door is drawn `-`/`|`, which is in
`WALL_GLYPHS`. That costs both R3 losses. The other 25 losses are the same untouched TASK-GAPs as before:
Boxoban has no push planner, Quest has no item/lava opener and no corridor hold, and CB-Dark has no fight
post. On top of that, the VLM copies `rec` on **1412/1415** model steps, so every model call is pure cost.

## 1. What this harness does (mechanism)
`agent.py:GroundedHarness.act` runs a stage ladder:
- **Stage 0, modal:** runs when a prompt is open, i.e. the message contains `[yn`/`--More--`/`what do you
  want to`, or cursor y==0. It skips `memory.observe`, books `(last_pos, last_move)` as refused on
  `Really attack`, and answers `more`/`esc`.
- **Stage 1, stairs:** standing on a charted `>`, it emits `down`.
- **Stage 2, chart:** runs when `ground()` returns a `recommended` whose `route_goal` is a stairs route or
  "nearest unexplored edge". It emits that first hop with no model call.
- **Stage 3, model:** everything else, meaning the "least-visited neighbour" fallback. The model gets
  the rendered menu plus `skills/spatial.md`, and `arbitrate` snaps and vetoes the reply.

`tools/grounding.plan_route` is a BFS over `seen ∪ grid` with `_passable(g) = g not in {'-','|','}',' '}`.
It goes to a drawn `>`, else to the nearest **unstood** frontier by the key `(dist, index)`. The unstood
rule is new this round.

Decision census over the 40 episodes:
| Stage | Steps |
|---|---|
| `chart:*` | 915 |
| `model` | 1415 |
| `modal` | 8 |

**`reply == rec` on 1412/1415 model steps.**

## 2. What actually happened (causal chains)

**A. MazeWalk 10/10, R3/00, /02, /04: pure chart.** These use 0 to 8k tokens, with no ABA orbit. The
stood-frontier retirement did what it predicted, so **keep it.**

**B. R3/01 and R3/03: the open door is a wall to the BFS (HARNESS-BUG, 2 episodes).**
- **R3/03, step 9 at (20,3):** `west`, and the message is `The door opens.`
- **Step 10 frame:** the prose says `vertical open door adjacent northwest`, and the door cell (19,3) is
  now drawn **`-`** (`-....|` on map row 2).
- `plan_route` sees `-` as impassable, so no frontier is reachable. `ground` falls to "least-visited
  neighbour", which hands the step to the model. **60 model steps (10–70)** then mow the 4×5 room:
  `south,south,south,east,north,…`, all copying `rec`.
- At step 70 the least-visited walk happens to pass west through the door. The chart immediately takes
  over again (`chart:route` 71–99), but the episode times out.
- **R3/01:** the door opens at step 53 at (20,3), and the very next step the chart walks **east**, away
  from it (steps 54–75), because the doorway is walled off in the BFS.
- It ends in a dead-end corridor stub at (20..23,7). There the "least-visited" fallback probes all 8
  bearings per cell (steps 76–98: `It's solid stone.` ×20). Each probe costs a step, because an unknown
  blank neighbour is not vetoed until it has been refused.
- In the parent (model every step), the model occasionally deviated, and R3 was 4/5. Here the chart is
  literal, so the bug became deterministic.
- r036 fixed door identity (doors passable, diagonal-through-door pruned) and got R3 4/5.

**C. Quest-Medium: 4/5 deaths at steps 23–26, and QM/02 hits the 100-step cap (TASK-GAP).**
- **QM/01, steps 0–12:** `chart:route` goes east along the corridor, straight **past** the horn `(`
  at (22,11) in the start room. The prose says `horn far westsouthwest`.
- **Step 12:** the agent enters the lit rat room `|@...r..rr......}>..|`. Nothing unstood borders
  undrawn space, so the case falls to the model.
- The model wanders to (35,13), then emits `north` into a rat 4 times while HP goes 16→13→8→5→1, and
  dies at step 22.
- There is no hold at the corridor mouth, HP is unread, and there is no item acquisition.

**D. Quest-Easy 0/5: VLM-inert room mowing (TASK-GAP).**
- 100/100 steps are `model`, covering 26 unique cells (a 5×5 room sealed by a `}` column).
- `horn adjacent south` appears **at step 0** of QE/00, and `horn` appears in 94–96 of 100 frames.
- The harness has no pickup/apply rung. About 440k tokens/ep are spent with the model copying `rec`
  every time.
- QE/04 had 8 modal steps. The `[yn]` fix did break the old `north,more,more` 54-step lock.

**E. CorridorBattle-Dark 0/5: all deaths at steps 23–30, 100% `chart:route`.**
- **CB/00:** the frontier leads east down the corridor and into the room mouth at (35,11).
- There the agent takes `giant rat adjacent east, southeast, and south`: three attackers. Death comes at
  step 26.
- The chart's "nearest unexplored edge" is the fight square. r026/r036/r037 got 1/5 here, which is
  dice at n=5, per the chain.

**F. Boxoban 0/10: the chart and the model alternate on `#`.**
- `#` is passable to the BFS (it is not in `BLOCK_GLYPHS`). The route takes 1 step (e.g. BH/00
  `south` "(2)"), then its next hop is noun-vetoed, and the step drops to the model.
- The pattern is: model `east`, model `north`, chart `south`, repeated.
- Episodes cover 5–21 cells, with 0–5 `in vain` pushes. There is no push plan. The model spends
  ~430k tokens/ep here, the largest share of the cost.

## 3. Failure breakdown (27 losses)
| Mode | Eps | Class | Mechanism |
|---|---|---|---|
| Boxoban: no push planner; `#` passable | 10 | TASK-GAP | no `tools/sokoban.py` |
| Quest-Easy: sealed room, horn adjacent, never acquired | 5 | TASK-GAP | no item/lava rung |
| Quest-Medium: advances into the open rat room and dies | 5 | TASK-GAP | no hold; route passes the horn |
| CB-Dark: melee in the open | 5 | TASK-GAP / dice | frontier = the fight square |
| R3: an opened door (`-`/`|`) is a wall to the BFS | **2** | **HARNESS-BUG** | `grounding.WALL_GLYPHS` + `_passable` |

The model decides 1415 steps and agrees with `rec` on 1412, so its contribution to primary is ≈0. It is
the whole cost bill.

## 4. Binding constraint
- **HARNESS-BUG (cheap, 2 eps):** door identity in `plan_route`. This one bug explains the entire
  regression vs the parent (R3 4→3) and vs r036.
- **TASK-GAP (dominant, 25 eps):** Boxoban solver, Quest opener + corridor hold, CB fight post. These are
  exactly the gaps the r001 and r036 diagnoses already listed; this round re-plumbed the control flow
  instead.
- **Not VLM-CEILING:** the VLM never disagrees with the planner. The fallback hands it steps where the
  planner itself has no goal, so no information reaches it that could help.

## 5. Concrete changes (ranked by leverage)

1. **Boxoban push planner, `tools/sokoban.py` plus a stage ahead of stage 2** (up to +10 eps; also
   removes ~430k tokens/ep on 10 eps).
   - **Detect:** boulders `` ` `` and targets `{` inside a `#` ring on frame 0.
   - **Search:** A* over (the player's canonical reachable region, box set). Use cardinal pushes only,
     prune dead corners, and use the sum of min Manhattan distances as the heuristic. Heap entries must
     not compare a `frozenset`, since `<` on a frozenset is a subset test.
   - **Pitfalls:** remember targets from frame 0 (a box on a target hides it). The square under `@` at
     frame 0 is floor.
   - **Execute:** compile to a move tape, and re-plan on any divergence (`in vain`, box not where
     expected).
   - Cross-chain, this gave 19/20 at 0 model calls.
2. **Door identity in `tools/grounding.py`** (+1–2 R3 eps, deterministic).
   - Keep a `memory.doors` set: record `step(pos, last_move)` when the message contains `door opens` /
     `crashes open`, plus any cell named `open door`/`doorway` in `percept.adjacent`, mapped to its
     coordinate.
   - In `_passable`/`_neighbours`, treat a door cell as passable regardless of its `-`/`|` glyph. Keep
     the existing no-diagonal-through-door prune, and apply it to door cells, not only to `+`.
   - Also render an open-door cell as passable in `ground()`'s wall-glyph veto (it already checks
     `opening` nouns, so the BFS is the only gap).
   - Predicted: R3/03 exits west at step 10 instead of step 70. R3/01 walks through (19,3) at step 54.
3. **Quest opener + corridor hold, new `tools/quest.py` rung before stage 2.**
   - **Trigger:** a `}` is charted and `pickup` is legal.
   - **Acquire:** the prose names the item (`horn adjacent south` at QE step 0; `horn far
     westsouthwest` in QM). Route to the nearest `(`, `pickup`, and answer the item menu via its
     letter. On a pile, `kick` **once**, never twice, and never toward lava.
   - **Cross:** `apply` the horn, answer the item prompt by letter, answer `Improvise?` with `northwest` (it sends the `y` key), and
     answer the direction with the bearing to the nearest `}` on the line to `>`. Stop on `solidifies`.
     This rung must own those prompts **before** stage 0's `more`/`esc`, or it cancels them.
   - **QM hold:** when ≥2 creatures are drawn within 6 squares and the agent is at a corridor mouth
     (≤2 passable neighbours), do not advance. Spend the turn with `search`/`wait` and swing only at an
     adjacent foe. Commit the post to the cell you stepped to.
   - Cross-chain: QM 0→3–5/5, QE ~2/5.
4. **Model fallback, `agent.py` stage 3** (cost only, primary-neutral).
   - Since reply==rec on 99.8% of steps, stage 3 can emit `g.recommended` directly with no model call.
   - Alternatively, keep the call only when `g.recommended is None`.
   - Predicted: cost goes from 188k to ~0–20k/ep, with identical actions. **Do this only after (1) and
     (3)**, because on Quest/Boxoban the least-visited fallback is the whole policy. It is a mower
     either way.
5. **Dead-end probing** (efficiency, R3/01 steps 76–98).
   - In `ground`, veto a bearing whose target is blank (`' '`) and not named in the prose, when the
     current cell is a stood corridor cell. Otherwise the fallback spends up to 7 steps per cell bumping
     `solid stone`.
   - Better: add a `search` rung at dead-ends when no frontier remains, with a budget keyed on the
     probed cell.

## 6. Exhausted — do NOT repeat
- **Re-plumbing the control flow** (model-first ↔ chart-first) without adding a planner. r026, r027,
  r036, r037 and this candidate all land at 0.325–0.375 with the same five 0/5 tasks.
- **Prompt/skill wording for the model:** the model copies `rec` 99.8% of the time, so prose changes are
  inert.
- **CB-Dark tweaks priced at n=5:** CB-Dark is dice; per the chain it needs ≥200 eps.
- **The frontier orbit fix** (stood cells retired) is done and works. Don't re-derive it.
