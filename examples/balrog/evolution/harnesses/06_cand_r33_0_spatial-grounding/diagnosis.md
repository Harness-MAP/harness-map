# Diagnosis — minihack iter 33 / cand_r033_0_spatial-grounding (GroundedAgent: model picks, ring-of-8 vets)

**Headline:** primary=0.175 (7/40, SE 0.060), progression 0.175, survival 0.750, step_eff 0.145,
cost **330k tok/ep**. Per task: MazeWalk-9x9 5/5, MazeWalk-15x15 2/5, and 0/5 on the other six tasks.
For comparison: the anchor scored 0.100 / 0.700 / 309k, and r027 (a field walker in which the model almost never
decides) scored 0.325 / 0.800 / 87.5k.
**Binding constraint:** the decision horizon is **one hop**. The map decode is correct (`pos` was decoded on
2758/2760 steps, and no move entered a wall or lava). But the only planner is "the VLM picks, then `vet` swaps
out lethal or stale single steps". Nothing routes toward a `>` that is drawn but not adjacent, toward a frontier
that is not adjacent, or toward an item/objective. The agent therefore paces locally until the 100-step timeout
(6/8 tasks), or wanders into open-room melee (CB-Dark, Quest-Medium: 10/10 deaths). This is a
**TASK-GAP made structural by a HARNESS design choice**: move the decision to a BFS router and keep the VLM as a
fallback.

## 1. What this harness does (mechanism)
`orchestrator.py:GroundedAgent.act` does the following:
1. `tools/grid.py:parse_glyph_map` reads the `(x=,y=)` cursor and the `map:` block (row = y−1) and classifies the
   8 neighbours as wall/lava/creature/stairs down/boulder/open/dark. It also reads prose `X adjacent <dirs>`
   into `lava` and `foes`, and records the last `>` it sees in `down`.
2. `learn` updates `visits[pos]` and `trail`, and adds `(prev_pos, dir)` to `refused` when a move left the
   position unchanged. Creature targets are exempt.
3. `ground` appends a block to the prompt: the 8 neighbours with their visit counts, "stairs down is dx east,
   dy south", and the `memory.md` trail. `AGENT.md`, `rules.md` and `skills/unstick.md` are prepended as priors.
4. The VLM makes one call. `extract_action` does a whole-word projection onto the legal set.
5. `vet` keeps the model's move unless `forbidden` returns True. `forbidden` covers: lava by prose, ring kind
   lava/wall, a refused (pos,dir), or a "stale" move (target == `trail[-2]` **and** `visits>=3`). If the move is
   forbidden, `vet` picks the minimum of `score` = (RANK kind, visits, ORDER).

Who decided each step (CSV `Reasoning` column, 2754 steps): **model 2383 (86.5%)**, grounded 387, fallback 34.
On Quest-Medium the model decided 127/127 steps.

## 2. What actually happened (causal chains)

**A. Corridor-R3 0/5: a drawn `>` 22 squares away is never reached.** In R3 run_02, frame 0 shows
`|.>.........     .......@|`. The hero is at (25,5), `>` is at (3,5), and a doorless gap at (18,5) leads
west. `ground` prints "stairs down `>` is -22 east, 0 south" on all 99 steps. The model walks west along row 7,
reaches x=16 at step 11, and turns back east at step 12. It then drifts south into other corridors (min x=4, but
on row 16) and times out. The `>` hint is a bearing with no route. The gap at x=13–17 is blank (`dark`), and the
model does not enter a blank square on purpose. On the other 4 R3 episodes the `>` is never drawn, and the agent
covers 218 distinct cells over 5 episodes with 273/495 revisits, so exploration stays local.

**B. MazeWalk-15x15 3/5 losses: blank squares are bumped, and a frontier that is not adjacent is invisible.**
- 94 model moves went into `dark` (blank) squares, and **89 of them were no-moves** (stone). `forbidden` does not
  treat `dark` as a wall. `refused` only books a (pos,dir) *after* a bump, so every blank neighbour costs one
  wasted turn. There were 92 no-moves out of 330 steps (28%). MW-9x9 shows the same pattern: 20 no-moves out of
  81 steps.
- On 168/287 model-decided MW15 steps, **no neighbour was unvisited**, so the nearest frontier was ≥2 hops away.
  When an unvisited open neighbour *was* adjacent, the model took it 82/119 times. The one-hop visit counts work
  when the frontier is adjacent. They carry no information once it is not, and the stale veto (`visits>=3` and
  `trail[-2]`) fires too late to break the drift.

**C. CorridorBattle-Dark 0/5, all deaths: the agent paces in the 3x3 start room until the rats arrive.** HP
traces (action+HP per step): run_00 spends 32 steps at HP 16 on n/s/e/w in the start room
`|...|/|@...##/|..<|`. The exit `##` east at (27,11) is never taken for good. Then HP goes 13→…→2→dead. Runs 01–04
show the same shape: 31–36 idle steps, then 6–40 steps of melee to death. The stale veto needs `visits>=3` *and*
an exact return to `trail[-2]`. In a 3x3 room the model cycles through 4–5 squares, so the veto rarely fires
(22 grounded steps out of 256). The rats get the agent while it stands in the open room, not in the corridor.

**D. Quest-Medium 0/5, all deaths: the agent charges into the rat pack in the open room.** Runs 00 and 01 are
nearly byte-identical: `north, east×4, south×2, east×9`. This walks the corridor `######` into the room at
x=32–36, where 4–5 `r` stand at step 17 (`...@%r.r...`, `rr` above). The agent swings `east` from (36,13) for 7
turns, with 36/36 no-moves being attacks, and dies at HP 16→0 around step 24. The model decided all 127 steps,
because `forbidden` never objects to a creature square (`RANK` puts creature at 1, and AGENT.md says "attack a
creature by moving into it"). The corridor mouth at (31,13) was a one-wide chokepoint the agent had just left.
Also, at step 0 the prose says **`horn adjacent south`**. The frost horn that solidifies the lava is one square
away and is never picked up.

**E. Quest-Easy 0/5 (timeouts, no deaths): the lava column is avoided correctly, but nothing crosses it.**
Run_00 frame 50: `|...@.}......---` with `horn` drawn as `(` at (26,9). The agent spends 100 steps in the
6x5 west room (125 distinct cells over 5 episodes, the top cell visited 10x). The legal set has
`pickup`/`apply`, but the harness scores only the 8 bearings and AGENT.md mentions only `>`, lava and creatures.
The lava veto works (0 lava deaths, versus 4/10 Quest lava deaths at the anchor), so it only moves survival.

**F. Boxoban 0/10: the goal is misdescribed.** AGENT.md says "reach `>`", but Boxoban has no `>`: it is won by
pushing boulders `` ` `` onto fountains `{`. `grid.py` lists `{` in `FLOORISH`, so fountains show up as plain
"open". In Boxoban-Hard, 414/495 moves were revisits and 56 formed period-2 orbits, and only 31/990 steps
targeted a boulder. The agent performs a random cardinal walk.

**G. Cost: 330k tok/ep (+7% over the anchor).** BALROG's 16-frame history is still replayed in full, and every
turn also adds a priors block and a grounding block. The model is consulted on 100% of steps.

## 3. Failure breakdown (40 episodes)
| category | eps | tasks |
|---|---|---|
| Timeout: one-hop walker, no route to a drawn/far target | 8 | R3 5, MW15 3 |
| Melee death after idle pacing in the open (no chokepoint) | 10 | CB-Dark 5, QM 5 |
| Timeout: objective needs an item verb (horn pickup/apply) | 5 | QE 5 (QM also has one) |
| Timeout: wrong goal model (Sokoban) | 10 | Boxoban-H/M |
| Wins | 7 | MW9 5/5 (3 ended in ≤6 steps), MW15 2 (5 and 30 steps) |

Numeric pre-pass cross-check: survival 0.750 matches exactly the 10 melee deaths. There are 0 lava deaths and
0 wall re-bumps, so the vet works as designed.

## 4. Binding constraint
**TASK-GAP (structural), fixable in the harness.** Perception is solved: `grid.py` decodes the map exactly and
the vetoes never misfire. What is missing is a *multi-hop goal-directed controller*. The model is used as the
planner, and a one-word Instruct reply at T=1.0 does not carry a 20-square route or a frontier 5 hops away,
however well the ring is grounded. That is the VLM-CEILING part: re-wording the ring will not fix it. r027
already showed that the same map decode plus a BFS/field walker scores 10/10 MazeWalk and 2/5 R3 with the model
mostly silent.

## 5. Concrete changes (ranked by leverage)
1. **Add a BFS router and let it decide; the model breaks ties or acts as fallback** (`orchestrator.py`, new
   `route()` next to `vet`, and let `grid.parse_glyph_map` return the full `rows` grid).
   - Build a passable set: floor-ish glyphs, `>`, doorways, `+` (closed doors are passable via auto-open;
     book a refusal when a move fails). Treat blank as passable at cost 3 so that gaps like R3/02 (13–17,5) are
     routable, and walls, `}` and bars as impassable. Keep every door/`-|` noun lesson from r027 (an open door is
     drawn `-`/`|`: use the prose `doorway`/`open door` noun to override).
   - Targets, in order: a drawn `>` (then emit `down` when standing on it), else the nearest *frontier* (a
     known passable cell with a blank neighbour that has not been refused), with ties broken by distance
     ascending in a total order. Re-elect every step.
   - Emit the first hop of the path. Consult the VLM only when no target is routable.
   - Predicted: R3 from 0/5 to 2–4/5 (R3/02 is solvable on frame 0), MW15 from 2/5 to 4–5/5, and cost down
     roughly 5–10x, because the model is skipped whenever a route exists.
2. **Make `dark` a soft wall in `forbidden`/`score`** (`orchestrator.py:forbidden`). If any open, unrefused neighbour
   exists, veto a model move into `dark`. Keep `dark` only as a router edge with a penalty. Predicted: MW
   no-moves drop from 28% toward under 5% (89/94 blank moves were bumps). Do *not* make blank hard stone
   everywhere: the known trap is that it starves the frontier on dark levels and blocks the R3 gaps.
3. **Add an item/objective rung for the Quest pair** (`orchestrator.py`, before the router). Gate it on the prose
   noun `horn`/`wand` being drawn and `pickup`/`apply`/`zap` being in `legal()`: route to the item square, emit
   `pickup`, then `apply`/`zap` toward the `}` bearing. Handle the `[yn]`/direction prompts with an explicit
   modal answerer (see r027's `y`/`more`/`esc`). Sibling lineages price this at QM 0→3/5 and QE 0→2/5. QM run 00
   starts with `horn adjacent south`.
4. **Hold the corridor mouth instead of charging into the room** (Quest-Medium, CB-Dark). When ≥2 foes are
   drawn and the hero is in a one-wide corridor square (walls on both lateral sides), swing only at an adjacent
   foe and otherwise emit `search`/`wait` rather than stepping into the room. For the idle start-room pacing on
   CB-Dark, change 1 already fixes it: the frontier router leaves the 3x3 room through `##`. Price CB-Dark on
   ≥200 episodes before trusting it (the melee axis is dice; see §6).
5. **Fix the goal description for Boxoban** (`grid.py` gives `{` its own kind `fountain`; AGENT.md gets a
   per-task goal line keyed on whether `{` and `` ` `` are drawn). The real lever is a push-space planner
   (siblings: one push-space A* gives 19/20 when the optimal plan is ≤99 steps; Boxoban is fully drawn and
   premapped, and `#` is BARS there). If the proposer has no budget for a solver, leave Boxoban alone. Prose
   alone will not move it.
6. **Cut cost** (`orchestrator.py:act`): after change 1, skip the VLM call whenever the router answers. When it is
   consulted, elide the replayed history down to the current frame plus the grounding block.

## 6. Exhausted — do NOT repeat
- **More prose or grounding in the prompt with the model still deciding.** This candidate is that experiment.
  86.5% of steps were model-owned and the result was 0.175, versus 0.325 for the r027 walker in which the model
  rarely decides.
- **One-hop visit-count ranking/veto as the explorer.** It fails on every frontier that is not adjacent (§2B).
  Raising or lowering the `visits>=3` threshold is a parameter tweak on the wrong horizon.
- **Melee retreat/disengage re-parameterisations on CB-Dark** (refuted, dice; needs 200–500 episodes).
- **Frontier ranking reformulations** (information gain, nub demotion) once a correct BFS explorer exists: flat.
- **Treating `-`/`|` as always impassable once doors are routed** (r027: an open door is drawn `-`/`|`).
