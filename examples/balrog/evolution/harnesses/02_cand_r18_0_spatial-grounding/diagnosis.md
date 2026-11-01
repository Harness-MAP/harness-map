# Diagnosis — minihack evolve_r018 / cand_r018_0_spatial-grounding (middleware pipeline)

**Headline:** primary = **7/40 = 0.175** (±0.060), survival 0.75, step-eff 0.159, ~305k tok/ep. The anchor scored 0.100.
- Wins by task: MazeWalk-9x9 4/5, MazeWalk-15x15 2/5, Corridor-R3 1/5. Every other task is 0/5.
- Lava deaths fell from 4 to **0**: Quest-Easy survives 5/5, and HazardStage fired 6 vetoes.
- The 10 deaths are all melee: CorridorBattle-Dark 5/5 and Quest-Medium 5/5, every one killed by a giant rat.
- The 23 capped episodes end in **deterministic 3/4/6-cycles**: 17 are strictly periodic in their last 40 steps.

**Binding constraint:** the pipeline has **no goal and no route**. Every stage only *reacts to the model's proposed direction*, so the frozen VLM still picks where to go. Worse, OrbitStage's perpendicular redirect gets copied by the model's in-context history and becomes a longer cycle. This is a HARNESS-BUG: the fix is a chart + BFS route (a frontier or a drawn `>`), which is already proven at 0.35–0.375 in this same chain (r001/r036/r041).

## 1. What this harness does (mechanism)
`orchestrator.py:PipelineHarness.act` works in five steps:
1. `context.build(obs)` builds a StepContext from the frame: cursor pos, `map:` cells, message, and adjacent-noun prose. It keeps no memory across steps.
2. `pre()` runs outer→inner: Legal, Modal, Hazard, Bump, Orbit. Only ModalStage can short-circuit. Hazard and Bump only append `ctx.notes` lines to the prompt.
3. One model call. The prompt is NaiveAgent's plus a "Checks on this frame:" block. The reply goes through `action_io.snap`: whole-word match against the legal set, falling back to the stock string.
4. `post()` runs inner→outer and each stage may rewrite the action:
   - `OrbitStage`: an immediate reversal onto a square visited ≥2× is sent to the least-visited open neighbour.
   - `BumpStage`: a repeat of a (pos, dir) pair that previously left the hero in place is replaced with `alt[0]`.
   - `HazardStage`: a step onto a `}` glyph or a lava/water noun is replaced with `safe[0]`.
   - `ModalStage`: no rewrite.
   - `LegalStage`: only logs illegal actions.
5. The trail is written to `reasoning`.

Stage state is limited to `Bump.blocked`, `Orbit.visits`/`Orbit.last`. Nothing plans. The prose files (AGENT.md, rules.md, skills/) are empty.

## 2. What actually happened (causal chains, with evidence)
Positions below come from decoding the CSV `Observation`, which is the observation *after* that row's action.

**A. OrbitStage turns period-2 orbits into period-3/4 orbits, and the model locks onto them (23 capped episodes).**
The stage only tests `self.last == OPPOSITE[action]`. Its redirect is perpendicular, and BALROG replays the *applied* action in the model's history. So the model now sees a 3-step rhythm and copies it, and no reversal ever occurs again to trigger the stage.
- **MazeWalk-9x9 r00:**
  - Steps 3, 6, 9, 12 and 15 all show `model:south | orbit south->east`.
  - From step 17 onward the model emits `east, southwest, north` with **zero** stage interventions until the cap.
  - Only 3 distinct cells are visited in the last 40 steps.
- **MazeWalk-15x15 r04:**
  - Step 4 is `orbit west->north`. From then on, `southwest, east, north` repeats for 96 steps over cells (43,14)`<`, (44,14) and (44,13).
  - The frame at step 3 shows an **11-cell corridor running west** from `<` and a N–S corridor through (44,·). The agent never enters either.
- **Corridor-R3 r00:** `orbit west->south` at step 4, then a `northwest, east, south` 3-cycle.
- **Boxoban-Hard r02:** `orbit east->north` at step 6, then `west, north, east, south`×23, a 4-cell square with no intervention.
- **Census of the 23 capped episodes' last 40 steps:**
  - Period-4: 9 (all Boxoban except Medium r01).
  - Period-3: 5 (R3 r00/r04, MW15 r00/r04, MW9 r00).
  - Period-6: 3.
  - Aperiodic but ≤6 cells: most of the rest.
- **Revisit rate among moves:** 0.83–0.93 on every task except QM (0.48). Distinct cells per 100-step episode: Boxoban 6–16, MW15 4–23.
- The non-trivial wins are MW15 r01 (20 steps) and R3 r02 (26 steps). Those are cases where the model happened to walk straight at the goal. The other 5 wins are 2–6 step starts next to `>`.

**B. BumpStage books a melee swing as a wall bump, then vetoes the next swing (the melee deaths).**
- An attack on an adjacent rat leaves the hero in place, so `BumpStage.pre` adds `(pos, dir)` to `blocked`.
- The next time the model swings the same way, `post` rewrites it to `alt[0]`, the first unblocked entry in `DELTA` order, which is usually `north`/`south`/`west`. The hero turns its back on the rat and gets bitten.
- **Quest-Medium:** 14 of the 15 bump-vetoes immediately follow a melee message.
  - Example: r04 step 24 is `model:east | bump-veto east->west` right after "You kill the giant rat!"
  - Deaths come at steps 23, 52, 29, 32 and 31, e.g. r00 steps 19–22: "The giant rat bites!" ×2 per turn while the agent alternates east/south/north.
- **CorridorBattle-Dark:** 5 of 11 bump-vetoes follow melee messages. No-move rate is 0.25 (CB) and 0.22 (QM).
- **Also:** a bump is booked without reading the message ("It's a wall." / "It's solid stone." vs "You miss the giant rat."). And `expect` is set *before* HazardStage may rewrite the action, so the ledger can book the wrong direction.

**C. `snap` cannot parse 16-point compounds (Quest-Easy r04).**
- The model emitted `southsoutheast` **51 times** (steps ~49–99). `\bsouth\b` has no word boundary inside `southsoutheast`, so `snap` returns the raw string.
- LegalStage only logs it (`illegal 'southsoutheast'`). BALROG then silently executes `north` and bumps "It's a wall." at every step.
- This accounts for 51 of the 61 illegal steps in the whole eval (QE 52, BM 6, QM 3).

**D. What worked.**
- HazardStage: 6 lava vetoes (e.g. QE r03 steps 5, 6, 62, 85, 86, all `east->north`) and 0 lava deaths, against 4 for the anchor.
- ModalStage answered the QE r04 `[yn]`/`--More--` frames with `more` (10×) without a model call.
- These are real survival gains, but a survived Quest-Easy episode still times out: it wanders 4–17 cells and never reaches the objective.

## 3. Failure breakdown (40 episodes)
| category | eps | evidence |
|---|---|---|
| capped in a model-driven 3/4/6-cycle (often seeded by an orbit redirect) | 18 | Boxoban 9, R3 4, MW15 2, MW9 1, QE 2 (periodic tails) |
| capped wandering ≤17 cells, no goal pull | 5 | QE r00/r02/r03, MW15 r03, BM r01 |
| killed by giant rat (bump-veto breaks melee in QM) | 10 | QM 5/5, CB-Dark 5/5 |
| of which: illegal compound `southsoutheast` → north | 1 ep, 51 steps | QE r04 |
| wins | 7 | 5 trivial (≤6 steps), MW15 r01 (20), R3 r02 (26) |

Stage-firing census over 2,801 steps:
- `orbit`: 203
- `bump-veto`: 116
- `modal`: 10
- `hazard-veto`: 6
- `illegal`: 61

The model was called on 2,791 of 2,801 steps (99.6%).

## 4. Binding constraint — classification
**HARNESS-BUG (primary).** Veto-only middleware cannot create a destination, and the one stage that alters the path (Orbit) is a local, period-2-only rule the model imitates. Two further code bugs apply:
- Bump-as-melee in `stages.py:BumpStage.pre`, which turns swings into flights and contributes to the 10 rat deaths (CB-Dark is otherwise largely dice).
- The compound-bearing parse in `action_io.py:snap`.

This is not a VLM ceiling: the frames contain enough to route. MW15 r04 draws the unexplored corridors and R3 draws its corridors. Siblings in this chain that route with BFS on the same model reach MW 10/10 and R3 4/5.

**TASK-GAP (secondary):**
- Boxoban needs a push planner; prior work on this benchmark found a push-space A* over the premapped board solves ~19/20.
- CB-Dark melee is largely dice.
- Quest needs the item/kick lever.

## 5. Concrete changes to try (ranked by leverage)
1. **Add a route stage that owns the decision (new `tools/chart.py` + a `RouteStage` placed innermost in `default_stack`).**
   - Keep a persistent chart across steps: terrain from the `map:` block (row = cursor.y−1; `.#<>{+` and open-door `-`/`|` passable; `}` lethal), visit counts, and the bump ledger. Reset it on `reset()` and on a Dlvl change.
   - In `pre()`: if `>` is drawn and BFS-reachable, return the first hop. Otherwise BFS to the nearest unvisited passable cell bordering unknown space, using a total order `(dist, visits, y, x)` so re-election each step is a fixed point.
   - Short-circuit so the model is not called. Consult the model only when no route exists.
   - Predicted: MW9/MW15 → ~5/5 each and R3 → ~4/5, i.e. primary ~0.35. Cost falls sharply (most steps make no model call).
   - Causal reason: 18 of 23 capped episodes are cycles over ≤6 cells with drawn, unvisited frontier cells (MW15 r04 has 11+ corridor cells in view).
2. **Fix BumpStage (`stages.py:BumpStage.pre`).**
   - Book `(pos, dir)` only when the message is a no-move refusal ("It's a wall.", "It's solid stone.", "You cannot pass", or an empty message with no creature).
   - Never book it when the message names a creature, contains "You miss"/"You hit"/"You kill", or when a letter glyph is drawn on the target square.
   - Move the `self.expect` assignment to after the final rewrite: let LegalStage record the *final* action into a shared `ctx.action`, and have Bump read that on the next `pre`.
   - Predicted: survival on QM/CB-Dark rises and bump-veto flights mid-fight disappear (14/15 QM vetoes are melee). A small win gain on CB-Dark is possible, though the memory warns CB-Dark is dice (~0.2 ceiling).
3. **Fix `action_io.snap` for compound bearings.**
   - If there is no whole-word hit, map 16-point compounds (`southsoutheast`, `northnortheast`, …) to the 8-point bearing they contain as a suffix (`southsoutheast` → `southeast`) or prefix.
   - Never pass an unlisted string through. Fall back to the route's first hop, or the least-visited safe move.
   - Predicted: 51 wall-bumping steps in QE r04 recovered, and illegal count → ~0.
4. **Delete OrbitStage or subordinate it to the route.** The perpendicular redirect demonstrably *seeds* the period-3/4 cycles (MW9 r00, MW15 r04, R3 r00, Boxoban-Hard r02). Once a route owns movement it is dead weight. If it is kept, detect cycles on the position history (period ≤6), not on action reversal.
5. **Keep HazardStage and ModalStage as they are.** They are the only stages with measured positive effect: lava deaths 4 → 0 and modal frames answered without the model. Note that a `pre()` short-circuit skips every `post()`, so HazardStage cannot veto a route hop. The route's BFS must exclude `}` and lethal-noun squares itself, or `orchestrator.py` must run the post chain on short-circuited actions too.

## 6. Exhausted — do NOT repeat
- Veto/rewrite layers around a model-chosen direction with no destination: this candidate, 0.175. The chain's scores separate by *whether a BFS route drives the step*: 0.35–0.375 with a route, ≤0.25 without.
- Period-2 reversal breaking by perpendicular redirect: it seeds longer cycles (§2A).
- Prompt "notes" (the "Checks on this frame" block): the model ignores them. QE r03 got the lava note and still proposed `east` 5 times, and bump notes did not stop repeat proposals.
- Prose priors: they remain empty and were not the lever here.
