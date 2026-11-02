# Diagnosis — minihack evolve r042 / cand_r042_0_spatial-grounding (pipeline + FieldStage)

**Headline:** primary (success) = **0.300** (12/40, se 0.072).
- **Wins:** MazeWalk-9x9 5/5, MazeWalk-15x15 5/5, Corridor-R3 2/5.
- **Zero on:** CorridorBattle-Dark 0/5, Boxoban-Medium 0/5, Boxoban-Hard 0/5, Quest-Easy 0/5, Quest-Medium 0/5.
- **Other metrics:**
  - progression 0.275.
  - survival 0.75, from 10 deaths: all 5 CB-Dark and all 5 Quest-Medium.
  - step_efficiency 0.239.
  - cost **60.2k tokens/episode**, versus 21.9k for r041. Almost all of it is Quest-Easy: about 420k tokens per episode across 4 episodes.
- **Chain context:** the chart+route plateau is 0.325–0.375 (r026/r036–r038/r041). This candidate reaches it on navigation but loses 2 R3 episodes and the single CB win.

**Binding constraint:** a **HARNESS-BUG in `tools/field.py`**.
- `SOLID = set("|- }0`")` makes an **opened door** (drawn `-`/`|`) into a wall.
- The refusal ledger also books "The door opens." as a refused edge.
- All 3 R3 losses come from this. The field loses its only route to a `>` it can see and hands 45–90 steps to the model, which never gets through.

Behind that bug, the remaining zeros are the same three TASK-GAPs as in r041's diagnosis:
- no Boxoban push planner;
- no horn/lava rung for Quest;
- the fight is taken on the room-mouth square.

## 1. What this harness does (mechanism)
`agent.py:PipelineHarness.act` builds a `StepContext` (`tools/context.py`). It then runs `pre()` on the stack in order `Legal > Modal > Hazard > Field`; the first stage that answers short-circuits the stack, and the outer `post()` hooks still run. If no stage answers, the model is called with `ctx.notes`, its reply is `snap`ped, and every `post()` runs.

- **ModalStage:** on `[yn` / `--More--` / a trailing `?`, it emits the first of `y`, `more`, `esc` that is legal. `y` is not in BALROG's vocabulary, so it always emits **`more`**.
- **HazardStage:** vetoes steps onto `}`, or onto squares whose prose names lava/water.
- **FieldStage.pre** (innermost) folds the frame into `Chart` (`tools/field.py`) and then decides in this order:
  1. On a remembered `>`, it emits `down`.
  2. On any adjacent alphabetic glyph, it **swings** (`field:swing`).
  3. It steps to the neighbour with the lowest potential: the stairs field if a `>` is drawn, otherwise the frontier field.
  4. If every neighbour is off-field, it returns None, and the model gets a "least trodden first" menu in the notes.
- **The Chart:**
  - `terrain` is glyphs by cell.
  - `open(c)` is `glyph not in SOLID`.
  - `blocked` is a set of `(square, dir)` edges. It is booked whenever the hero did not move and the message isn't a fight message.
  - `_frontier()` = open, **unvisited** cells that have an undrawn neighbour.
  - `flood()` is a visit-weighted reverse Dijkstra.

Deciding stage per task (all 4 000 steps):

| task | field:frontier | field:stairs | field:swing | model:* | modal->more |
|---|---|---|---|---|---|
| MW9 / MW15 | 21 / 140 | 11 / 15 | 0 | 0 | 0 |
| R3 | 177 | 22 | 0 | **~200** (R3/00, /01, /03 only) | 0 |
| CB-Dark | 99 | 0 | 47 | 0 | 0 |
| Boxoban M / H | 499 / 499 | 0 | 0 | 1 / 1 | 0 |
| Quest-Easy | 0 | 0 | 40 | ~393 | 64 |
| Quest-Medium | 65 | 0 | 31 | 16 | 0 |

The field owns MazeWalk (0 no-moves in 177 moves, 10/10 wins) and wins R3 whenever it never meets a door (R3/02, R3/04).

## 2. What actually happened (causal chains, with evidence)
In the CSVs, each row is (action, the observation AFTER that action), and `pos` is the cursor `(x,y)`.

### A. R3: an opened door becomes a wall, so the field goes flat and the model wanders (HARNESS-BUG, 3/3 R3 losses)
- **R3/00:**
  - The field owns steps 0–54. Step 54 gives "The door opens.", and from step 55 on, **every step is `model:*`** (45 of them).
  - The map at step 55 shows `@-.......` with `....>|` one row up: the opened door (`-`) is directly east of the hero, and `>` is **7 squares away and drawn**.
  - `-` ∈ `SOLID`, so the stairs flood can't pass through the door, and the unexplored cells beyond it are unreachable too. The field returns None, and the model walks around the corridor for 45 steps. It times out with `>` on screen.
- **R3/01:** the door opens at step 53. The room `|>....-###` has its door `-` at the east wall, 5 squares from `>`. The field keeps exploring the rest of the map until it runs dry at step 76, and the model then burns the last 24 steps.
- **R3/03:**
  - Step 9: the field chose `northwest` into the door square, which gives "The door opens." with no move.
  - `Chart.observe` then books `((20,4),'northwest')` into `blocked`, because "door opens" is not in `MELEE`.
  - Step 10: the model proposes `northwest`, and `FieldStage.post` rewrites it with `field-veto northwest->north`.
  - The door is now drawn `-` (SOLID), and the only other exit is a dead-end corridor (two `#` squares, both visited, so not frontier). The field is flat, and **90 model steps** follow, with 23 "It's a wall." and 39 "It's solid stone." bumps across R3.
- There are two stacked bugs here:
  1. the open-door glyph is treated as SOLID;
  2. "The door opens." is booked as a refusal.
- A third problem is that the field steps **diagonally** into doorways. "You can't move diagonally into an intact doorway." appears once, and the door at R3/03 was approached `northwest`.
- r041's chart fixed exactly this: doors are passable, with no diagonal moves through doors. In that lineage it gave "0/176 MazeWalk no-moves, 6/269 R3" and R3 4/5.

### B. Boxoban: the frontier field chases bars forever (HARNESS-BUG + TASK-GAP, 0/10)
- 998 of 1 000 steps are `field:frontier`, and **174 of 990 moves (17.6%) are "You cannot pass through the bars."**
- Boxoban's `#` is **bars**, but `#` ∉ SOLID. Each bar cell is "open", unvisited, and borders undrawn space, so **every bar is a frontier seed**.
- `blocked` convicts only the `(square,dir)` edge. The bar cell stays a seed, so the field walks round to another edge and bumps it again.
- The episodes visit only 3–34 unique squares, and no boulder is ever deliberately pushed. `` ` ``/`0` ∈ SOLID, so the field routes around boulders.
- Even with bars fixed, there is no push planner. With no frontier and no `>`, the field goes flat and the model gets the menu, and the model cannot solve Sokoban.

### C. Quest-Easy: sealed behind the `}` column; one episode is a peaceful-goblin prompt loop (TASK-GAP + HARNESS-BUG, 0/5)
- **QE/00–03:**
  - The start room `|.....}......` has the lava column at x=30.
  - The seeds beyond the lava are unreachable through `}` (SOLID), so the field is flat from about step 7, and the model decides about 93 of 100 steps.
  - In total there are 114 "It's a wall." bumps, and 25–26 unique squares per episode.
  - QE/00 step 40 says "horn adjacent west", but **no stage ever emits `pickup`/`kick`/`apply`**.
  - About 420k tokens per episode are spent for nothing: that is 83% of the whole bill.
- **QE/04: a 97-step 3-cycle.**
  - Step 3: `field:swing southeast` at a `G` produces "Really attack the goblin? [yn] (n)".
  - ModalStage emits `more` (Enter = the default `n`), which produces "Unknown command '^M'.".
  - `more` again, then `field:swing southeast` again.
  - This repeats **32 times** (steps 3–99).
  - The swing rung has no peaceful memory. r041 marked the last swing glyph peaceful on `[yn]`.

### D. Every death is taken on the room-mouth square (TASK-GAP: fight posture, 10/10 deaths)

| episode | death square | neighbours N..NW | how it got there |
|---|---|---|---|
| CB/00–03 | (35,12) | `.%(r.\|.\|`, `.r.r.\|.\|`, `.rrr.\|.\|`, `.r[r.\|.\|` | `field:frontier east`, first swing at step 19–20 in all 5 |
| CB/04 | (34,12) | `\|r%r\| # ` | same |
| QM/00–03 | (36,12) | `.rrr....` ×4 | the field goes flat at (33,12); then **`model:east` ×3** walks into the lit rat room |
| QM/04 | (37,12) | `.rrr....` | same |

- **CB-Dark:**
  - The frontier descent is deterministic. It spends steps 0–10 exploring the dark start room, then marches east down the corridor, arriving at the room mouth at step 19 in **every** episode.
  - In CB/00–02 it takes one more `field:frontier east` hop **after** the first kill, off the corridor square and into (35,12), which has 3 open bearings toward the pack.
  - In CB/03 the hero was already standing on (35,12) when the first rat reached it (step 19).
  - Its HP went 16→13→10→7→4→2→dead within 6 swings.
- **QM:**
  - At step 12 the map shows 7 `r` in the lit room (`|@.....rrr.r...}.>.|`).
  - The level is fully drawn and `>` lies beyond `}`, so both fields are flat.
  - The model is asked, says `east`, and walks the hero into the pack.
  - The stack has no "don't advance toward drawn letters" guard for the flat-field case.

## 3. Failure breakdown (28 losses)
| category | eps | class | fixable? |
|---|---|---|---|
| R3: open door `-`/`\|` ∈ SOLID and "door opens" booked as a refusal → flat field → model timeout | 3 | **HARNESS-BUG** | yes, cheap (change 1) |
| Boxoban: bars are frontier seeds (17.6% bumps); no push planner | 10 | HARNESS-BUG + TASK-GAP | the bar fix is cheap; wins need a solver (change 4) |
| Quest-Easy: sealed behind `}`; no pickup/kick/apply rung | 4 | TASK-GAP | yes (change 5) |
| Quest-Easy/04: peaceful-goblin `[yn]` → `more` → re-swing, 32 cycles | 1 | HARNESS-BUG (cost only; the episode is unwinnable without the horn) | yes (change 2) |
| Quest-Medium: flat field → model walks into the rat room → death | 5 | TASK-GAP (posture) + missing flat-field guard | partly (change 3) |
| CB-Dark: fight taken on the room mouth | 5 | TASK-GAP (posture), dice-heavy | partly (change 3); price it on ≥200 eps |

## 4. Binding constraint
**HARNESS-BUG first:** the chart's passability model is glyph-scoped in the wrong direction.
- `-`/`|` are always walls, even when the square is an open door.
- `#` is always open, even when it is bars.
- Refusals convict edges instead of squares, and a "door opens" event counts as a refusal.

Fixing this recovers the r041-level navigation: R3 moves toward 4–5/5, and the Boxoban bump storm stops.

**Beyond the bug: TASK-GAP.** The frozen model decides only on flat-field residue (Quest, and R3 after the door bug), and it never contributes a win. The next points need goal mechanisms (push planner, horn/lava rung, a hold posture), not better prompting. The model is **not** the ceiling: on the tasks that score, it is not in the loop at all.

## 5. Concrete changes to try (ranked by leverage)

1. **Doors in `tools/field.py`: square-scoped doors, and "door opens" is not a refusal.** Predicted R3 2/5 → 4–5/5 (+0.05…+0.075 primary), zero risk to MW/CB.
   - Add `self.doors = set()` to `Chart`, and make three changes in `observe()`:
     - When the pending move produced "The door opens." (or the prose says "open door"/"doorway" adjacent on that bearing), add `ctx.target(pending[1])` to `doors` and **do not** book `blocked`.
     - Treat "This door is locked." as a refusal that can be kicked (chain memory: `kick` is a two-step verb, +7/−0 on R3). At minimum, don't let it poison the field.
     - A `+` glyph goes into `doors` as well.
   - `open(c)`: return True if `c in self.doors`, whatever the glyph is (an open door is drawn `-`/`|`).
   - In `flood()` and in `FieldStage.pre`'s `open_dirs`, **forbid diagonal edges into or out of a door cell**. Check the `(dx,dy)` with both non-zero against `prev in doors or cur in doors`.
   - R3/03 step 9 would then open the door orthogonally and walk through it, and R3/00 would reach `>` at 7 hops.

2. **Refusal messages convict the SQUARE; peaceful swings are remembered.** This is `Chart.observe` plus `FieldStage.pre`.
   - "You cannot pass through the bars." / "It's a wall." / "It's solid stone." → add the target cell to `self.solid`, and make `open()` return False for it.
     - Keep "boulder … in vain" edge-scoped.
     - Do **not** make `#` SOLID by glyph: `#` is corridor on R3 and MW. Chain memory: "a bump convicts the square, never the glyph".
     - Predicted: Boxoban bumps 174 → ≤~20. R3 "solid stone" bumps (39) drop too. No primary gain on Boxoban by itself.
   - On a `[yn]` prompt matching "Really attack", record the swing target (square + glyph) as peaceful, and exclude it from the swing rung until the glyph moves.
     - Also stop ModalStage from answering `more` twice in a row. The second `more` is what produces "Unknown command '^M'".
     - Predicted: QE/04's 97 wasted steps become exploration; cost and efficiency improve, primary does not.

3. **Flat-field guard plus a corridor-hold posture (`FieldStage.pre`, before the model call).** This targets the 10/10 deaths; survival should rise.
   - **Flat field and a letter drawn within Chebyshev 4:** emit `wait` (or `search`) instead of calling the model.
     - That removes the QM `model:east ×3` walk into the pack.
     - Chain memory: "stall-contact gate is the survival lever" (+0.125 survival; primary and efficiency identical).
   - **Hold the post:** if the hero's square has ≤2 open neighbours, a letter is drawn within radius 3, and the chosen field hop enters a cell with ≥3 open neighbours, emit `wait` instead of the hop.
     - This addresses CB/00–02, where one more `field:frontier east` after the first kill stepped onto (35,12).
     - Release the hold only after no letter has been within radius 3 for 2 steps, and don't count the walk to the post.
   - **Do not** add HP-based retreat (the dose-response refutes it).
   - CB-Dark is dice at n=5. Price it on ≥200 stub-rig episodes before trusting its sign. QM is **not** dice: 5/5 deaths share one square and one trigger.

4. **Boxoban push planner: a `SokobanStage` before FieldStage, triggered by `{` and `` ` ``/`0` drawn with no `>`.** Up to +10 episodes (+0.25).
   - Decode the `map:` block exactly (chain memory: 2696/2696 exact decodes).
   - Run A* in push space:
     - state = (canonical hero region, boulder set);
     - heuristic = sum of min Manhattan distances to targets;
     - prune dead corners using walls only.
   - Emit a committed cardinal tape; Boxoban has no diagonals. Re-plan on any unexpected no-move.
   - Chain memory: 19/20 on the stub rig and 10/10 live; it wins iff the optimal plan is ≤99 steps. Mind the phantom boulder or masked target under `@` at step 0.
   - Cost drops to 0 tokens on Boxoban.

5. **Quest horn/lava rung (flat field and `}` drawn, no reachable frontier).** Up to +5 QE; QM once (3) keeps the hero alive.
   - The recipe from chain memory:
     1. walk to the `(` pile or the "horn" noun;
     2. `kick` it **once** to split it, and never toward lava;
     3. `pickup`;
     4. face the lava and `apply`/`zap`;
     5. answer the direction prompt with the lava bearing.
   - ModalStage needs a direction-prompt branch ("In what direction?" → the committed bearing) rather than `more`.
   - This also removes the ~420k-token model orbits on QE: cost per episode falls from 60k toward about 10k.

6. **Cost: when the field is flat and none of (3)–(5) fires, stop calling the model.**
   - Take `menu[0]` (least trodden, excluding blocked and solid cells), or `search` when the menu is empty.
   - The model's flat-field decisions produced 0 wins and all of the 114 QE wall bumps.
   - Predicted: primary identical, cost down by more than 80%.

## 6. Exhausted — do NOT repeat
- **Re-plumbing where the decision lives** (model vs. chart vs. learner vs. pipeline order). r026/r027/r036–r038/r041/r042 all land at 0.30–0.375, and navigation is solved once doors work.
- **Prose / AGENT.md / notes wording.** The model decides no winning step anywhere.
- **Glyph-level `#` = wall.** It breaks the R3/MW corridors. Convict squares from the refusal message instead.
- **The OrbitStage / BumpStage perpendicular redirect.** It seeded the 3–4-cycles (the prior diagnosis in this lineage), and the visit-weighted field supersedes it.
- **HP-triggered retreat or disengagement in melee.** Refuted (9→13→17→42 deaths).
- **Pricing CB-Dark on 5 episodes.**
