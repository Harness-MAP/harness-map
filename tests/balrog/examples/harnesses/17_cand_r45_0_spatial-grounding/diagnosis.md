# Diagnosis — minihack evolve r045 / cand_r045_0_spatial-grounding (middleware stack over a map-memory Chart)

**Headline:** primary = **0.300** (12/40, se 0.072). Survival is 0.85, step_efficiency 0.242, cost **320.7k tokens/ep**.

| Task | This candidate | Chain best (r036/r041/r044) |
|---|---|---|
| MazeWalk-9x9 | 5/5 | 5/5 |
| MazeWalk-15x15 | 4/5 | 5/5 |
| Corridor-R3 | 3/5 | 4/5 |
| CorridorBattle-Dark | 0/5 (**5 deaths**) | 1/5 |
| Boxoban-M/H | 0/10 | 0/10 |
| Quest-E/M | 0/10 | 0/10 |

The cost is about 15x r041's 21.9k tokens/ep.

**Binding constraint:** the model copies the BFS ROUTE hop on **100% of the steps that have one** (0 disagreements outside swings, across all 40 episodes). That makes `Chart.plan()` the whole policy. Two Chart bugs then decide the losses:
- **HARNESS-BUG, R3.** An **open door is drawn `-`/`|`**, and `passable()` treats that glyph as wall. The route dies behind every opened door and never reaches a charted `>`.
- **HARNESS-BUG, CB-Dark.** The route walks **away from adjacent, drawn rats**. SwingStage only *protects* a swing the model chose; it never *forces* one.

Everything else is the known TASK-GAP residue (Boxoban and Quest).

## 1. What this harness does (mechanism)
- `agent.py:PipelineHarness.act` runs this sequence each step:
  1. `Chart.observe(text)` updates the per-level glyphs, visits, prose-named wall squares, and terrain-refused `(square, dir)` edges.
  2. `Chart.plan(legal)` runs a BFS to a charted `>`, or else to the nearest frontier (never stood on AND borders an uncharted square).
  3. The stage `pre` hooks run: Veto, Swing, Route.
  4. The model is called **once, every step**.
  5. `extract_action` projects the reply onto the legal set.
  6. The `post` hooks run: Route, Swing, Veto.
- `RouteStage.pre` writes `ROUTE: nearest … first move toward it: <hop>` into the prompt. `RouteStage.post` overrides a move onto a visited square. `SwingStage.pre` adds `CREATURE adjacent: …`. `SwingStage.post` only *logs* a swing.
- Prose (`AGENT.md`) says: follow ROUTE unless a creature is adjacent, then keep attacking.

**What actually drives behaviour:** trail census over all 40 CSVs:
- `route:` overrides: **0**. `veto:` rewrites: **0**. `swing:` tags: 71 (all on CB-Dark and Quest).
- Wherever `route≠None`, `model == hop` except on 12 swing steps.

So the post-hoc override machinery is inert. The prompted ROUTE line is the lever, and the model is a transcription layer for it.

## 2. What actually happened (causal chains)

### A. Corridor-R3: an open door is a wall to the Chart (HARNESS-BUG). This costs both losses.
- **R3/03.** Steps 0–8 follow the route around the start room. At step 8 the agent is at pos (20,3) and moves `northwest` at the closed door `+`, and at step 9 the message is `The door opens.` The glyph becomes `-`: the map row goes from `+....|` to `-....|`.
  - `chart.passable('-')` is False, so the only exit from the room is no longer passable, and the east corridor stub `#` at (25,5)/(25,6) has been stood on.
  - From step 10 on, `route=None` for **90/100 steps**. With no ROUTE line, the model orbits the room (period-2 revisits 27, 15 unique squares) and times out.
- **R3/01.** The `>` is charted from about step 54, at map row `|>....-#######`. Its room's east door is **already open**, drawn `-` with `.` on the west and `#` on the east.
  - The BFS can never enter the room, so `plan()` keeps returning frontier hops elsewhere. At step 77, after `It's a wall.`, it returns None.
  - The stairs are visible and 7 squares away, but the route never targets them.
- **R3/00.** The door opens at step 62, and route=None from step 64. The agent won anyway at step 72, because the model walked in unaided (8 no-hop steps).

The door-opening move itself works: `+` is passable and a bump opens it. What breaks is the **drawn glyph after opening**. This is the square-vs-glyph seam already known from the chain ("an OPEN DOOR is drawn `-`/`|`").

### B. CorridorBattle-Dark: the route walks away from adjacent rats (HARNESS-BUG). 5/5 deaths.
- The rats **are drawn** on the map as `r`. For example, CB/02 step 16 (x=39,y=13) shows `....rrr` / `..@r`, and the prose says `giant rat adjacent north, northeast, east, and northwest`.
- `plan()` excludes creature squares (`passable` rejects letters), so the hop always points *away*: `southeast`, then `west`, then `west`, `west`, `west`. The model copies it.
- HP goes 16→12→10→7→3→dead in 8 steps with **zero swings** (CB/02 has 0 `swing:` tags).
- Across the 5 CB episodes, **57 frames have a rat adjacent and 47 of them (82%) emitted a non-attack move**. A retreating hero takes a free bite from every adjacent rat each turn.
- The `CREATURE adjacent … keep attacking` note is in the same prompt, but the ROUTE line beats it every time.
- CB/04 shows the contrast: when the model did swing (8 swings), it killed 4 rats, but it walked away in between (steps 16, 18–23, 25–26) and died at step 33.

### C. MazeWalk-15x15/03: a dice loss with a correct mechanism
- The model follows the route 100/100 steps, and there are 62 unique squares. The nearest-first frontier produces 9–14-step backtracks (`west@9`, `north@14`), and the 100-step cap arrives before the `>` is charted.
- Nearest-first is already the priced optimum in this chain (extremity-first measured worse once model-free). Do not chase this episode.

### D. Quest: no route at all, plus a `[yn]` sink
- **Quest-Easy:** `route=None` on **500/500 steps**. The start room is sealed by a lava column `}`, and the BFS (correctly) will not cross lava. The model free-walks.
  - The `horn` (`(`) is **adjacent west at step 0** in QE/01 and QE/04, yet nothing ever picks it up.
- **QE/04 (HARNESS-BUG):** `Really attack the goblin? [yn] (n)`, with the goblin peaceful and adjacent south. The model emits `south` **97 times**, re-raising the prompt every step (94 no-moves).
  - `Chart.observe` never books the edge, because `COMBAT` matches `Really attack` and the target is a creature. `Veto` never fires either.
- **Quest-Medium (4 of 5 identical):** the agent follows the route through the corridor. At about step 19 the `>` is charted beyond the lava, and route=None for 81 steps. The agent walks up to the lava edge (`lava adjacent northeast, east, southeast` at step 60) and orbits.
  - QM/00 died at step 41 in the rat room (11 swings).

### E. Boxoban: no frontier and no stairs, so the model orbits
- Route=None on 54–82% of steps. The room is premapped, so every frontier is exhausted in the first ~20–45 steps, and there is no `>`.
- The model then orbits: period-2 revisits 17–44 per episode, and 13–39 unique squares.
- 0/10, as in every lineage without a Sokoban solver.

## 3. Failure breakdown (40 eps; 28 losses)
| Failure | Eps | Class |
|---|---|---|
| Boxoban: no push planner, and the model orbits after the frontier empties | 10 | TASK-GAP |
| Quest-E/M: stairs beyond lava; no horn pickup/apply rung (QE/04: also 94 steps lost to the peaceful `[yn]` loop) | 10 | TASK-GAP (+1 HARNESS-BUG) |
| CB-Dark: the route retreats from adjacent rats (82% of rat-adjacent frames) | 5 | **HARNESS-BUG** |
| R3: open door `-`/`|` is impassable in the Chart | 2 | **HARNESS-BUG** |
| MW15: 100-step cap with nearest-first backtracking | 1 | dice |

## 4. Binding constraint
**HARNESS-BUG in `tools/chart.py` and `tools/stages.py`, on the navigation and melee tasks.**

The VLM is not a constraint here: it obeys the ROUTE line perfectly. Every point this candidate lost against the chain-best 0.375 comes from where the Chart sends it:
- R3: 2 points, from the door glyph.
- CB-Dark: 1 point, from the retreat.
- MW15: 1 point, from dice.

Recovering those restores about 0.35–0.375. Only the TASK-GAPs (Boxoban planner, Quest horn/lava) move the ceiling.

## 5. Concrete changes (ranked by leverage)

1. **`tools/chart.py`: open doors are passable, orthogonal-only squares.** *(R3 +1–2 eps; should also help Quest-Medium's corridor rooms.)*
   - Keep a `self.doors` set of squares. Add a square when any of these is true:
     - (a) the square was `+` and is now `-`/`|` (door opened), or the message says `The door opens.` (add `self.last` target);
     - (b) the prose says `open door adjacent <dirs>`, `broken door adjacent <dirs>` or `doorway adjacent <dirs>` (add a `DOOR_NOUN` regex beside `WALL_NOUN` and convict those squares);
     - (c) a **structural test** on every charted `-`: a real horizontal wall never has passable ground (`.`, `#`, `<`, `>`) on *both* its west and east sides. Likewise a `|` with passable ground on both north and south is a door. This catches R3/01's already-open `|>....-###`.
   - In `_open(sq)`, return True for `sq in self.doors`.
   - In `plan()`, forbid diagonal edges into or out of a door square (`dx and dy` and either endpoint in doors), since NetHack refuses them.
   - Keep the conviction square-scoped, never glyph-scoped: `-` everywhere else is still wall.

2. **`tools/stages.py:SwingStage`: make the attack code-owned, not merely protected.** *(CB-Dark back to at least the 1/5 chain level; it also removes ~80% of the free bites.)*
   - In `pre`: when `foes` is non-empty, set `ctx.route = (foes[0], "creature", 1)` (or a fixed total order over foes), set `ctx.fallback = foes[0]`, and **do not emit the ROUTE note**. RouteStage must run after Swing's pre, or check `ctx.foes`.
   - In `post`: if foes exist and the action is a move to a non-foe square, rewrite it to the foe direction.
   - Do **not** add HP retreat or disengage: the dose-response in the chain refutes it.
   - Exclude peaceful creatures (see 3), and treat a `%` corpse as not a foe.

3. **`tools/chart.py` / `stages.py`: the peaceful `[yn]` prompt.** *(Quest-Easy survival and cost only; no wins.)*
   - On `Really attack .*\? \[yn\]`, add the last direction to a per-square `peaceful` set, and veto that direction while the creature glyph stays there.
   - If the prompt line is still open, answer it with a vocabulary action that dismisses it. Use `more` once; `esc`/`y` are not in the vocabulary.
   - Also: with `route=None` and a foe tagged peaceful, the fallback must pick another open direction.

4. **New `tools/quest.py` rung: horn → lava.** *(TASK-GAP, up to +10 eps.)* The chain has already priced the Quest recipe on seeds: pickup → apply → bearing solidifies the lava, and a kick splits the spawn pile. The rung:
   - fires only when the chart shows `}` between the hero and a charted `>`, or `route=None` with lava in the prose;
   - (a) routes to the nearest `(` square (the BFS target is the item glyph) and emits `pickup`;
   - (b) emits `apply`, and answers the item-letter prompt and `In what direction?` with the bearing toward the nearest `}` on the stairs side;
   - (c) re-plans once the lava square turns to floor.

   The model is not needed for any of this.

5. **New `tools/sokoban.py` push-space A\* for Boxoban.** *(TASK-GAP, the largest pool: 10 eps.)*
   - Boxoban decodes to an exact Sokoban from the `map:` block: `#` is bars, `` ` ``/`0` are boulders, `{` goals, 4 cardinals.
   - A push-space A\* over the decoded grid solved 19/20 on the stub rig.
   - Walls-only deadlock pruning. Treat the hero's masked start square as floor.

6. **Cost: skip the model call when the route is decisive.**
   - Measured agreement is 100%, so whenever `route` has a hop and `foes` is empty, return the hop without calling the model. r036/r040/r041 did this with 0-token descents.
   - Expected effect: cost from ~320k to <60k tokens/ep, with primary identical by construction.
   - Also expected: the chart then owns Boxoban/Quest only via 4–5, so the model is consulted only on route=None residue.

7. **Remove or repurpose `RouteStage.post`.** It fired 0 times in 40 episodes. Either drop it, or turn it into the "route decisive, skip the model" branch from 6.

## 6. Exhausted — do NOT repeat
- **The code-owned post-hoc route override** (this candidate). It is inert, because the model already copies ROUTE.
- **Re-plumbing chart / stage / model order** (r026/r036–r044): plateau at 0.325–0.375. Navigation is saturated apart from the two Chart bugs above.
- **Prompt, AGENT.md or skills wording.** The model is a transcription layer for the ROUTE line: the `keep attacking` prose lost to ROUTE on 47/57 rat frames.
- **Glyph-level `#` = wall or `-` = wall in either direction.** Convict squares, not glyphs.
- **HP-triggered retreat / disengage** (dose-response 9→13→17→42 deaths). Change 2 is the opposite: attack.
- **Frontier re-ranking** (extremity, gain rate, nub demotion): nearest-first is the optimum. MW15/03 is dice.
- **Pricing CB-Dark on 5 episodes.** Judge change 2 by the rat-adjacent non-attack rate (currently 82%), not by the win count.
