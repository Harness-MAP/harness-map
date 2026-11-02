# Diagnosis — minihack evolve r043 / cand_r043_0_spatial-grounding (rung ladder on r001)

**Headline:** primary (success) = **0.325** (13/40, se 0.074).
- **Per task:**
  - MazeWalk-9x9: 5/5
  - MazeWalk-15x15: 5/5
  - Corridor-R3: **3/5**
  - CB-Dark: 0/5
  - Boxoban-Medium and Boxoban-Hard: 0/5 each
  - Quest-Easy: 0/5
  - Quest-Medium: 0/5
- **Other metrics:** progression 0.300. Survival 0.775 (9 deaths: all 5 CB-Dark and 4 of 5 Quest-Medium). step_efficiency 0.248.
- **Cost:** 176.8k tokens per episode. Almost all of it is Boxoban (~420k per episode) and Quest-Easy (~450k per episode).
- **Versus the parent r001** (0.350; R3 4/5; survival 0.85; 301k tokens): R3 is down one episode and survival is down. Cost is −41%.
- **Chain plateau:** r026/r036–r038/r041 sit at 0.325–0.375.

**Binding constraint:** there are **two deterministic HARNESS-BUGs on R3**:
1. An opened door (drawn `-`/`|`) is treated as a wall by the BFS.
2. The search rung's dead-end test makes a period-2 orbit.

These two bugs account for both R3 losses, which is the only headroom left on the navigation tasks. Behind them are the same three TASK-GAPs as in r041's diagnosis:
- no Boxoban push planner;
- no horn/lava rung on Quest;
- the fight is taken on the room-mouth square.

## 1. What this harness does (mechanism)
`agent.py:GroundedHarness.act` runs these steps in order:
1. `perception.parse`, then `SpatialMemory.observe`.
2. `action_io.modal_response`: `--More--`, `[yn`, or "what do you want" → `more`/`esc`. "staircase down here" → `down`.
3. **Route rung:** `grounding.ground` → `plan_route` runs a BFS over `seen`. The target is a drawn `>`, or else the nearest *unstood* frontier. Its first hop is emitted with no model call.
4. **Search rung:** `grounding.search_plan`. It fires only when there is no route, no `>` anywhere on the map, and `search` is legal. It walks to the nearest dead end and searches there, with 8 searches per cell.
5. **Otherwise:** one `client.generate` with the rendered grounding block, then `arbitrate` (legal snap, veto override).

There is **no melee rung**. A creature letter is not in `BLOCK_GLYPHS`, so `_passable` lets the BFS route straight through monsters. Fighting only happens when a rat happens to sit on the route's first hop.

Rung census from the CSV `Reasoning` column (2 338 steps):

| task | rung:route | rung:to-dead-end | rung:search | model | modal |
|---|---|---|---|---|---|
| MW9 / MW15 | 32 / 155 | 0 | 0 | 0 | 0 |
| R3 | 205 | **90** | **0** | 26 | 0 |
| CB-Dark | 133 | 0 | 0 | 0 | 0 |
| Boxoban M / H | 160 / 136 | 0 | 0 | 340 / 364 | 0 |
| Quest-Easy | 0 | 0 | 0 | 446 | 54 |
| Quest-Medium | 65 | 0 | 0 | 132 | 0 |

- The model runs **only** on the 20 episodes that score 0: Boxoban, Quest-Easy and Quest-Medium. Every model token was spent on tasks with 0 wins.
- `rung:search` never fired once. The 90 `to-dead-end` steps are all one orbit (§2B).

## 2. What actually happened (causal chains)
A CSV row is (action, the observation AFTER that action). `pos` is `(x, cursor.y)`, and map row = cursor.y.

**A. R3/01: the opened door is a wall, so the drawn `>` is unreachable (HARNESS-BUG).**
- **Step 53:** at (20,3) the agent moves `west` → "The door opens.". The door at (19,3) is now drawn `-` (row 3: `|>....-#######`). `>` sits at (14,3), 6 squares west through that door.
- `_passable('-')` is False because `-` is in `WALL_GLYPHS`, so the BFS cannot enter the door. The route immediately turns back **east** (step 54, "nearest unexplored edge, 7") and explores the far corridor.
- **Steps 76–98:** it reaches the dead-end corridor (20..23,7). The search rung is **disabled** because `>` is drawn (`search_plan` returns None whenever any `>` is on the map). So the model is called, and it bumps each of the 8 bearings in turn: "It's solid stone." ×18. It times out.
- The `ground()` menu does mark the door move allowed, because the prose noun "open door" sets `opening`. Only `plan_route` treats the door as a wall. This is exactly chain memory's "an OPEN DOOR is drawn `-`/`|`".

**B. R3/03: the same door bug, followed by a period-2 orbit in `search_plan` (HARNESS-BUG).**
- **Step 9:** `west` at (20,3) → "The door opens." (the prose says "vertical open door adjacent west"). The door is again BFS-impassable, and the only other frontier is the dark corridor (25,6)–(25,7), which is already stood on. So `plan_route` returns None.
- **Steps 10–99:** `rung:to-dead-end` alternates `north`/`south` between (25,6) and (25,7) **88 times**, and `search` is never emitted.
- **Cause:** `search_plan` builds `seen = {..., pos: "."}`. The current square therefore never matches the corridor dead-end test (`seen.get(cell) == "#" and undrawn >= 3`). With 2 passable neighbours it also fails `walk <= 1`.
- The *other* corridor square does pass the test, so the agent walks there. Next step the roles swap. The cell it stands on can never be the dead end, so `c == pos → "search"` is unreachable.
- Had the door been passable, the route would have gone west through (19,3), and the search rung would never have fired.

**C. CB-Dark 5/5 and Quest-Medium 4/5 deaths: every death is a fight on a room-mouth square the route stepped onto (TASK-GAP: posture).**
- In **CB-Dark** (every step is `rung:route`), the corridor is (27..34,12). In all five episodes the first rat is killed from the corridor square (34,12). The route then steps `east` to (35,12) and/or `north` to (35,11), where three or four rats adjoin.
  - CB/04 step 23 shows `rr` / `.@r` / `.r` with "giant rat adjacent north, northeast, east, and southeast".
  - HP falls 16→7→5→1 within 3–6 steps.
  - The route's hop is a frontier heuristic, not a target choice. In CB/04 steps 23–24 it emits `north` while bites arrive from E/SE.
- In **Quest-Medium** (the model decides steps 13+ and copies `rec` 100% of the time), the corridor square is (34,13).
  - **QM/02 is the only survivor.** It held (34,13) for **8 consecutive `east` swings** at steps 17–24 and made 5 kills.
  - The 4 deaths (QM/00, 01, 03, 04) all stepped to (35,13) → (35,12)/(35,11) after 0–2 kills and died there within 4–6 steps.
- This is the **same room-mouth pattern** as r041 §2E. One corridor square has exactly one open bearing toward the pack, and every death is on a square with ≥3.

**D. Quest-Easy/04: the `[yn]` peaceful-blocker loop was NOT fixed (HARNESS-BUG in `perception` + `spatial_memory`).**
- **Steps 21–59:** there are 13 identical triples: `north` (model) → "Really attack the goblin? [yn]" → `more` → "Unknown command '^M'." That is 39 of 100 steps, plus 54 `modal` rows in the episode.
- NOTES' temporary conviction never fires. On the `[yn]` frame the cursor sits on the **prompt line**, so `pos` is parsed as **(35,0)** (visible in every `[yn]` row). `observe()` then sees `last_pos != pos`, books nothing, and sets `last_pos = (35,0)`.
- The next two frames follow a `more`, and `note_action("more")` sets `last_move = None`. So `(29,12), north` is never convicted, and the model re-emits `north` straight back into the goblin.
- Chain memory already records this: "an open modal prompt is `cursor.y==0`".
- Even when fixed, it frees about 40 steps on a task that is sealed behind lava (§2E). It is a cost and cleanliness fix, not a primary fix.

**E. Quest-Easy 0/5: sealed behind the `}` column (TASK-GAP).**
- In QE/00–03 the agent visits 26 unique squares in 100 steps, all in the start room (x 25–29). The lava column is at x=30.
- The prose says "horn near westnorthwest". The `(` pile is drawn at (26,10). **No rung emits `pickup`/`kick`/`apply`.**

**F. Boxoban 0/10: no push planner; the model dithers (TASK-GAP).**
- The route rung finishes the few frontiers in 30–40 steps. The model then owns 68–73% of steps.
- Period-2 reversals run **24–58 per 100 steps**, with only 5–21 unique squares per episode.
- 0–5 boulder pushes per episode ("With great effort you move the boulder."), none planned.
- Bar bumps are now rare (0–4 no-moves per episode). The leftover problem is the lack of a plan, not bumping.

## 3. Failure breakdown (27 losses)
| category | eps | class | fixable in harness |
|---|---|---|---|
| Boxoban: no push planner, model dithers | 10 | TASK-GAP | yes: exact decode + push-space A* (chain memory: 19/20 stub, 10/10 live) |
| Quest-Easy: sealed start room, horn never picked up | 5 | TASK-GAP | yes: kick-once → pickup → apply toward lava (chain memory: 5/5 QM, 2/5 QE) |
| CB-Dark: death on the room-mouth square | 5 | TASK-GAP (posture), dice-heavy | partly; price at ≥200 eps |
| Quest-Medium: death on the room-mouth square | 4 | TASK-GAP (posture) | partly; this must hold before the lava lever can pay on QM |
| Quest-Medium/02: survived, never crossed | 1 | TASK-GAP | the lava lever |
| R3/01 + R3/03: opened door is a BFS wall (+ dead-end orbit on /03) | 2 | **HARNESS-BUG** | **yes, deterministic, zero risk elsewhere** |

## 4. Binding constraint
- **For the next +0.05: HARNESS-BUG.** R3 3/5 → 5/5 needs only the door fix. The orbit fix is insurance for the hidden-corridor levels.
- **For anything beyond the 0.325–0.375 plateau: TASK-GAP.** Boxoban (10 eps) and Quest (10 eps) need new goal-directed rungs.
- **Not VLM-CEILING.** The model's reply equals `rec` on essentially every model row; it is inert, so re-wording the prompt cannot move primary. The model's only measurable effect is cost.

## 5. Concrete changes (ranked by leverage)

1. **Doors are passable, orthogonal-only cells (`tools/spatial_memory.py`, `tools/grounding.py`). Predicted R3 3/5 → 5/5, +0.05 primary; MazeWalk and CB unchanged.**
   - In `SpatialMemory`, add `self.doors = set()`. In `observe()`:
     - For each `d, nouns in percept.adjacent.items()`, if any noun contains `open door`, `doorway` or `broken door`, add `step(pos, d)` to `doors`.
     - On "The door opens." / "door crashes open", add `step(last_pos, last_move)`. `pos` is unchanged on that frame, because the opening turn does not move the hero.
   - Pass `memory.doors` into `plan_route`/`_bfs`/`_neighbours`. In `_passable(g, cell)`, return True for any `cell in doors` regardless of glyph.
   - In `_neighbours`, forbid a diagonal step when `n in doors or cell in doors`. This replaces the `'+'`-only check, which never fires on `-`/`|`/`.` doors.
   - Keep the edge refusal: "This door is locked." stays refused, and a locked door goes to `kick` (chain memory: +15/−3).
   - **Why it works:** in R3/01 the route from (20,3) to `>` (14,3) is 6 hops through (19,3). In R3/03 the rest of the level lies beyond (19,3).

2. **Fix the dead-end 2-cycle in `grounding.search_plan`. Insurance for R3 hidden corridors; turns 88 wasted steps into searches.**
   - Evaluate `_dead_end` on the square's **remembered** glyph: `memory.seen.get(c)`. `observe()` already skips `@`, so `seen[(25,7)]` is `'#'`. Do not use the `pos: "."` override. Keep the `"."` override only for BFS passability.
   - Additionally, keep a committed dead-end target in memory: walk to it until you arrive, then search. Never re-elect a different cell mid-walk.
   - Also allow the search rung when a `>` is drawn but **unreachable** (R3/01 steps 76–98). Change the gate from `any('>')` to "no BFS route to any `>`".

3. **The prompt frame must not move the hero (`tools/perception.py:parse`, `spatial_memory.observe`). Fixes the QE/04 `[yn]` loop (39 steps); cost and cleanliness only.**
   - If the cursor `y == 0`, the cursor is on the prompt line. Set `p.pos = None`, and make `observe()` return early **without** touching `last_pos`/`last_move`.
   - On the frame whose message contains `really attack`, convict `(last_pos, last_move)` in `temp` **before** the modal answer runs. `last_move` is still the compass move at that point.
   - Make `note_action` of `more`/`esc` leave `last_move` untouched while a prompt is open.

4. **Melee posture rung before the route rung (`agent.py`, new `tools/melee.py`). Price on ≥200 CB-Dark + QM episodes; the target key is survival first.**
   - Swing: if a letter glyph (`isalpha`, not `@`) is adjacent in `percept.grid`, emit that bearing. When several are adjacent, prefer the letter on the bearing the hero entered from, i.e. along the corridor axis. Never step onto a non-letter square while any letter is adjacent.
   - Hold: if a letter is drawn within Chebyshev distance 3 and the current square has ≤2 passable neighbours (a corridor square), emit `search` (spend the turn, don't move) instead of advancing. The route's next hop would open ≥3 bearings.
   - **Evidence:** QM/02 is the sole survivor, and it did exactly this: 8 `east` swings from (34,13) with HP 16→11. All 9 deaths stepped to (35,x).
   - **Chain memory caveats:**
     - CB-Dark is dice at n≤200.
     - HP-triggered retreat is refuted, so do NOT add one.
     - The fight post must be a square you stepped TO, and it must be committed.
     - r036/r041 already had a swing rung and got CB 1/5.

5. **Boxoban `sokoban_plan()` rung (TASK-GAP, the largest pot: 10 eps, up to about +0.25).**
   - **Trigger:** `` ` `` and `{` are drawn and there is no `>`.
   - **Decode:** `#` is bars/wall, `` ` `` a boulder, `{` a target. The square under the start `@` may be a masked target.
   - **Search:** push-space A* with cardinal moves only (no diagonals in Boxoban). Prune corner squares using walls only.
   - **Execution:** emit from a committed tape, and re-plan on any no-move.
   - It also removes about 700 model calls, roughly 60% of the token bill.

6. **Quest horn rung (TASK-GAP, up to 5 QE + QM survivors).**
   - **Trigger:** no route, `}` drawn, and "horn" in the prose.
   - **Sequence:**
     - walk to the pile;
     - `kick` it **once**, never toward lava;
     - `pickup`;
     - `apply`, and answer "In what direction?" with the lava bearing.
   - `modal_response` currently answers every direction prompt with `esc`/`more`. It needs a committed-bearing branch.

7. **Cost (secondary key): replace the model fallback with a model-free residue owner.** Use least-visited, with ties broken by level coordinates. The model decides no winning step anywhere; r041 ran at 21.9k tokens per episode with identical primary. This drops cost from about 177k to under 30k tokens per episode.

## 6. Exhausted — do NOT repeat
- **Re-plumbing where the decision lives** (model vs. chart vs. rung order). r026/r027/r036–r038/r041/r042/r043 all land at 0.30–0.375. This candidate's rung ladder is exactly that axis, and it moved cost, not primary.
- **Prompt, AGENT.md or skills wording.** The model copies `rec`.
- **Glyph-level `#` = wall.** It breaks R3/MW corridors. Convict squares from the refusal message instead.
- **HP-triggered retreat or disengagement.** Refuted (9→13→17→42 deaths).
- **Pricing CB-Dark on 5 episodes.**
- **A `"."` override of the current square when classifying it.** It creates the §2B orbit. Classify by the remembered glyph.
