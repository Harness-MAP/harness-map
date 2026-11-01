# Diagnosis — minihack iter r064 / cand_r064_0_spatial-grounding (tools/ loader + `sokoban` stage)

**Headline:** primary = 0.875 (35/40), valid. Parent r058 scored 0.625, so this is **+10 episodes**, and all of them are Boxoban (0/10 → 10/10).

| Task | Wins |
|---|---|
| Boxoban-Hard | 5/5 |
| Boxoban-Medium | 5/5 |
| MazeWalk-9x9 | 5/5 |
| MazeWalk-15x15 | 5/5 |
| Quest-Easy | 5/5 |
| Quest-Medium | 5/5 |
| Corridor-R3 | 4/5 |
| CorridorBattle-Dark | 1/5 (4 deaths) |

- **Tokens:** 6 476 over the whole eval, from one consult on Boxoban-Medium run_04. The other 39 episodes made 0 model calls.
- **Rung census:** chart 740, sokoban 567, quest 369, reflex 143, learner 6.

**Binding constraint:** the 5 remaining losses have two causes.
- **1 × HARNESS-BUG, fixable.** `route()` ignores a drawn but unreachable `>` and picks among equal-distance frontiers by bearing order. This was already flagged in r058 §3 and never fixed.
- **4 × CB-Dark melee deaths.** Chain memory classes this as dice, a closed axis.

## 1. What this harness does (mechanism)

`orchestrator.py:LearnerAgent.act()` parses the frame with `parse_glyph_map`. It runs `forget()` when `Dlvl` changes, then runs `STAGES` in order. The first stage that answers wins, and its rung is written to `reasoning`.

1. **`prompt`:** `more` (at most 2 in a row), `esc`, and Quest direction/item answers.
2. **`orient`:** `observe`, `credit`, `deadly`, and `Quest.observe` / `goal_kind`.
3. **`sokoban`** (new, `tools/sokoban.py`):
   - Arms only when the move vocabulary is exactly the 4 cardinals and the frame draws a boulder `` ` `` and a fountain `{`.
   - Runs a push-level A* once, then pops one move per step.
   - Replans when the observed (hero, boulders) differs from the expected state.
4. **`stairs`:** emits `down` on `>`.
5. **`swing`:** attacks any adjacent letter.
6. **`cross` / `hold` / `kit` / `fetch`:** `tools/quest.py`, armed on `{pickup, kick, zap}`.
7. **`route`:** BFS over charted passable cells to a drawn `>`, otherwise the **nearest** unstood frontier.
8. **`learner`:** a TD value table. It consults the VLM only on ties it has not seen before (never on Quest).

The prose modules (AGENT.md, a single tie-break sentence) are effectively inert because the VLM is almost never called.

## 2. What actually happened (causal chain, with evidence)

### A. Boxoban 10/10: the new stage works exactly as designed
- In 8/10 episodes, steps = first plan length + 1 with 0 replans. For example, Hard run_00 has plan:80 and 81 steps; Medium run_00 has plan:37 and 38 steps.
- Two episodes briefly fell through to `learner`:
  - Hard run_03: 2 `learner:argmax` steps at steps 34 and 36, then 2 replans.
  - Medium run_04: steps 33–39 alternate `learner` and `sokoban:plan:9`. This includes the eval's only VLM consult (6.5k tokens). This is the level that draws only 3 `{` and uses the start square as the 4th goal.
- In both, `soko.step` returned None for one frame right after a push. When a boulder covers a `{`, the goal set / state check fails once, and the cached "failed solve" falls through to the learner. A ping-pong follows (`west`, `east`, `west`) until the next frame re-solves.
- Cost: about 6 steps and 1 consult. No wins were lost.

### B. Corridor-R3 run_03 (timeout at 100 steps, 100% `chart:route`): a drawn `>` is ignored
- **Step 23.** The hero is at (15,7) on a corridor `#`. The `>` is drawn at (8,7) inside the dark room `|....>.`, 7 columns west. The cells (10..13,7) are undrawn, so the `>` is **not reachable on the chart**.
- **The choice.** `route()` falls through to "nearest unstood frontier". Both (14,7) (west, leading to the room) and (16,7) (east) are 1-hop frontiers. BFS expands in `ORDER = north, east, south, west, …`, so **east is elected purely by bearing order**.
- **Steps 24–60.** The hero follows the eastern corridor network, a sequence of nearest frontiers each 1 hop further away: (18,7) → (22,9) → (27,11) → … → (30,7) at step 60.
- **Step 98.** It has only drifted back to (24,10) and times out. The `>` stayed drawn the whole time, "stairs down far west" is in the prose on every frame, and there were 71 unique squares and only 3 no-moves.
- This is exactly r058 §3's "R3: drawn `>` unreachable, frontier chosen by BFS order". The proposer did not touch `route()` this round.

### C. CorridorBattle-Dark 1/5 (4 deaths at steps 27–34): melee dice
- In every episode, `chart:route` walks the hero east to the corridor mouth, and `reflex:swing` then fights in place.

| Run | Fight squares | Peak simultaneous rats | Final HP / outcome |
|---|---|---|---|
| 00 | (34,11)→(35,11), 13 swings on (35,11) | 3 | HP 16→1, death |
| 01 | (34,11)→(35,11) | 3 | death |
| 02 | (34,11)→(35,11) | 3 | death |
| 03 | (38,10), 11 swings | 3 | death |
| 04 (win) | (38,10), 18 swings | 2 | won with HP 3–4 |

- **Run_00 detail.** After killing the first rat from the walled corridor square (34,11) (prose `vertical wall adjacent north and south`), `route` steps east onto the corpse square (35,11). That square is open to N/NE/E/SE/S, and three rats then attack at once.
- **Sweep length.** Runs 00–02 also spend steps 0–9 sweeping the spawn closet (`N N W S S W N N E E`) before heading east, while run_04 heads east at step 1. But run_03 also heads east early and dies. This matches the chain's finding that arrival time is refuted as a separator (7 arms, 0/3).
- Memory: "CB-Dark has NO null move", so a `wait` / `search` hold is not available here.

### D. MazeWalk, Quest: saturated
All 20 episodes won. Quest runs on `tools/quest.py`, byte-identical with r058, which also went 10/10. MazeWalk is chart-only. Do not touch either.

## 3. Failure breakdown

| Failure | Episodes | Cause | Class |
|---|---|---|---|
| CB-Dark melee death at the corridor mouth, 3-on-1 | 4 | §2C | dice / closed axis |
| R3: drawn `>` unreachable, frontier picked by bearing order | 1 (run_03) | §2B, `orchestrator.py:route` | HARNESS-BUG |
| Boxoban one-frame solver dropout → learner ping-pong | 0 losses (~6 steps, 1 consult) | §2A, `tools/sokoban.py:step` | efficiency bug |

## 4. Binding constraint

- **HARNESS-BUG (R3, 1 episode):** this is the only loss with a deterministic, code-level fix. It is worth +0.025 on this draw, and chain memory measured +2/−0 over 125 R3 levels for the same fix.
- **CB-Dark (4 episodes):** each loss is 3 rats at once on a 5–6-lane mouth square. Retreat, HP-disengage, lookahead recoil, arrival time and the `wait` hold have all been priced and refuted or unavailable (no null move). Treat it as dice unless a proposal is priced on at least 200–500 episodes.
- **Everything else is at ceiling.** The next keys after primary are cost (already about 0) and steps. The Boxoban dropout is the only visible step waste.

## 5. Concrete changes to try (ranked by leverage)

### 5.1 `orchestrator.py:LearnerAgent.route`: make a drawn but unreachable `>` attract the frontier (R3 +1, low risk)

**Current behaviour:** when `stairs` is non-empty but BFS never reaches one, `route` still breaks on the first frontier in BFS order.

**Change:**
- Keep the BFS running to completion, and record `dist[cell]` for every frontier cell found.
- If a stair was reached, return its first hop, as today.
- Otherwise, if any `>` is drawn, elect the frontier that minimises this key:

  ```
  (dist[f] + cheb(f, nearest_drawn_gt), dist[f], f[1], f[0])
  ```

  This is a total order. Re-elect it every step, with no commitment state.
- Otherwise, keep today's nearest frontier.

**Gate:** apply the rule only when `not c["quest"]`, i.e. pass a flag in from `st_route`. On Quest the `>` sits beyond lava and `tools/quest.py` owns movement; memory says "gate on `}`".

**Check on R3 run_03, step 23 at (15,7):**
- West frontier (14,7): 1 + 6 = 7.
- East frontier (16,7): 1 + 8 = 9.
- So the rule emits `west` into the room corridor, and the `>` is about 7 steps away. Today's harness emits `east`.

**Predicted effect:**
- R3 4/5 → 5/5, primary +0.025.
- MazeWalk is unaffected: a drawn `>` there is reachable once drawn, so the branch never fires.
- CB-Dark and Boxoban are unaffected: no `>` is drawn, or the sokoban stage answers first.
- Memory ("A DRAWN but unreachable `>` is an attractor the frontier must aim at"): R3 4/5 → 5/5, +2/−0 over 125 levels, 0 arms changed on other tasks.

### 5.2 `tools/sokoban.py:Sokoban.step`: do not drop out for one frame after a push onto a goal (steps and tokens only)

**Where:** Hard run_03 steps 34 and 36, and Medium run_04 steps 33–39, where `learner` alternates with `sokoban:plan:9`.

**Change:**
- (a) A goal covered by a boulder, or by the hero, must stay in the accumulated goal set. Never let a covered `{` shrink the goal set; it already accumulates, so check the start-square fallback when `len(goals) == len(boulders)` after a cover.
- (b) If `solve` fails on the current frame but a cached plan exists whose next move is legal and non-bumping, keep popping it instead of returning None.
- (c) When `armed` is true, never let `learner:consult` fire: set `c["no_consult"]` so the VLM tie-break is skipped.

**Predicted effect:**
- Primary: no change (Boxoban is already 10/10).
- About −6 steps and −6.5k tokens (the eval's entire token bill goes to 0).
- It removes a latent risk: a ping-pong on a level with a tighter step margin (for example, Hard run_00 used 81 of the roughly 100-step budget) could cost a win.

### 5.3 (Optional, low EV, price at ≥200 eps) CB-Dark: do not advance off a walled corridor square while a rat is named "very near"

**Where:** `st_route`. In run_00 step 20, the hero stepped from (34,11) (prose `vertical wall adjacent north and south`) onto the open corpse square (35,11) and met 3 attackers.

**Change:** a one-way anchor.
- If the hero's square has a wall/corner noun on two opposite bearings, and a creature noun appears in the prose "very near" band but is not adjacent, then do not take a route hop onto a square with more open bearings.
- Since there is no null move, emit the swing toward the named bearing if it is adjacent. Otherwise leave the move alone.

**Prior evidence:** memory records a one-way anchor at `ARM_LANES=2` at +5/−1 over 120 episodes and a committed fight post at +9/200 (p = 0.14). Both are on other lineages, and the chain also records reversals at n = 500.

**Risk:** do NOT add a retreat or HP gate (see §6).

**Expected value:** about +0 to +1 episode per 5. Ship it only if a stub/CRN rig at ≥200 CB-Dark levels agrees.

## 6. Exhausted — do NOT repeat
- **Boxoban via frontier walking, value learning or VLM tie-breaks.** This is now solved by the exact push planner. Keep `tools/sokoban.py` and its arming gate (4 cardinals + drawn `` ` `` + `{`) as they are.
- **Re-deriving the Quest plan.** `tools/quest.py` is 10/10 on two consecutive candidates; keep it byte-identical.
- **CB-Dark melee tuning:** retreat, HP-crit disengage, lookahead recoil, streak caps, arrival time, and threat memory. Dose-response shows that more disengagement means more deaths. A `wait` hold is not available because CB-Dark has no null move.
- **R3/MazeWalk frontier re-rankings by gain rate, extremity or nub demotion** when no `>` is drawn. These were refuted repeatedly; the goal is one hop away on 66% of steps. Change 5.1 applies **only** when a `>` is drawn but unreachable. It is a different trigger, and the only frontier change with a replicated positive result.
- **Re-wording AGENT.md / the consult.** The VLM is called once in 40 episodes, so prose is not a lever.
