# Diagnosis — minihack iter r068 / cand_r068_0_spatial-grounding (r064 stage table + `>`-attracted frontier in `route()`)

**Headline:** primary = **0.900 (36/40)**, valid. That is +1 over r064 (35/40), and the whole gain is Corridor-R3 run_03 (a timeout at 100 steps in r064, won in 31 steps here).
- **Model use:** 1 VLM call in 40 episodes, 6 476 tokens, all on Boxoban-Medium run_04. The other 39 episodes made 0 calls.
- **Rung census:** chart:route 669, sokoban 567, quest:* 369 (walk 180, hold 106, …), reflex:swing 141, learner 6 (5 argmax, 1 consult), reflex:more 2.
- **Binding constraint:** all 4 losses are CorridorBattle-Dark melee deaths. They happen on the square **just outside** the 1-lane corridor, where 3 rats attack at once. The chain has priced this axis as dice, so primary is at the chain's practical ceiling.
- **Deterministic headroom left:** only on the next keys, cost and steps. One code bug, the Boxoban arm gate (§2B), causes the eval's entire token bill and 6 wasted steps.

| Task | Wins | Notes |
|---|---|---|
| Boxoban-Hard | 5/5 | 53–81 steps, 0 tokens |
| Boxoban-Medium | 5/5 | run_04: 6 learner steps + 1 consult (§2B) |
| Corridor-R3 | **5/5** (r064 4/5) | run_03 fixed by the attractor (§2A) |
| MazeWalk-9x9 / 15x15 | 5/5 / 5/5 | chart only |
| Quest-Easy / Medium | 5/5 / 5/5 | `tools/quest.py`, unchanged from r064 |
| CorridorBattle-Dark | 1/5 | 4 deaths, steps 27–34 (§2C) |

## 1. What this harness does (mechanism)

`harness.py:LearnerAgent.act()` parses the frame (`parse_glyph_map`), runs `forget()` on a `Dlvl` change, and then runs `STAGES = (prompt, orient, sokoban, stairs, swing, cross, hold, kit, fetch, route, learner)`. The first stage that returns a reply wins, and its rung is written to `reasoning`.

- **`st_prompt`:** `more` (at most 2 in a row), `esc`, and Quest item/direction answers.
- **`st_orient`:** `observe` (doors, bars, bumps), `credit` (updates `drawn` and the TD values), `deadly`, and `quest.observe` / `goal_kind`.
- **`st_sokoban`:** `tools/sokoban.py`. It is gated by `Sokoban.armed(moves, fr["rows"])`, which requires a **4-cardinal vocabulary, a drawn boulder and a drawn `{` in THIS frame**. It runs a push-level A* with a plan cache and expect-and-replan.
- **`st_stairs`:** emits `down` on `>`.
- **`st_swing`:** attacks any adjacent non-peaceful letter.
- **`cross` / `hold` / `kit` / `fetch`:** `tools/quest.py`, armed on `{pickup, kick, zap} ⊆ legal`.
- **`st_route`:** BFS over charted passable cells to a drawn `>`, otherwise to a frontier.
  - **New this round:** when a `>` is drawn but unreachable and no `}` is charted, the frontier is elected by `(dist + cheb(f, nearest >), dist, y, x)`. This is a total order, re-elected every step.
- **`st_learner`:** the TD value table. It consults the VLM only on first-time ties, never on Quest.

The prose (AGENT.md, one tie-break sentence) is effectively inert, because the VLM is reached once in 40 episodes.

## 2. What actually happened (causal chain, with evidence)

### A. Corridor-R3: the attractor fixed exactly the r064 loss, with no collateral change
- **r064 run_03:** at step 23 the hero was at (15,7), the `>` was drawn 7 columns west behind undrawn cells, and the bearing order elected the east frontier. It drifted east and timed out at 100 steps.
- **This candidate, run_03:** the route rungs run `… west, west, south, west×7`. It wins in **31 steps**, all 31 of them `chart:route`, with return 1.00.
- The other R3 runs win in 23–62 steps.
- The branch is gated off whenever a `}` is charted, and Quest stays 10/10, so the gate held.
- MazeWalk episodes are unchanged: every drawn `>` there is reachable, so the new branch never fires.

### B. Boxoban-Medium run_04: `Sokoban.armed()` is a per-frame gate, and it drops out whenever the hero covers the last visible `{` (HARNESS-BUG, steps and cost)

I replayed `tools/sokoban.py:Sokoban.step` offline on the episode's frames (the decision at step k reads the observation of CSV row k−1):
- At steps 33, 35, 37 and 39, the solver's cached plan is **intact and valid**. `step()` returns `north` (`sokoban:plan:8`), and `expect` matches the state.
- But live, those steps were `learner:argmax`, `learner:consult`, `learner:argmax` and `learner:argmax`.

**Cause:** at those decisions the hero stands on (39,14), the only goal not yet covered. The other three goals (35,12), (38,10) and (37,14) sit under boulders. The frame therefore draws **no `{` at all**, so `Sokoban.armed()` (`GOAL in flat`) returns False and `st_sokoban` abstains before `step()` is reached. The learner then ping-pongs:

| Step | Action | Rung |
|---|---|---|
| 32 | `east` | sokoban |
| 33 | `west` | learner |
| 34 | `east` | sokoban |
| 35 | `east` | **learner:consult** (the 6 476 tokens) |
| 36 | `west` | sokoban |
| 37 | `east` | learner |
| 38 | `west` | sokoban |
| 39 | `north` | learner, which happens to match the plan |

**Cost:** about 6 wasted steps plus the eval's whole token bill. r064's diagnosis §5.2 guessed at a goal-set or solve failure. The real cause is the **arming gate**, not the solver.

**Risk:** any level whose plan walks the hero across the last uncovered goal repeats this. With a tight step margin (Hard run_00 used 81 steps), it could cost a win.

### C. CorridorBattle-Dark 1/5: every death is fought on a square with 3 open lanes
Runs 00–02 share a **byte-identical prefix through step 18**: the same 10-step closet sweep, then east along the corridor row y=12. They diverge only on dice.

| Run | Fight square(s) | Outcome |
|---|---|---|
| 00 | (34,12) doorway: 1 kill, then `chart:route` east onto the corpse square **(35,12)** at step 20 | Prose shows `giant rat adjacent northeast, east, and southeast`, so 3 attackers from step 21. HP 16→13→9→3→1, death at step 33. |
| 01 | Bitten at (34,12) at step 18; 3 rats there, HP 14→6→3 by step 21 | Stepped to (35,12) at step 23, death at step 26. |
| 02 | Same as 00: route onto (35,12) at step 21 | HP 16→1 in 4 steps, death at step 28. |
| 03 | Went straight east, then fought at (38,11). Prose: `horizontal wall adjacent north, northeast, and northwest`, which leaves 5 open lanes (E, SE, S, SW, W) | Rats at E + SE, HP 16→1, death at step 27. |
| 04 (win) | Same square (38,11) as run_03 | Won with HP 3, 18 swings. |

**The structural fact:** the corridor `#` squares (27..33, 12) have exactly one open neighbour toward the room (the east hop). A rat can reach the hero there only single file. The doorway (34,12) is doorless (drawn `.` once vacated), so it already exposes NE/E/SE. (35,12) and (38,11) are open room squares. The harness never fights from inside the corridor:
- `st_route` walks the hero **out** of the corridor toward the nearest frontier, which is always east.
- `st_swing` then fights wherever the hero happens to be.

**Evidence from the one win:** run_04 got lucky at (38,11), with a peak of 2 adjacent rats and 8 misses by the rats.

**Memory caveats** (see §6):
- CB-Dark has no null move (`wait` and `search` are not legal), so "hold" means pacing.
- Disengage/retreat variants measured as killing more.
- A committed fight post was +9/200 at p=0.14.
- n=200 results have reversed at n=500.

### D. Quest and MazeWalk: saturated
- Quest-Medium: `quest:hold` fires 106 times in total (the corridor-mouth `wait` hold), and all 5 wins cost 71–81 steps.
- Do not touch either task.

## 3. Failure breakdown

| Failure | Episodes lost | Other cost | Cause | Class |
|---|---|---|---|---|
| CB-Dark 3-on-1 on an open room square | 4 | — | §2C; `st_route` leaves the 1-lane corridor, and `st_swing` fights in place | TASK-GAP (dice-dominated) |
| Boxoban disarmed while the hero covers the last visible `{` | 0 | 6 steps, 1 consult (6.5k tokens, 100% of the bill) | §2B; `tools/sokoban.py:Sokoban.armed` is evaluated per frame | HARNESS-BUG |
| R3 drawn-unreachable `>` | 0 (fixed) | — | §2A | resolved |

## 4. Binding constraint
- **Primary:** CB-Dark, 4 episodes.
  - This is a **TASK-GAP with dice on top**. The harness has no notion of *where* to fight. The one lever not refuted in chain memory is fight-square exposure: memory records a one-way anchor at ARM_LANES=2 at +5/−1 over 120 episodes.
  - The chain also records that the melee axis reverses sign between n=200 and n=500. So nothing here can be confirmed on a 5-episode draw.
  - Treat any CB-Dark change as a **priced bet** (at least 500 stub/CRN episodes), not a sure gain.
- **Cost/steps (the next lexicographic keys): HARNESS-BUG, deterministic and fixable.** One line in the arming gate makes the eval use 0 tokens and saves about 6 steps. It does not change any win.

## 5. Concrete changes to try (ranked by leverage)

### 5.1 `tools/sokoban.py` + `harness.py:st_sokoban`: latch the arm per level (cost → 0, −6 steps, zero primary risk)
**Change:**
- Add `self.live = False` to `Sokoban.__init__`. It already resets on `forget()`, because `forget()` rebuilds `Sokoban()`.
- In `st_sokoban`:
  ```python
  if not (self.soko.live or Sokoban.armed(c["moves"], fr["rows"])):
      return None
  self.soko.live = True
  ```
- Also make the learner never consult on a latched Boxoban level (`and not self.soko.live` in `st_learner`'s consult condition), as a belt-and-braces guard.

**Predicted effect:**
- At Medium run_04 steps 33, 35, 37 and 39, `step()` returns the cached plan move. I checked this by offline replay: it returns `north` at `plan:8`. The episode drops from 48 to about 42 steps.
- The learner consult disappears, so the eval uses **0 tokens in total**.
- Hard run_03's 2 learner steps (see r064 §2A) very likely have the same cause. Verify with the same replay.
- **Transfer check:** the Boxoban vocabulary is 4-cardinal only. The latch is set only after a real arm (boulder plus `{` drawn), so no other task can latch.

### 5.2 CB-Dark: fight from the corridor, not from the room (priced bet, ≥500 episodes before shipping)
**Where:** a new stage `post` between `swing` and `route`, owned by a new `tools/melee.py`.

**Arm:**
- Arm only when the vocabulary is plain movement with no `pickup`/`kick`/`zap`, no boulder is drawn, and the level has shown a creature or a `bites` / `misses` message.
- In practice, arm once the first contact message appears.

**Mechanism:**
1. **The post.** It is the last corridor square you STEPPED ON (memory: the post must be one you stepped to) whose drawn neighbourhood has exactly one open non-corridor lane, i.e. `#` with blank or wall on every side except the two corridor hops. In run_00 that is (33,12).
2. **When to hold.** Hold while contact is recent: fewer than K quiet steps since the last bite, kill or adjacent-letter frame, with K around 6. Memory says the release clock must not count the walk back to the post.
3. **How to hold with no null move.** If no rat is adjacent, pace between the post and the corridor square behind it (west, then east). Adjacent rats are still handled first by `st_swing`, which runs before `post`.
4. **Retreat trigger.** If the hero stands on a square with ≥3 open lanes and a rat is adjacent on only 1 lane, do NOT recoil; memory says a one-hop recoil is dice (+9/−10). Only use the post *before* contact spreads: when `st_route` would step from the post/doorway **east into the room** while contact is recent, emit the pace-back move instead.

**Why it could work:**
- From (33,12), at most one rat is adjacent at a time. Runs 00–02 all lost HP in 3-attacker frames on (34,12) or (35,12).
- The r064 diagnosis §5.3 and the memory anchor at ARM_LANES=2 point the same way.

**Risks (memory):**
- "Every degree of disengagement kills more."
- "CB-Dark threat horizon is one square."
- A reversal at n=500.

So this must be priced on a CRN stub rig at ≥500 CB-Dark levels and must not touch any other task. That gate is safe: a creature and a bite message only co-occur with a plain-movement vocabulary on CB-Dark. Expected: +0 to +1 episode per 5, with high variance.

### 5.3 (Keep) Do not touch `route()`'s attractor, `tools/quest.py` or MazeWalk
- They are at 100% on this draw.
- The attractor is a total order, so it cannot orbit.
- Its gate on a charted `}` is what keeps it out of Quest. Do not widen it to fire on Quest.

## 6. Exhausted — do NOT repeat
- **Boxoban via frontier walking, the learner or VLM tie-breaks.** The exact push A* solves 10/10. Only fix the arm gate (§5.1).
- **R3 frontier ranking.** The attractor landed, and chain memory lists 7 other frontier reformulations as refuted. R3 is at its ceiling.
- **CB-Dark:**
  - HP-gated retreat and disengage (a dose-response that kills more).
  - Hop-lookahead recoil (+9/−10, +8/−11).
  - Arrival time and closet-sweep length (7 arms, 0/3; here run_03 went straight east and still died).
  - A `wait`/`search` hold (not legal on CB-Dark).
  - Threat memory in any placement.
  - None of these should be re-proposed without ≥500-episode pricing.
- **Prose / AGENT.md edits.** The VLM is called once in 40 episodes, so prose has no causal path to the score.
