# Diagnosis — minihack evolve r061 / cand_r061_0_spatial-grounding (chart-routed learner)

**Headline:** primary = **0.350** (14/40, se 0.075). Survival 0.825, step_efficiency 0.262, cost 34.8k tok/ep.
The parent r035 scored 0.150, so the chart + BFS graft recovered **+8 episodes**. The lineage elite r058 scores 0.625.

| Task | Wins | Who decided |
|---|---|---|
| MazeWalk-9x9 / 15x15 | 5/5 + 5/5 | `chart:route` on 100% of steps, 0 tokens |
| Corridor-R3 | 2/5 | route, then the learner once the route dies (3 timeouts) |
| CorridorBattle-Dark | 2/5 | route + `reflex:swing`, 0 tokens; 3 deaths |
| Quest-Easy | 0/5 | `learner:*` on 97–99 of 100 steps; route fires once |
| Quest-Medium | 0/5 | 4 deaths to giant rats at steps 22–31, 1 timeout |
| Boxoban-Hard / Medium | 0/5 + 0/5 | learner on 52–81% of steps; no push model |

**Binding constraint:** there are two, and they are different kinds.
- **HARNESS-BUG in `credit()`:** a `The door opens.` turn does not change the hero's position, so it is booked as a wall bump in `self.blocked`. That permanently seals the only exit. This causes **all 3 Corridor-R3 losses**.
- **TASK-GAP on Boxoban and Quest (20/40 episodes, 0 wins):** the harness has no mechanism for either task. Both have known, observation-only mechanisms: r058's `tools/quest.py` scores 10/10, and an exact push-space Sokoban planner scores about 19/20.

## 1. What this harness does (mechanism)
Everything is in the single module `agent.py:LearnerAgent.act`. `AGENT.md` is one line and is used only by `consult`. `rules.md` and `memory.md` are empty.

Each step runs in this order:
1. **Prompt reflexes:** `y` for an attack `[yn]` prompt, `more` for `--More--`, otherwise `esc`.
2. **`credit()`:** records drawn glyphs and applies a TD(λ) update. It also books a no-move as `(s,a)` in `self.blocked` and pins q to −1 (`agent.py:265-268`).
3. **`overlay()`:** writes each `<noun> adjacent <dirs>` clause onto its square, as `block` (wall, corner, **dark area**, bars, stone…) or `open` (doorway, open door, floor…).
4. **`reflex:stairs`:** emits `down` on a remembered `>`. It fired **0 times**, because stepping onto `>` already ends a MiniHack episode.
5. **`deadly()`:** vetoes lava/water nouns and adjacent `}`. **`foe_bearing()`:** swings at any adjacent letter.
6. **`route()`:** BFS over `passable()` cells to a drawn `>`, or else to the nearest unstood `frontier()` cell, and emits the first hop.
7. **Fallback when there is no route:** a learner argmax, plus a once-per-square model consult on ties.

The legal vocabulary is limited to the 8 compass bearings plus the reflex verbs. `pickup`, `kick`, `apply` and `zap` are **never** emitted, and nothing reasons about pushes.

## 2. What actually happened (causal chains)
CSV rows are (action, obs AFTER it).

### 2a. Corridor-R3: a door-open is booked as a bump, and that seals the exit (3/3 losses)
- **run_00:** the `>` is drawn in the SE room from about step 35. At step 38 the agent goes `east` from (12,17) into a closed door, and the message is `The door opens.`. The position is unchanged, so `credit()` adds `((12,17),'east')` to `blocked`.
  - The next frame says `vertical open door adjacent east`, but `route()` skips the blocked edge.
  - At step 39 the agent turns `north`. At step 40 it tries `southeast` into the doorway: `You can't move diagonally into an intact doorway.`, which is booked again.
  - By step 55 no frontier is reachable. The learner then retraces the corridor for 44 steps, and the episode times out at 100 with the `>` 7 squares away.
- **run_03:** at step 10, `northwest` gets `The door resists!`. At step 12, `west` from (20,3) gets `The door opens.`, and that is the room's only exit.
  - From step 13 on, **87 of 87 steps are `learner:*`**, bouncing inside the 4×5 start room with `route()` returning None.
- **run_01:** at step 52, `west` from (20,3) gets `The door opens.`, and the route turns back east.
  - It then gets stuck at (20–22,7), where every frame reads `dark area adjacent north, northeast, southeast, south, southwest, (west,) northwest`.
  - `overlay()` tags all of those squares `block`, so they are neither passable nor counted as frontier (see 2b). The learner ping-pongs `east`/`west` from step 74 to 99. The `>` is drawn at `far westnorthwest`.
- **Wins:** runs 02 and 04 met no door, or only a `resists`, on their path, and won in 25 and 40 steps, all by `chart:route`.

### 2b. `dark area` is in `BLOCK_NOUNS`
`dark area` is NetHack's word for an *unlit, unseen* square, and that is very often floor: dark room floor or the far side of a doorway.
- Tagging it `block` removes it from `passable()`. It also makes it "known" in `frontier()`, which only counts neighbours that are neither drawn nor in `self.nouns`. So dark ground can never become a frontier.
- The noun appears in 251/365 R3 frames, 57/190 CB-Dark frames, 60/201 QM frames, and every MazeWalk frame.
- MazeWalk survives this because the `>` is drawn in a lit maze. R3 run_01 and the dark-room legs do not survive it.

### 2c. Quest-Easy and Quest-Medium: no acquisition layer (10/10 losses)
- **QE run_00:** the hero spawns with `horn adjacent …` and a potion. A `}` lava column at x=30 splits the level.
  - `route()` finds nothing: `}` is LETHAL and there is no frontier on the left bank. So from step 1, **98/100 steps are `learner:*`** over just 20 distinct squares, at about 700 tokens per consult.
  - The episode times out without a single `pickup`, `kick`, `apply` or `zap`.
- **QM run_02:** step 0 frame reads `horn adjacent south`, and the agent routes *away* from it (`north`, then `east`).
  - Steps 1–13 route through the dark corridor into the rat room. At (35,12) inside the room, 2–3 giant rats are adjacent.
  - `reflex:swing` takes steps 16–21, with HP 12→9→5→4→2→dead. The corridor mouth at x≈32–33 was 1-wide and had been passed at step 12.
  - Runs 01, 03 and 04 are the same trajectory: death at steps 22–31.

### 2d. CorridorBattle-Dark (3 deaths)
- **run_01:** at step 18 the agent fights at the corridor mouth (34,12) (`vertical wall adjacent north and south`). It kills three rats but drops to HP 3/16.
  - At step 23, `route()` steps it `southeast` into the room at (35,13), where two rats are adjacent. It dies at step 26.
- The other deaths look the same. The chain's CB-Dark record says this task is mostly dice (§6), so 2/5 is within noise.

### 2e. Boxoban (0/10)
- Route runs on 18–48 steps, and the rest are argmax/consult. That is 60–119k tokens per episode wandering a 4-boulder puzzle with no push model.
- No boulder reaches a `{` in a controlled way: returns are only +0.001/−0.099.

## 3. Failure breakdown (26 losses)
| Category | Episodes | Class |
|---|---|---|
| Boxoban: no solver | 10 | TASK-GAP (solved mechanism exists) |
| Quest: never picks up or uses horn/wand; QM advances into the rat room | 10 (4 deaths) | TASK-GAP (r058 `tools/quest.py` = 10/10) |
| R3: `The door opens.` booked as a bump → exit sealed (+ `dark area` blocking) | 3 | HARNESS-BUG |
| CB-Dark melee deaths | 3 | mostly dice |

Token cost is entirely learner consults on Boxoban and Quest-Easy, the tasks where `route()` has no answer.

## 4. Binding constraint
- **HARNESS-BUG (cheap, certain): +2–3 R3 episodes.** `credit()` equates "position unchanged" with "blocked edge". NetHack has several no-move turns that change the world and must not seal an edge: `The door opens.`, `The door resists!`, `This door is locked.` (which should be kicked), `You can't move diagonally into/out of an intact doorway.` (which only rules out the diagonal), and attacks.
- **TASK-GAP (large): Boxoban + Quest = 20 episodes at 0.** The route/learner stack cannot win these at any tuning. Both need a task mechanism, armed from the observation: `{`/`0` glyphs and 4-cardinal legal set for Boxoban, and `pickup`/`kick`/`zap` in the printed legal verbs plus `horn`/`wand` prose for Quest.

## 5. Concrete changes (ranked by leverage)

1. **Graft r058's Quest module**
   - **Where:** `tools/quest.py`, plus a stage in `act()` before `route()`. Source: `candidates/prop_r058/cand_r058_0_spatial-grounding/tools/quest.py` (`Quest.observe/answer/kit/cross/hold/targets/toward`) and its wiring at `agent.py:396-468` there.
   - **What it does:** kicks the spawn pile apart, `pickup`s one item at a time (wand first, so its letter is answerable), then answers the `zap`/`apply` letter prompt with the letter-equivalent verb (`h j k l y u b n a c e o q s z`) and fires at the lava toward the goal bearing.
   - **In QM:** hold the 1-wide corridor mouth for the rat fight; **never advance into the room while a rat is adjacent**.
   - **Prompt reflex:** stop `esc`-ing item prompts when Quest is armed.
   - **Predicted:** Quest 0/10 → 7–10/10, so primary +0.18 to +0.25. It also removes about 70k tokens per QE episode and should turn 4 QM deaths into survivals.

2. **Fix the no-move classifier in `credit()`** (`agent.py:252-268`)
   - Book `(s,a)` in `blocked` **only** when the message does NOT contain `door opens`, `door resists`, `door is locked`, `diagonally`, or an attack verb.
   - On `door opens`, set `self.nouns[target] = "open"` and `discard((s,a))` from `blocked`.
   - On `diagonally`, block only that diagonal edge (current behaviour), and tag the target as a doorway so `route()` never offers a diagonal into or out of it.
   - On `This door is locked.`, if `kick` is legal, emit `kick` + bearing (a two-step verb) instead of routing away.
   - **Predicted:** R3 2/5 → 4–5/5 (+0.05 to +0.075). Each of the 3 losses had the `>` or the only exit one door away.

3. **Take `dark area` out of `BLOCK_NOUNS`** (`agent.py:32`)
   - Treat it as *unknown*: don't write it to `self.nouns`, or write `"dark"`, which `passable()` rejects but `frontier()` counts as unexplored.
   - A real wall still gets caught by the bump ledger once the fix in item 2 is in.
   - **Predicted:** gets R3 run_01-style dark-room stalls moving again, and lets frontiers extend through dark doorways. MazeWalk is unaffected because it already routes to a drawn `>`.

4. **Add a Boxoban push planner** (new `tools/sokoban.py`, armed when the legal set is exactly the 4 cardinals and the map shows `0` + `{`)
   - **Push model:** `#` = bars (impassable), blank = stone, and `.`/`{`/`<` = floor. A push succeeds iff the square beyond the boulder is floor and empty.
   - **Search:** A*/BFS in push space, with one player-reachability BFS per node and paths rebuilt from parent links.
   - **Execution:** follow the plan move by move, rolling the plan state forward on every emitted move. A boulder standing on a `{` hides that fountain, so union the fountains drawn across frames.
   - **Predicted:** Boxoban 0/10 → about 8–10/10 (plans of 28–97 steps fit the 100-step cap on roughly 96% of levels), so +0.20 to +0.25. It also removes 60–119k tokens per Boxoban episode.

5. **(Low leverage) CB-Dark:** don't let `route()` leave a 1-wide square (walls on both flanks per prose) while HP < 50% and a creature was adjacent within the last 2 frames. Spend the turn with `search` instead. Expect ±1 episode; price it on ≥200 episodes before believing it.

6. **Dead code:** `reflex:stairs` never fires, because the episode ends on `>`. Harmless, but can be deleted.

Items 1, 2 and 4 are independent and can go in one candidate. Together they bound primary at about 0.75–0.85, because MW is already 10/10.

## 6. Exhausted — do NOT repeat
- **Pure learner as the primary policy** (r035: 0.150). Optimistic trial-and-error cannot pay for itself under a 100-step cap. Keep it only as the no-route fallback.
- **Model consult as a tie-breaker:** inert. The model's pick doesn't change outcomes, and it is the whole token bill.
- **CB-Dark melee reformulations** (retreat, recoil, hold, lookahead): the chain's record is that they are dice rerolls at n=5. Don't spend a round on them.
- **Re-tuning learner rewards or TIE/BONUS constants:** they don't touch the three causes above.
