# Diagnosis — minihack iter 47 / cand_r047_0_spatial-grounding (prose-noun neighbour ledger + code arbitration)

**Headline:** primary = **0.200** (8/40, SE 0.063). Survival 0.800, step_efficiency 0.161, cost 348k tok/ep.
Per task:
- MazeWalk-9x9: 5/5
- MazeWalk-15x15: 2/5
- Corridor-R3: 1/5
- Boxoban-Hard, Boxoban-Medium, CorridorBattle-Dark, Quest-Easy, Quest-Medium: 0/5 each

Comparisons:
- It beats the naive anchor (0.100) and r029 (0.150), which both use the same single-call design.
- It is far below the chain frontier, r041/r046 at **0.375**. Those candidates use a chart + BFS route and score MazeWalk 10/10 and R3 4/5, at ~17k tok/ep.

Four wins are MazeWalk episodes where the `>` is adjacent to spawn and the episode ends in ≤4 steps: 9x9 /01 /02 /04 and 15x15 /02. They carry no signal. The three "real" wins are MazeWalk-9x9/00, MazeWalk-15x15/01 and R3/02.

**Binding constraint:** two HARNESS-BUGs in the new ledger turn the code arbiter into the main source of wasted steps.
1. **`dark area` is classified as OPEN.** 310/310 "It's solid stone." bumps targeted a square the prose called `dark area`. **210 of them were chosen by code** (orbit-break 143, fallback:veto 67). This is 56–69% of all steps on MazeWalk-15x15 /00 /03 /04.
2. **An unanswerable `[yn]` prompt is never booked as a failure.** The prompt text contains "attack", which matches the `COMBAT` regex. The result is a 69-step freeze on Quest-Easy/04.

Beneath both bugs sits a structural TASK-GAP. A one-hop orbit-break cannot explore: 56–84% of steps revisit a square, and every R3 / MazeWalk-15 loss ends with the `>` never charted.

## 1. What this harness does (mechanism)
`agent.py:GroundedHarness.act` runs these steps each turn:
1. `Ledger.observe`
   - Parses cursor `(x, y-1)`, the `map:` block, `Dlvl` and "`<noun> adjacent <dirs>`" prose.
   - Counts visits.
   - Books `(pos, dir)` as failed if the cursor did not move AND `COMBAT` does not match the frame.
2. `Ledger.rows` builds 8 rows `(dir, glyph, visits, failed, kind, is_back)`. `kind` comes from `classify(nouns)`:
   - `deadly` if the noun contains lava or water.
   - `blocked` for wall, stone, bars or tree.
   - `open` for any noun in `TERRAIN`, **which includes `"dark"`**.
   - `creature` otherwise.
3. `modal_reflex` answers `[yn` with `y` and `--More--` with `more`, but **only if that verb is legal**.
4. One VLM call is made on the naive transcript + AGENT.md/memory.md + `render_table`. The model emits about 2 output tokens per step and gives no reasoning.
5. `action_io.arbitrate`:
   - Takes the first whole-word legal mention.
   - Vetoes it if `failed` / `deadly` / `blocked`, then uses `rank_open[0]` (`fallback:*`).
   - If the pick is the square just left and no creature is adjacent, replaces it with `rank_open[0]` when that square has strictly fewer visits (`orbit-break`).
6. `rank_open` sorts by `(g in "|- " and kind != "open", g != ">", visits, back, i)`.

Rule census over 2948 steps:
| Rule | Steps | Share |
|---|---|---|
| model | 1971 | 67% |
| orbit-break | 684 | 23% |
| fallback:veto | 221 | 7.5% |
| fallback:model | 72 | 2.4% |
| modal | 0 | 0% |

## 2. What actually happened (causal chains)
CSV row = (action, observation AFTER it). "No-move" means `pos[i] == pos[i-1]`.

**A. Code walks into stone because `dark area` counts as passable** (`tools/ledger.py:TERRAIN` contains `"dark"`, so `classify` → `open`).

MazeWalk-15x15/04, steps 20–33, position (43,6) and then (42,6):
- The prose is `dark area adjacent north, northeast, south, southwest, and northwest`. The only real floor is east/west (`..@..` on the map row).
- The agent emits, in order: north [orbit-break], northeast [fallback:veto], south [orbit-break], southwest [orbit-break], northwest [model].
- Each emission returns "It's solid stone."
- It moves one square west and repeats the same 5–6-bump sweep.

The chain has four links:
- The unseen `' '` squares get `kind='open'`, so `rank_open` does not demote them: the demotion clause is `kind != "open"`.
- Those squares have `visits = 0`, so they always beat the known corridor square behind the agent.
- `orbit-break` therefore replaces every correct "go back out of the dead end" with an untried stone square.
- `failed` books each dir only after its bump, so the sweep costs up to 7 steps per new square.

Totals:
- No-moves are 226/366 steps on MazeWalk-15x15 (62%). Of those, orbit-break chose 112 and fallback:veto chose 45.
- On R3, CB-Dark and Quest-Medium, 100% of stone bumps also target `dark area`.
- `memory.md` / `render_table` tell the model these squares are "passable". That is why the model also chose 69 of the MazeWalk bumps itself.

**B. `[yn]` freeze: the modal reflex cannot answer it, and the ledger cannot see it.**

Quest-Easy/04: at step 29 the agent stands at (27,9) next to a peaceful goblin `o` at (26,10). From step 30 on, 69 consecutive frames show `Really attack the goblin? [yn] (n)`. The model emits `south` 72 times, and the episode ends 0/1 at 100 steps.

The chain:
- `y` is not in the BALROG minihack action set (`BALROG/balrog/environments/minihack/__init__.py:ACTIONS`), so `modal_reflex` returns None. It fired 0 times all eval.
- The cursor jumps to the prompt line `(x=35, y=0)`, so the ledger stores `pos = (35, -1)`.
  - `rows()` then reads `mp[-1+dy]` through Python negative indexing, which gives garbage glyphs.
  - The move also updates `prev_pos` and `visits` for a phantom square.
- The prompt text contains "attack", which `COMBAT` (`\battacks?\b`) matches, so the stuck move is **never booked as failed**. Arbitration therefore can never break the loop.

**C. Exploration has no memory beyond one hop (TASK-GAP).**
- R3: 4 losses, 56–81 revisits per 100 steps.
- MazeWalk-15x15: 3 losses, 72–84 revisits.
- The `>` never appears on the map in any of these 7 losses. In every win it was charted.

The orbit-break removes exact A→B→A (the `aba` count is 0–2 on MazeWalk and R3), but longer cycles remain. A one-step "least-visited neighbour" rule is not a frontier search. This matches the chain's established fact: win ⇔ `>` ever charted.

**D. Deaths (8/40) are melee on CB-Dark (4/5) and Quest-Medium (4/5).**
- Quest-Medium/03, steps 22–29: 6 consecutive `east` swings at a rat chain. HP falls 16→5, then the agent dies.
- Quest-Medium/01, steps 35–45: 11 swings at the `eastnortheast` rat queue while standing in a pocket. HP falls 13→1.
- CB-Dark/04, steps 30–37: HP falls 14→3.

The r029 swing-booking fix works: repeat swings happen. This is the same dice-dominated melee the chain has already closed.

**E. Boxoban is 0/10 as expected.**
- 60 "boulder in vain" steps.
- 131 orbit-break rewrites (27% of the 495 Boxoban steps). They randomise the walk.
- No solver.

## 3. Failure breakdown (40 episodes)
| Category | Episodes | Evidence |
|---|---|---|
| Timeout, `>` never charted (explore gap + stone sweeps) | MazeWalk-15 ×3, R3 ×4 = 7 | revisit 56–84%, no-move up to 69% |
| Melee death | CB-Dark ×4, Quest-Medium ×4 = 8 | HP drains during swing streaks |
| Quest lava never crossed (no item logic) | Quest-Easy ×5, Quest-Medium /02 = 6 | 0 `apply`/`zap`/`pickup` emitted; one `[yn]` freeze (69 steps) |
| Boxoban, no solver | 10 | |
| CB-Dark timeout | /03 = 1 | aba 32 |
| Wins | 8 | 4 of them trivial spawn-adjacent `>` |

## 4. Binding constraint — classification
- **HARNESS-BUG (primary, cheaply fixable):**
  - `dark` in `TERRAIN`.
  - The `[yn]` prompt is not escaped with a legal verb.
  - `COMBAT` matches "Really attack".
  - A `y=0` cursor is accepted as a position.
- **TASK-GAP (structural):** no chart and no BFS frontier. MazeWalk-15 and R3 need a global route, which r041 already has (10/10 and 4/5).
- **Not a VLM-CEILING:** the model's own picks cause only 317 of the 573 no-moves, and the table misleads it ("passable").

## 5. Concrete changes (ranked by leverage)
1. **Do not keep extending this lineage. Graft its useful parts onto the r041/r046 stage pipeline** (0.375).
   - This single-call design is dominated on every navigation task.
   - The only transferable pieces are the prose-noun classification (bars/walls blocked before the first bump) and the combat-aware failure booking.
   - Check them against r041's `tools/chart.py` blocked-set logic. Predicted effect: neutral to +, since r041 is already near its MazeWalk/R3 ceilings.
2. **If this lineage continues, fix the ledger:**
   a. `tools/ledger.py:TERRAIN` — remove `"dark"`. Make `classify` return `''` (unknown) for `dark area`.
      - In `action_io.rank_open`, demote unknown `' '` squares below known-passable glyphs `.#><+{`. Change the key's first element to `g == ' ' or (g in "|-" and kind != "open")`.
      - Predicted: MazeWalk-15 no-moves drop from 62% to under 10%, which frees ~60 steps per episode. Some of the 3 timeouts should convert.
   b. `action_io.arbitrate` orbit-break — never override toward a square whose glyph is `' '` or whose kind is not `open`. The back step out of a dead end is correct maze behaviour.
   c. `harness.modal_reflex` — when `[yn]` is open and `y` is not legal, emit `more` if it is legal (the chain has shown `more` escapes a modal prompt).
      - Also add `"Really attack"` to a **refusal** list, checked before `COMBAT`. Book `(pos, dir)` failed so the peaceful goblin's square is vetoed.
      - Predicted: removes the 69-step freeze on Quest-Easy/04. Primary gain is 0, because Quest still needs lava crossing, but it saves cost and steps.
   d. `Ledger.observe` — ignore any cursor with `y == 0` (the prompt line). Keep the last real position, and do not touch `visits` or `prev_pos`.
3. **Replace orbit-break with a BFS-to-nearest-unstood-frontier override** on the charted map (the r041 `route` rung). This is the lever that actually moves MazeWalk-15 and R3, because win ⇔ `>` charted. Predicted: MazeWalk-15 → ~5/5 and R3 → ~4/5, as on r041.
4. **Cost:** at 348k tok/ep, this candidate costs 20× r046 for about half its primary score. Any continuation should use the model only on ties, as the chain's reply-independence result shows.

## 6. Exhausted — do NOT repeat
- Advisory tables or prose for the single-call model (r029, and here): the model ignores visit counts and follows "passable" labels blindly.
- One-hop anti-orbit rules as a substitute for a frontier search: exact A→B→A is gone, but revisits are still 56–84%.
- HP-triggered retreat or disengagement on CB-Dark/QM. Pricing CB-Dark on 5 episodes: it is dice.
- Prompt or AGENT.md wording changes. `pickup` on the Quest spawn pile without a kick split (see the r046 diagnosis).
