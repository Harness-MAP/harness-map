# Diagnosis — minihack iter r035 / cand_r035_0_spatial-grounding (in-episode value learner)

**Headline:** primary = **0.150** (6/40). MazeWalk-9x9 5/5, Corridor-R3 1/5, and every other task 0/5.
MazeWalk-15x15 is 0/5, where sibling chart+route harnesses in this run score 5/5.
Survival is 33/40: CB-Dark lost 4/5 to death and Quest-Medium 3/5.
**29% of all move emissions do not move the hero** (876/3052). **Binding constraint:** the value
table starts every untried (square, move) pair at the same optimistic value and **never reads the
glyph on the target square**. It spends about half of each dark maze bumping into blank stone the
frame had already drawn as impassable. It has no route to a drawn `>`. The failure is
**HARNESS-BUG**: the observation predicts 98.5% of the bumps, and the model is not involved.

> Trace alignment note: a BALROG CSV row is (action, obs AFTER it). All numbers below pair
> action *i* with the frame of row *i−1* (the frame the agent decided from).

## 1. What this harness does (mechanism)
`orchestrator.py:LearnerAgent.act`, one module, no tools:
1. **Reflexes.** If `fr["prompt"]` is set, emit `y` for `[yn]`+attack, `more` for `--More--`, or `esc`
   for anything else. On a remembered `>`, emit `down`.
2. **`credit()`** computes a TD(λ) update over `(square, move)` values. The rewards are +0.05 per new drawn
   cell, +0.5 for an unstood square, −1 for a bump (which also pins that pair to −1), and +0.4 per
   Chebyshev step closer to `>`.
3. **`deadly()`** vetoes lava/water named in the prose and any `}` next to the hero.
4. **`foe_bearing()`** swings at any adjacent letter except `@`.
5. **Ranking.** Moves are ranked by `value + 0.6/sqrt(1+tries)`. Every untried pair scores 1.0 + 0.6 = 1.6,
   **whatever the glyph on that square is**. Ties within 0.05 → `consult()` asks the VLM once per
   square, and the reply is projected onto the tied set.

The constants `WALL = set("|-")` and `TAME` exist, but **`WALL` is never referenced**.
`glyph()` is used only for `}` and creature letters.

## 2. What actually happened (causal chains, with evidence)

**A. Blind wall-probing (the dominant failure, all tasks).** A newly entered square has up to 8 untried
pairs, and all of them score 1.6. In a corridor, 5–6 of those bearings are blank stone. The optimism
bonus makes the learner try each one once before the −1 pin removes it. The same probe repeats on
every new square.

| task | moves | no-move | blocked target glyph (blocked / moved) |
|---|---|---|---|
| MazeWalk-15x15 | 495 | **243 (49%)** | blank `' '` 243 / 2 |
| MazeWalk-9x9 | 83 | 32 (39%) | blank 32 / 0 |
| Corridor-R3 | 462 | 121 (26%) | blank 65/2, `-` 35/0, `|` 18/0, `+` 3 |
| CB-Dark | 200 | 83 | blank 31, `-`/`|` 19, `r` 33 (swings) |
| Quest-Medium | 327 | 102 | `-`/`|` 46, blank 20, `r` 36 (swings) |
| Quest-Easy | 495 | 51 | `-` 39 |
| Boxoban-H/M | 990 | 244 | `#` (bars) 191/2, boulder `` ` `` 53/57 |

Pooled over all tasks, a **blank target is blocked 397/403 (98.5%)**, a `-`/`|` target 157/159, and
a Boxoban `#` 191/193. The frame the agent decided from shows every one of these. Excluding swings,
roughly 800 of the 876 no-moves were predictable. MazeWalk-15x15 has only 12–26 unique squares per
100-step episode.
- Example, CB-Dark/00 steps 3 and 6: the hero is at (27,11), then (29,11), in a `#` corridor
  running east. North is blank, and the learner emits `north` and gets "It's solid stone." It does
  this once per new corridor square.
- Example, Quest-Medium/04 steps 18–21 at (31,12): `north`, then `south`, then `north` again. All
  three are blank or solid stone. The −1 pin covers only the one (square, move) pair, and every other
  blank bearing still reads 1.6.

**B. The TD trace punishes productive corridors.** A bump's delta is about −1 + 0.9·1.0 − 1.0 = −1.1.
It is multiplied by α=0.5 and pushed along the eligibility trace (γλ=0.72 per step). Each blind probe
therefore subtracts about 0.4 from the move that ENTERED the square. A corridor with 5 blind
bearings per square makes its own entry move look worse than an untried backward move (still 1.6).
The result is reversals and revisits: max 7–9 visits per square in MW15, R3, CB-Dark and QM.

**C. No route to a drawn `>`.** The goal signal is +0.4 per Chebyshev step closer, credited after the
fact. It does not plan through walls. The `>` was drawn in 3/5 R3 episodes and in MW15 runs 01/02,
but it was reached only in R3/04 (67 steps). It was never reached in MW15.

**D. The `[yn]` reflex can never fire, so a modal prompt swallows steps (Quest-Easy/04 steps 42–50).**
`foe_bearing` swings at a peaceful goblin `o`, which opens `Really attack the goblin? [yn] (n)`.
`fr["prompt"]` is True, but `y` and `esc` are **not in the MiniHack legal set**. `more` is gated on
`msg.endswith("--More--")`. No reflex matches, and the code falls through with `pos = (35, -1)`
(the cursor is on screen row 0). The learner then emits `east/north/south/west/...` for 9 steps.
The CSV shows 0 `reflex:confirm-attack`, 0 `reflex:escape` and 0 `reflex:more` across all 40
episodes. Per the known MiniHack fact, `more` is the one legal verb that closes both `[yn]` and item
prompts.

**E. Deaths (CB-Dark 4/5, Quest-Medium 3/5): walking into the pack.** In CB-Dark/00 steps 10–16, the
learner walks the corridor east, EXITS it into the lit room (`....r`), and is then flanked by 3 rats.
HP goes 16→3→dead within 4 swings. Quest-Medium/04 steps 22–30 are the same: the hero walks from the
corridor into a room with `rr` visible 2 squares ahead, then swings at them from the open floor.
Nothing in the ranking reads a drawn creature until it is adjacent. `consult` sent the hero east
444 of the 751 times it was asked (59%).

**F. The VLM consult is pure cost.** It made 751 calls, and each one is only a tie-break between moves the
learner already scored as equal. The pick distribution (east 444, north 210, south 81, west 10)
is a bearing prior, not spatial reading. The consults make up nearly all of the ~30–165k input
tokens per episode.

## 3. Failure breakdown
- **Map-predictable bumps: ~26% of all moves** (≈800/3052). MazeWalk-15x15 alone spends 49%, which
  explains its 0/5.
- **Revisits/reversals from the trace penalty** (B): these are most of the remaining MW15 and R3 waste.
- **A drawn `>` that is never routed to** (C): 4 episodes had it charted and did not win.
- **Modal `[yn]` step sink** (D): 9 steps in one episode, caused by the reflex set.
- **Pack deaths** (E): 7 episodes, on CB-Dark and QM.
- **Boxoban:** 0/10. A learner cannot solve Sokoban in 100 steps. The chain's 1.0 Boxoban harnesses
  use a push-space solver.
- **Quest:** 0/10, with no lava deaths, so the prose veto works. There is no lava-crossing mechanism
  (such as wand/horn/kick-the-pile).

## 4. Binding constraint
**HARNESS-BUG.** The failures come from the decision structure, not from perception. The target-square
glyph (blank, `-`, `|`, `#` bars) predicts ≥98% of the no-moves, yet it is never read. Credit
assignment then turns those bumps into penalties on good moves. A secondary **TASK-GAP** is the
missing route-to-`>` and the missing Boxoban/Quest mechanisms. The VLM is not the constraint: it is
consulted only on ties, and its answer does not change the outcome.

## 5. Concrete changes to try (ranked by leverage)
1. **A glyph-gated action set (`orchestrator.py:act`, before ranking).** Build
   `passable(fr, pos, d)` from the drawn target glyph. Drop any bearing whose target is blank `' '`
   (stone), `-`, `|`, or, on Boxoban, `#` (bars). Treat `+` as a door, since it may be locked: allow
   it and let the one-shot bump pin handle it. Keep the existing −1 pin as the fallback for anything
   the map misses. *Predicted:* MW15 no-move 49% → <5%, and MW15 0/5 → 3–5/5, because the
   same 100 steps now reach roughly 2× as many new squares. R3 gains about +1 as well. The causal reason:
   398/403 blank targets and 157/159 wall targets were blocked, so this removes about 26% of all wasted steps.
2. **Initialise values from the target square, not a constant (`value()`).** Give an unstood
   passable target 1.0 and an already-stood target 0.3. Do not use a flat `OPTIMISM` for both. This
   removes the backward-move tie with an unexplored forward move, which currently goes to the model.
   Also stop pushing the **bump** delta along the eligibility trace. A bump says nothing about the
   move that led here, so apply only the pin (`credit()`: skip the trace loop when
   `pos == s and kind != "swing"`). *Predicted:* reversals and max-visits drop, and corridors are
   followed to their end, which gives DFS-like exploration.
3. **Route to a drawn `>` (new `route(fr, target)` BFS over drawn passable glyphs).** When `>`
   appears in `fr["rows"]`, emit the first hop of the BFS path. Re-elect it every step and put it
   ahead of the learner. When there is no `>`, BFS to the nearest unstood passable square that
   borders blank (the frontier), ranked distance-ascending with a fixed bearing tiebreak. This is the
   mechanism the MW15/R3 1.0/0.8 harnesses in this run use. *Predicted:* R3 1/5 → 3–4/5, and
   MW15 converts every run where `>` is charted.
4. **Fix the prompt reflex (`act`, prompt branch).** Detect a prompt as `pos[1] < 0`
   (cursor on row 0) OR the current test. Emit `more` whenever it is legal, and do this before any
   `y`/`esc` branch, since neither is in the MiniHack legal set. Also stop `foe_bearing` from
   re-swinging at a square whose last swing produced `Really attack` (the target is peaceful, so skip
   it). *Predicted:* no more 9-step `[yn]` sinks on Quest-Easy. The survival effect is neutral.
5. **Don't leave the corridor when creatures are drawn (`foe_bearing`/ranking).** If any creature
   letter is drawn within 3 squares and the hero is on `#`/a doorway, penalise moves whose target
   square has more than 2 passable neighbours. The effect is to hold the corridor mouth rather than enter
   the room. Price this on ≥200 CB-Dark episodes. CB-Dark is high-variance, and the chain's CB-Dark
   has never exceeded 0.4. *Predicted:* fewer QM/CB-Dark flank deaths, which helps survival more than primary.
6. **Drop `consult()`; break ties deterministically (by the ORDER or distance to the frontier).** The consult
   sent the hero east 444 of 751 times and carries no spatial information. *Predicted:* cost
   falls by about 95% with no primary change. This matters if cost is a later lexicographic key.
7. **(Task gaps) Boxoban: a push-space A\* solver over the lit, fully drawn map** (Boxoban has no
   diagonals, and `#` is bars). **Quest: a lava-crossing mechanism** (pick up the pile item, then
   `apply`/`zap` it toward the `}`, or `kick` the spawn pile). These are the only routes to the 1.0
   rows the chain's top harnesses (r064/r068) score.

## 6. Exhausted — do NOT repeat
- A **memoryless or VLM-per-step policy** (anchor): 30% period-2 orbits, lava walking, and an illegal
  `y` coerced to `north`.
- **Optimistic tabular values without map priors** (this candidate). Optimism alone cannot stand in
  for the map, because it pays one bump per blind bearing per square.
- **Re-wording AGENT.md for the tie-break**: the model's pick is a bearing prior, and prose moves nothing.
- **The melee/retreat axis on CB-Dark** is dice at n=5. Do not select on the CB-Dark row alone.
