# Diagnosis — minihack iter 46 / cand_r046_0_spatial-grounding (stage pipeline + Quest lava-crossing stage)

**Headline:** primary = 0.375, survival = 0.775, efficiency = 0.273, cost = 16.6k tok/ep.
Per task:
- Boxoban-Hard / Boxoban-Medium: 0 / 0
- MazeWalk-15x15 / MazeWalk-9x9: 1.0 / 1.0
- Corridor-R3: 0.8
- CorridorBattle-Dark: 0.2
- Quest-Easy / Quest-Medium: 0 / 0

This is **identical** to the parent r041 (0.375 / same vector). All 23 navigation and CB-Dark episodes replay r041's action sequence byte for byte. The new Quest stage fired in 10/10 Quest episodes and **never got an item it could use**: `apply`/`zap` were emitted 0 times.

**Binding constraint:** HARNESS-BUG in `tools/quest.py`. The stage's premise, that `pickup` on the horn square collects the horn, is false. The horn and wand lie in a **2-object pile** at spawn, and `pickup` on a pile is **silent** under NLE: no menu, no message, no inventory change. `Quest` never detects this, never kicks the pile apart, and marks the square `tried` forever.

## 1. What this harness does (mechanism)

`harness.py:LearnerAgent.act` runs the stages `STAGES = (prompt, orient, stairs, fight, quest, fetch, route, learner)` in order. The first stage that answers wins, and its rung name goes into `reasoning`.

- **`orient`** folds the frame into `tools/chart.py:Chart`, credits the TD learner, calls `Quest.observe`, and computes `route`.
- **`quest` and `fetch`** are new in this candidate. They are gated by `Quest.armed = "pickup" in legal and a } is charted`.
  - `Quest.step` fires `pickup` once, labelled "spawn". In practice this is wherever the hero stands when lava is first charted.
  - It also fires `pickup` on any charted `ITEM` glyph (`()/!?="[*$`). Each square is tried once.
  - Once `usable()` finds a horn or wand on a typeable letter, it fires `apply`/`zap` across the shortest cardinal lava run.
- **`fetch`** walks to uncollected `ITEM` squares, then to lava banks.
- **`route`** is BFS to `>`, or to the nearest unstood frontier.
- **`learner`** ranks moves by TD value + optimism and consults the VLM once per tied square.

AGENT.md holds one tie-break sentence, and rules.md and memory.md are empty. The VLM is used only by `learner:consult`.

## 2. What actually happened (causal chain, with evidence)

CSV row = (action, observation AFTER it).

**Quest-Medium, all 5 episodes, with the same trajectory each time.**
- `armed` only becomes true at **step 7**, when the `}` column first enters the chart.
- So `quest:pickup:spawn` fires on a corridor square: `There is nothing here to pick up.`
- `fetch` then walks 5 hops back to the `(` at the spawn room (x=22, y=12), steps 8–12.
- Arriving on it prints **no `You see here …`**. That is the tell for a pile.
- `quest:pickup:item` at step 13 gets an **empty message**. The frame is unchanged, and after the hero leaves at step 14 the `(` is still drawn (`horn adjacent west`).
- The square is now in `Quest.tried`, so `targets()` drops it, and the stage is done for the episode.
- `route` takes over and walks east to the rat room.

**Quest-Easy, 5/5.** In runs 01/02/03 the hero spawns on the pile, and `quest:pickup:spawn` at step 0 gives an empty message. At step 1 the prose reads `horn adjacent south` and the map shows `(` on the square just left (run_01, frames 0–1).
- The ITEM set then sends `fetch` to the **wrong objects**:
  - run_00: `f - a sky blue potion`
  - run_02: `j - 2 darts`, then `j - a dart`
  - run_03: `i - a crude dagger`
  - run_04: `f - a fizzy potion`
- None is a horn or wand, so `usable()` stays None. These pickups also **burn the low free letters** that a later horn or wand would need (`f` is not typeable).
- After that, `route` is None on the lava-locked bank. `learner` orbits the left pocket for about 90 steps, with 71–79 `learner:argmax` and 18–19 `learner:consult` per episode.
- That orbit is **66–80k tokens/episode**, about 44% of the whole run's token bill, and it buys nothing.
- run_04 also spends 12 steps on `Really attack the goblin? [yn]` → `more`. This is the known `[yn]` 2-cycle.

**Quest-Medium deaths, 5/5.** This is why survival stays at 0.775. In QM run_00:
- The hero reaches the 1-wide corridor mouth (x=31, y=13) at step 21 and kills the first rat there (steps 23–24).
- At step 25, with no rat *adjacent*, `st_route` (`chart:route:frontier`) steps **east into the room**, to x=32.
- There two or three rats bite per turn: HP 13 → 9 → 7 → 4 → 1, and the hero dies at step 33.

Runs 01–04 die the same way at steps 29–42. This matches the chain's prior finding that holding the mouth at x=31 wins and advancing into the room dies. `fight` only answers when a foe is adjacent, so on every rat-free turn between kills the explorer advances.

**Everything else is byte-identical to r041:**
- 23/40 episodes replay the same action sequence.
- Boxoban differs only through temp-1.0 consults.

So the stage split changed nothing outside Quest, which is the intended outcome. But it also means nothing outside Quest moved.

## 3. Failure breakdown (40 eps)

| Bucket | Eps | Cause |
|---|---|---|
| Quest pile never split | 10 | silent `pickup` on 2-object pile; no kick; `tried` blocks retry — 0 `apply`/`zap` emitted |
| └ wrong item taken | 4 (QE) | `ITEM` includes `!` `)`; potion/darts/dagger consume letters |
| └ QM death at the room mouth | 5 (QM) | `route:frontier` advances x=31→32 between swings |
| Boxoban | 10 | no Sokoban planner; learner + consult wander (25–52k tok/ep) |
| CB-Dark deaths | 4 | same as r041 (melee dice; chain says price only at ≥200–500 eps) |
| R3/03 timeout | 1 | same as r041 |

## 4. Binding constraint

**HARNESS-BUG, in `tools/quest.py`.** It shows up two ways, both fixable in code.

**(a) The acquisition layer assumes that `pickup` works on the spawn square.** The chain has already measured the fix, the kick opener, and replicated it on two lineages: Quest-Medium 0/5 → 3–5/5, Quest-Easy 2/5. The mechanism was ported without it.

**(b) The QM fight has no hold.** Without one, a crossing would not help: the hero dies in the rat room before it ever reaches the lava.

It is not VLM-CEILING, because the VLM never touches Quest decisions except the orbit consults.

## 5. Concrete changes to try (ranked by leverage)

**1. `tools/quest.py`: kick the pile apart, once, away from lava.** *(Up to +5 QM, +2 QE; the chain has measured 7/10 on these seeds.)*
- **Detect a pile.** A `pickup` whose next message is empty, contains no `x - …` assignment, and is not `nothing here` means this square is a **pile**. Also treat arrival on a square charted `(`/`/` with no `You see here` as a pile. Set `self.pile = pos`. Do not put it in `tried`.
- **Kick it.**
  - Step off to a neighbour `n` whose ray *from n through the pile* hits a wall before any `}`. Use the same walk as `crossing()`, but require no `LETHAL` before `-|`.
  - Emit `kick`, and set `self.aim` to the bearing n→pile. `answer()` already returns `self.aim` on `In what direction?`.
  - **Kick exactly once.** A second kick along the same ray re-piles both objects at the wall.
  - Kicking toward the lava burns the horn (`It burns up!`).
- **Pick up both squares separately.** After the kick, add the pile square and the landing square (any newly drawn `(`/`/`) to targets.
  - `fetch` + `pickup` must reach `x - a horn.` / `x - a … wand.`
  - Keep a **stood-on ledger**, not just a picked-up ledger. A kicked object can land out of sight, leaving a stale glyph.
- **Arm on the verb list, not on `}` being charted:** `{"zap","pickup","kick"} ⊆ legal`. This is true only on the Quest pair. It lets the opener run from step 0 on QM too, instead of step 7 in a corridor.

**2. `tools/quest.py:ITEM`: restrict it to `(` and `/`.** *(QE letter arithmetic.)* Potions, darts and daggers never help and eat letters, so drop `!`, `)`, `?`, `=`, `"`, `[`, `*` and `$`.
- If the first tool lands on an untypeable letter (`f`, `g`, `i`, …), the second pickup gets the next letter. Pick up **both** tools before firing, and let `usable()` choose whichever is typeable. It already reads letters off `x - …` messages and checks them against the `[..]` prompt.

**3. Add a mouth hold for the rat room: `harness.py:st_fight`, or a new `st_hold` between `fight` and `quest`.** *(QM survival 0/5 → up to 3–5/5; required for change 1 to score on QM.)*
- Arm it when a hostile letter is drawn within about 3 squares, or named `very near`/`near` in the prose, and the hero stands on a square with ≤ 2 passable neighbours: a corridor or door mouth.
- In that case emit `search` (a free wait, and it is in the Quest vocabulary) instead of letting `route` advance, and keep swinging whatever is adjacent.
- Release it once no creature is drawn within the radius.
- This is a hold, not a retreat. It is the one melee form the chain has measured positive: hold the 1-wide mouth; advancing dies at step ~39 and holding wins at 68. Do not add HP-triggered retreat.
- Gate it on the Quest verb set, so CB-Dark stays byte-identical until it is priced separately.

**4. After a crossing, stop firing and route.** Set `crossed` on `The lava cools and solidifies.`, which `observe` already does via `solidif`. Keep ZAP_CAP at 1–2 per lava column: the chain measured ZAP_CAP=4 wasting 15 steps.

**5. Cost: stop the Quest orbit consults.** When `Quest.armed` and `route is None`, have `st_learner` skip `consult` and just take argmax. That cuts about 70k tokens on each QE episode, roughly 40% of the run's tokens, without changing primary: the consult never produced a winning move. For Boxoban, the chain's measured lever is a push-space A* (`tools/sokoban.py`, 19/20 on the stub rig); it is still untried on this lineage.

**6. `[yn]` answered with `more` is a 2-cycle** (QE run_04, 12 steps). Keep the peaceful-veto: after a `Really attack` prompt, `st_fight` must skip that glyph. It already adds it to `peaceful`, but only when `last[2] == "swing"`; verify that `self.last_foe` is set on the swing that triggered the prompt.

**Validation.** Check the stage on the stub rig before live eval.
- Each QM episode must show `kick` → `quest:aim:*`, then two `x - …` messages, then `apply`/`zap` → `solidifies`.
- Every non-Quest CSV must stay byte-identical to r041.

## 6. Exhausted — do NOT repeat

- **Stage-table / module rewiring by itself** (r026, r036–r046). The r046 split is exactly neutral outside Quest, on 23/23 byte-identical eps. Structure is not a lever; the mechanism inside the stages is.
- **`pickup` without a pile split.** It is silent on 10/10 here, and on every earlier lineage that tried it.
- **Multiple kicks along one ray** (re-piles), and **kicks toward lava** (burns the horn).
- **Prompt, AGENT.md or tie-break wording.** The VLM decides no winning move.
- **Frontier re-ranking on MazeWalk/R3.** They are saturated at 10/10 and 4/5.
- **HP-triggered retreat or disengagement on CB-Dark/QM.** Deaths rise with the dose. Change 3 is a hold, which is different.
- **Pricing CB-Dark on 5 eps.** It is dice. Any CB mechanism needs ≥ 200 eps.
