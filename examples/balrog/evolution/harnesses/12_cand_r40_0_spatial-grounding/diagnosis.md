# Diagnosis — minihack iter 40 / cand_r040_0_spatial-grounding
**Headline:** primary = **0.25** (10/40). MazeWalk-9x9 5/5 and MazeWalk-15x15 5/5. Every other task is 0/5.
- **MazeWalk is fully solved and costs nothing.** All 10 wins are 100% `field:descend` with 0 model tokens: MW15 in 2–70 steps, MW9 in 2–18 steps.
- **Corridor-R3, CB-Dark, Quest-Easy and Quest-Medium are 0/20.** These losses come from **a single parser bug**:
  - `tools/grid.py:parse_glyph_map` classifies every prose `"<noun> adjacent <dirs>"` line as a creature ("foe") unless the noun is in a short exact-match exclusion list.
  - `horizontal wall`, `vertical wall`, `southeast corner`, `northeast room corner` and items such as `horn` and `sky blue potion` therefore all become foes.
  - `orchestrator.py:GroundedAgent.reflex` rung 3 (`reflex:swing`) then attacks the wall **every step for 100 steps**.

**Binding constraint: HARNESS-BUG.** A wall/item noun misread as a creature pins the agent into an infinite swing that no veto can override. It is fixable with a one-line change to the foe classifier.

## 1. What this harness does (mechanism)
`orchestrator.py:GroundedAgent.act` runs these steps each turn:
1. `fold_chart` merges the `map:` rows into a per-Dlvl chart.
2. `parse_glyph_map` produces `pos`, `ring` (the 8 neighbours classified by map glyph), `lava` and `foes` (both from the prose lines), and `down`.
3. `learn` books a `(square, bearing)` refusal after a no-move.
4. `flood_field` does a reverse relaxation from `>`, or else from unstood open squares that touch blank rock. Squares that have been visited cost more.
5. `reflex` walks the rungs in order: modal prompt, then `down` on `>`, then **swing at any `d in frame["foes"]` or ring kind `creature`**, then `field:descend`, then dead-end `search`.
6. Only if no rung fires does the model get called, with the grounded ring plus priors. `vet` then projects its reply onto a legal action and applies the hard and soft vetoes.

The rung order is the whole policy. Rung 3 comes before the field, and **rung 3 checks neither `self.refused` nor whether the swing did anything.**

## 2. What actually happened (causal chain, with evidence)
**Corridor-R3 run_00, step 0.** The prose contains `horizontal wall adjacent north, northeast, and northwest`, and the map shows `------` above `@` at (3,4).
- In `parse_glyph_map`, the noun `horizontal wall` is not in the exclusion tuple, which only has the exact string `"wall"`.
- The noun is therefore recorded as `foes = {north, northeast, northwest}`.
- `reflex` returns `north` with reason `reflex:swing`. The env replies `It's a wall.` and the agent does not move.

`learn` does book `((3,4), north)` as refused, because the map kind is `wall`, not `creature`. But the swing rung never consults `refused`. The same frame repeats for 100/100 steps, and the episode return is −1.00: each step is a no-op that incurs the step penalty.

Census of the nouns that triggered `reflex:swing` across all traces:

| Task | Swing steps | Nouns that fired it |
|---|---|---|
| Corridor-R3 | 499/500 | vertical wall 395, horizontal wall 198, northeast room corner 99 |
| CB-Dark | 499/500 | horizontal wall 494, vertical wall 396, southeast corner 297, northeast room corner 99 |
| Quest-Medium | 495/500 | vertical wall 490, **horn** 490 (the item pile at spawn) |
| Quest-Easy | 418/500 | **horn** 345, horizontal wall 288, sky blue potion 46, fizzy potion 13; real monsters only 13 (grid bug 8, goblin 3, fox 2) |

Only 13 of these triggers were real creatures.

- **Every R3 and CB-Dark episode is exactly one action repeated 100 times:** `north`×100 or `east`×100.
- **Quest-Easy run_00** alternates `east`/`west` 52/48. The items are on both sides.
- **Quest-Easy run_04** is the only episode that escapes into the model path (76 model steps). It still dies at the cap.
- **MazeWalk never triggers the bug.** Its prose uses bare nouns (`wall`, `dark area`) that are in the exclusion list, so the field owns 100% of its steps and solves 10/10.
- **Boxoban does not trigger it either.** Boxoban never reaches a reflex: the field is empty on a premapped level and the level has no `>`. It goes to the model on 70–93% of steps at about 620k input tokens per episode (the entire token bill of this candidate), and scores 0/10. That is expected, because a model walking Boxoban cannot solve it.

## 3. Failure breakdown (30 losses)
- **20/30: infinite swing into a wall or item noun.** This covers R3 5, CB-Dark 5, Quest-Medium 5 and Quest-Easy 4–5. It is a harness bug and fully explains why these four tasks went to zero.
- **10/30: Boxoban.** The model walks and there is no push planner. This is a task gap, and it also accounts for the cost.
- **0 deaths recorded as dying.** All losses hit the 100-step cap. Survival is inflated only because the agent never moves.

The field mechanism itself is not refuted on R3, CB-Dark or Quest. It never got a single step there: `field:descend` appears exactly once per episode, on the step before the swing fires.

## 4. Binding constraint
**HARNESS-BUG.** The foe detector uses a **blacklist of prose nouns**, while NetHack's language observation uses an open vocabulary for walls, corners and items. The two consumers of the result compound it:
- `reflex` swings unconditionally.
- `learn`'s refusal is ignored by the swing rung.

The result is an absorbing state. This is the same failure class as the known "a creature is a ranking term, never a refusal" lesson and the "`passable` conflates terrain and occupancy" lesson: a noun was trusted over the glyph.

## 5. Concrete changes to try (ranked by leverage)
1. **`tools/grid.py:parse_glyph_map`: a foe must be a LETTER glyph on the map, not a prose noun.**
   - Delete the `else: frame["foes"][d] = noun` branch.
   - Keep the prose loop only for `lava`/`water`.
   - After computing `ring`, set `frame["foes"] = {d: ch for d, (k, ch, _) in ring.items() if k == "creature"}`. That uses the existing `ch.isalpha() and ch != "@"` test.
   - If you want to keep prose, add a whitelist instead: accept a noun only if the ring glyph at that bearing `isalpha()`.
   - **Predicted effect:** R3, CB-Dark and Quest leave the swing trap and fall through to `field:descend`, which is the same chart-plus-field that solved MW 10/10. The chain's chart-plus-route candidates (r001, r036) hit R3 at about 4/5. **Expected primary is about 0.35–0.45** (MW 10 + R3 3–4 + CB-Dark 1–2), up from 0.25.
2. **`orchestrator.py:reflex` rung 3: swing only if the previous swing was not a no-op into a non-letter.**
   - Skip `d` if `(pos, d) in self.refused`.
   - Also skip it if the last message was `It's a wall.`/`It's solid stone.`
   - This is a belt-and-braces guard so that any future misclassification costs one step, not 100. Refusals already book correctly for walls because `learn` uses the map kind.
3. **`orchestrator.py:reflex` rung 3 on Quest: never swing at a lava-adjacent or item glyph.** Quest's `horn`, potions and wands are `(`, `!` and `/`, which are not letters. Fix 1 therefore handles this automatically. If items are ever re-added from prose, route them through a pickup rung, not the swing rung. The chain's memory says Quest needs pickup, then `apply`/`zap` toward the lava; that is a separate, later lever.
4. **Boxoban cost (secondary key):** the model is called on about 85% of Boxoban steps at about 7k input tokens per call for zero wins.
   - When `any(ch == "`" for ch in terrain.values())` and no `>` is drawn, a cheap model-free rung makes cost drop by about 10× without changing primary. Examples: the least-visited ring neighbour, or a push-space A* toward `{`/`^` targets. Prior chain work reached 19/20 on Boxoban with a push-space A*.
   - This is the biggest remaining *cost* lever once primary is recovered.

## 6. Exhausted — do NOT repeat
- **Prose/AGENT.md wording:** it is inert, since the code decides 100% of steps on 6/8 tasks.
- **Re-plumbing model-first vs chart-first without a new mechanism:** r026/r027/r036–r038 all scored 0.325–0.375.
- **Rebuilding the chart or frontier from scratch:** r039 regressed to a 2-cycle this way. This candidate's `flood_field` with visit-weighted cost has **no** orbit on MazeWalk (10/10), so **keep it and fix only the foe classifier**.
- **Pricing CB-Dark tweaks at n=5.** The result is dice.
