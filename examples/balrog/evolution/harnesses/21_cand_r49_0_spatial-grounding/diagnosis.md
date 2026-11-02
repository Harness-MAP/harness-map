# Diagnosis — minihack iter 049 / cand_r049_0_spatial-grounding (Chart BFS walker in front of the model-plus-vet path)

**Headline:** primary = **0.325** (13/40, se 0.074). Survival 0.825, step_efficiency 0.235, cost **279k tokens per episode**.
Per task:
- MazeWalk-9x9 5/5 and MazeWalk-15x15 5/5, both at 0 tokens.
- Corridor-R3 2/5.
- CB-Dark 1/5.
- Boxoban-M/H 0/10, Quest-E 0/5, Quest-M 0/5 (4/5 deaths).

That beats the r033 parent (0.175), mainly because MW15 went from 2/5 to 5/5. It is still below the chain elites (r036/r041/r044/r046/r048), which score 0.375 with R3 at 4/5.

**Binding constraint: a HARNESS-BUG in `tools/grid.py:parse_glyph_map`.** Every prose noun that is not on a short whitelist gets booked as a *foe*. That includes `vertical wall`, `horizontal wall`, `southeast corner`, `vertical open door`, `horn`, `dart` and `giant rat corpse`. `GroundedAgent.walk` then returns None whenever the hero is next to any wall. So the new chart walker is switched off in every room and doorway, and the model plus the old vet path own those steps, including the vet's "`-`/`|` = wall" door bug. After that fix, the remaining 20 losses are the known Boxoban and Quest TASK-GAP.

## 1. What this harness does (mechanism)
`agent.py:GroundedAgent.act` runs these steps in order:
1. `parse_glyph_map`: finds the cursor, classifies the 8 neighbours from the map glyphs, and parses the `X adjacent <dirs>` prose into `lava` and `foes`.
2. `learn`: updates visits and refused edges.
3. `Chart.fold`: folds terrain for the current Dlvl and books the verdict on `pending` (door opens, stone, or blocked edge).
4. **`walk`**: code-owned, 0 tokens.
   - It returns None if `pos is None`, **if `frame["foes"]` is non-empty**, or if a ring glyph is a letter.
   - Otherwise, when standing on `>`, it emits `down`.
   - Otherwise `Chart.plan` does a BFS: `chart:route` to a reachable `>`, else `chart:seek` (the frontier nearest a drawn but unreachable `>`), else `chart:explore` (the nearest frontier).
5. Otherwise the model gets the stock prompt plus the AGENT.md/rules/skills priors plus the `ground()` ring listing. `vet` then projects and replaces the reply: it forbids lava, ring `wall`, refused edges and stale back-steps, and falls back to the ring sorted by `score`.

Step owners (Reasoning column, 40 episodes):

| task | model | grounded/fallback (vet) | chart:* |
|---|---|---|---|
| MazeWalk 9x9 + 15x15 | 0 | 0 | **165 (100%)** |
| Corridor-R3 | 173 | 76 | 197 |
| CB-Dark | 230 | 35 | 33 |
| Quest-Medium | 157 | 10 | 41 |
| Quest-Easy | 455 | 45 | 0 |
| Boxoban-M/H | 790 | 210 | 0 |

## 2. What actually happened (causal chains; CSV row = action_i, obs AFTER it)

### A. False foes switch off the chart next to every wall, and R3 orbits between two owners (R3/03, R3/01, R3/04)
I replayed `parse_glyph_map` on the observation before each non-chart step. **Every** R3 model or vet step (245/245, excluding step 0) had a non-empty `frame["foes"]`, and none had a real creature. The top "foe" sets were:
- `vertical wall` 65
- `vertical open door` 47
- `horizontal wall` 32
- `horizontal wall, southeast corner, vertical wall` 18
- `vertical closed door` 7
- `... room corner ...` about 30

The cause is `grid.py:parse_glyph_map`: the exclusion tuple only contains the bare nouns `"wall"` and `"door"`. NLE prose says `vertical wall`, `horizontal wall`, `northwest room corner`, `vertical open door`, `vertical closed door` and so on, so all of them fall into `frame["foes"]`.

**R3/03** (22 distinct squares in 100 steps):
- The chart heads for the frontier at the room's NW corner (20,3), where a door is drawn `-` at (19,3).
- As soon as the hero stands next to a wall, `walk` returns None, and the model says `south` or `east`.
- On the next open-floor square the chart takes over again with `north`.
- Rows 12–98 are one period-3 to period-5 orbit: `north/chart:explore → south/model → south/model → east/model → north/chart:explore`, 19 laps.
- At row 38 `The door opens.` fires at (19,3). The chart would take that door, but every frame at (20,3) is a "foe" frame, so it never gets the turn.
- `ground()` also prints `vertical wall (attack by moving west)` to the model, because the foe label replaces the kind. The prompt literally tells the model to attack walls.

**R3/01:**
- Rows 37–61 are a clean `chart:explore` run to the same NW door (20,3).
- Rows 62–69: the model emits `open`, the game asks `In what direction?`, and on the next step the model emits `open` again (`What a strange direction! Never mind.`). That is 8 wasted steps.
- Row 70 `west`: `The door opens.`
- Rows 71–99 are a strict 2-cycle:
  - At (21,3), `chart:route` says `west` toward the now-reachable `>`.
  - At (20,3), the wall "foes" hand the step to the model. `vet` forbids `west`, because `ring_of_eight` classifies the opened door's `-` glyph as `wall`, and falls back to `east`.
  - Also mixed in: 4× `It's solid stone.` and 2× `It's a wall.` bumps chosen by the vet.
- The `>` was reachable through that door for 29 steps. This is **r027's "open door drawn `-` = wall" bug, surviving in the vet path**. `Chart.passable` knows about `opened`, but `GroundedAgent.forbidden` never asks the chart.

### B. Quest-Medium: `chart:seek` walks straight into the rat room (4/5 deaths). This is the known TASK-GAP.
**QM/00:**
- Rows 7–11 are `chart:seek` east toward the drawn `>` beyond the lava.
- From row 12 on, the rats are adjacent, so the model owns the steps and swings `east` 11 times.
- HP goes 16→13→10→6→5→1, and the hero dies at row 23.
- The seek attractor is a lure here: the `>` sits behind `}`, and the only crossing is the horn or wand from the spawn room. (The parse also books `horn` as a foe in QE/QM: 56+36+34 frames.)

### C. Quest-Easy (0/5, about 455k tokens per episode) and Boxoban (0/10, about 620k tokens per episode): the chart has nothing to plan
- **QE:** 140 of the model frames are `plan None`. The lit room is fully drawn and the `>` is behind lava, so there is no frontier and no reachable `>`. The rest are false-foe frames (`horn`, `dart`, `orcish dagger`, walls).
- **Boxoban:** the board is premapped with no `>`, so 1000/1000 steps go to model or vet. That is about 6.2M of the 11.1M total tokens (56%).

### D. CB-Dark (1/5, 3 deaths): melee dice, as in every lineage
`giant rat corpse` is booked as a foe too (the chain law says a corpse is a phantom foe and the current-frame glyph separates), which hands corridor steps to the model.

## 3. Failure breakdown (27 lost episodes)
| mode | eps | owner | class |
|---|---|---|---|
| Boxoban: no push planner; model or vet orbits | 10 | model/vet | TASK-GAP |
| Quest-E: sealed by lava, plan None, model wanders | 5 | model | TASK-GAP |
| Quest-M: seek lures into the rat room, model swings until dead | 4+1 timeout | chart:seek then model | TASK-GAP (+ lure) |
| R3 timeout: wall/door nouns are foes → chart/model orbit; vet treats the opened `-` door as a wall | 3 | model/vet | **HARNESS-BUG** |
| CB-Dark melee death or timeout | 4 | model | dice |

## 4. Binding constraint
**HARNESS-BUG first** (R3 3 eps, plus cost everywhere). The chart this candidate added is correct: MW is 10/10 at 0 tokens, and R3/02 reaches the `>` by `chart:seek`/`chart:route`. But the foe parser gates the chart off in every room, so R3 reverts to the r033 model orbit.

After the fix, the binding constraint is the same **TASK-GAP** as r048: Boxoban has no Sokoban planner and Quest has no acquire-and-freeze routine, which is 20 of 40 episodes. Neither is a VLM ceiling; both are solved model-free in sibling lineages.

## 5. Concrete changes (ranked by leverage)

### 1. `tools/grid.py:parse_glyph_map`: foes come from the GLYPH, never the prose noun. Expected R3 2/5 → 4/5 (+0.05), R3 tokens −70%, and fewer model steps on CB/QM.
- Drop the prose→foes branch entirely. Define `frame["foes"] = {d: ch for d, (kind, ch, _) in ring.items() if kind == "creature"}`, where the ring already marks a letter glyph as a creature.
- Keep the prose parse **only** for `lava`/`water`.
- If you want a prose fallback for dark squares, use a positive rule: accept a noun only if it has no suffix `wall|corner|door|doorway|corridor|area|floor|bars|boulder|stairs|fountain|corpse`, **and** the ring glyph in that direction is a letter.
- Predicted effect: on the 245 R3 non-chart frames, `walk` would have answered on all of them, since the census found 0 real creatures. That removes the R3/03 orbit, and R3/01 goes through the opened door at row 71 (29 steps before the cap).
- Also fix `ground()`: never relabel a non-creature as `(attack by moving ...)`.

### 2. `agent.py:forbidden` / `vet` must ask the chart, not the ring glyph
- Replace `kind in ("lava","wall")` with `not self.chart.passable(target, self.bars)`. The ring may still veto lava, but a square in `chart.opened` must be allowed.
- Keep the diagonal-door rule: reuse `self.chart.step_ok(pos, d, self.bars) is None` as the veto.
- Predicted effect: R3/01's 29-step `east/west` 2-cycle is gone even if a real creature hands the turn to the model at a doorway. It also stops vet-chosen `solid stone` bumps (R3/01 rows 75–90).

### 3. `agent.py:act`: never send `open` to the model; answer `In what direction?` in code
- If the message contains `In what direction?`, emit the bearing of the adjacent `+` (from `chart.terrain`), else `esc`.
- Better still, remove `open`/`close` from what the model may emit (project `open` to the `+` bearing). Bumping a door auto-opens it, and the chart already books `The door opens.`
- This saves the 8 wasted steps in R3/01 rows 62–69.

### 4. Port the two TASK-GAP levers (the real ceiling: up to +20 eps). Specs are in the r048 diagnosis §5.1–5.2.
- **`tools/sokoban.py` + a `board` step before the model.**
  - Gate: a `` ` `` boulder and a `{` are drawn and no `>` is charted.
  - Planner: push-space A* with cardinal moves only, `#` = bars = wall, and dead squares pruned on walls only.
  - Replay, and re-plan on `in vain` or when the boulder set differs from the prediction.
  - Working implementations to lift: `runs/adaptive_k16_crn/candidates/prop_r014/cand_r014_0_spatial-grounding/tools/sokoban.py` and `runs/adaptive_k2_crn/.../prop_r017/*/tools/sokoban.py`.
  - This also takes ~55% of the token bill off the model.
- **Quest routine.**
  - Gate: `}` charted and no reachable `>`.
  - Steps: pick up at the spawn item square; on a silent pickup, step off and kick ONCE, **not toward lava**; then `apply` the horn or `zap` the wand **across** the lava on the shortest cardinal run.
  - Reference: `runs/adaptive_k8_c1/quest_opener_k8c1.py`.
  - **Gate `chart:seek` off while `}` is charted and the item is not used yet.** In QM, seek is what walks the hero into the rat room (QM/00 rows 7–11).

### 5. Seek pull tie-break (small)
`Chart.plan.key` puts `pull` first, which can prefer a far frontier near the `>` over a 1-step one. That is correct for R3/02 (a win). Keep it, but leave it gated off when `}` is charted (see #4).

## 6. Exhausted — do NOT repeat
- Prose, priors or menu rewording aimed at the model (AGENT.md/rules/skills/memory.md). The model is net harmful on every residue step: it attacks walls it was told are foes, swings into rat packs, and orbits Boxoban.
- More navigation re-ranking. The MW 10/10 and R3 4/5 elite ceiling is already reached by r036–r048. What costs R3 here is the foe gate plus the vet's door-as-wall, not the BFS order.
- Bars-as-floor for Boxoban cost (0/10 either way); Quest pickup without pile detection; a double kick or a kick toward lava.
- Melee tweaks priced at n=5 CB-Dark (HP retreat, disengage, hold). The chain calls these dice; they need 200 or more episodes.
- Treating a creature as a refusal: creature = ranking term, never a veto.
