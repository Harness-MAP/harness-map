# Diagnosis — minihack iter r052 / cand_r052_0_spatial-grounding (chart walker + tools/quest.py)

**Headline:** primary = **0.600** (24/40, se 0.077). Progression 0.575, survival 0.900, step_efficiency 0.357, cost **152k tok/ep**. The parent lineage sat at 15/40 (r041/r046/r048/r051), so this is **+9 episodes**, and every one of them comes from the new Quest module: QM 0/5 → 5/5, QE 0/5 → 4/5.

| Task | Result | Stage that answered |
|---|---|---|
| MazeWalk-9x9 / 15x15 | 5/5, 5/5 | `field:*`, 0 tokens |
| Corridor-R3 | 4/5 | `field:*`; run_03 timed out at 100 steps, all of them `field:frontier` |
| CorridorBattle-Dark | 1/5 (4 died) | `field:frontier` + `reflex:swing` |
| Quest-Medium | **5/5** | quest stages (71–81 steps) |
| Quest-Easy | **4/5** | quest stages; run_04 used 65 `quest:menu` steps and timed out |
| Boxoban-Hard / Medium | **0/5, 0/5** | `model:consult` on 492/500 steps |

**Binding constraint:** Boxoban is now the only big hole left: 10 episodes (0.25 primary) and **100% of the token bill** (~600k tok/ep × 10 ≈ 6.07M of 6.08M). The field is flat on a premapped, stairless level, so every step falls through to a VLM that random-walks the four cardinals. Classification: TASK-GAP (no push planner) plus VLM-CEILING (the consult cannot plan pushes). It is fixable in code.

## 1. What this harness does (mechanism)
`harness.py:GradientAgent.act` parses the frame with `tools/field.py:parse_tty_frame`. It then runs `STAGES = (prompt, orient, stairs, swing, cross, hold, kit, fetch, descend, consult)`, and the first stage that answers wins.

- **Chart and field.** `st_orient` folds the frame into `tools/chart.py:Chart` and computes `Chart.field(pos)`:
  - a BFS potential seeded on a *reachable* `>`;
  - failing that, seeded on reachable frontier squares;
  - failing that, `({}, None)`, which means the field is flat.
- **Quest stages.** `cross`, `hold`, `kit` and `fetch` are armed only when `{pickup, kick, zap} ⊆ legal`. They drive `tools/quest.py:Quest`:
  - detect the pile and kick it once;
  - pick up the horn first, then the wand, with a filler pickup for untypeable letters;
  - fire across the lava;
  - hold at the corridor mouth.
- **Consult.** `st_consult` is model-free on Quest (`quest:menu`, the least-trodden open edge). On every other task it calls the VLM with a least-trodden menu.
- **Where the model is still called.** Only when the field is flat on a non-Quest level. In this eval that happened on Boxoban alone. All other 30 episodes used 0 tokens.

## 2. What actually happened (causal chain, with evidence)

**Boxoban, 10/10 timeouts.**
- **Frame** (Hard run_00 step 0): premapped room. `#` is bars (the message reads `You cannot pass through the bars.`), `` ` `` is a boulder, `{` is a fountain/target, and the hero is at (35,16) with a boulder adjacent east.
- **Field.** No `>` exists and no frontier survives (every cell is drawn), so `Chart.field` returns flat after 1–2 `field:frontier` steps.
- **Consult.** `st_consult` → `ask` handles steps 1–99 with `model:consult`. Action histograms are near-uniform, e.g. run_00: east 32 / west 27 / south 21 / north 20.
- **Env outcomes over all 1000 Boxoban steps:**
  - 241 `You cannot pass through the bars.`
  - 68 `You try to move the boulder, but in vain.`
  - 40 `With great effort you move the boulder.`
  - 639 silent steps
  - This matches the r048 figures (225/64/32), so the consult behaves identically and makes no push progress.
- **Two separate defects:**
  - There is no sokoban mechanism at all.
  - The consult menu (`c["open"]`) still lists bar and wall moves, because `open` is not filtered by `Chart.edge`/`passable`. That is 24% of steps wasted on bumps.

**Corridor-R3 run_03, timeout.**
- The final frame shows `|....>.` at map row 7, x≈8: the `>` is **drawn**, in a partially seen room west of the hero.
- `Chart.field` seeds only on a `>` inside `reach(pos)`. The room's connection is not charted, so the kind stays `frontier` on all 100 steps.
- The walker orbits the eastern rooms instead (positions revisited 2–3×). It also hits a `The door resists!` at step 79.
- The drawn stairs never attract the frontier choice.

**CorridorBattle-Dark, 4/5 deaths.**
- The pattern is `field:frontier` east into the corridor, then 7–14 consecutive `reflex:swing` while HP drains (e.g. run_00: 13→11→9→3→3→1→dead).
- The chain has priced this axis as dice at n=5, and melee mechanisms as rerolls (see §6). It is not diagnosable at this n.

**Quest-Easy run_04, timeout.**
- step 2: `kick` → `quest:kick:south`, OK
- step 8: `f - a horn.`
- step 16: `g - an oak wand.`
- **Both letters are untypeable**: `LETTER_VERB` has no `f`/`g` verb. So `Quest.usable` returns None and `cross` never fires.
- No filler object was drawn, so the filler rule could not pre-burn `f`/`g`.
- The rest of the episode is 65 `quest:menu` moves beside the lava (hero at (28,11), `}` column at x=30).
- There is also one wasted `reflex:swing` into a peaceful goblin at step 11 (`Really attack the goblin?`), which `peaceful` then learns.
- NOTES.md predicted exactly this loss. It is 1 episode.

## 3. Failure breakdown (16 lost episodes)
| Category | Eps | Cause |
|---|---|---|
| Boxoban: flat field → VLM compass shuffle | 10 | No push planner; the consult menu offers bars (241 bumps) |
| CB-Dark melee deaths | 4 | Dice (chain-established) |
| R3: drawn `>` outside the charted reach | 1 | `Chart.field` ignores unreachable drawn stairs |
| QE: horn/wand on untypeable letters, no filler | 1 | Inventory-letter trap |

The Boxoban fix is worth 10× any other remaining lever, and it also removes essentially **all** of the cost.

## 4. Binding constraint — classification
- **TASK-GAP + VLM-CEILING (Boxoban).**
  - The harness has no representation of push-space.
  - The frozen VLM, with the full map in view, produced a coherent push sequence 0 times in 1000 steps on this lineage and on r048.
  - Re-wording the prompt is exhausted. The plan must be **pre-computed**.
  - Boxoban is fully observable and deterministic: the `map:` block decodes exactly. A solver therefore decides from the observation only, which is anti-cheat clean.
- **HARNESS-BUG (R3, minor):** `Chart.field`'s seed rule.
- **HARNESS-BUG (Boxoban, secondary):** the unfiltered consult menu.

## 5. Concrete changes to try (ranked by leverage)

**1. New `tools/sokoban.py` plus an `st_push` stage placed after `swing` and before `descend`.** *(Up to +10 eps ≈ +0.25 primary; cost 152k → ~0 tok/ep. Chain stub-rig evidence: one push-space A* scored 19/20, and 10/10 live at 0 model calls. Boxoban wins iff the optimal plan is ≤99 steps.)*
- **Arming.** Arm on the observation: boulders `` ` `` and targets `{` are drawn, the legal direction set is exactly the 4 cardinals, and there is no `>`. This mirrors how `Quest.armed` keys on the verb list. It leaves all 30 other episodes byte-identical.
- **Decoding.** Decode the `map:` block (row = cursor.y−1):
  - `#`, `|`, `-` and blank are walls. `#` is bars here; `Chart` already knows this via `bars`.
  - `` ` `` is a boulder, `{` is a target, `.` is floor.
- **Hero square.**
  - Treat it as floor.
  - **Never assume a boulder under `@`.** The chain found that this phantom-boulder bug scores 0/10 even with a correct solver.
  - If #targets < #boulders, the hidden target is under `@`.
- **Search.** Run A* or BFS over the state (hero-reachable region canonicalised, frozenset of boulders), with push moves only.
  - Prune dead squares on **walls only**: non-target corners.
  - The "unwinnable from drawn count" prune is REFUTED.
  - Cost = hero steps + pushes, because the step cap counts moves.
- **Execution.**
  - Compile the plan to a tape of cardinals and emit one per step with rung `push:tape:<i>`.
  - Re-plan if the observed boulder set diverges from the prediction, or on `in vain`.
- **State.** Clear the solver state and memo in `GradientAgent.reset()`. The agent object is reused across tasks per worker; the chain measured memo-clear as 2/10 → 10/10 on an adversarial order.
- **Budget.** Cap search at a node budget, e.g. 200k. If it fails, fall through to `st_consult` as today.
- **Verify:** Boxoban CSVs should show `push:*` rungs, 0 `cannot pass through the bars`, and `input_tokens`≈0.

**2. `tools/chart.py:Chart.field`: attract toward a drawn but unreachable `>`.** *(R3 run_03 → win; chain measured +2/−0 over 125 levels, 0 arms off R3.)*
- If some cell has `terrain == ">"` but is not in `region`, seed each frontier cell with potential `k·manhattan(seed, >)` (k≈1) instead of 0.
- Break ties in the level frame.
- Keep `kind = "frontier"`.
- Gate it off on Quest levels (`}` drawn), where the `>` is behind lava and the Quest stages own the approach.

**3. `harness.py:st_consult`: filter the menu by the chart.** Use `menu = [d for d in menu if self.chart.edge(pos, d) is not None] or menu` on **every** task, not only Quest. This is a fallback-quality fix for whatever Boxoban states the solver cannot solve: 24% of Boxoban steps are bar bumps. It is cheap, but it will not win Boxoban by itself.

**4. (Low, 1 ep) QE untypeable-letter trap.**
- When no filler object is drawn and the next free letter is untypeable, try a `drop`/re-pickup cycle to walk the letter pointer forward, if `drop` is legal.
- **Price this on the stub rig first.** NetHack may keep a dropped object's old invlet on re-pickup, in which case it is inert.
- Also: do not `reflex:swing` at an `o`/`G` on Quest before the first `Really attack` answer. The goblin is peaceful on this level. It is minor, costing 3 steps here.

**Predicted result if #1 and #2 land:** about 32–34/40 (0.80–0.85), with cost near 0. The residue would be CB-Dark dice plus the rare Boxoban plan longer than 99 steps.

## 6. Exhausted — do NOT repeat
- **CB-Dark melee/retreat/hold mechanisms.** The chain priced these repeatedly as dice rerolls: recoil +9/−10 and +8/−11 over 100 eps; HP-gated disengage is dose-response negative. Leave `st_swing` alone.
- **Prompting the VLM on Boxoban** (menus, least-trodden hints, priors prose). This produced 0 push plans across r041–r052.
- **Stage-table rewiring alone.** Four rewires landed on the identical 15/40 vector. Only new mechanisms (this candidate's `tools/quest.py`) moved it.
- **The Quest module itself.** It is at 9/10. Do not restructure it; the only residue is the letter trap in #4.
