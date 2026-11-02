# Diagnosis — minihack iter r048 / cand_r048_0_spatial-grounding (potential-field walker as a STAGE table over tools/field + tools/chart)

**Headline:** primary = **0.375** (15/40, se 0.077). Progression 0.35, survival 0.775, step_efficiency 0.278, cost **207k tok/ep**.

| Task | Result | Stage that answered most steps |
|---|---|---|
| MazeWalk-9x9 | 5/5 | `field:*` (model-free) |
| MazeWalk-15x15 | 5/5 | `field:*` (model-free) |
| Corridor-R3 | 4/5 | `field:*` (model-free); run_03 timed out |
| CorridorBattle-Dark | 1/5 (4 died) | `field:frontier` + `reflex:swing` |
| Boxoban-Hard / Boxoban-Medium | 0/5, 0/5 | `model:consult` on 982/1000 steps, ~600k tok/ep |
| Quest-Easy | 0/5 (all timed out) | `model:consult` on 476/500 steps, ~420k tok/ep |
| Quest-Medium | 0/5 (all died by step 27) | frontier → consult → swing |

This is the same per-task vector as r041, r046 and r051: the lineage has a stable plateau at 15/40.

**Binding constraint:** the field goes flat on every Boxoban and Quest frame, and the flat case falls through to a model consult that only ever emits compass directions (0 `pickup`/`kick`/`apply`/`zap`/push-plans in 1000 steps). The missing 25 points are 10 Boxoban, 10 Quest, 4 CB-Dark (dice) and 1 R3. Of these, 21 need a **code mechanism for a regime the field cannot represent**: TASK-GAP plus HARNESS-BUG, not the navigation.

## 1. What this harness does (mechanism)
`agent.py:GradientAgent.act` parses the frame (`tools/field.py:parse_tty_frame`). It then runs `STAGES = (prompt, orient, stairs, swing, descend, consult)`; the first stage that answers wins.
- `st_prompt` answers every modal frame with `more`, or `esc` if `more` is not legal.
- `st_orient` does four things:
  - folds the frame into `tools/chart.py:Chart`;
  - bans lava and bump-blocked edges;
  - builds `c["open"]`, which is **all directions minus lethal/blocked edges. It is not filtered by `Chart.passable`**;
  - calls `Chart.field(pos)`.
- `Chart.field` builds a reverse shortest-path potential over the region reachable from the hero:
  - it seeds at 0 on a *reachable* `>`;
  - otherwise it seeds at 0 on every reachable unvisited cell that has an undrawn neighbour (`frontier`);
  - otherwise it returns `({}, None)` = flat.
- `st_swing` hits any adjacent letter. `st_descend` takes the argmin of the field plus 0.25·visits.
- `st_consult` runs only when the field is flat. It asks the VLM with the priors, plus a menu "Open moves, least trodden first", and projects the reply onto the legal set.

The prose (`AGENT.md`, `rules.md`) is seen only by `consult`, and `skills/` is empty.

## 2. What actually happened (causal chain, with evidence)

**MazeWalk and R3 (14/15 wins) are entirely model-free** (0 tokens). The frontier field is at the DFS optimum here, as the chain has already established.

**R3/03, the one loss: a drawn `>` has no pull.**
- The `>` is first drawn at **step 21**, 7 squares west of the hero (`|....>.` with the hero at `#@#`).
- The `>` room's entrance is not yet charted, so the `>` is not in `reach(pos)`. `field()` falls back to *uniform* frontier seeds.
- The walker then spends 78 steps exploring the NE corridors and the SE room (step-98 map). The only messages are `The door opens.` and `The door resists!`. It times out at 100.
- The frontier seeding in `Chart.field` has no term for the drawn `>`.

**Quest-Medium, 5/5 deaths, all on the same trajectory** (runs 00–04 are near-identical):
- Steps 0–12 are `field:frontier` moves east through the corridor into the lit rat room. At step 12: `giant rats far east`, `stairs down very far eastnortheast`, and a lava column `}` in front of the `>`.
- **Step 13: the field goes flat.** The room is fully drawn, so there is no frontier, and the `>` is behind lava, so it is unreachable. `consult` is called, and the VLM says `east` (3–4 times), into the room.
- From step 16 the hero stands at (36,12) in the **open room**, with rats adjacent NE/E/SE. `reflex:swing` hits east every turn.
- HP goes 16 → 14 → 10 → 3 → death (`Killed by a giant rat`, step 19–27).
- The 1-wide mouth at the room entrance, where only one rat can reach the hero, was passed at step 12.
- The horn/wand pile at spawn (`horn adjacent west` at step 0, `(` at x=22) is never touched.

**Quest-Easy, 5/5 timeouts:**
- Run_00 steps 0–6 are `reflex:swing` on a fox and a grid bug (HP 15 → 6), which is fine.
- From step 7 the lava column (x=31) splits the room, so the field is flat on **every remaining step** and the VLM is consulted 93 times.
- It emits only compass moves (the action frequency across all 5 QE episodes is 100% directions plus 16 `more`). It shuffles in front of the lava next to `horn near west` / `sky blue potion very near west` (step 30) until the timeout.

**Boxoban, 10/10 timeouts:**
- The map is premapped, so the frontier dies after 7–11 steps, and there is no `>`, so the field is flat for about 90 steps per episode.
- The consult menu is `c["open"]`, which **includes wall/bar moves** because `open` is not filtered by `Chart.passable`/`edge`.
- Over 1000 Boxoban steps: **225 `You cannot pass through the bars.`**, 64 `boulder … in vain`, and only 32 successful pushes.
- Each episode costs ~600k tokens and makes zero progress.

**CB-Dark 1/5.** 59 `reflex:swing` steps and 4 deaths. The chain has priced this as dice at n=5, so it is not diagnosable here.

## 3. Failure breakdown
| Failure | Eps lost | Mechanism |
|---|---|---|
| Boxoban: flat field → VLM compass shuffle | 10 | No sokoban planner; the consult menu offers walls (22.5% of steps are bar bumps) |
| Quest-Easy: flat field behind lava, pile never split | 5 | No item/lava stage; the VLM emits 0 non-movement verbs |
| Quest-Medium: walks into the lit rat room on a flat-field consult, dies in the open | 5 | Flat field → `consult` → `east`; no mouth hold; no Quest opener |
| CB-Dark deaths | 4 | Dice (chain-established) |
| R3/03: drawn-but-unreachable `>` ignored | 1 | `Chart.field` frontier seeds are uniform |

Cost: Boxoban + QE consults are about 96% of all tokens. `consult` has produced **0 wins across 20 Boxoban/Quest episodes**, both here and on the whole lineage.

## 4. Binding constraint — classification
- **TASK-GAP (primary)**: Boxoban and Quest need non-navigation plans (push sequences; kick/pickup/apply across lava). The walker operates legally but has no lever for them. Delegating to the VLM is a **VLM-CEILING**: across 1000 steps it never produced a single non-movement verb or a coherent push plan. The fix has to *pre-compute* the plan, not re-word the prompt.
- **HARNESS-BUG (secondary, cheap)**:
  - `st_consult`'s menu lists impassable moves (225 bar bumps).
  - The consult fires into a monster room on QM.
  - `Chart.field` ignores a drawn `>` that is not yet reachable (R3/03).
  - `Chart.fold` also writes creature letters into `terrain` permanently. The effect is minor, but a stale `r` stays "passable floor" after the rat moves.

## 5. Concrete changes to try (ranked by leverage)

**1. New `tools/quest.py` + `st_quest` stage, armed on the verb list `{"kick","pickup","zap"} ⊆ legal` (Quest pair only).** *(Up to +5 QM, +2–5 QE. The chain measured QM 0→3/5 on two lineages with this exact opener.)*
- **Place it BELOW `swing`**, so fighting outranks errands (a chain-priced law: an errand rung above the fight rung gets the hero eaten).
- **Opener from spawn** (the hero stands ON the 2-object pile; `pickup` on a pile is silent under NLE, and autopickup is False):
  1. step off one square, away from lava;
  2. `kick`, then answer `In what direction?` with the bearing back to the pile. This needs `st_prompt` to return a stored `self.aim` instead of `more` on direction prompts. Currently `st_prompt` answers `more` to *every* modal, which would swallow the kick direction.
  3. walk to each newly drawn `(`/`/` and `pickup` → `i - a horn.` / `j - a … wand.`
- **Kick exactly once.** A second kick along the same ray re-piles the objects. Never kick toward `}` (`It burns up!`).
- **Keep a stood-on ledger** so a stale drawn glyph cannot 2-cycle the errand.
- **Cross the lava.** Walk to the square adjacent to the nearest cardinal lava run and `apply`/`zap`. Answer the item prompt with the verb whose keystroke equals the item letter, and the direction prompt with the bearing to the lava. On `solidif`, clear the lava from `Chart.terrain` so `field()` reseeds on the now-reachable `>`.
- Restrict pickups to `(` and `/`, so letters are not burned on potions.

**2. `agent.py`: QM mouth hold.** *(Survival 0.775 → up to 0.90; required for #1 to score on QM.)*
- Gate it on the Quest verb set so CB-Dark stays byte-identical.
- When ≥2 creature letters are drawn and the hero stands on a cell with ≤2 passable neighbours (the corridor mouth at step 12), do not advance: swing if something is adjacent, else `search` (a free wait in the Quest vocabulary).
- Never let `consult` answer while creatures are drawn. The QM deaths start exactly at the step-13 consult.

**3. New `tools/sokoban.py` + `st_push` stage, armed on Boxoban's vocabulary (4 cardinals only) with `` ` `` and `{` drawn.** *(Up to +10 eps, ≈ +0.25 primary; chain stub-rig 19/20; Boxoban wins iff the optimal plan is ≤99 steps.)*
- Decode the premapped `map:` block exactly: `#` = bars/wall, `` ` `` = boulder, `{` = target, `.` = floor.
- Run an A*/BFS over (hero, boulder-set) push-space with dead-square pruning on walls only.
- Compile the plan into a committed tape and replay it. Re-plan only if the observed frame diverges.
- This also removes ~600k tok/ep: Boxoban becomes model-free.

**4. `tools/chart.py:Chart.field` — `>` attraction when drawn but unreachable.** *(R3 4/5 → 5/5; chain measured +2/−0 over 125 levels, 0 arms off R3.)*
- If any `>` is in `terrain` but not in `region`, give each frontier seed a potential of `manhattan(seed, >)·k` (k≈1) instead of 0.
- Gate it on no `}` drawn, so Quest is untouched. Tie-break in level coordinates.

**5. `agent.py:st_consult` — the cost/hygiene fix.** *(Primary neutral; cost −90% if #1/#3 are not shipped.)*
- Filter the menu with `self.chart.edge(pos, d) is not None`, which removes the 225 bar bumps.
- When the flat regime is Boxoban/Quest (armed on the verb list), skip the VLM: take `menu[0]`, or `search` on Quest. The consult has produced 0 wins in 20 episodes on every lineage.

**6. `tools/chart.py:Chart.fold`: don't persist creature letters.** Store `.` for a cell drawn as `[a-zA-Z&']` if its terrain is unknown, and never overwrite known terrain with a letter. This is minor hygiene that keeps stale monsters from being charted.

**Verification signals:**
- Every non-Quest/non-Boxoban CSV should be byte-identical to this run (23 eps).
- QM traces should show `kick` → aim → two `x - …` pickups → `apply`/`zap` → `solidifies` → `field:stairs`.
- Boxoban traces should show `push:*` rungs with 0 bar bumps.

## 6. Exhausted — do NOT repeat
- **Stage-table / module rewiring by itself** (r026, r036–r048). This candidate is the fourth rewire to land on the identical 15/40 vector.
- **Prompt, AGENT.md, rules or menu wording.** The model decides only flat-field residue frames and never emits a non-movement verb.
- **Frontier re-ranking on MazeWalk/R3** (gain rate, extremity, nub demotion, visit constants). They are saturated at 10/10 and 4/5. Only `>` attraction (#4) is still open.
- **CB-Dark melee posture** (retreat, hold, recoil or HP gates) priced on 5 eps. It is dice; it needs ≥200–500 eps.
- **HP-triggered retreat/disengagement.** Deaths rise with dose. The QM mouth hold (#2) is a *hold*, not a retreat.
- **Quest `pickup` without a pile split, a double kick, or a kick toward lava.** All are refuted.
- **Answering `Really attack? [yn]` with `n`.** It is a 2-cycle once anything swings.
