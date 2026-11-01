# Diagnosis — minihack iter r065 / cand_r065_0_spatial-grounding (chart walker + Quest module)

**Headline:** primary = 0.625 (25/40), survival 0.900, step_eff 0.366, cost 148 510 tok/ep.
Per task: Quest-Easy 5/5, Quest-Medium 5/5, MazeWalk 9x9 5/5, MazeWalk 15x15 5/5, Corridor-R3 4/5,
CB-Dark 1/5, **Boxoban-Hard 0/5, Boxoban-Medium 0/5**.
**Binding constraint:** Boxoban has no stage owner, so the VLM decides 1000/1000 Boxoban steps. It
produces an unbiased random walk: 0/10 wins, and those steps account for **~100% of the token bill**.
This is a TASK-GAP with a known, already-validated fix (an exact push planner).

## 1. What this harness does (mechanism)
`orchestrator.py:GroundedAgent.act` walks `STAGES = (prompt, orient, stairs, fight, cross, hold, kit, fetch,
route, probe, model)`, and the first stage that answers wins.
- `tools/frame.py` parses message, prose, cursor and map.
- `tools/chart.py` holds per-level map memory and does BFS `route` (stairs first, then the nearest
  unstood frontier) and a dark-room `probe`. `set_regime` makes `#`/`` ` `` walls when the moves are
  the 4 cardinals.
- `tools/quest.py` (new this round, ported from r058) is armed only when `{pickup,kick,zap} <= legal`.
  It handles the pile kick, horn/wand pickup, apply/zap across lava, typed prompt answers and the
  mouth hold. On an armed Quest level `st_model` never calls the VLM (`quest:wander`).
- AGENT.md, rules.md, skills and memory.md are empty. The model gets only the naive instruction plus
  a "closed or deadly" move list.

## 2. What actually happened (causal chain, with evidence)
**The Quest port worked exactly as predicted.** Quest went from 0/10 in the parent (r060–r063) to
10/10 here, with 0 model tokens. Every Quest episode runs the full plan. The rung census over the 10
episodes:

| Rung | Count |
|---|---|
| `quest:walk` | 180 |
| `quest:hold` | 106 |
| `quest:pickup` | 32 |
| `quest:aim` | 20 |
| `quest:kick` | 10 |
| `quest:zap` | 9 |
| `quest:letter` | 10 |
| `quest:apply` | 1 |

That is one kick per episode, zap/apply across the lava, then `chart:route-stairs`. Only 2 generic
`reflex:prompt-more` fired, so the quaff⇄`more` cancel loop is gone. Episodes take 41–81 steps.

**Boxoban is the whole loss (10 of 15 losses).**
- Rung census on all 10 Boxoban CSVs: **1000/1000 steps are `model`**. No chart stage answers,
  because the board is fully drawn and lit: there is no frontier and no `>`, so `route` and `probe`
  abstain and the model is asked.
- The VLM's action histogram is flat:
  - Hard: S 142, N 132, E 118, W 108.
  - Medium: S 137, N 127, W 118, E 118.
- The whole Hard+Medium run yields only 14 "With great effort you move the boulder" pushes, against
  16 "in vain" pushes and 30 "cannot pass through the bars" bumps. The random walk cannot finish a
  38–81-push-move plan in 100 steps.
- All 10 episodes hit the 100-step cap (`end_reason -1`, progression 0.0).
- Each episode costs 560–620k input tokens, about 5.94M in total. The other 30 episodes cost 0, so
  Boxoban is the entire `cost_per_episode` of 148.5k.

**CB-Dark 1/5 is the known dice.** Runs 00–03 die in the same pattern: `route-frontier` east along
the corridor, then `reflex:swing` at one square (for example run_00 at (35,12) for 12 consecutive
swings) while HP falls 16→1. Run_04 survives at HP 3–4 and walks out. This matches the chain's
closed melee axis (see §6).

**Corridor-R3 run_03** hit the 100-step cap while still on `route-frontier`. It walked long diagonal
corridors (for example (33,15)→(21,9) over the last 13 steps) with no bumps or orbits. That is
cap-limited exploration, not a bug.

## 3. Failure breakdown (15 losses)
| Category | Eps | Cause |
|---|---|---|
| Boxoban timeout | 10 | No push planner. The VLM does a random walk (flat action histogram, ~3% net pushes). |
| CB-Dark death | 4 | Melee dice at the fight square. The melee axis is closed. |
| R3 timeout | 1 | Exploration ran out of steps under the 100-step cap. |

## 4. Binding constraint
**TASK-GAP (plus VLM-CEILING on Boxoban), fixable in the harness.** The frozen model cannot plan
Sokoban pushes. The fix must remove the decision from the model entirely, not re-word the prompt.
This is already demonstrated in this chain:
- **Sibling cand_r064** is this same Quest module plus `tools/sokoban.py`. It scored **0.875** with
  Boxoban 10/10 (Hard 5/5, Medium 5/5) and every other task at or above this candidate.
- In hier_k8_c1, r017's `tools/sokoban.py` also gave Boxoban 10/10.

## 5. Concrete changes to try (ranked by leverage)
1. **Port `prop_r064/cand_r064_0_spatial-grounding/tools/sokoban.py` and add a `sokoban` stage
   directly after `orient`.** Set `STAGES = ("prompt","orient","sokoban","stairs","fight",...)`.
   - Arming: use the existing `Sokoban.armed(c["moves"], fr["rows"])`, which requires the move set
     to be exactly the 4 cardinals and the frame to draw both `` ` `` and `{`. Every other task
     offers diagonals, so the stage is inert there, and MazeWalk, R3, CB-Dark and Quest stay
     byte-identical.
   - Stage body: `got = self.soko.step(fr["rows"], fr["pos"], self.chart.drawn)`. Return
     `(got[0], "sokoban:plan:%d" % got[1])` if `got[0]` is legal. Set `self.last = None`: a push that
     moves nothing should not close a chart edge, because the planner re-plans from the observed
     state.
   - Instantiate `self.soko = Sokoban()` in `__init__` and clear it in `forget()`.
   - Keep r064's state rules:
     - Boulders come from the **current frame** only. `chart.drawn` accumulates stale boulder
       squares, so never read boulders from it.
     - Goals `{` are **accumulated** across frames, because a boulder on a target hides it.
     - The hero square is floor, never a boulder (the phantom-boulder trap).
     - If fewer goals than boulders are drawn, add the never-seen start square as a candidate goal.
     - Expect-and-replan whenever the observed (hero, boulders) differs from the plan's prediction.
   - Predicted effect: Boxoban 0/10 → about 10/10, primary **0.625 → ~0.875**, and cost_per_episode
     **148k → ~0**. The chain's memory records Boxoban as decoding to an exact Sokoban, with wins
     when the optimal plan is at most 99 steps. r064's offline check solved all 10 frames in ≤0.54 s
     with 38–81-move plans.
2. **If the solve fails or times out, fall through to the model, not to `route`.** The chart has no
   frontier on Boxoban, so the model is the only fallback. Keep the solver's time budget (4 s at
   w=1, then 6 s at w=3) so the eval does not stall.
3. **Optional: R3 cap-limited exploration (1 episode).** Only pursue this after change 1 lands. The
   chain has already measured the R3 ceiling at cap as ~0.94, so the remaining upside is at most
   about one episode in 40.

## 6. Exhausted — do NOT repeat
- **Melee changes on CB-Dark.** Retreat, recoil, hold, the committed fight post and HP-gated
  disengagement were all priced at n≥100–500. Each is a dice reroll, so a 1/5 → 2/5 swing at n=5 is
  noise, not signal.
- **Prompt or prose re-wording for Boxoban.** The VLM's action distribution is flat, so there is no
  signal for prose to shape.
- **Treating `#` as walkable on Boxoban.** `#` is iron bars. `chart.set_regime` already makes it a
  wall in the 4-cardinal regime, so keep that.
- **The Quest module.** It is now at 10/10 with 0 tokens. Do not touch it, and keep its stages
  gated on `Quest.armed`.
