# Diagnosis — maze3d_pro iter 2 / cand_r002_1_action-planning

**Headline:** primary `success` **9/25 = 0.360** (parent `cand_r001_1_action-planning` 7/25 = 0.280;
sibling `cand_r002_0_spatial-grounding` 7/25 = 0.280; staged anchor 2/25 = 0.080). Wins **L1, L2, L3,
L4, L6, L7, L11, L13, L17** = parent's win set **+L2 +L4 +L11 +L13 / −L15 −L20**. All 16 losses
terminate at exactly 100/100 decisions; 1880/1880 decisions parsed to a legal action
(`default_actions = 0` in all 25); `vlm_calls == steps_taken` and `used_rich_harness` in all 25, no
void/oracle flag — the contract is clean.

**The axis landed.** This is the first harness in the lineage to change floor at all: **16 effective
floor transitions**, 13 of 25 episodes reach floor 2, and **3/12 of the exit-on-floor-2 levels are
won (parent 0/12)**. The stair detector is not approximately right, it is *exact* on the live tape:
over 1880 decisions the `StairSense` latch agrees with the engine's own `cell_under_foot` **110 true
positives, 1 false positive, 0 false negatives**, and **15 of 15 harness-grounded climbs actually
changed floor**. Nothing below asks for it to be re-derived.

**Binding constraint (one sentence):** the stair is solved and the clock is not — the rewritten prose
collapsed the model's stride to `small` (**94%** of forward proposals vs the parent's 64%; **1.02 vs
1.54 cells gained per forward**), so at the win set's own cost of **2.3–5.7 decisions per
shortest-path cell (median 3.5)** a 100-decision budget buys ≈29 cells of route while **5 of the 16
losses have shortest routes of 32–54 cells**, and only 36% of each level's reachable cells are ever
visited — a **TASK-GAP in travel throughput and coverage**, on top of **two HARNESS-BUGS** (a 0.2%
frame-diff tolerance that single-handedly lost L20, and a floor-price gate that polices the harness's
climb but not the model's) that between them account for both regressions.

---

## 0. The instrument (so every number below is checkable)

No frames are kept by the eval, so the harness's internal state was recovered with an **open-loop
replay rig** (`/tmp/rig/replay.py`, outside the candidate dir): the model's **own recorded
answers** from the live run are fed back in order into the **real** `Maze3DProOpenAIEvaluator` with
the **real** candidate harness installed (`evaluator.client` stubbed; `asyncio.sleep` collapsed, since
`decision_frequency=0.5` is 2 s of real sleep per decision → **4.8 s per 100-step episode**).
`ActionPlanner.observe/govern/apply` are patched **at the class, after `build_agent`** (it purges
`sys.modules["tools"]`), and engine state (`get_state()` → `player_x/y/z`, `player_direction`,
`cell_under_foot`, `distance_to_exit`) is read **by the rig only**.

It is **calibrated on both harnesses**: it reproduces this candidate's live 9/25 with the identical
win set *and* identical step counts (11, 40, 17, 28, 11, 28, 46, 67, 32), and the parent's live 7/25
likewise (12, 21, 5, 16, 46, 14, 29). Caveat that governs §5: a mechanism change that alters the
executed stream cannot generate *new* model answers, so replay `success` is a **lower bound** —
price mechanism changes on mechanism metrics (false nulls, dwell, inert climbs, coverage), not on
`success` alone.

Ground truth for attribution comes from the level files (`data/maze3d_pro_merge/Level_NN.txt`,
rig-side only): **the exit is on the start floor on 13 of 25 levels and upstairs on 12**. Note that
the engine's `distance_to_exit` is **Manhattan in (x, y, floor) and ignores walls** (verified against
poses on L5/L9/L20) — it is not a route length, so every "how close did it get" figure below is a
**BFS distance over the real tower graph** computed in the rig.

## 1. What this harness does (mechanism)

`agent.py:build_agent(evaluator)` composes the system prior from `AGENT.md` + `skills/*.md` +
`rules.md`, opens **one** `tools.plan.ActionPlanner` per episode, and returns
`agent(frame_data) -> vlm.decide(...)`. Each decision is

    plan.observe(frame)  ->  vlm._inject_memory (append the block to conversation_history[0])
                         ->  evaluator.baseline_step (the pristine VLM call + JSON parse)
                         ->  plan.apply(proposal)  ->  engine Action

Two readings of `frame_data.image` drive it:

* **whole-frame fingerprint** (`plan.fingerprint` / `plan.screen_changed`): md5 of every 2nd pixel.
  A translation that leaves the screen unchanged is filed as `refused[view] += action`.
* **exact-colour census** (`tools/sight.py`): `stair ⇔ R==0`, `exit ⇔ B==0`; paint area → cells away;
  *saturated stair ahead + a translation + the frame changed ⇒ the stair is underfoot*
  (`StairSense`), the one fact the fingerprint cannot express.

`plan.govern` then displaces the proposal in two ways: a **positive** branch (stair underfoot ∧
`floor_age ≥ FLOOR_PATIENCE=24` ∧ `stale ≥ STALE_FLOOR=1` ∧ no exit in sight ⇒ take the climb), and
the **negative** chain in `refusal_reason` (inert climb / undo ban / proven-refused translation /
stall), whose re-aim is one step of a right-hand wall follower (`_reaim`). `memory.md` renders the
ledger back to the model every turn.

The §5.4 purity fix from the parent gradient **worked exactly as claimed**: ESCAPING mode is now
**124 of 1880 decisions (6.6%)** against the parent's **1042 of 1943 (53.6%)**.

## 2. What actually happened (causal chains, with evidence)

### (a) L20 — one 0.2% tolerance turns the start cell into an absorbing state  *(regression: parent won it in 29)*
`plan.screen_changed` calls a frame "unchanged" when identical **or when fewer than
`_NULL_FRAC = 0.002` of sampled pixels differ**. On **L20 decision 2** the agent really moves
`(0,1,1) → (0,1,2)` straight down a long open corridor; the two frames differ by **less than 0.2% of
sampled pixels**, so `_score` records a *null* and files `move_forward` into
`refused["618eec6aefe3707d"]` — the view of **(0,1,1) facing EAST, the only productive action from the
start cell of that level** (row 1 is a dead-end corridor whose sole branch is at col 9).

The agent walks east to (0,1,9), is refused by the wall north of it, is re-aimed west, and returns to
the start cell at decision 21. From decision 26 the governor overrides `move_forward` **16 times**
with *"`move_forward` already left this exact screen unchanged"* — a fact that is simply false.
Result: **74 of 100 decisions standing in cell (0,1,1)**, 9 distinct cells of 98 reachable (**9%**),
BFS distance never better than 18 against 26 at the start, 46 overrides (46% of decisions, the
sweep's highest). The view id is *not* a collision — it maps to that one pose and heading; the fact
itself was manufactured by the tolerance.

Measured over all 2040 translations in both harnesses: **a refused translation leaves the frame
byte-identical 83/83 times (parent 69/69) and a real move leaves it identical 0/1077 times (parent
0/814)**. Exact md5 identity is a perfect one-bit outcome oracle; the tolerance branch contributes
**nothing but 4 false nulls** (L20 step 2; L25 steps 7, 90, 91).

### (b) L15 — the floor gate fires ten decisions after the exit was one cell away  *(regression: parent won it in 46)*
L15's exit is at floor 0 (7,2). At **decision 19** the agent stands at (0,7,3) — **BFS distance 1**,
the exit directly WEST — facing SOUTH, with `exit` paint **0.0** in the frame (a tile is drawn only
while it is ahead, so an adjacent exit behind the shoulder is invisible). It turns LEFT to EAST, walks
the row away (distance 2, 3, 4, 5), turns north up the stair column, and at **decision 29**
`govern`'s grounded branch fires (`floor_age` ≥ 24, stale ≥ 1) and climbs. BFS distance then grows
monotonically to 18; the agent finishes upstairs with 36 cells seen and never returns.

### (c) L23 — the floor price does not apply to the model's own climb
`FLOOR_PATIENCE`/`STALE_FLOOR` live in `sight.StairSense.wanted`, which only gates the **harness's**
grounded branch. `refusal_reason` has no floor-price clause, so a **model-proposed** climb on a stair
is obeyed instantly. On L23 (exit on the start floor) the model proposes `climb_up` at **decision 13**
with `floor_age = 13`; it is obeyed, works, and the agent spends **86 of 100 decisions on the wrong
floor** (13 decisions on the exit floor in total), including decisions 15–19 stuck on the stair tile
itself. Across the sweep the two paths are starkly different in quality: **grounded climbs 15/15
effective; the model's own climbs 1/13 effective** — and the one that worked is this level-losing one.

### (d) L5 — the gate missed its only chance by three decisions
L5's exit is upstairs; the level has exactly one stair. At decision 19 the stair is saturated ahead
(`stair = 0.506`) and the model proposes `climb_up` one cell early; it is obeyed and does nothing. At
decision 20 the model repeats it, the inert-climb veto re-aims to `move_forward small`, and decision
21 lands the body **on** the stair (`cell_under_foot = 'T'`, latch set). `floor_age = 21` against
`FLOOR_PATIENCE = 24` ⇒ refused; the model's next `move_forward` carries it off, and **the stair is
never in view again in the remaining 79 decisions** (3 stair-sighted frames in the whole episode).

### (e) The stride collapse — the quiet regression inside a winning iteration
What the **model asked for** on its forward proposals: candidate **1348 small / 70 medium / 17 large**
(94% small); parent **868 small / 316 medium / 175 large** (64%). Cells actually gained per forward:
**1.02 vs 1.54**. Only **11** of the candidate's 1115 forwards were harness re-aims and the `_trim`
guard never fired on a model `medium`/`large`, so this is **the model obeying the new prose**, not the
harness. The responsible lines are new this iter: `rules.md` §Movement Granularity gained *"Use
`small` whenever the thing you are aiming at is the very next cell; a longer stride would walk
straight over a stair or past a turning"* (the parent's section had no such bullet), `skills/navigation.md`
went from 1 to **3** `small` recommendations, `AGENT.md` from 0 to 1, and `memory.md` repeats *"Moving
one cell at a time (`small`) proves what is really ahead"* **every single turn**. The stair-approach
caution generalised into a global default. The irony: `plan._trim` already clips `medium`/`large` to
`small` whenever the frame shows a stair or exit adjacent, and the engine stops a long stride early
at a wall (`('large',1): 1`, `('large',2): 27` in the parent's histogram) — the harness already makes
the long stride safe, and the prose then taught the model not to use it.

### (f) The search half — the stair is simply not found
A cyan stair is in view on **141 of 1880 decisions (7.5%)** and is the adjacent cell on 52 (2.8%). On
**L8, L10, L16, L18, L19 the stair paint is 0 in all 100 frames** — the agent never walked within line
of sight of it, and all five have their exit upstairs. Coverage is **36%** of reachable cells
(parent 22%), **22% on the exit floor in the losses**, and **60% of decisions re-execute a
(view, action) transition already taken from that exact view** (31% repeat one taken ≥2 times).

### (g) The model does not act on the block — and that is fine
When the block lists something as PROVEN REFUSED for the current view, the model re-proposes it
**333/574 = 58%** of the time (parent 61%). Stating the fact back changed nothing; the governor
absorbing it is what works. Likewise the gold exit appears in **0 frames of all 16 losses** (and
exactly once, adjacent, in each win) — re-confirming the parent's finding that an exit-commit lever
has no opportunity in the loss set.

## 3. Failure breakdown (16 losses, primary cause, mutually exclusive)

| # | cause | levels | evidence |
|---|---|---|---|
| 5 | **never found a stair, exit upstairs** | L8, L10, L16, L18, L19 | 0 stair-painted frames in 100 decisions; 0 decisions on the exit floor |
| 1 | **gate missed the only stair arrival** | L5 | arrival at decision 21, patience deficit 3, stair never re-sighted |
| 5 | **climbed off the exit floor** | L15, L21, L23, L25, L24 | departures at 29/27/13/30/41; 13–30 of 100 decisions on the exit floor (L24 76); 4 of 5 finish upstairs |
| 3 | **reached floor 2, out of route** | L9, L12, L14 | 18–61% exit-floor coverage, end BFS 10/10/32 |
| 1 | **absorbed by a false refusal** | L20 | 74/100 decisions in the start cell (§2a) |
| 1 | **big maze, right floor, out of clock** | L22 | 64/100 on the exit floor, 47% coverage, min BFS 5 |

Clock arithmetic for the same set: the win set spends **2.3–5.7 decisions per shortest-path cell
(median 3.5)**, so 100 decisions ≈ 29 cells of route. **L12 (49), L18 (47), L24 (54), L19 (39), L14
(32)** are beyond that at the sweep's own efficiency; **L11 won a 34-cell route in 46 decisions
(1.4 dec/cell)**, which is the existence proof that the throughput is the lever and not the maze.

Per-decision mix: `move_forward` 59%, turns 36%, `move_backward` 3%, climbs 1.5%; **83 null
translations (4.4%)**, down from the anchor's pathology and no longer a cost centre.

## 4. Binding constraint — classified

* **TASK-GAP (dominant).** Travel throughput + coverage. The agent is legal, unstuck, and now
  multi-floor; it runs out of clock. Two independent handles are measured and untouched: the stride
  (1.02 → 1.54 cells/forward is a **+51%** range already demonstrated by the parent on the same model)
  and the revisit rate (60% of decisions repeat a transition the harness has already recorded).
* **HARNESS-BUG (two, both priced below).** (i) `plan.screen_changed`'s `_NULL_FRAC` tolerance
  manufactures false "wall ahead" facts — 4 in 1160 translations, one of which cost L20 outright.
  (ii) The floor price gates only the grounded climb, so the model's own climb is a free, ungated
  floor departure — 13 presses, 12 inert, 1 effective and level-losing (L23).
* **VLM-CEILING (confirmed, already absorbed — do not re-prompt).** "Am I on a stair" is solved by
  `sight.py`, not by prose. Adherence to the PROVEN-REFUSED line is 58% and did not move. An adjacent
  exit off the facing axis is invisible to both model and census (L15 decision 19). The fix for these
  is always a harness that acts, never a sentence.

## 5. Concrete changes to try (ranked by leverage)

### 5.1 Give the stride back — prose only, the code guard already exists  *(highest leverage)*
*Files: `rules.md` §Movement Granularity, `skills/navigation.md` (first 3 bullets), `AGENT.md`
(distance paragraph), `memory.md` (the "How to use this" tail).*
Delete the four new *use-`small`* recommendations and state the economics instead: a stride costs one
decision whatever its length, it stops early at a wall so it is never worse than `small`, and the
harness itself clips it to one cell when the frame shows a stair or exit in the next cell
(`plan._trim` — say so, so the model does not have to be careful). Keep exactly one `small` rule: the
stair/exit **adjacent** case, which `_trim` enforces anyway. Optionally widen `_reaim`'s probe from
`"small"` to `"medium"` (the null test is unaffected — a 1-cell partial stride still changes the
frame), though it only touches 11 decisions.
**Predicted effect:** forward proposals back toward the parent's 64% `small`, +0.4–0.5 cells per
forward ≈ **+400–500 cells of travel across the sweep at an unchanged decision count**, which is
exactly the deficit of the 5 out-of-budget losses (L12, L14, L18, L19, L24) and the 3 floor-2
out-of-route losses (L9, L12, L14). **Price it on cells-visited / exit-floor coverage and on BFS end
distance, not on `success`** — it is a budget change, and the replay rig cannot show its wins.

### 5.2 Make the outcome test exact — delete the tolerance  *(free correctness, priced)*
*File: `tools/plan.py:screen_changed` (+ `_DIFF_EPS`, `_NULL_FRAC`, and the `_thumb` downsample copy
in `fingerprint`, all then dead).* Return `prev_view != view`.
**Measured on the replay rig (variant A, all 25 levels):** false nulls **4 → 0**; view-keyed
"forward is a wall" facts in force at decision time **17 wrong of 574 (3.0%) → 0 wrong of 533
(0.0%)**; L20's worst-cell dwell **74 → 18 decisions**, overrides **46 → 27**, end BFS distance
**26 → 20**; **win-neutral and regression-free on every other level**. Justification: 83/83 refused
translations leave the frame byte-identical and 0/1077 real moves do.

### 5.3 Hold the model's own climb to the same floor price  *(priced, closes the gate asymmetry)*
*File: `tools/plan.py:govern` (or a clause in `refusal_reason` reading
`sight.STALE_FLOOR`/`FLOOR_PATIENCE`).* If a climb is proposed while
`stale < STALE_FLOOR or floor_age < FLOOR_PATIENCE`, re-aim it and say why.
**Measured (variant B):** climbs executed **28 → 17** (11 of the 12 inert model climbs and L23's
decision-13 departure removed), cells visited 451 → 457, **win-neutral, no regression**. It converts
~11 wasted decisions into probes and removes the one path by which the harness can leave the exit
floor before the price has been paid.

### 5.4 The transition ledger — the parent's §5.3, still untouched and now licensed
*File: `tools/plan.py` — a `transitions[(view, action)]` counter; `govern` re-aims a proposal that
would repeat a transition already taken **≥2** times from this view onto the lowest-count untried
bearing; surface it in `fields()`.* `tried[view]`/`views[view]` exist and `govern` reads **neither**.
Sized on this run: **591 decisions (31%) propose a (view, action) already executed ≥2 times from that
view; 1126 (60%) repeat one at all; 840 decisions (45%) sit on a view already seen ≥4 times.** §5.2
is the precondition: after it the view-keyed ledger is **0-wrong in 533 facts**, so extending the key
to transitions is now sound.
**Predicted effect:** attacks L20, L22, L16, L10, L8 (coverage 9–47% on the right floor) and the
"never found a stair" five. Price on coverage and revisit rate.

### 5.5 Do not walk off the stair you are standing on
**42 of 43 abandoned stair arrivals were the model's own `move_forward`** (one was a harness
override), and only **17 of 43 arrivals** were ever spent on a climb. The latch is already exact, so
remember it: record the view/step at which the latch was set, and when the floor *does* go stale
re-aim the sweep back to that view (the ledger already knows the bearings). **Do not simply loosen
the gate:** variant C (let the arrival decision satisfy the staleness clause, since `stale` is 0 by
construction on an arrival) was replayed and **lost L13** — a start-floor-exit win — while raising
climbs to 31. The staleness clause is load-bearing for the 13 start-floor levels. If the gate is
loosened at all, make the departure **revocable** first (a return budget), because 4 losses (L15,
L21, L22, L25) end upstairs having never come back.

### 5.6 Structural axis if 5.1–5.4 do not close the gap: read the corridor, not just the tiles
`sight.py` measures only two tile families; the frame also carries **corridor depth** in the vertical
extent of the paint down the centre column. That gives (a) the legal stride length for free — the
guard §5.1 asks the model to judge — and (b) a frontier preference (prefer the bearing with the
deepest unexplored corridor), which is the one thing that would put a stair into view on L8/L10/L16/
L18/L19, where it never was.

## 6. Exhausted — do NOT repeat

* **The on-stair detector.** Now measured on the live tape: 110 TP / 1 FP / 0 FN in 1880 decisions,
  15/15 grounded climbs effective. It is finished. Extend it; do not re-derive or re-threshold it
  (`ADJACENT = 0.30` sits in a factor-4 gap and has no gradient).
* **JSON parsing / the parse fallback / the key table / granularity spelling.** 1880/1880 legal,
  `default_actions = 0` in all 25, third iteration running. Zero headroom.
* **The per-turn memory channel.** Verified again live; do not re-plumb `conversation_history[0]`
  and never splice the base turn.
* **An exit-line-of-sight commit lever.** Re-confirmed: **0 exit-painted frames in all 16 losses**,
  exactly one (adjacent) in each win. No opportunity in the loss set.
* **Prose that states the ledger back to the model.** Adherence to PROVEN-REFUSED is 58% here vs 61%
  in the parent — unmoved by a full rewrite. Likewise "avoid turning in the same spot" and
  `view_visits` did not stop the orbits. Mechanisms, not sentences.
* **Letting the grounded climb fire on the arrival decision** (dropping the `stale` clause from
  `StairSense.wanted`): replayed and **refuted, −L13**.
* **The `_reaim`/sweep latching fix.** Shipped and confirmed (ESCAPING 53.6% → 6.6%); don't revisit.
* **The granularity trim as a scoring lever.** It fired on ~0 live decisions because the model stopped
  asking for long strides at all (§2e). Keep it as the safety that *licenses* §5.1; do not price it.
* **`FLOOR_PATIENCE`/`STALE_FLOOR` as a blind sweep.** The curve is not the issue — the gate is
  asymmetric (§5.3) and irrevocable (§5.5). Fix those two before touching the constants; the measured
  trade-off is 9 useful departures (exit upstairs) against 6 harmful ones (exit on the start floor),
  and no observable in the frame distinguishes the two cases, so a *return* path beats a better
  threshold.
