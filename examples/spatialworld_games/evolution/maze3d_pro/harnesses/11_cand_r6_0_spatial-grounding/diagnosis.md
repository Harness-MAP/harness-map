# Diagnosis — maze3d_pro / `anchor_maze3d_pro` (staged rich anchor, `maze3d_pro__k2_claude_pv`)

**Headline:** primary `success` = **2/25 = 0.0800** (wins **L6** in 7 decisions, **L17** in 33).
All 23 losses burn the full 100/100 budget. The response contract is spotless — **2340/2340
decisions parsed to a legal action, `default_actions = 0` in 25/25 episodes, `vlm_calls ==
steps_taken` everywhere, `used_rich_harness: true` everywhere.** Nothing is wrong with the plumbing.

**Binding constraint (one sentence):** **HARNESS-BUG, compound.** (a) The harness holds **no state
between steps**, so the one bit that separates a move that happened from a move that was refused —
*did the frame change?* — never reaches the model, and **1829 of 2340 decisions (78.2%) re-press a
wall this same episode has already proven at the same cell and the same heading**; (b) `AGENT.md`
and `rules.md` describe a **HUD and stair prompts that this build does not draw** (measured: **0
pixels ≥ 235 in 4,236 frames** of this run's own recordings), which makes the stated
`climb_up`/`climb_down` precondition literally unsatisfiable — **0 climb actions in 2340 decisions**
on levels where **17 of 25 require a staircase**, capping the anchor at a geometric **8/25** before
a single navigation decision is made.

---

## 0. Method — why every pose claim below is exact, not inferred

I rebuilt the anchor's 25 episodes by exact replay: parse `Level_NN.txt` (`#` wall / `S` start /
`T` stair / `E` exit / `===` floor separator), start at `S` on floor 0 facing **north**, apply each
recorded `action_taken` with the adapter's semantics (`small|medium|large` = 1|2|3 cells, stopping
before the first wall; turns 90°; a climb only on a `T` with the target floor present).

The replay reproduces the engine **exactly**:

| check | agreement |
|---|---|
| `success` (did it end on `E`) | **25/25** |
| `distance_to_exit` (= Manhattan, confirmed; BFS agrees on only 6/25) | **25/25** |

Initial heading is identified, not assumed: north gives 0/25 success mismatches, the other three
bearings give 3, 3 and 2. Everything below is therefore a measurement of the real trajectory.
Oracle level data is used for **this analysis only** — §5 never routes it into a decision.

Second instrument: the run's own recorded video. 44 `maze3d_pro_openai_evaluation_20261005_13*.mp4`
files cover the anchor's window (13:26–13:50), 4,236 frames total; the L17 win is identified
unambiguously (its session log starts 13:39:53, its video is `..._133953.mp4`, and both are 33
frames long).

---

## 1. What this harness does (mechanism)

`agent.py:build_agent(evaluator)` is the agent factory the extended evaluator installs as
`ai_model_func`. Exactly two things happen in it:

1. `evaluator.current_prompt = compose_system_prompt()` — concatenates `AGENT.md` + `skills/*.md`
   (sorted) + `rules.md`. The engine's `build_full_prompt` appends the JSON RESPONSE FORMAT + tool
   list on top. I diffed the result against the shipped `system_prompt.txt`: **content-identical in
   order, differing only by 3 blank lines** the `.strip()` in the join removes. The anchor really
   does score like the prompt-only baseline.
2. it returns `async def agent(frame_data): return await vlm.decide(evaluator, frame_data)`.

`tools/vlm.py:decide` is a two-line pass-through — `step = evaluator.baseline_step; return await
step(frame_data)` — and `baseline_step` is the *pristine* SpatialWorld per-step fn: capture the
frame, maintain the rolling image+text history, call the VLM, parse the JSON, return an engine
`Action`. **The decision path is byte-for-byte the prompt-only baseline.**

Three consequences, all verified in this tree:

* **Zero state.** There is no object, no counter, no previous frame, no ledger anywhere in
  `agent.py` + `tools/vlm.py`. **There is no place in this harness where a loop-break could
  live.**
* **`tools/action.py` and `tools/frame.py` are dead.** `agent.py` imports `tools.vlm` and nothing
  else; `key_for`, `granularity_steps` and `capture` appear only at their own `def` lines.
  `memory.md` is read by no `.py` at all, and its own header says so. These three are exactly the
  surfaces §5.1–§5.4 need — they are already in the genome, so wiring them costs no new module.
* **The frozen model is `Qwen3-VL-30B-A3B-Instruct`, not a Thinking build.** Every response is bare
  JSON: `completion_tokens` is **19** for a translation and 8–15 for a turn, `reasoning=None` in all
  2340 raw responses. There is no chain of thought to read, and none to appeal to.

---

## 2. What actually happened (causal chain, with evidence)

### 2.1 The anchor spends its budget pressing walls it has already disproved

| quantity | value |
|---|---|
| decisions | 2340 |
| `move_forward` | 2090 (89.3%) |
| `turn_right` / `turn_left` | 128 / 122 (10.7% combined) |
| `climb_up` / `climb_down` | **0 / 0** |
| `move_backward` | **0** |
| granularity: small / medium / large / absent | 1207 / 410 / 473 / 250 (absent = exactly the 250 turns) |
| **translations that moved ZERO cells** | **1904 / 2090 = 91.1%** |
| of those: genuine first probe of a `(cell, heading)` | **75** |
| of those: **re-press of a wall already proven this episode at the same cell and heading** | **1829 = 78.2% of ALL decisions** |
| distinct cells visited, mean per episode | **5.2** |

Six episodes have a **100% null rate** — every translation a no-op (L2, L5, L10, L15, L18, L22) —
and four of them (**L2, L10, L15, L18**) never left the **start pose** in 100 decisions.

### 2.2 Three absorbing orbits, entered early

Taking "last decision at which a *new* pose `(floor,x,y,heading)` first appeared" as the absorption
point: **1928 of 2300 loss decisions (83.8%) come after it**, the median absorption step is **8**,
and **17 of 23 losses are absorbed by step 20**. The orbits are of three kinds.

**Type A — the frozen forward (18 of 23 losses).** The episode ends in one unbroken run of the
*identical* forward action, ≥37 long. Run-length-encoded (`F1/F2/F3` = forward small/medium/large,
`R`/`L` = turns):

```
L 2   F1x100                         <- never moved; 100 byte-identical replies
L10   F2x100                         <- never moved
L15   F1x100                         <- never moved
L18   F1x100                         <- never moved
L 3   R F2 R F1x2 L F1x94            <- 94 blocked presses after 3 real moves
L13   R F2 R F1x2 L F1x5 L F1x88
L21   R F1x5 R F1x93
L16   R F3x2 L F3x3 R F3x92
L22   R F2x2 L F2x3 L F2x92
L 7   R F2x2 L F2x96
L25   R F2x2 L F2x2 R F2x2 R F1x2 F2x3 F1 L F3x83
L20   R F2x4 L F2x5 R F2x6 R F3x81
L 8   R F2x2 L F3 F1x4 R F1x4 L F3x85
L24   … L F2x3 L F1x70
L 9   … R F1x64 ;  L12 … F1x53 ;  L11 … F1x37
```

L2 is the purest case: start `S(1,1)` facing north into `#`, and the model emits
`{"action": "move_forward", "granularity": "small"}` **100 times**, byte-identical, never once
turning. L18 is the same, on an 11×11 maze.

**Type B — the two-heading flip (L1, L4).** Three distinct poses in 100 decisions. The agent reaches
cell `(3,1)` and then occupies it at **heading north (50×) and heading east (49×) only**:

```
L1/L4   R F2 L F2 R F2 L F2 R F3 L F2 …      49/50 translations null
        cell (0,3,1): headings tried [N, E] of [N, E, S, W]
```

North and east are both walls there. **South is open** — on L1 it leads `(3,2) → (3,3)=stair`, and
the exit sits at `(1,3)`. The model turns, is refused, turns *back*, is refused, forever: it has no
record of which headings it has already refuted, so it alternates between two of four and never
samples the other two. L1 was lost for want of **one more turn in the same direction**.

**Type C — the successful shuttle (L23).** Every move succeeds and the agent still loses:

```
L23   Rx2 F1x6 Lx2 F1x6  (×12)  Rx2 F1x2      0 nulls in 74 moves, 7 distinct cells
```

A 180° about-face, six cells up a corridor, about-face, six cells back — a period-16 limit cycle.
A null-move ledger alone would be **blind** to this one.

### 2.3 Why the loop is self-reinforcing: the harness feeds its own failure back as a demonstration

`prompt_tokens` on L2 goes `1986 → 2504 → 3022 → … → 17045` and then **freezes at 17045/17046** from
roughly step 29 on — the engine's 29-turn image+text window has saturated.

The renderer is a pure function of the pose, so a refused move leaves the frame **byte-identical**.
From step ~29 onward the model is therefore handed a **29-shot prompt consisting of the same image
paired with its own `move_forward / small` reply, 29 times**, followed by that same image again. The
next-token distribution has one obvious answer. Nothing in the observation says the move failed —
detecting it requires diffing two images in the history, and the model is spending 19 output tokens.

**This is the harness's bug, not the model's.** The one bit needed (*the frame did not change*) is
free, exact, and already in `frame_data.image`; the anchor simply has nowhere to put it.

### 2.4 The climb precondition is unsatisfiable, and it costs 17 of 25 levels

`AGENT.md` tells the agent:

> **Top HUD (Top-Left)**: Displays your current `POS` … `FLOOR` … and `DIR` … **Rely on this to map
> your progress.**
> **Stair Prompts (Bottom-Center HUD)**: … If it shows `[Q] UP`, the higher floor is accessible.

and `rules.md` makes it a hard gate:

> `climb_up`: **CRITICAL: ONLY use this action if you are standing on a stair AND see "[Q] UP" in
> the bottom HUD.**

**None of that is drawn.** Measured over all 4,236 frames of the anchor's own 44 recordings:
**0 pixels ≥ 235 in any channel**; per-video max channel is 160–162 for plain corridor views and
240–242 where the gold exit face is on screen. The sole exception is the L17 win video, frames
**31–32**, with 2,289 then 4,787 near-white pixels forming a growing centred square outline
(rows 199–401 → 99–501) — that is the **exit tile's white outline**, in the 3D view, not a HUD. The
top-left band carries **zero** near-white pixels in every frame of every episode, and the
bottom-centre band carries **zero** saturated cyan.

So the harness's own prose forbids both climb actions unconditionally, and the trace agrees:
**0 climbs in 2340 decisions.** The cost is structural:

* exit physically on the **upper** floor: **12/25** — L2, 4, 5, 8, 9, 10, 11, 12, 14, 16, 18, 19;
* exit on floor 0 but floor 0 **disconnected** from it: **5/25** — L21, 22, 23, 24, 25;
* ⇒ **a staircase is required on 17 of 25 levels**; the exit is reachable with no staircase on
  **8/25** — **L1, 3, 6, 7, 13, 15, 17, 20**.

**The anchor's geometric ceiling is 8/25 = 0.32. It scored 2/25**, losing six of the eight
same-floor levels to the §2.2 orbits (L1, L3, L7, L13, L15, L20).

*This refines the earlier statement in the chain that "12 of 25 levels have their exit on floor 2":
true, but the operative number is **17**, because L21–L25 also need the stair as a bridge.*

### 2.5 The strategy prior in `skills/navigation.md` is almost entirely inert

Its priority ladder is: exit visible → stair visible → path open → wall ahead, turn. Measured over
the same 4,236 frames:

| cue | frames carrying it |
|---|---|
| gold exit face (>50 px) | **2 / 4236 (0.05%)** — both in the L17 win |
| cyan stair (>50 px) | 176 / 4236 (4.2%) |

Rules 1 and 2 therefore fire on ~0% and ~4% of decisions. The ladder collapses to rule 3, *"if the
path ahead is open … move_forward"* — whose condition the frozen model **cannot evaluate**, so it
executes the consequent unconditionally. Rule 4 (*"if there is a wall directly ahead, turn"*) and
the explicit anti-spin line (*"Avoid turning repeatedly in the same spot"*) are both already in the
prompt, and 23/23 losses are absorbed anyway.

### 2.6 The budget is never the constraint

Shortest route measured in **legal decisions** (BFS over `(floor,x,y,heading)`, unit cost per
decision, strides of 1–3 cells costing one decision each):

| | max over 25 levels | budget |
|---|---|---|
| forward-only (no reversals) | **40** (L24) | 100 |
| with `move_backward` admitted | 38 (L24) | 100 |

The six lost same-floor levels needed **3, 5, 9, 14, 14, 16** forward-only decisions (L1, L3, L7,
L13, L15, L20) and got 100. And **L1 is a one-decision win**: from the start pose,
`move_backward / medium` walks `(1,1) → (1,2) → (1,3) = E`. The anchor emitted
**`move_backward` 0 times in 2340 decisions** — a legal action, documented in `rules.md`, that the
frozen model never reaches for. Admitting it saves **41 decisions across the 25 levels**.

---

## 3. Failure breakdown

| category | rate | evidence |
|---|---|---|
| **RE-PRESSED WALL** — translation into a `(cell,heading)` wall already proven this episode | **1829 / 2340 decisions = 78.2%** | exact replay; only 75 of 1904 null forwards were first probes |
| **ABSORBED ORBIT** — decisions after the last newly-seen pose | **1928 / 2300 loss decisions = 83.8%**, median entry step 8, 17/23 by step 20 | pose-novelty replay |
| ↳ Type A frozen forward | 18 / 23 losses (terminal identical run ≥37; 4 of them 100/100) | RLE above |
| ↳ Type B two-heading flip | 2 / 23 (L1, L4 — 3 poses, 2 of 4 headings ever tried) | pose census |
| ↳ Type C successful shuttle | 1 / 23 (L23 — 0 nulls in 74 moves, 7 cells) | RLE + null mask |
| **NEVER CLIMBS** — levels made unwinnable | **17 / 25 levels**; 0 climbs in 2340 decisions | frame census + flood fill |
| **PARSE / CONTRACT FAILURE** | **0** | `default_actions = 0` in 25/25; 2340/2340 legal |
| **BUDGET EXHAUSTION as a true cause** | **0** | forward-only optimum ≤ 40 vs budget 100 |

The numeric pre-pass agrees where it overlaps (`steps_taken`, `vlm_calls`, `successful_actions =
100 / default_actions = 0`, `used_rich_harness`); `distance_to_exit` is a Manhattan proxy and is
**misleading as a progress signal** — the anchor's "1 cell away" on L3 and L5 is an artefact of
never having moved off a start cell that happens to sit near the exit.

---

## 4. Binding constraint — classified

**HARNESS-BUG (primary), two independent bugs, both fixable in this directory.**

1. **No state ⇒ no loop break.** The refusal bit is free (frame identity), exact, and already in
   `frame_data.image`; this harness has no object to hold it. 78.2% of decisions are spent
   re-pressing proven walls. A code fix recovers them.
2. **The prose describes a UI that does not exist.** The HUD and the `[Q] UP` / `[E] DOWN` prompts
   are not drawn (0 near-white px in 4,236 frames), yet `rules.md` makes them the climb's
   precondition. Result: 0 climbs, 17/25 levels unwinnable. An edit to two `.md` files removes the
   prohibition; grounding *"am I on a stair"* makes the climb land.

**VLM-CEILING (secondary, and it dictates the *shape* of the fix).** The frozen Instruct model
emits ≤19 completion tokens with no reasoning and provably does not extract "the cell ahead is a
wall" from the frame — it pressed one into a wall 100 consecutive times. So the fix must
**scaffold the decision with a computed observation** (a measured fact stated back to it, or a
proposal the harness displaces), **not re-word the prompt**. Re-wording is already exhausted (§6).

**NOT TASK-GAP.** The forward-only decision-optimal route is ≤40 on every level against a 100-step
budget, and the win set is exactly ⊂ the no-stair set — the agent is not running out of road, it is
standing still.

### What this same arm has already proved about the fix ordering

Read straight off the sibling evaluations in `$TRACE_DIR` (clustered by episode mtime; each round
is 2 candidates × 25 levels):

| round | wins | climb actions proposed | episodes ending on the upper floor | win set |
|---|---|---|---|---|
| **anchor** | **2/25** | **0** | **0/25** | {6, 17} |
| r001 | 8/25, 7/25 | 293 | **0/50** | **{1,3,6,7,13,15,17,20}** and {1,3,6,7,15,17,20} |
| r002 | 7/25, 9/25 | 39 | 18/50 | 10 distinct levels |
| r003 | 9/25, 7/25 | 130 | 15/50 | 11 distinct levels |
| r004 | **19/25, 22/25** | 52 | **25/50** | 24 distinct levels |
| r005 | 41/50 | 43 | 23/50 | 24 distinct levels |

Two clean facts: **r001's win set is *exactly* the 8-level no-stair set** — breaking the orbit alone
recovers the whole same-floor ceiling and not one level more, because its 293 climb proposals changed
floor **0 times in 50 episodes**. The jump to 19–22/25 arrives in **r004**, the first round where
floor changes actually land. That is the gradient, in order.

---

## 5. Concrete changes to try (ranked by leverage)

### 5.1 Give the episode a state object and a `(cell, heading) → blocked` ledger with a forced escape
**Files:** `agent.py:build_agent` (construct one ledger per episode and close over it),
`tools/vlm.py:decide` (fold the previous step's outcome, then govern the proposal),
`tools/action.py` (the chart + the escape ranking — it is already in the genome and currently dead).

**Change.** `build_agent` is called once immediately before the episode, so a plain object created
there is episode-local and leaks nothing between levels. Keep a dead-reckoned pose
`(floor, x, y, heading)`, a set of proven walls keyed on `(cell, heading)`, and the previous frame's
hash. Each step: hash `frame_data.image`; if it equals the previous hash, the press emitted last
step **moved nothing** — write the wall fact; then delegate to `baseline_step`, and if the model's
proposal is a translation into a fact-proven wall, **substitute** a turn instead of emitting it.

**Mechanics that are already priced in this lineage and must not be re-derived:**
* **Key the facts on the dead-reckoned CELL, never on the view digest.** The digest aliases badly
  (257 distinct digests over 5624 distinct poses downstream); a cell is a function of the maze, so
  two diverged runs reconverge on it.
* **Settle "did it move" ONE decision late, by exact frame equality.** My replay reproduces
  `success` and `distance_to_exit` 25/25 under a one-decision model, so there is **no observation
  lag on this build** (an older chain note describing a two-slot pipeline is superseded). A
  diff *tolerance* is refuted — exact equality is correct; 0.2% manufactures false walls.
* **Rank the escape openness-first, then a constant `right`-first, and keep it path-independent.**
  A visit counter in the tiebreak costs both wins and run-to-run repeatability; novelty-first burns
  the escape rotating through headings the frame already refuted.
* **Fold a translation pessimistically (0 cells) and settle it from the emitted cell count**, or the
  believed pose runs onto a phantom cell and the harness re-presses a blocked heading inside the fix
  for re-pressing blocked headings.
* **Arm the escape on a discovery stall** (decisions since a first-time pose), not on a visit count;
  the flat optimum is 2–4, so ship 4 — it overrides the frozen model least.
* **Record climb edges too, even though they cannot be frame-proven** — that is what makes L23's
  shuttle and a run of dead climbs visible as a self-loop. Type C (every move succeeds) is invisible
  to a null-move ledger; only the pose-transition map catches it.

**Predicted effect:** `success` **0.08 → ≈0.32** (the full no-stair set), and the 78.2% re-press
share collapses. **Why it works:** it supplies the exact bit the observation withholds, and it is
the one change r001 made that recovered 8/8 of the reachable levels immediately.

### 5.2 Rewrite `AGENT.md` + `rules.md` to describe the build that exists
**Files:** `AGENT.md` (the perception section), `rules.md` (the two climb bullets),
`skills/navigation.md` (the priority ladder).

**Change.** Delete the "Top HUD" and "Stair Prompts" paragraphs outright and say instead: *there is
no HUD, no coordinates, no direction readout and no `[Q] UP` / `[E] DOWN` text; a climb works only
while standing on a stair tile and is otherwise a silent no-op; never refuse to climb because you
saw no prompt — the prompt does not exist.* Keep the measured palette facts that **are** true (walls
brown, exit gold `(255,200,0)` with a white outline, stair cyan; the exit and stairs are only drawn
straight down the corridor you face). Demote rules 1–2 of the ladder, which fire on 0.05% and 4.2%
of frames, and make rule 3 conditional on the fact block §5.4 injects rather than on a judgment the
model cannot make.

**Predicted effect:** unlocks the climb action at all — a precondition for §5.3, worth nothing on
its own. **Why it works:** the current text is the *only* reason a documented legal action was
emitted 0 times in 2340 decisions. In r001 the candidate that replaced this prose with a measured
description went from 0 to ~7% climb decisions.
**Caution:** the pre-eval gate case-insensitively text-scans `AGENT.md` / `rules.md` / `memory.md` /
`skills/*.md` for the privileged vocabulary — see the trap list below.

### 5.3 Ground "am I standing on a stair", and commit the climb into a fresh map epoch
**Files:** `tools/frame.py` (sight the cyan stair ahead/beside), `tools/action.py` (the chart entry
and the climb decision).

**Change.** The tile under the agent's own feet is **never drawn**, so "am I on a stair" is *not* a
pixel question. Chart it: when cyan is sighted down the forward ray or in a flank column, mark that
cell `stair`; when the dead-reckoned pose later enters a cell marked `stair`, the climb's
precondition holds. Commit the climb **optimistically into a new map epoch** and settle it
afterwards by comparing per-pose frame digests across epochs.

**Mechanics already priced — do not re-derive:**
* **Frame identity is NOT a climb-success test**: 19.2% of successful climbs (46 of 240) leave the
  frame byte-identical. Treat a held digest as *no evidence*, never as proof of a no-op.
* **The stair prompt's content was a property of the FLOOR, not the square** — do not scope a
  floor-exists verdict per-cell, or a *sighted* stair never stood on will offer a climb off the top
  floor forever.
* **Do not let the climb fire on the arrival decision** that just stepped onto the stair (replayed
  and refuted downstream: −L13), and order the rungs so a re-goal pass cannot read arrival-on-stair
  as a dead goal and walk off it.
* Deferring the climb by discounting the departure is a **cliff, not a curve** — L25 must depart at
  5 of 23 cells stood and L13 must not, and nothing map-derived separates them. Do not build a
  departure heuristic; commit instead, and keep a committed return route.

**Predicted effect:** `success` **≈0.32 → ≈0.80**. **Why it works:** 17 of 25 levels are gated on
exactly this, and r004 — the first round in this arm where episodes actually ended on the upper
floor (25/50) — is the round that reached 19–22/25.

### 5.4 State the measured facts back to the model through **message 0**, not `current_prompt`
**File:** `tools/vlm.py:decide` (before delegating), with the wording template in `memory.md` (which
is already in the genome, already documented as this lever, and currently read by nothing).

**Change.** Render one short block — *what is ahead and how far, which headings at this cell are
proven walls, whether the last press moved, the step budget* — and inject it by mutating
`conversation_history[0]["content"][0]["text"]` **in place** as `stashed_base + block`.

**Non-negotiable wiring facts (each one has cost a round elsewhere):**
* `build_full_prompt` returns `current_prompt` **unchanged on every non-first call**, and
  `build_conversation_history` reuses `previous_history[0]` *by reference* — so re-rendering
  `evaluator.current_prompt` after step 1 **reaches nobody**. Message 0 is the only per-turn channel.
* The engine folds `AVAILABLE ACTIONS` + the JSON `RESPONSE FORMAT` into message 0 **on the first
  call only**. Always write `stashed_base + block`; **never** truncate-at-a-marker-then-append, or
  the response contract is deleted from every later call.
* Stash the base only once a conversation exists, so turn 1 stays byte-identical to the anchor's —
  a mechanism must never speak on turn 1.
* A lone appended user memo is dropped (`truncate_history` trims to an even message count); append a
  (user memo, assistant ack) **pair** if you use that path instead.

**Predicted effect:** raises adherence on the §5.1 substitutions and lets the model exploit facts the
harness measures rather than having them imposed. Note adherence to an explicitly stated
proven-refused line measured only **58–61%** downstream, and did not move under a full re-wording —
so treat injection as a *supplement* to displacement, never a replacement for it.

### 5.5 Admit and ration `move_backward`
**Files:** `tools/action.py` (the route cost), `rules.md` (one sentence legitimising it).

**Change.** Price routes in **decisions** and let the reversal compete. Then ration it: only plan a
reversal over ground the chart has already **verified** passable.

**Predicted effect:** +41 decisions of headroom over the 25 levels and an outright one-decision win
on L1. **Why it works:** the reversal is legal and documented and the frozen model emitted it
**0 times in 2340 decisions** — it is pure harness-side headroom. The gate that only plans a
*verifiable* reversal is worth more than admitting the reversal itself, and it throttles the
"stride over unseen ground" failure at the same time.

### 5.6 (low) Clamp the granularity to a measured free run — behind a named switch
**File:** `tools/action.py`. **Change.** Clamp a proposed `large` to the corridor's measured free
run. **Predicted effect: 0 wins.** Priced at exactly 0 wins and 0 truncations downstream. It is a
decisions-efficiency lever, not a win lever; ship it behind a named switch and say so in NOTES so
the next iteration *flips* it rather than re-measuring it. The universal pattern here is stride
**freezing** (22 of 23 losses end on a single constant granularity, 13 of them on `small` from step
0), not stride inflation — a mechanism aimed only at inflated strides addresses a minority.

---

## 6. Exhausted — do NOT repeat

* **The response contract.** JSON parsing, the parse fallback, the key table, granularity spelling:
  **2340/2340 legal, `default_actions = 0` in 25/25**, and the same at every point downstream. Zero
  headroom. Do not spend an iteration here.
* **Prompt-only re-wording as the cure for the orbit.** `skills/navigation.md` *already* says
  *"Avoid turning repeatedly in the same spot — if you have turned several times without moving,
  commit to a direction and move forward"*, and 23/23 losses are absorbed anyway. Downstream,
  adherence to an explicitly stated proven-refused line was 58–61% and did not move under a full
  rewrite. **Mechanisms, not sentences.**
* **Asking for reasoning / a scratchpad field.** This is an *Instruct* build: `reasoning=None` in
  2340/2340 responses, median 15 and max **19** completion tokens. Budget a structured observation
  instead of prose.
* **Re-asking the model after a refusal.** Priced: +2 changed answers in 44 on the first re-ask,
  **0 in 42** on the second. One re-ask maximum, then substitute.
* **A frame-diff tolerance.** Refuted — exact equality is correct; a 0.2% tolerance manufactures
  false walls and cost a level.
* **A VIEW-keyed ledger as the state key.** Refuted: **257 distinct digests over 5624 distinct
  poses.** Key on a dead-reckoned cell.
* **Frame identity as a climb-success test.** Refuted: **46 of 240 successful climbs (19.2%) are
  frame-identical.**
* **Letting a grounded climb fire on the stair-arrival decision.** Replayed and refuted: −L13.
* **Discounting the climb departure / any "how much of this floor is explored" gate.** Refuted as a
  **cliff**: caps 1–2 win L13+L20 but cost −26/−29 episodes elsewhere; ≥5 is inert; no interior value
  exists. L25 must depart at 5 of 23 cells stood and L13 must not.
* **Re-offering the traversed stair link as a re-elected goal.** Refuted at both ends of its credit
  range — inert when capped, oscillating when not. The cure is **commitment**, not price.
* **An unbounded "model casts the deciding vote" tiebreak.** Refuted: a degenerate
  always-`turn_right` proposer swings the target every step and spins in place (11/25 vs 22/25).
  Bound the vote to the turn-minimal subset.
* **An exit-line-of-sight commit lever.** **2 gold-painted frames in 4,236** here, both adjacent to
  the win. No opportunity in the loss set.
* **`distance_to_exit` as a progress signal.** `success` is the score, and the metric is Manhattan:
  the anchor's "1 cell away" on L3/L5 is an artefact of never having moved. A harness that converts
  standing-still into real motion *raises* it on some levels. It is also a **privileged key** — see
  the traps.
* **Stride escalation as the story.** Minority pattern (5/23). The universal one is stride
  *freezing* (22/23).
* **`max_steps` / budget framing.** Forward-only decision optimum ≤ 40 against a budget of 100 on
  every level. The budget is not binding.

### Dead weight to wire, not to carry
`tools/action.py` (`key_for`, `granularity_steps`), `tools/frame.py` (`capture`) and `memory.md` are
referenced by nothing. They are exactly the four surfaces §5.1–§5.5 need: put the chart and the
escape ranking in `action.py`, the pixel reads in `frame.py`, the fold/govern in `vlm.py`, and the
injected wording template in `memory.md`.

### Gate and rig traps (each has cost a round somewhere in this tree)
* **The pre-eval gate text-scans the agent-facing priors case-insensitively** — `system_prompt.txt`,
  `AGENT.md`, `rules.md`, `memory.md`, `skills/*.md` — for
  `gt.json`, `maze_file`, `distance_to_exit`, `distance_to_food`, `cube_solved`, `current_floor`,
  `target_blocks`, `matched_blocks`, and **AST-scans `agent.py` + `tools/*.py`** for the same names
  used as an attribute or a string subscript. Never let those spellings into prose or code — not even
  in a comment that discusses them. `NOTES.md` and `diagnosis.md` are proposer-facing and are **not**
  scanned (which is why they appear freely above).
* **Decide from `frame_data.image` only.** It is a numpy `(H,W,C)` array and the legal perception
  surface. Under `ANTICHEAT_GUARD=1`, `evaluator.game` allows **only** `.screen`; any other read or
  write — even one swallowed by `getattr(..., default)` or `hasattr` — voids the score to 0.0, and a
  disk guard refuses harness-caller reads under the benchmark's `data/`. Reading your own harness dir
  (`memory.md`) is fine.
* **A harness with 0 `baseline_step` calls is VOIDED** as a hand-coded solver. The model must stay in
  the loop; displace its proposals, never replace the call.
* **`build_agent` purges `sys.modules["tools*"]`**, so any lever override set *before* the call is
  thrown away and your "control" silently measures the shipped config. Set rig levers as **class
  attributes after `build_agent`**.
* **`evaluator.game` does not exist yet inside `build_agent`** — `run_evaluation` builds it after the
  call. Nothing may touch engine state at construction time.
* **The evaluator resolves its level file relative to CWD** — an offline pricing rig must run from
  the benchmark repo root, and must live outside the candidate dir (or in a `traces_*/` dir) so it
  never enters the genome.
* **Price on a context-free stub panel, not on one live run.** Sampling is `temperature=1.0,
  top_p=0.9` with no seed; four identical requests gave 3 distinct actions. A panel whose arms are
  transcribed from the actual losing streams is what makes a reversal/escape cure measurable at all —
  a panel with no stuck arm cannot price a stuck cure.
