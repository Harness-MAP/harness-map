# Diagnosis — snake iter 0 / `staged_anchor` (parent tag `anchor_snake`)

**Headline:** primary = **0.0000** (0/20 episodes scored, **0 meals in 67 engine steps**).
**20/20 episodes end by walking out of the 4×4×4 box at step 3 or 4**, consuming 3.4 % of the
100-step budget. Binding constraint: **HARNESS-BUG** — `tools/vlm.py:decide` is a bare pass-through
to `evaluator.baseline_step`, so *nothing in the harness tests the one predicate that kills every
single episode*: "does this key take the head outside `[0,3]³`?". That test needs **no pixels and no
oracle** — the spawn cell is fixed and the head is exactly dead-reckonable from the harness's own
emitted keys — and on this very seed set it is worth **+0.375 … +0.4375 primary**. A food-ward
ranking on top saturates the metric at **1.0000**.

Everything below is measured, not inferred. `traces_diag/replay.py` replays the 20 recorded action
tapes through a 45-line adapter replica (spawn head `(2,3,2)`, neck `(1,3,2)`, heading `+X`, G=4,
`random.seed(seed)` + `random.randint(0,3)` x/y/z food draw) and reproduces **steps_taken 20/20 AND
distance_to_food 20/20** — two independent fields, so the offline rig is byte-exact and every
counterfactual below is *priced*, not stubbed.

---

## 1. What this harness does (mechanism)

The anchor is a **prose refactor with an inert code surface**. Control flow:

```
eval_runner._apply_harness
  → harness.build_agent(evaluator)
      · evaluator.current_prompt = compose_system_prompt()   # AGENT.md + skills/*.md + rules.md
      · returns  agent(frame_data) = await tools.vlm.decide(evaluator, frame_data)
  → tools/vlm.py:decide  →  evaluator.baseline_step(frame_data)      # the PRISTINE baseline step
```

`compose_system_prompt()` reproduces `system_prompt.txt` (verified: identical modulo 4 blank-line
characters, 1624 vs 1628 bytes), so the agent is shown exactly the prompt-only baseline text.

**Three of the eight modules are dead code.** `grep` over the whole tree finds exactly one
executable import (`from tools import vlm`):

| module | status |
|---|---|
| `tools/action.py` (`DIRECTION_VECTORS`, `KEY_MAPPING`, `is_reversal`, `key_for`) | **never called** — referenced only inside a `tools/vlm.py` docstring |
| `tools/frame.py` (`capture`) | **never called** — docstring reference only |
| `memory.md` (the working-memory template) | **never read** — `compose_system_prompt` globs `skills/*.md` only |

So the harness owns **zero** mechanism: no legality filter, no dead reckoning, no frame decode, no
substitution, no memory injection. The decision is 100 % the frozen model's, unreviewed. Notably
`tools/action.py` *already contains the exact six direction vectors a wall test needs* — the data is
sitting in the tree, unwired.

The prose shapes the decision as: `AGENT.md` (role + 4 "verified" environment facts) →
`skills/strategy.md` (4 strategy bullets) → `rules.md` (6 legal moves, the no-180 rule, the control
mapping, the "observation tips"). The engine appends its JSON RESPONSE FORMAT + tool list on top
(composed text ≈ 813 tokens; the frame is 507 tokens — `prompt_tokens` 1320 → 1827 → 2334, exactly
+507/turn).

---

## 2. What actually happened (causal chain, with evidence)

### 2a. The tape is food-independent — the episode is decided before perception could matter

Across 20 seeds with 20 **different** food placements (`distance_to_food` spans 2…8) the model
emitted only **four distinct action tapes**, and only **three distinct opening pairs**:

| tape | seeds | n | dead-reckoned trajectory | death |
|---|---|---|---|---|
| `F,F,F` | 1,6,7,8,9,12,14,16,17,20 | 10 | (2,3,2)→(2,3,1)→(2,3,0)→**(2,3,−1)** | wall `z=−1`, step 3 |
| `F,R,F,R` | 2,3,4,13,15,18,19 | 7 | (2,3,2)→(2,3,1)→(3,3,1)→(3,3,0)→**(4,3,0)** | wall `x=4`, step 4 |
| `R,F,D` | 5,10 | 2 | (2,3,2)→(3,3,2)→(3,3,1)→**(3,4,1)** | wall `y=4`, step 3 |
| `R,F,R` | 11 | 1 | (2,3,2)→(3,3,2)→(3,3,1)→**(4,3,1)** | wall `x=4`, step 3 |

Every one of those 20 trajectories is reproduced exactly by the rig, and every recorded
`distance_to_food` falls out of it. **The food position changes; the tape does not.** The
`P(applied move closes L1 to food)` over all 67 applied moves is **12/67 = 0.179** — *below* what
uniform-over-the-safe-set would give, i.e. the agent is not steering at all. A safe
distance-reducing move existed on **63/67** decisions.

### 2b. The model never deliberates

`completion_tokens = 8` on **67/67** calls, `reasoning=None` on all of them (it is
Qwen3-VL-30B-A3B-**Instruct**). Every response is the bare string `{"action": "<name>"}`.
`default_actions = 0` and `successful_actions = 3` in every
`snake_model_responses_*.json`: **zero parse failures, zero fallbacks**. The agent is answering
confidently and instantly from an image it is not reading — `ep snake_1 steps 0/1/2`, three
identical `{"action": "move_forward"}` at `prompt_tokens` 1320/1827/2334, is the canonical instance.

### 2c. Death was avoidable on every single decision

At each of the 67 decisions the mean number of **safe** moves (in-box ∧ non-body ∧ non-reversal) was
**3.45** (min 2 — never 0). The model's pick was inside that safe set on **47/67 (70.1 %)** and
outside on **20/67 (29.9 %)** — and because the body is only 2 cells long and the engine's only
*refusal* is the 180, **every out-of-safe pick was immediately fatal** (0 refused reversals in 67
steps; `vlm_calls == steps_taken` in all 20 episodes). At the moment of the fatal pick a safe
**distance-closing** alternative existed in **18/20** episodes.

### 2d. The prose is actively wrong about the three facts that decide steps 1–4

- **`AGENT.md`: "Snake starts at the center" — FALSE.** The head spawns at `(2,3,2)`, flush against
  the `y=3` face, heading `+X`. `move_down` (+Y, rendered *downward*) exits the cube immediately.
  Seeds 5 and 10 died by exactly that, `move_down` at `(3,3,1)` → `(3,4,1)`. The prose tells the
  model it has room in six directions when it has room in three.
- **`rules.md`: six × `Look for the "X" letter on screen` + "Indicators only show for moves that
  won't cause immediate collision" — points at a HUD this build does not render.** Re-confirmed
  black-box on this same adapter and the same day (2026-10-10): the pure-`(0,0,0)` glyph-box mask is
  **empty on every frame**. So the harness's single stated perception procedure resolves to nothing,
  and its single stated legality affordance does not exist. The model is told to read a safety oracle
  out of the picture, finds none, and emits a canned direction. *(This run's frames were not
  persisted — only the 4 JSON/log files per episode — so I could not re-mask these exact frames; the
  claim rests on the same-day re-confirmation on the same engine.)*
- **`AGENT.md`: "Maximum possible score: 64" — 16× off-target.** `eval_runner.py:78`
  `SNAKE_EDGE_LENGTH = 4` and `primary = min(food/4, 1)` with `early_stop_score = 4`. The objective
  is **four meals**, not filling the grid; `skills/strategy.md`'s "space management" framing is
  budget advice for a goal that does not exist here.

---

## 3. Failure breakdown (20 episodes, 67 engine steps)

| category | count | rate |
|---|---:|---:|
| **WALL exit (head leaves `[0,3]³`)** | **20 / 20 eps** | **100 % of deaths** |
| body self-collision | 0 | 0 % |
| refused 180 reversal (wedge) | 0 | 0 % |
| parse failure / default action | 0 | 0 % |
| meals eaten | 0 | — |
| episodes reaching `max_steps = 100` | 0 | 0 % |
| mean engine steps / episode | **3.35** | **3.4 % of budget** |
| out-of-safe-set picks | 20 / 67 | 29.9 % |
| fatal picks with a safe closing alternative | 18 / 20 | 90 % |
| repeat-own-last-direction rate | 20 / 47 | **42.6 %** |

Death face inventory: `z=−1` ×10, `x=4` ×8, `y=4` ×2. Fatal step index: step 3 ×13, step 4 ×7.

**This is a different regime from the long-episode snake lanes.** There is no wedge (0 refused
reversals vs 201/1009 on `snake__k32_claude`), no orbit, no body death — the snake is dead before a
cycle can form. Any lever aimed at periodicity, freshness vetoes, orbit-breaking or
reversal-guarding is **unreachable** here: it operates on steps the agent never takes.

---

## 4. Binding constraint — **HARNESS-BUG** (not VLM-CEILING, not NAVIGATION-GAP)

The frozen model is genuinely poor at this task (0.179 closing rate, 8-token non-answers, a
food-independent tape), which *looks* like a ceiling. It is not the binding constraint, for a
measured reason: **the harness has no mechanism at all**, and the cheapest conceivable mechanism —
one `all(0 <= c < 4 for c in head + delta)` test — recovers most of the metric *without the model
improving at all*. `tools/vlm.py:decide` forwards the pick unreviewed; `tools/action.py` holds the
vectors needed to review it and is never imported. That is a code defect, not a model limit.

Classification note: the *residual* after the clamp (rung 2 ≈ 0.40–0.45, the snake surviving ~100
steps but eating only ~1.7 pellets) **is** a NAVIGATION-GAP, and it closes with a frame-decoded
food read (rung 3 = 1.0000). But diagnose it in that order — survival first. Nothing downstream of
step 4 is observable until the snake stops exiting the box.

### Priced ladder — 20 lane seeds × 5 trials, `max_steps=100`, `early_stop=4`

Stand-ins replace the live model offline. Per the standing rule, the primary stand-in is
**deterministic and memoryless, fitted from this anchor's own 67 emissions** (`argmax P(pick | last
applied)`: `last=+X → move_forward` 27/30; `last=−Z → move_forward` 20/37) — a random stand-in
flatters any rung whose escape is noise. `uniform` is the control.

| rung | mechanism it needs | cycler | uniform | steps/ep | honour % |
|---|---|---:|---:|---:|---:|
| 0 — pass-through (**= this anchor**) | none | **0.0000** | 0.0150 | 3.0 | 100 % |
| 1 — in-box + no-180 clamp, least-visited substitute | **dead-reckoned head only — NO pixels** | **0.3750** | **0.4375** | 95.5 | 26 % / 60 % |
| 2 — rung 1 + body model, least-visited | + `trail[:2+score]` | 0.4000 | 0.4500 | 99.5 | 26 % |
| 3 — rung 2 + food-ward ranking (honour the model on distance-optimal ties) | + food cell from the frame | **1.0000** | **1.0000** | 15.6 | 17 % / 28 % |

Observed anchor primary **0.0000** matches rung 0/cycler exactly. Read the ladder as an **ordering**,
not an absolute forecast. Two structural facts in it:

- **Rung 1 is free of perception.** It needs only the fixed spawn `(2,3,2)` and the keys the harness
  itself emitted — no decode, no `evaluator.game` attribute (the anti-cheat `_GameGuard` allows only
  `.screen`), no disk read. It is the single highest-leverage edit available and it is ~10 lines.
- **Honouring the model stays safe.** Honour rate never falls below 17 %, and `vlm_calls` counts
  *delegations* to `baseline_step` (`eval_runner.py:287-291`), not honoured picks — so the
  `_vlm_calls[0] == 0` void at `eval_runner.py:317` cannot fire as long as `decide` keeps calling
  `baseline_step` once per step. Keeping the frozen model in the loop costs nothing: rung 3 = 1.0000
  with the model honoured on ties.

---

## 5. Concrete changes to try — the gradient, ranked by leverage

### L1 · `tools/vlm.py:decide` — dead-reckon the head and veto an out-of-box key  (+0.375 … +0.44)

The whole first rung. In `decide`, keep module-level state `(head, last_dir)` initialised to
`(2,3,2)` / `(1,0,0)`; import the already-present `tools.action.DIRECTION_VECTORS`; after
`baseline_step` returns the `Action`, read the model's own name from
`act.metadata["function_called"]`, and if `head + DIRECTION_VECTORS[name]` leaves `[0,3]³` (or
reverses `last_dir`), **rewrite `act.key`** to the least-visited in-box non-reversing alternative
(`tools.action.KEY_MAPPING[sub]`). `act.key` is the entire override surface — the engine presses
only that. Then advance `head`/`last_dir` by the key you actually shipped.
*Why it works:* it converts 20/20 wall deaths into ~100-step episodes (mean steps 3.35 → 95.5),
which is the only thing that makes any other lever observable. **Trap:** a refused 180 does *not*
increment `steps_taken` and does *not* move the head — so only advance the ledger when the key you
shipped is non-reversing, or the reckoning silently drifts one cell on the first refusal. Also store
the head the snake left **FROM** (pre-commit), never the post-commit head.

### L2 · `tools/vlm.py` — add the body model, drop the body deaths  (+0.025 … +0.075 on top of L1)

Rung 1 still loses 12–15 % of episodes to self-collision. The body is legitimately
`([head] + trail)[:2 + meals]`, most-recent-first (measured exact 596/596 on this engine). Detect a
meal frame-side (the food disc jumps cell) or infer it from the length the trail must have; at ≤ 4
meals the body is ≤ 6 cells. Mirror the **full-snake** test (the engine would accept the vacating
tail, but test conservatively so predictions stay comparable). Keep `least-visited` as the
tie-break — it beat first-legal (0.100), keep-heading (0.025) and uniform (0.235) on the cycler by a
wide margin; `keep-heading` is near-worthless precisely because the cycler's heading is what walks
into the wall.

### L3 · a new `tools/perceive.py` — decode the food cell from the frame, then RANK  (→ 1.0000)

The saturating rung. Mask the raw `frame_data.image` (an `(H,W,3)` uint8 — the pixels the base64 is
encoded from, so exact colour works): food = the only `R>150 & G>120 & B<90` cluster; head = the only
`R>150 & G>150 & B>140` cluster of area ≥ 40 in rows 60…514. Invert the projection (centre on the
cube, rotate X by 0.5 rad then Y by 0.6 rad, pinhole `s = 600/(600+z_c)`, `px = x_r·s + 400`,
`py = y_r·s + 300` on 800×600) and **gate on the blob diameter** — head ring `2·int(12·scale)`, food
disc `2·int(14·scale)` — so the four near-coincident projected pairs (6.76/6.95/8.54/9.26 px apart)
separate by ≥ 6 px of *size*. Accept a cell only if the blob is the size that cell would be drawn at;
otherwise **abstain** and fall back to L2. Measured 1562/1566 food and 1561/1566 head with **0 wrong**
on real frames; the abstains are occlusion at length ≥ 4. Then: honour the model's pick iff it is safe
**and** L1-distance-optimal, else take the least-visited distance-optimal safe move. The food read is
from **pixels**, never `evaluator.game` — `_GameGuard` whitelists only `.screen`, and any other
attribute read voids the candidate.

### L4 · `AGENT.md` + `rules.md` — stop asserting three false facts  (small, but it is sign-correct)

Cheap, and it removes the prose cause of the `move_down` deaths and the 16×-off objective:
- Replace "Snake starts at the center" with the measured spawn: head `(2,3,2)`, neck `(1,3,2)`,
  heading `+X`; `y=3` is a face and `+Y` renders **downward**, so `move_down` at spawn leaves the
  cube.
- Replace "Maximum possible score: 64" with the real target: **4 pellets ends the episode**
  (`early_stop_score=4`, `primary = min(food/4, 1)`); there are 100 steps for them.
- **Delete** the six `Look for the "X" letter on screen` clauses and the "Indicators only show for
  moves that won't cause immediate collision" promise. They describe a HUD this build does not
  render; keeping them trains the model to answer from an affordance that is not there. Replace with
  the honest statement that the frame shows only the lattice, the white head sphere, the orange food
  disc and the green body, and that **nearer cells are drawn larger**.
*Caveat:* judge L4 by the closing rate and the `move_down`-at-spawn count in the next trace, not by
the scalar. Prose alone cannot fix a wall death the harness does not test for — ship it **with** L1.

### L5 · `harness.py:compose_system_prompt` — actually wire `memory.md`  (speculative)

`memory.md` is a documented lever that nothing reads. Once L1 exists the harness *has* self-derived
state (head cell, heading, recent keys, visit counts) worth rendering into the prompt each turn. Low
confidence on its own — the model's 8-token non-answers suggest it is not conditioning on text
detail — but it is the natural way to let the frozen model participate in the ranking instead of
being overridden 74–83 % of the time, and the honour rate is a clean instrument for whether it lands.

---

## 6. Exhausted / do NOT repeat

- **"License the repeat" in `rules.md`.** This is the standing untried prose lever on the
  long-episode lanes, where the model had absolutised "no 180 turn" into *never repeat a direction*
  (0.08 % safe repeats). **It does not apply here:** this tape's repeat rate is **42.6 %** (20/47
  transitions), and `F,F,F` — the modal tape, 10/20 seeds — is pure repetition. There is no
  self-veto to remove. Measured, closed.
- **Reversal-only guarding.** 0 refused reversals in 67 steps, so a reversal clamp has literally
  nothing to act on (0.000 here; priced at +0.004 elsewhere). Only the **in-bounds** half moves the
  metric.
- **Anything targeting periodicity / orbits / freshness vetoes / 4-cycle detection.** Episodes last
  3.35 steps; the shortest closed walk on a cubic lattice is 4. These levers are unreachable until
  L1 lands, and a span-`w ≤ 3` freshness veto can only ever veto the neck, already barred.
- **Re-wording `skills/strategy.md`'s four bullets.** The agent emits 8 tokens with no reasoning
  trace; prompt-polish axes are noise-dominated on this metric. Change prose only where it asserts
  something **false** (L4).
- **Reading `evaluator.game.*` for the head/food/body, or anything under `SPATIALWORLD/data/`.**
  `_GameGuard` whitelists `screen` alone and records-then-raises on every other attribute;
  `_DiskGuard` patches `os.open` and catches importlib too. Either sets `score = 0.0` and
  `valid = false`. The head is dead-reckonable and the food is decodable — there is no need.
- **A harness that stops delegating.** `eval_runner.py:317` voids any episode with
  `_vlm_calls[0] == 0`. Call `baseline_step` once per step even when you intend to override the pick;
  rung 3 shows overriding 72–83 % of picks still scores 1.0000.

---

### Rigs left in the tree (φ-free: `traces_diag/` is skipped by the descriptor pass)
- `traces_diag/replay.py` — the 45-line adapter replica; **20/20 on `steps_taken`, 20/20 on
  `distance_to_food`**. Use it to price a mechanism offline instead of spending an eval round.
- `traces_diag/measure.py` — per-decision safe-set size, closing rate, repeat rate, death inventory.
- `traces_diag/rungs2.py` — the priced ladder above (survival, meals, honour rate, death mix per rung).
