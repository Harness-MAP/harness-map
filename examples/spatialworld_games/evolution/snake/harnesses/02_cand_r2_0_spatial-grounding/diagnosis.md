# Diagnosis — snake iter 0 / `staged_anchor` (parent tag `anchor_snake`)

**Headline:** primary = **0.0000** (0/20 seeds, **0 meals in 67 engine steps**). 20/20 episodes end by
walking the head out of the 4x4x4 box at step 3 or 4 — 3.4 % of the 100-step budget. Binding
constraint: **HARNESS-BUG** for the deaths (`tools/vlm.py:decide` is a bare pass-through, so nothing
tests the one predicate that kills every episode), with a **NAVIGATION-GAP** immediately behind it
(once the box test lands, the frozen model asks for an illegal move on **65.1 %** of 1 953 real
decisions and still eats only ~1.05 pellets per episode).

Everything below is measured on this box today. Two independent instruments:
* `traces_diag/replay.py` — a 45-line offline replica of the adapter (spawn head `(2,3,2)`, neck
  `(1,3,2)`, heading `+X`, G=4, `random.seed(seed)` + 3x`randint(0,3)` food draw) reproduces
  **steps_taken 20/20 AND distance_to_food 20/20** off the recorded tapes. Two fields, and
  `distance_to_food` separates the food axis order uniquely (20/20 at `(x,y,z)`, next best 16/20).
* **The live engine.** I re-ran the anchor plus three mechanism rungs through
  `game_config/eval_runner.py` against the real `Qwen3-VL-30B-A3B-Instruct` on :8000
  (`traces_diag/run_ladder.sh` + `vlm_rung.py` + `decoder.py`), on this lane's own 20 seeds — 64
  live episodes and 2 508 real decisions in total. Every number in §5 is a real-engine measurement,
  not a stand-in extrapolation, and the offline stand-in ladder turned out to **overprice** the first
  rung by ~1.6x (see §5). I also probed the harness's five candidate injection channels black-box
  (§4d) — three of them are dead.

---

## 1. What this harness does (mechanism)

```
eval_runner._apply_harness
  -> harness.build_agent(evaluator)
       - evaluator.current_prompt = compose_system_prompt()    # AGENT.md + skills/*.md + rules.md
       - returns agent(frame_data) = await tools.vlm.decide(evaluator, frame_data)
  -> tools/vlm.py:decide -> evaluator.baseline_step(frame_data)  # the PRISTINE baseline step
```

`compose_system_prompt()` reproduces `system_prompt.txt` exactly modulo 4 blank lines (1624 vs 1628
bytes; identical after whitespace normalisation — verified by importing the module). So the agent is
shown the prompt-only baseline text, and the anchor scores like the prompt-only baseline by design.

**The harness owns zero mechanism, and three of its eight modules are dead code.** `grep` over the
tree finds exactly one executable import in the whole harness: `from tools import vlm`.

| module | status |
|---|---|
| `tools/action.py` (`KEY_MAPPING`, `DIRECTION_VECTORS`, `key_for`, `is_reversal`) | **never imported** — named only inside a `tools/vlm.py` docstring |
| `tools/frame.py` (`capture`) | **never imported** |
| `memory.md` | **never read** — `compose_system_prompt` globs `skills/*.md` only |

No legality filter, no dead reckoning, no frame decode, no substitution, no memory. The decision is
100 % the frozen model's, unreviewed — and `tools/action.py` already holds the exact six direction
vectors a box test needs, sitting unwired two files away.

---

## 2. What actually happened (causal chain, with evidence)

### 2a. The tape is food-independent — the episode is over before perception could matter

Across 20 seeds with 20 different food cells (`distance_to_food` spans 2…8) the model emitted only
**four distinct tapes** and three distinct opening pairs:

| tape | seeds | n | dead-reckoned trajectory | death |
|---|---|---:|---|---|
| `F,F,F` | 1,6,7,8,9,12,14,16,17,20 | 10 | (2,3,2)→(2,3,1)→(2,3,0)→**(2,3,−1)** | wall `z=−1`, step 3 |
| `F,R,F,R` | 2,3,4,13,15,18,19 | 7 | (2,3,2)→(2,3,1)→(3,3,1)→(3,3,0)→**(4,3,0)** | wall `x=4`, step 4 |
| `R,F,D` | 5,10 | 2 | (2,3,2)→(3,3,2)→(3,3,1)→**(3,4,1)** | wall `y=4`, step 3 |
| `R,F,R` | 11 | 1 | (2,3,2)→(3,3,2)→(3,3,1)→**(4,3,1)** | wall `x=4`, step 3 |

The replica reproduces all 20 trajectories and every recorded `distance_to_food` falls out of them.
**The food moves; the tape does not.** `P(applied move closes L1 to food)` over all 67 applied moves
is **12/67 = 0.179**, and a safe distance-reducing move existed on **63/67** decisions. The agent is
not steering.

### 2b. The model never deliberates

`completion_tokens = 8` on **67/67** calls; `reasoning=None` on all of them (this is Qwen3-VL-30B-A3B-
**Instruct**). Every response is the bare string `{"action": "<name>"}`. `default_actions = 0`,
`successful_actions == total_responses` in all 20 episodes: **zero parse failures, zero fallbacks**.
`snake_1` steps 0/1/2 — three identical `{"action": "move_forward"}` at `prompt_tokens`
1320/1827/2334 (a flat +507/turn, i.e. the image and nothing else) — is the canonical instance.
There is no `<think>` block to read on this lane; any gradient that assumes one is misaimed.

### 2c. Death was avoidable on every decision

Mean safe-set size (in-box AND non-body AND non-reversal) at the 67 decisions: **3.45**, min 2, never
0. The pick was inside the safe set on **47/67 (70.1 %)** and outside on **20/67 (29.9 %)** — and
because the body is 2 cells and the engine's only *refusal* is the 180, **every out-of-safe pick was
immediately fatal** (0 refused reversals; `vlm_calls == steps_taken` in all 20 episodes). At the
fatal pick a safe *distance-closing* alternative existed in **18/20** episodes.

### 2d. `rules.md` points the model at a HUD that does not exist — confirmed on a real frame

I dumped `frame_data.image` from a live episode (seed 2, frame 0; `traces_diag/decode_check.py`) and
inventoried it. The frame is 600x800x3. Its entire text content is **two** bands: `Score: 0` at rows
21–32 / cols 21–81, and a small status line at rows 571–579 / cols 21–131. The axis gizmo at rows
526–565 / cols 59–85 is **three unlabelled arrows** (`(255,80,80)` +X, `(80,255,80)` +Y,
`(80,80,255)` −Z). The whole palette is: background `(20,20,25)`, grid `(80,80,100)`/`(60,60,70)`,
food gold `(229,197,0)`, body greens `(54,175,54)`/`(88,187,88)`/`(42,142,42)`, head white
`(220,220,220)`, HUD grey `(150,150,150)`, and the three gizmo colours. **There is no `R`, `L`, `U`,
`D`, `W` or `S` glyph anywhere in the frame**, and no in-frame legality affordance of any kind.

So `rules.md`'s six `Look for the "X" letter on screen` clauses and its promise that *"Indicators only
show for moves that won't cause immediate collision"* describe a build that is not this one. The
harness's single stated perception procedure resolves to nothing, and its single stated safety oracle
does not exist. The model is told to read legality out of the picture, finds none, and emits a canned
direction — which is exactly the observed behaviour.

### 2e. Two more false facts in the priors

* **`AGENT.md`: "Snake starts at the center" — FALSE.** The head spawns at `(2,3,2)` with the neck at
  `(1,3,2)`, heading `+X` (decoded from the real frame: the only bright-white blob snaps to `(2,3,2)`
  at the exact drawn diameter). `y=3` is the bottom face and `+Y` renders **downward**, so
  `move_down` leaves the cube from the spawn row. Seeds 5 and 10 died by precisely that.
* **`AGENT.md`: "Maximum possible score: 64" — 16x off.** `eval_runner.py:78` `SNAKE_EDGE_LENGTH = 4`
  and `primary = min(food/4, 1)`, with `early_stop_score = 4`. The objective is **four pellets**, not
  a grid fill, and `skills/strategy.md`'s "space management" is budget advice for a goal that does
  not exist at this scale.

---

## 3. Failure breakdown (20 episodes, 67 engine steps)

| category | count | rate |
|---|---:|---:|
| **WALL exit (head leaves `[0,3]³`)** | **20 / 20 eps** | **100 % of deaths** |
| body self-collision | 0 | 0 % |
| refused 180 reversal | 0 | 0 % |
| parse failure / default action | 0 | 0 % |
| meals | 0 | — |
| episodes reaching `max_steps = 100` | 0 | 0 % |
| mean engine steps / episode | **3.35** | 3.4 % of budget |
| out-of-safe-set picks | 20 / 67 | 29.9 % |
| fatal picks with a safe closing alternative | 18 / 20 | 90 % |
| repeat-own-last-direction rate | 20 / 47 | 42.6 % |

Death faces: `z=−1` x10, `x=4` x8, `y=4` x2. Fatal step: 3 x13, 4 x7.

**This is a short-episode regime, unlike the other snake lanes.** No wedge (0 refused reversals), no
orbit, no body death — the snake dies before a cycle can form. Levers aimed at periodicity, freshness
vetoes, orbit-breaking or reversal-guarding have *no steps to act on* here.

---

## 4. Binding constraint

### 4a. The deaths are a HARNESS-BUG

`tools/vlm.py:decide` forwards the model's key unreviewed. The cheapest conceivable mechanism — one
`all(0 <= c < 4 for c in head + delta)` test — needs **no pixels and no oracle**: the spawn is fixed
and the head is exactly recoverable from the keys the harness itself emitted. I measured that
reckoning against an independent frame decode over **1 953 real decisions on the live engine:
1 944/1 951 exact agreement, 2 decoder abstains, and all 7 disagreements were the DECODER being
wrong, not the reckoning**. The argument is structural: the ledger advances deterministically by the
key the harness shipped, so a drift would be PERMANENT — every subsequent frame would disagree. All
7 disagreements are isolated single frames whose successor agrees again (e.g. seed 3 steps 58/84:
reckoned `(3,0,1)`, decoded `(2,0,2)`; seed 20 steps 40/44: reckoned `(0,2,2)`, decoded `(0,2,3)`),
so the error is on the pixel side every time. Dead reckoning is the most reliable state channel on
this lane, and it is free. `tools/action.py` already contains the vectors it needs. That is a code
defect, not a model limit.

### 4b. What is behind it is a real NAVIGATION-GAP — and it is bigger than the offline rig said

Once the box test lands and episodes run ~95 steps instead of 3.35, the live model's behaviour is
visible for the first time, and it is worse than the dead tape suggested. Over **1 953 decisions on the
live engine under the clamp** (20/20 seeds, mean 97.7 steps/episode):

* the model asked for a move that was **out-of-box or a 180 reversal on 1 271/1 953 = 65.1 %** of
  decisions (vs 29.9 % on the 3-step tapes) — so the clamp substitutes two picks in three;
* the asked-action mix collapses to **`move_forward` 700 / `move_backward` 650 / `move_down` 291 /
  `move_right` 253 / `move_up` 55 / `move_left` 4**. **69 % of all asks are on the Z axis**, and
  `move_forward`/`move_backward` are exact opposites, which is where the 19.7 % reversal share comes
  from: the model repeatedly requests its own 180, a request-level 2-cycle the engine refuses;
* the legal set was never empty (sizes 2:238, 3:780, 4:730, 5:205).

So the surviving snake does not steer; it is pushed around by the substitute rule. **Survival is a
harness fix; scoring is not.** Diagnose in that order — nothing past step 4 is observable until the
head stops leaving the box.

### 4c. The frozen model's compliance ceiling, measured

I tested whether the model can be *scaffolded* rather than overridden, by splicing a per-turn
directive into the system message (the one live text channel — see §4d): *"The harness has computed
the only safe move for THIS turn: "<x>". Ignore the image. Respond with exactly {"action": "<x>"}"*.
It was obeyed on **1 of 4** spliced turns (seed 19). The text does arrive — the obeyed turn emitted
`move_up`, an action absent from all 67 baseline emissions — but the model reverts to
`move_forward`/`move_right` most of the time. **Advice is not control on this model; `act.key` is.**
Plan for a harness that *ranks and vetoes*, and treat any prompt-side scaffold as a bonus, not a
mechanism.

### 4d. The channel map — three of the five obvious injection points are DEAD

This is the part worth reading before writing any mechanism. All five probed black-box on the live
engine; the instrument is the `prompt_tokens` ladder, which is a flat **+507/turn** (image only) for
the pristine anchor. Full detail and the probe sources: `traces_diag/channels.md`.

| channel | verdict | evidence |
|---|---|---|
| `evaluator.current_prompt` mutated per step | **DEAD** | `len` 1624→3241→4858, ladder unchanged at 1320/1827/2334 |
| `frame_data.image` replaced | **DEAD** | swapped in a 300x400 array; image cost still +507 — `baseline_step` re-captures the surface itself |
| `conversation_history.append(user msg)` | **DEAD** | 1080 chars/turn, ladder unchanged; the appended message is dropped |
| **`evaluator.capture_game_screenshot` wrapped** | **LIVE** | returning a 300x400 PNG moved the image cost 507 → **140**/turn |
| **`conversation_history[0]["content"][0]["text"]` mutated in place** | **LIVE** | 1080 chars/turn moved the ladder to +**827** (507 image + 320 text) |

Consequences for the gradient:
* **`memory.md` cannot be wired the way its own comment says.** "harness.py can fill this per step and
  inject it into the prompt" via `compose_system_prompt`/`current_prompt` measures **exactly zero**
  after turn 1. The working route is the in-place system-message mutation, and it only lands from
  **turn 2** (`conversation_history` is empty on turn 1). Write it as
  `t.split(MARK)[0] + MARK + block`, never `+=`.
* **Annotating the observation means wrapping `capture_game_screenshot`,** not touching
  `frame_data.image`. `tools/frame.py` is the natural home and is currently dead code.
* Anti-cheat boundary (`ANTICHEAT_GUARD=1` is on in this lane): `evaluator.game.<anything but
  `screen`>` is recorded and raises → `score = 0.0, valid = false`; so is any read under
  `SPATIALWORLD/data/` from inside the harness dir. `evaluator.last_score` is engine score state that
  the `.game` guard does **not** sink — do not read it; a meal is observable from the frame. And
  `_vlm_calls[0] == 0` over the run voids the candidate, so `decide` must call `baseline_step` every
  step (it may still override `act.key`).
* `Action(type='key_press', key=..., metadata={... 'function_called': 'move_forward', ...})`. Only
  `act.key` is pressed; read the model's own choice from `metadata['function_called']` —
  `metadata['direction']` is unreliable (it read `'forward'` on a `move_right` press).

---

## 5. The priced ladder — measured on the LIVE ENGINE, not a stand-in

Each rung is a different `tools/vlm.py` dropped into a copy of this harness and run through
`game_config/eval_runner.py` against the real `Qwen3-VL-30B-A3B-Instruct`, 20 lane seeds, 8-way
parallel (`traces_diag/run_ladder.sh` + `vlm_rung.py`). `illegal-ask` = the frozen model named a move
that was out-of-box or a 180; `honour` = the fraction of decisions the harness pressed the model's own
key.

| rung | mechanism it needs | n | primary | meals/ep | steps/ep | illegal-ask | honour |
|---|---|---:|---:|---:|---:|---:|---:|
| **0** - pass-through (**= THIS anchor**) | none | 20 | **0.0000** | 0.00 | 3.5 | 0.290 | 1.000 |
| **1** - in-box + no-180 clamp, least-visited substitute | dead-reckoned head only - **NO pixels** | 20 | **0.2625** | 1.05 | 97.7 | 0.651 | 0.349 |
| **2** - rung 1, but the substitute is **food-ward** | + the food cell decoded from the frame | 7 | **0.8214** | 3.29 | 54.4 | 0.588 | 0.412 |
| **3** - **NO clamp**; decoded state panel drawn onto the frame | + wrapped `capture_game_screenshot` | 17 | **0.0441** | 0.18 | 6.0 | 0.181 | 1.000 |

Per-seed detail and decision counts: `traces_diag/LADDER.md`. Rungs 0 and 1 are the full 20 seeds;
rungs 2 and 3 are at n=7 and n=17 because I stopped the rig to give the box back (a clamped episode
runs ~100 steps x a 29-image history, so each costs minutes of GPU). Rung 2's 7 seeds are
1.00/1.00/1.00/1.00/0.75/0.75/0.25 — the ordering is not in doubt, but treat 0.8214 as a floor-ish
estimate with a wide interval, not a forecast.

**Read three things off it.**

1. **The box test is the whole first rung and it is free of perception.** 0.0000 → **0.2625** with
   nothing but the fixed spawn and the keys the harness itself emitted; mean episode length 3.35 → 97.7.
   Every other lever is invisible until this one lands, because the anchor never reaches step 5.
2. **The food read is where the remaining metric is.** Changing *only* the substitute rule — rank the
   legal set by L1 to the frame-decoded food instead of by visit count — takes 0.2625 → **0.8214**
   (n=7: 1.00, 1.00, 1.00, 1.00, 0.75, 0.75, 0.25) and cuts episode length from 97.7 to 54.4 because
   episodes now *early-stop on the 4th pellet* instead of running out the 100-step budget. The clamp
   buys survival; the ranking buys the score.
3. **Scaffolding the observation is real but not sufficient.** Rung 3 wraps
   `capture_game_screenshot` to draw the harness's head cell / food cell / legal-move list onto the
   frame and then presses whatever the model says — **no clamp at all**. It barely scores (0.0441),
   but it is far from inert: the **illegal-ask rate falls 0.290 → 0.181** (a 38 % relative drop in the
   same no-clamp regime, where an episode can contain at most one illegal ask — the fatal one), mean
   episode length rises **3.5 → 6.0**, two seeds ate pellets (0.25 and 0.50), and the action
   vocabulary opens up — the anchor asked `move_left` **0/67** times and the clamped runs 4/1 953,
   while rung 3 asks it on ~20 % of decisions. Seeds 6 and 9 were still alive at 60 and 52 steps when
   I stopped the rig. **The frozen model does read the panel** (8-px glyphs and all); it just cannot
   convert that into reliable survival on its own.

**A correction to the offline ladder.** `traces_diag/ladder.py`/`rungs2.py` priced rung 1 at
**0.375–0.4375** against a deterministic stand-in fitted from the anchor's own 67 emissions. The live
engine says **0.2625** — the stand-in overprices it ~1.6x, because a fitted memoryless cycler wanders
differently from the real model (the real model asks an *illegal* move on 65 % of clamped decisions,
vs 30 % on the 3-step tapes). The offline rig got the **ordering** exactly right and the **magnitudes**
wrong; use it to rank mechanisms, never to forecast a score.

### Why the clamped snake still does not steer — the model is worse than random at it

Measured over all **1 953** clamped decisions (a food read was available on every one of them):

| | out-of-box | 180 reversal | legal |
|---|---:|---:|---:|
| share of the model's asks | **45.4 %** | 19.7 % | 34.9 % |

and the ask distribution is essentially **independent of where the food is**:

| action | food lies that way | model asked it |
|---|---:|---:|
| `move_left` | **52.5 %** | **0.2 %** |
| `move_backward` | 43.3 % | 33.3 % |
| `move_forward` | 44.3 % | 35.8 % |
| `move_up` | 42.7 % | 2.8 % |
| `move_down` | 34.9 % | 14.9 % |
| `move_right` | 26.1 % | 13.0 % |

69 % of all asks are on the Z axis (`forward`/`backward`, exact opposites — which is where the
19.7 % reversal share comes from), and `move_left` is asked **4 times in 1 953 decisions** while
being a food-ward axis on **52.5 %** of them. Restricted to the 682 decisions where its ask *was* legal, the
model is distance-optimal **28.2 %** of the time against **51.1 %** for picking uniformly among the
legal moves, and strictly reduces the distance **26.1 %** against **49.1 %**. **It is about half as
good as random at steering toward the food.** That is a VLM-CEILING on food localisation, and it is
why rung 2's gain has to come from the harness computing the direction rather than from asking better.

### Where the frame decode is and is not trustworthy
* **Head — use arithmetic, not pixels.** Dead reckoning vs an independent decode over 1 953 live
  decisions: **1 944/1 951 agree**, with 2 decoder abstains and 7 isolated decoder errors (seed 3 steps 58/84:
  reckoned `(3,0,1)`, decoded `(2,0,2)` — a near-coincident projected pair slipping through the ±4 px
  diameter gate), and no detectable reckoning drift.
* **Food — decodable but it FLICKERS.** Across those same episodes the decoded food cell changed
  **39 times against only 10 real meals**; 38 of 83 runs are 1–2-frame excursions to one specific
  near-coincident neighbour (`(0,1,2)↔(0,1,3)`, `(3,2,2)↔(2,3,3)`, `(1,2,3)↔(0,2,3)`,
  `(1,1,2)↔(0,1,3)`) and then snap back. Partial occlusion by the passing snake perturbs the blob's
  bounding box and flips the size gate. See L2 for the fix, which costs no pixels.
* Validation that the decoder is right when it is unoccluded: on a real dumped frame (seed 2, frame 0)
  the head snaps to `(2,3,2)` and the food to `(0,0,0)`, both at the exact drawn diameter, and `(0,0,0)`
  is independently what the 20/20-validated replica says seed 2's food is.

---

## 6. Concrete changes to try (the gradient — ranked by leverage)

> Note on what is already claimed: `prop_r001_908c91/cand_r001_0_spatial-grounding` (built from the
> earlier read of this parent) already ships **L1 + the body model + the prose repair**. Treat L1 as
> *done* and start at **L2**; L1 stays here because it is the mechanism every later rung sits on.

### L1 (done in `cand_r001_0`) · `tools/vlm.py:decide` — dead-reckon the head, veto an out-of-box key
Keep module state `(head, last_dir)` from `(2,3,2)` / `(1,0,0)`; after `baseline_step` returns, read
`act.metadata["function_called"]`; if `head + DIRECTION_VECTORS[name]` leaves `[0,3]³` or reverses
`last_dir`, rewrite `act.key` to `KEY_MAPPING[substitute]`; advance the ledger by the key you
*shipped*. **0.0000 → 0.2625 primary, 3.35 → 97.7 steps/episode, measured on the live engine over 20 seeds /
1 953 decisions** (1 body death in 20, so the body model is worth little on its own — see §5).
*Traps:* a refused 180 does not move the snake and does not increment `steps_taken`, so never advance
the ledger on a reversal; store the cell the head left **FROM**, not the post-commit head.

### L2 · `tools/vlm.py` — make the SUBSTITUTE food-ward, and gate the food read on a meal  (the big one)
This is where the remaining metric is. Two parts, and the second is not optional:
1. **Rank, don't wander.** The clamp's substitute currently resolves ties by visit count, which is a
   random walk with extra steps: 65 % of picks are substitutes, so the substitute rule *is* the
   policy. Rank the legal set by L1 distance to the food cell and take the nearest (visit count only
   as the tie-break). Honour the model whenever its pick is legal *and* distance-optimal.
2. **Stabilise the food read with a self-consistency gate.** My decoder read the food's cell on
   1 953 live frames and its value changed **39 times against only 10 real meals** — 38 of 83 runs are
   1–2-frame flickers between a true cell and one near-coincident neighbour (`(0,1,2)↔(0,1,3)`,
   `(3,2,2)↔(2,3,3)`, `(1,2,3)↔(0,2,3)`): partial occlusion by the passing snake perturbs the blob's
   bounding box by a pixel or two and flips the ±4 px diameter gate. A food-ward ranking that chases
   a flickering target oscillates. **The fix needs no extra pixels:** the food cell can only change
   when the snake has just eaten, and the harness knows that exactly — accept a new food read **only
   when the previous head equalled the previously accepted food cell**, otherwise hold the last
   accepted value. (Secondary guard: require the blob's area to match `pi*r^2` for its accepted
   diameter, and ABSTAIN rather than snap when it does not.)
**Measured: 0.2625 → 0.8214 on the live engine** (§5 rung 2, n=7: 1.00 x4 / 0.75 x2 / 0.25),
changing nothing but the substitute rule. Expect the gain as *meals*, not survival — survival is
already ~98 steps — and expect mean episode length to *fall* (97.7 → 59.2) because episodes start
early-stopping on the 4th pellet.

### L3 · `tools/frame.py` + `tools/vlm.py` — annotate the OBSERVATION (wrap `capture_game_screenshot`)
`tools/frame.py` is dead code and `capture_game_screenshot` is the **only** live perception channel
(§4d). Wrap it to draw the harness's own state onto the frame the model is shown: the head cell, the
food cell, and the legal-move list. Keep the real lattice underneath — the panel is an addition, not a
replacement. *Judge this by the honour rate and the illegal-ask rate, not by the scalar*: its job is
to raise the fraction of decisions the frozen model gets right so the harness overrides less.
**Measured on the live engine with NO clamp behind it (§5, rung 3): illegal-ask 0.290 → 0.181, mean
episode 3.5 → 6.0 steps, two seeds ate pellets, and `move_left` — asked 0/67 times by the anchor and
4/1 953 under the clamp — is asked on ~20 % of decisions.** So the frozen model does read an
8-px-font panel; it just cannot convert that into survival alone (primary 0.0441). Ship it *behind*
the clamp,
not instead of it, and expect the gain in the honour rate rather than the score. Likely further
upside: draw larger glyphs, and encode legality **geometrically** (recolour the lattice dots of the
illegal neighbour cells) rather than as text.

### L4 (done in `cand_r001_0`) · `AGENT.md` + `rules.md` — stop asserting three false facts
Replace "starts at the center" with the measured spawn `(2,3,2)`/neck `(1,3,2)`/`+X` and the fact that
`+Y` renders downward from a `y=3` face; replace "Maximum possible score: 64" with "4 pellets ends the
episode, `primary = min(food/4, 1)`, 100 steps"; **delete** the six `Look for the "X" letter` clauses
and the "indicators only show for safe moves" promise — §2d confirms on a real frame that no such
glyph exists. Sign-correct and cheap, but judge it by the `move_down`-at-spawn count and the closing
rate, never by the scalar: prose cannot fix a wall death the harness does not test for.

### L5 · `tools/vlm.py` — a per-turn state block in the system message (the memory lever, re-aimed)
`memory.md` is a documented lever that nothing reads, and the route its own comment describes
(`compose_system_prompt` / `current_prompt`) is **dead after turn 1** (§4d). The route that works is
mutating `conversation_history[0]["content"][0]["text"]` in place right before `baseline_step`:
`t.split(MARK)[0] + MARK + block`, from turn 2 on. Render the self-derived ledger — head cell,
heading, legal moves, the recent trail — and nothing privileged. **Low confidence as a scorer**: the
measured compliance with an explicit per-turn directive is 1/4 (§4c). Its value is as an *instrument*:
the honour rate becomes a clean readout of whether the frozen model is participating at all, and the
override fraction is currently 65 %, which is the part of this harness a judge is most likely to call
"the harness playing the game".

### L6 · `tools/vlm.py` — a second ask on a vetoed pick (cheap, untested here)
When the pick is illegal, re-ask `baseline_step` once with the state block naming the legal set, and
only fall back to the ranked substitute if the second answer is also illegal. This keeps the frozen
model nominally in the loop on the two-thirds of decisions the harness currently takes from it, and
costs one extra VLM call per veto (`vlm_calls` is a floor, never a ceiling). Price it against the
honour rate; I did not measure it.

---

## 7. Exhausted / do NOT repeat

- **Per-step injection through `evaluator.current_prompt` or `frame_data.image`.** Both measured
  **dead** on the live engine (§4d). A candidate that wires either will score exactly like the anchor
  and look like a null result for the wrong reason.
- **Appending a message to `conversation_history`.** Dropped before the request (§4d). Only the
  in-place system-message mutation lands.
- **"License the repeat" in `rules.md`** (the standing prose lever on the long-episode snake lanes,
  where the model had absolutised "no 180" into "never repeat a direction"). Does not apply here: the
  repeat rate is already **42.6 %** and the modal tape `F,F,F` (10/20 seeds) is pure repetition. There
  is no self-veto to remove. Closed.
- **Reversal-only guarding.** 0 refused reversals in the anchor's 67 steps — a reversal clamp has
  nothing to act on until episodes get long. The **in-bounds** half is the whole gain.
- **Periodicity / orbit / freshness-veto / 4-cycle levers.** Episodes last 3.35 steps; the shortest
  closed walk on a cubic lattice is 4. Unreachable until L1 lands (and once it does, the real pattern
  is a *request-level* forward↔backward 2-cycle the engine already refuses — see §4b).
- **Re-wording `skills/strategy.md`'s four bullets.** 8 completion tokens, `reasoning=None`, zero
  parse failures: there is no deliberation to steer. Change prose only where it asserts something
  false (L4).
- **Reading `evaluator.game.*` (beyond `screen`), `SPATIALWORLD/data/*`, or `evaluator.last_score`.**
  The first two are sinked by `_GameGuard`/`_DiskGuard` and void the candidate; the third is engine
  score state the guard happens to miss, and it is not needed — a meal is observable from the frame
  and from the head/food coincidence the ledger already computes.
- **A harness that stops delegating.** `eval_runner.py` voids any run with `_vlm_calls[0] == 0`.
  Call `baseline_step` every step even when you intend to override the key.
- **Trusting the frame decode for the HEAD.** Dead reckoning beat it **1 944/1 951** (§4a) — 2 decoder
  abstains and 7 isolated decoder errors, 0 reckoning drift. Use pixels for the **food** (which is
  not reckonable) and arithmetic for the head.

---

### Rigs left in the tree (`traces_diag/` is skipped by the descriptor pass, so these are φ-free)
- `replay.py` — the 45-line adapter replica; **20/20 `steps_taken`, 20/20 `distance_to_food`**.
- `measure.py` — per-decision safe-set size, closing rate, repeat rate, death inventory.
- `ladder.py` / `rungs2.py` — the offline stand-in ladder (read it as an ordering; it **overpriced**
  rung 1 by ~1.6x — see §5).
- `decoder.py` + `decode_check.py` — the head/food decode, validated on a real frame (head `(2,3,2)`,
  food `(0,0,0)` on seed 2, both exact) and then measured over 1 953 live frames.
- `channels.md` — the probed channel map and the Action/FrameData contract.
- `vlm_rung.py` / `run_ladder.sh` — the live-engine ladder: drop a rung's `tools/vlm.py` into a copy
  of the harness and run `eval_runner.py` per seed, 8-way parallel. `LADDER.md` has the per-seed table.
- `vlm_advise.py` — a ready-to-run L5 design (clamp + per-turn system-message state block). **Never
  run**; no number here comes from it.
