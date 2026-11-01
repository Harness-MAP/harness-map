# Diagnosis — maze3d_pro iter 0 / `anchor_maze3d_pro` (staged rich anchor)

**Headline:** primary `success` **2/25 = 0.080** (wins **L6** at 7 steps, **L17** at 33 steps).
All 23 losses burn the full 100/100 decision budget; 2340/2340 decisions parsed to a legal action
(`default_actions = 0` in all 25 episodes, `vlm_calls == steps_taken == 100`, `used_rich_harness`
true everywhere) — the contract is clean and the parser is not the problem.

**Binding constraint (one sentence):** this harness holds **no state between steps**, so the one bit
that discriminates a successful move from a blocked one — *did the frame change?* — never reaches the
model, and **23 of 23 losses collapse into a terminal periodic orbit (80.3% of all decisions, median
entry at step 12)**; compounding it, the two prose priors describe a **HUD that this build does not
draw**, which makes the `climb_up`/`climb_down` precondition literally unsatisfiable and yields
**0 climbs in 2340 decisions on 25 two-floor levels**.

---

## 1. What this harness does (mechanism)

`harness.py:build_agent(evaluator)` is the agent factory the extended evaluator installs as
`ai_model_func`. Two things happen in it and nothing else:

1. `evaluator.current_prompt = compose_system_prompt()` — concatenates `AGENT.md` +
   `skills/*.md` (sorted) + `rules.md` into the system prior. The engine's `build_full_prompt`
   appends the JSON RESPONSE FORMAT + tool list on top.
2. it returns `async def agent(frame_data): return await vlm.decide(evaluator, frame_data)`.

`tools/vlm.py:decide` is a two-line pass-through:
`step = evaluator.baseline_step; return await step(frame_data)`. `baseline_step` is the *pristine*
SpatialWorld per-step function — it captures the frame, maintains the rolling image+text history,
calls the VLM, parses the JSON and returns an engine `Action`.

Consequences that matter, all verified in this tree:

* **The closure holds zero state.** No per-episode object, no previous frame, no step counter, no
  ledger. `grep -n "self\.\|nonlocal\|= {}\|= \[\]" harness.py tools/vlm.py` finds nothing but a
  docstring. There is no place in this harness where a loop-break *could* live.
* **`tools/action.py` and `tools/frame.py` are dead.** `key_for`, `granularity_steps` and
  `capture()` are referenced only from docstrings; no module calls them. 2 of the 3 tool modules are
  inert decoration.
* **`memory.md` is dead too.** `compose_system_prompt()` reads `AGENT.md`, `skills/*.md`, `rules.md`
  — never `memory.md`; no `.py` mentions it outside a docstring. The declared working-memory lever
  is unwired.
* The composed prompt is **content-identical** to `system_prompt.txt` but not byte-identical: 3280
  vs 3283 bytes, differing only in a leading newline and two blank separator lines that the
  `p.strip()` join eats. Scoring-irrelevant; NOTES.md's "byte-compatible" is 3 bytes off.
* The observation channel is a **rolling ~29-turn image+text window**: `prompt_tokens` climbs
  1986 → 17037 over frames 0–30 (≈518 tok/turn) and then saturates at 17045 (L2,
  `maze3d_pro_2_3fo7ydfa`). So the agent *does* see ~30 prior frames — which turns out to be the
  problem, not the cure (§2).
* The frozen model is **Qwen3-VL-30B-A3B-Instruct**, not a Thinking variant: `completion_tokens`
  over all 2340 decisions is mean 15.8 / median 15 / max 19, and **0 of 2340 responses exceed 80
  characters**. Every answer is a bare `{"action": ..., "granularity": ...}`. `rules.md`'s closing
  "Analyze the screen carefully step by step" buys **zero** tokens of deliberation.

## 2. What actually happened (causal chain, with evidence)

### 2a. The absorbing orbit — the dominant mechanism

Action mix over the anchor's own 25 episodes (2340 decisions):

| action | count | share |
|---|---|---|
| `move_forward` | 2090 | **89.3%** |
| `turn_right` | 128 | 5.5% |
| `turn_left` | 122 | 5.2% |
| `move_backward` | **0** | 0% |
| `climb_up` / `climb_down` | **0 / 0** | 0% |

Fitting the smallest period `p ≤ 30` whose repetition runs to the end of the tape:

| level | episode | period | entry step | % of decisions in the orbit | the orbit |
|---|---|---|---|---|---|
| L2 | `maze3d_pro_2_3fo7ydfa` | 1 | 0 | **100%** | `move_forward small` ×100 |
| L10 | `maze3d_pro_10_ca2ard2r` | 1 | 0 | **100%** | `move_forward medium` ×100 |
| L15 | `maze3d_pro_15_be4v5uqz` | 1 | 0 | **100%** | `move_forward small` ×100 |
| L18 | `maze3d_pro_18_qcuqewol` | 1 | 0 | **100%** | `move_forward small` ×100 |
| L5 | `maze3d_pro_5_dlaejbkc` | 1 | 1 | 99% | `move_forward small` ×99 |
| L3 | `maze3d_pro_3_hctt0rh_` | 1 | 6 | 94% | `move_forward small` ×94 |
| L1 | `maze3d_pro_1_m0f_8ise` | 4 | 16 | 84% | `R · F large · L · F large` ×21 |
| L4 | `maze3d_pro_4_zglohwpa` | 4 | 18 | 82% | `R · F large · L · F large` ×20 |
| L19 | `maze3d_pro_19_41zfga02` | 8 | 27 | 73% | `R · F med · F small · L · F med · F small` ×12 |
| L23 | `maze3d_pro_23_s7jpsyzu` | 16 | 0 | **100%** | `R R F F F F F F L L F F F F F F` ×6 |

**23 of 23 losses end inside a terminal periodic orbit. 1878 of 2340 decisions (80.3%) are inside
one. Median entry step 12; 17 of 23 are absorbed by step 20.** The only two non-periodic episodes in
the whole run are the two wins (L6, L17).

The orbits are provably displacement-free, from the engine's own progress line (logged at steps 50
and 100 only): `distance_to_exit` is **identical at step 50 and step 100 in 21 of 23 losses**. L3 is
the sharpest instance — `Progress: Step 50, Distance: 1` and `Progress: Step 100, Distance: 1`
(`session_20261005_132659.log:265,517`): the agent sat **one cell from the exit** and pressed
`move_forward small` into a wall 94 times. Of the two exceptions, one improved (L11 6→4) and **two
got worse** (L14 2→6, L23 4→6).

The causal chain, step by step, on L2:

> frame 0 → model answers `{"action":"move_forward","granularity":"small"}` → the move is blocked →
> **the rendered frame is byte-identical** → frame 1 is the same image, and the history now contains
> that image *plus* the model's own previous answer → frame 1 → identical answer → … ×100.

This is not a planning failure. The rolling 29-turn window, which is the only memory in the system,
fills with ~30 copies of one image and ~30 copies of one answer, and **conditions the model to
repeat**. The single observable that would break the tie — "your last action changed nothing" — is
exactly the observable that is degenerate, because a refused translation in this engine produces a
byte-identical frame. Round 1 of this chain measured that directly by replaying these very answers
through the real evaluator: **99/99 inert actions and 1 distinct view in 100 steps on L10 and L18;
93/99 on L3; 48/99 on L1 and L4** — versus 12/32 on the L17 win and 2/6 on the L6 win.

A second, subtler tell: when blocked, the model **escalates** the stride rather than questioning it.
L1 goes `medium` ×7 → `large` ×43; L8 `large` 36/46 → `large` 50/50; L16, L20, L25 all end at 100%
`large`. It behaves like an agent that believes it is moving. Nothing tells it otherwise.

### 2b. Zero floor transitions — an unsatisfiable visual precondition

`rules.md` gates both climbs on a bottom-center HUD string:

> `climb_up`: … **CRITICAL: ONLY use this action if you are standing on a stair AND see "[Q] UP" in
> the bottom HUD.**

and `AGENT.md` promises a top-left HUD with `POS` / `FLOOR` / `DIR` plus "Stair Prompts
(Bottom-Center HUD) … This text ONLY appears when you are standing exactly ON a stair tile."

**This build draws neither.** The round-1 sibling ran an exact pixel census on the live frames: the
800×600 render uses 7 brown/grey shades with a maximum channel value of 147 and **not one bright
text pixel**; two provably different poses render byte-identically. Behavioural corroboration from
this run alone: 2340 decisions across 25 levels that all report `total_floors: 2`, every episode
ending `Final floor: 1/2`, and **not a single `climb_up` or `climb_down` ever emitted**. A later
candidate in this chain then showed the agent *does* stand on stairs routinely (an exact
frame-derived latch fires 110 true positives / 1 FP / 0 FN in 1880 decisions) — so the opportunity
was always there and the prose veto is what suppressed it.

The prose was written against an older engine copy that did draw the HUD. On this build the model is
told to read coordinates that do not exist (so "where am I" is pure hallucination) and is forbidden
from climbing unconditionally.

The price is structural, not marginal. Scoring the anchor against the exit floor as revealed by *wins
observed anywhere in this arm* (9 levels whose win ends on floor 1; 5 whose win ends on floor 2; 11
never won by any harness):

| level set | anchor |
|---|---|
| exit observed on the start floor (L1,3,6,7,13,15,17,20,23) | **2 / 9** |
| exit observed on floor 2 (L2,4,5,9,11) | **0 / 5** |
| never won by any harness in the arm (L8,10,12,14,16,18,19,21,22,24,25) | 0 / 11 |

A harness that cannot change floor cannot win a floor-2 level at all. Every one of this chain's six
descendants scored 0.28–0.36 against this 0.08, and the first one to actually change floor
(`cand_r002_1`, 16 effective transitions) is also the first to win 3 of the 12 floor-2 levels.

### 2c. What the two wins look like

L6 wins in **7** decisions and L17 in **33** — both start-floor exits, both non-periodic, both with a
genuinely moving view (2/6 and 12/32 inert). They are the levels where a myopic forward-biased
policy happens to walk into the exit before the orbit closes. They are not evidence of navigation.

## 3. Failure breakdown (23 losses, primary cause, mutually exclusive)

| # | cause | levels | evidence |
|---|---|---|---|
| **13** | **frozen against a wall** — one action repeated to the step cap, net displacement 0 | L2, L3, L5, L7, L8, L10, L13, L15, L16, L18, L21, L22, L25 | period-1 orbit entered at step 0–17; identical `distance_to_exit` at steps 50 and 100 |
| **4** | **turn/move limit cycle** — period 4–16, zero net displacement | L1, L4, L19, L23 | `R·F·L·F` and friends repeating 6–21× to the cap |
| **4** | **late-absorbing wander** — real motion for 30–90 steps, then absorbed | L9 (s36), L12 (s47), L11 (s63), L24 (s30) | orbit covers 37–70% of the tape; final distance 4–9 |
| **2** | **moved, out of clock, never absorbed** | L14, L20 | orbit from step 88 / 19 but distance *worsened* (L14 2→6) or stayed at 12 (L20) |
| — | *parse / illegal action / default fallback* | **none** | 2340/2340 legal, `default_actions = 0` in 25/25 |
| — | *floor transition attempted and refused* | **none** | 0 climbs ever emitted (§2b) |

Cross-check against the numeric pre-pass: `steps_taken == max_steps == 100` in all 23 losses,
`current_floor == 1` in all 23, `distance_to_exit` 1–15 (median 6).

## 4. Binding constraint — classified

* **HARNESS-BUG (dominant, and cheap to fix).** Two distinct ones.
  **(i) No feedback channel.** `tools/vlm.py:decide` is a pass-through, so no component of this
  harness can observe that an action did nothing — and in this engine "did nothing" is exactly
  "frame unchanged", a free, exact, one-bit signal sitting in `frame_data.image`. 80.3% of decisions
  are spent in orbits that this single bit dissolves. The dead `memory.md` / `tools/action.py` /
  `tools/frame.py` show the author anticipated the surface and never wired it.
  **(ii) Prose describing a HUD that is not drawn.** `AGENT.md`'s perception section and
  `rules.md`'s climb gate were written for a different build. The climb gate is a *hard veto* in
  practice: 0/2340. Both are plain text edits.
* **VLM-CEILING (real, secondary, do not attack with prose).** The model cannot tell "I moved" from
  "I was blocked" out of two near-identical first-person renders, and it cannot tell it is standing
  on a stair. Median 15 completion tokens means asking for step-by-step reasoning gets none. Both
  are *harness* jobs: measure it and hand it over. Round 1 priced the obedience ceiling too — after
  a harness refusal, a first re-ask changed the model's answer **2 times in 44** and a second
  changed it **0 times in 42**.
* **TASK-GAP (latent, invisible at this score).** Travel throughput and coverage become the binding
  constraint only once the orbits are gone; at 0.08 they are not yet measurable here. The chain's
  later rounds hit it at 0.36.

## 5. Concrete changes to try (the gradient — ranked by leverage)

> This anchor is the start of the chain, and rounds 1–3 have already run. §5.1–5.3 are **confirmed
> landed** (0.08 → 0.28–0.36); reproduce them, don't re-derive them. §5.4–5.6 are the live frontier.

### 5.1 Give the harness a per-episode state object and an exact outcome bit *(highest leverage; CONFIRMED)*
*Files: `harness.py:build_agent` (construct one state object per episode and close over it),
`tools/vlm.py:decide` (observe → inject → delegate → govern), a new `tools/<state>.py`.*
Hash `frame_data.image` each step; `prev == cur` ⇒ the last action was inert. Keep a
`(view_digest, action) → outcome` ledger, dead-reckon heading (a turn is never blocked) and a local
cell position, and mark `(cell, bearing)` WALL on an inert translation / OPEN on a live one. Wrap the
whole step in `try/except` falling back to the pristine `baseline_step`, so the mechanism can never
cost an episode.
**Use an exact comparison, not a tolerance.** Measured downstream: 83/83 refused translations leave
the frame byte-identical and 0/1077 real moves do, so `prev_view != view` is exact while a 0.2%
`_NULL_FRAC` manufactured 4 false "wall ahead" facts and lost L20 outright.
**Predicted effect:** dissolves the period-1 and period-4 orbits = the 17 losses in rows 1–2 of §3.
Price on *inert-action rate* and *distinct views per episode*, not on `success` alone.

### 5.2 Delete the HUD from the prose; replace it with measurement *(free; CONFIRMED)*
*Files: `AGENT.md` §Visual Elements, `rules.md` `climb_up`/`climb_down` entries,
`skills/navigation.md` last bullet.*
State what the frame actually carries (brown depth-ring corridor, gold exit `RGB(255,200,0)`, cyan
stair `RGB(0,204,204)`, byte-identical frame ⇒ nothing happened) and drop the `[Q] UP` / `[E] DOWN`
precondition — on a cyan stair, try the climb and let the frame diff answer.
**Predicted effect:** unblocks the 12 floor-2 levels, i.e. ~48% of the benchmark that is currently
unwinnable by construction. Confirmed: the first descendant to do this *plus* §5.3 made 16 effective
floor transitions and won 3/12 floor-2 levels (anchor 0/12).

### 5.3 Wire `memory.md` as the per-turn channel — appended, never spliced *(CONFIRMED)*
*Files: `tools/vlm.py` (`_inject_memory`), `memory.md` placeholders.*
The system turn (`conversation_history[0]`) is the only per-turn text channel the engine has. Render
last-action outcome, heading, local position, proven-wall bearings and a stuck counter into it.
**Append to a stashed pristine base every step** — truncate-then-append destroys the JSON RESPONSE
FORMAT contract the parser depends on. Note `memory.md` is excluded from the evolvable bundle by the
adapter, so keep a `_FALLBACK_TEMPLATE` copy inside the `.py` that *is* bundled.

### 5.4 A transition ledger that re-aims repeats *(live frontier, sized)*
*File: the state module — a `transitions[(view, action)]` counter; re-aim a proposal that repeats a
transition already taken ≥2 times from this view onto the lowest-count untried bearing.*
Sized on a descendant run: **31% of decisions propose a (view, action) already executed ≥2× from
that view; 60% repeat one at all; 45% sit on a view already seen ≥4×.** §5.1's exact comparison is
the precondition (it makes the view-keyed ledger 0-wrong in 533 facts). Attacks the five
"never found a stair" losses and the 9–47% coverage levels.

### 5.5 Do not walk off the stair you are standing on; make departure revocable *(live frontier)*
Measured downstream: **42 of 43 abandoned stair arrivals were the model's own `move_forward`**, and
only 17 of 43 arrivals were ever spent on a climb. Record the view/step where the on-stair latch
set, and re-aim back to it when the floor goes stale. **Do not simply loosen the gate** — letting the
climb fire on the arrival decision was replayed and **lost L13**, a start-floor-exit win. Add a
return budget before touching the staleness threshold: 4 losses end upstairs having never come back.

### 5.6 Read the corridor, not just the tiles *(structural, untried)*
The frame carries corridor depth in the vertical extent of the paint down the centre column. That
yields (a) the legal stride length for free — which removes the judgement call that §2a shows the
model getting wrong in the escalating direction — and (b) a frontier preference (prefer the bearing
with the deepest unexplored corridor), the one lever that would put a stair *into view* on
L8/L10/L16/L18/L19, where none was ever seen.

### 5.7 Stride economics, stated as economics *(cheap, and a known regression trap)*
A stride costs one decision whatever its length and stops early at a wall, so it is never worse than
`small` — say that instead of recommending `small`. One descendant's rewrite collapsed the model to
94% `small` (1.02 vs 1.54 cells per forward) and ran out of clock on 5 losses with 32–54-cell
routes. Pair it with a harness guard that clips the stride to one cell when the next cell holds a
stair or exit, and **price it on cells-visited / coverage / end BFS distance, never on `success`.**

## 6. Exhausted — do NOT repeat

* **JSON parsing, the parse fallback, the key table, granularity spelling.** 2340/2340 legal and
  `default_actions = 0` in 25/25 here, and the same at every point downstream in this chain. Zero
  headroom. Do not spend an iteration on the response contract.
* **Prompt-only rewording as the cure for the orbit.** `skills/navigation.md` *already* says "Avoid
  turning repeatedly in the same spot — if you have turned several times without moving, commit to a
  direction and move forward", and this run is 23/23 absorbed anyway. Downstream, adherence to an
  explicitly stated PROVEN-REFUSED line was 58–61% and did not move under a full rewrite.
  Mechanisms, not sentences.
* **Asking for reasoning.** This is an *Instruct* model: 0 of 2340 responses exceed 80 characters,
  median 15 completion tokens. Chain-of-thought prose buys nothing; budget it as a structured
  observation instead.
* **Re-asking the model after a refusal.** Priced: +2 changes in 44 on the first re-ask, 0 in 42 on
  the second. One re-ask maximum; beyond that, substitute.
* **The on-stair detector.** Solved exactly downstream (110 TP / 1 FP / 0 FN in 1880 decisions,
  15/15 grounded climbs effective). Extend it; do not re-derive or re-threshold it.
* **An exit-line-of-sight commit lever.** 0 exit-painted frames in the loss sets measured
  downstream, exactly one (adjacent) in each win. No opportunity in the loss set.
* **A frame-diff tolerance.** Refuted and priced: exact equality is correct, a 0.2% tolerance
  manufactures false walls and cost a level.
* **Letting the grounded climb fire on the stair-arrival decision.** Replayed and refuted: −L13.
* **`distance_to_exit` as a progress signal.** `success` is the score. A harness that converts
  standing-still into real motion *raises* final `distance_to_exit` on some levels (the anchor's
  "1 cell away" on L3 and L5 is an artefact of never having moved). Do not read it as a regression.

### Dead weight to reuse or delete while you are in here
`tools/action.py` (`key_for`, `granularity_steps`) and `tools/frame.py` (`capture`) are referenced
only from docstrings, and `memory.md` is read by no `.py` at all. They are exactly the three surfaces
§5.1–§5.3 need, so wire them rather than carrying them as decoration.
