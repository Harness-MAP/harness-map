# Diagnosis — maze3d_pro iter 0 / `anchor_maze3d_pro` (staged rich anchor)

**Headline:** primary `success` **2/25 = 0.080** (wins **L6** in 7 decisions, **L17** in 33).
All 23 losses burn the full 100/100 budget. The contract is spotless: 2340/2340 decisions parsed to
a legal action, `default_actions = 0` in 25/25 episodes, `vlm_calls == steps_taken` everywhere,
`used_rich_harness: true` everywhere. Nothing is wrong with the plumbing.

**Binding constraint (one sentence):** this harness holds **no state between steps**, so the one bit
that separates a move that happened from a move that was refused — *did the frame change?* — never
reaches the model, and **23 of 23 losses terminate inside a periodic orbit (1882/2340 = 80.4% of all
decisions, median entry step 12, 17 absorbed by step 20)**; compounding it, `AGENT.md` and `rules.md`
describe a **HUD this build does not draw**, which makes the `climb_up`/`climb_down` precondition
literally unsatisfiable — **0 climbs in 2340 decisions on 25 two-floor levels**, and 12 of those 25
levels have their exit on floor 2.

---

## 1. What this harness does (mechanism)

`harness.py:build_agent(evaluator)` is the agent factory the extended evaluator installs as
`ai_model_func`. Exactly two things happen in it:

1. `evaluator.current_prompt = compose_system_prompt()` — concatenates `AGENT.md` + `skills/*.md`
   (sorted) + `rules.md`. The engine's `build_full_prompt` appends the JSON RESPONSE FORMAT + tool
   list on top.
2. it returns `async def agent(frame_data): return await vlm.decide(evaluator, frame_data)`.

`tools/vlm.py:decide` is a two-line pass-through —
`step = evaluator.baseline_step; return await step(frame_data)` — and `baseline_step` is the
*pristine* SpatialWorld per-step fn: capture frame, maintain the rolling image+text history, call the
VLM, parse JSON, return an engine `Action`. **The decision path is therefore identical to the
prompt-only baseline.** Everything else in the directory is scaffolding.

Verified properties of that mechanism, all re-measured in this tree:

* **Zero state.** `grep -nE 'self\.|nonlocal|= \{\}|= \[\]|prev|hash|md5|digest|counter' harness.py
  tools/vlm.py` returns exactly one hit, inside a docstring. There is no object, no counter, no
  previous frame, no ledger — **no place in this harness where a loop-break could live.**
* **`tools/action.py` and `tools/frame.py` are dead.** `harness.py:44` imports `tools.vlm` and
  nothing else; `key_for`, `granularity_steps` and `capture()` appear only at their own `def` lines.
  Two of three tool modules are inert decoration.
* **`memory.md` is dead.** No `.py` in the candidate mentions it — `compose_system_prompt()` reads
  `AGENT.md`, `skills/*.md`, `rules.md` and stops. The declared working-memory lever is unwired.
* The composed prompt is **content-identical but not byte-identical** to `system_prompt.txt`: 3280
  vs 3283 bytes, differing only in three blank lines the `p.strip()` join eats (whitespace-normalised
  comparison is `True`). Scoring-irrelevant; NOTES.md's "byte-compatible" is 3 bytes off.
* The observation channel is a **rolling ~30-turn image+text window**. On L2
  (`maze3d_pro_2_3fo7ydfa`) `prompt_tokens` climbs 1986 → 4576 → 7168 → 9763 → 12358 → 14953 → 17037
  across frames 0,5,…,30 (≈518 tok/turn) then saturates at 17046. The agent *does* see ~30 prior
  frames. §2a shows that is the disease, not the cure.
* The frozen model is **Qwen3-VL-30B-A3B-Instruct**, not a Thinking variant. Over all 2340
  decisions: `completion_tokens` mean **15.8**, median 15, max **19**; `response_text` max **57
  chars**; **0 of 2340** responses exceed 80 chars. Every answer is a bare
  `{"action": …, "granularity": …}`. `rules.md`'s closing "Analyze the screen carefully step by step"
  buys **zero** tokens of deliberation.

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

Fitting the smallest period `p ≤ 30` whose repetition runs unbroken to the end of the tape (coverage
≥ 2p), per episode:

| level | episode | period | entry step | share of tape | the orbit |
|---|---|---|---|---|---|
| L2 | `maze3d_pro_2_3fo7ydfa` | 1 | 0 | **100%** | `move_forward small` ×100 |
| L10 | `maze3d_pro_10_ca2ard2r` | 1 | 0 | **100%** | `move_forward medium` ×100 |
| L15 | `maze3d_pro_15_be4v5uqz` | 1 | 0 | **100%** | `move_forward small` ×100 |
| L18 | `maze3d_pro_18_qcuqewol` | 1 | 0 | **100%** | `move_forward small` ×100 |
| L23 | `maze3d_pro_23_s7jpsyzu` | 16 | 0 | **100%** | `R R F1×6 L L F1×6` ×6 (a corridor shuttle) |
| L5 | `maze3d_pro_5_dlaejbkc` | 1 | 1 | 99% | `move_forward small` ×99 |
| L3 | `maze3d_pro_3_hctt0rh_` | 1 | 6 | 94% | `move_forward small` ×94 |
| L21 | `maze3d_pro_21_uqw3n8nx` | 1 | 7 | 93% | `move_forward small` ×93 |
| L16 | `maze3d_pro_16_h6fa_dqc` | 1 | 8 | 92% | `move_forward large` ×92 |
| L1 | `maze3d_pro_1_m0f_8ise` | 4 | 16 | 84% | `R · F large · L · F large` ×21 |
| L4 | `maze3d_pro_4_zglohwpa` | 4 | 18 | 82% | `L · F large · R · F large` ×20 |
| L19 | `maze3d_pro_19_41zfga02` | 8 | 27 | 73% | `F1 L F2 F1 F1 R F2 F1` ×9 |

**23 of 23 losses end in a terminal orbit; 1882 of 2340 decisions (80.4%) are inside one; median
entry step 12; 17 of 23 absorbed by step 20.** The two wins are the only non-periodic episodes (the
fitter assigns them coverage 2 — the final two `move_forward`s into the exit; every loss has coverage
≥ 12).

Entropy collapse, measured directly: **in 17 of 23 losses the final 50 decisions contain exactly ONE
distinct `(action, granularity)` pair.** The remaining 6 contain 3 or 4. The policy stops being a
policy.

The orbits are provably displacement-free, from the engine's own progress line (emitted at steps 50
and 100 only): `distance_to_exit` is **identical at step 50 and step 100 in 20 of 23 losses**. The
sharpest instance is **L3** — `Progress: Step 50, Distance: 1` and `Progress: Step 100, Distance: 1`:
the agent sat **one cell from the exit** and pressed `move_forward small` into a wall **94 times**.
L5 is the same at distance 1. Of the three exceptions one improved (L11 6→4) and **two got worse**
(L14 2→6, L23 4→6) — L14 was *two cells from the exit at step 50* and wandered to six.

The causal chain, step by step, on L2:

> frame 0 → model answers `{"action":"move_forward","granularity":"small"}` → the translation is
> refused by a wall → **the rendered frame is byte-identical** → frame 1 *is* the same image, and the
> history now holds that image *plus* the model's own previous answer → frame 1 → identical answer →
> … ×100.

This is not a planning failure. The rolling 30-turn window — the only memory in the system — fills
with ~30 copies of one image and ~30 copies of one answer and **conditions the model to repeat**. The
single observable that breaks the tie ("your last action changed nothing") is exactly the observable
that is degenerate here, because a refused translation in this engine renders byte-identically. Round
1 of this chain replayed these very answers through the real evaluator and measured it: **99/99 inert
actions and 1 distinct view in 100 steps on L10 and L18; 93/99 on L3; 48/99 on L1 and L4** — against
12/32 on the L17 win and 2/6 on the L6 win.

Stride behaviour, corrected against the measurement (a prior pass overstated this): the tail stride
is a **single constant value in 22 of 23 losses**, but the model mostly *starts* there rather than
escalating. Mean forward-stride rank over the last 20 vs first 20 decisions rises by >0.25 in only
**5 losses** (L1 2.30→3.00, L4 2.30→3.00, L8 1.88→3.00, L20 2.06→3.00, L25 2.00→3.00); **13 losses
freeze on `small`** from the start, 1 on `large` (L16, rank 3.0 throughout), 3 on `medium`. Aggregate
forward-stride mix actually shifts *toward* `small` in the tail (51% → 62%). So the failure is stride
**freezing**, not stride inflation; what is uniform is that the model never varies the thing that is
not working.

### 2b. Zero floor transitions — an unsatisfiable visual precondition

`rules.md` gates both climbs on a bottom-center HUD string:

> `climb_up`: … **CRITICAL: ONLY use this action if you are standing on a stair AND see "[Q] UP" in
> the bottom HUD.**

and `AGENT.md` promises a top-left HUD with `POS` / `FLOOR` / `DIR` — "Rely on this to map your
progress" — plus "Stair Prompts (Bottom-Center HUD) … This text ONLY appears when you are standing
exactly ON a stair tile."

**This build draws neither.** Downstream in this chain the UI path was identified exactly: the HUD
text is behind an `if False:` in `_draw_ui` and the pose line is `info = ""`. An independent pixel
census of live frames found 7 brown/grey shades with a maximum channel value of 147 and **not one
bright text pixel**, and two provably different poses rendering byte-identically. Behavioural
corroboration from this run alone: 25 episodes all reporting `total_floors: 2`, every one ending
`Final floor: 1/2`, and **not a single `climb_up` or `climb_down` ever emitted in 2340 decisions.**

So the prose does two kinds of damage: it tells the model to read coordinates that do not exist (so
"where am I" is pure hallucination, which is also why the dead-reckoning surface was never built),
and it imposes a precondition that can never be satisfied — a **hard, unconditional veto** on half
the action set.

The price is structural, not marginal. Attributing each level by the floor on which a win was
*observed anywhere in this arm* (9 harnesses × 25 levels):

| level set | levels | anchor |
|---|---|---|
| exit reached on the start floor | L1,3,6,7,13,15,17,20,21,22,23,25 | **2 / 12** |
| exit reached on floor 2 | L2,4,5,8,9,10,11,12,14,16,18,19 | **0 / 12** |
| never won by any harness in the arm | L24 | 0 / 1 |

**Exactly 48% of the benchmark is unwinnable by construction** for a harness that cannot change
floor. The first descendant in this chain to delete the HUD gate made 16 effective floor transitions
and won 3 of the floor-2 levels; the best descendant wins 9/12 of them.

### 2c. What the two wins look like

L6 wins in **7** decisions, L17 in **33** — both start-floor exits, both non-periodic, both with a
genuinely moving view (2/6 and 12/32 inert). They are levels where a myopic forward-biased walk
happens to hit the exit before the orbit closes. They are luck, not navigation.

### 2d. The cost side

Each decision costs ≈2.1 s wall-clock (206 s / 100 steps on L10). At 80.4% of decisions inside
orbits, the anchor spends **≈165 s per losing episode rendering the same frame to itself**. Any
mechanism that refuses a move *before* spending it buys budget and latency together.

## 3. Failure breakdown (23 losses, primary cause, mutually exclusive)

| # | cause | levels | evidence |
|---|---|---|---|
| **13** | **frozen against a wall** — one `(action, granularity)` repeated to the step cap, net displacement 0 | L2, L3, L5, L7, L8, L10, L13, L15, L16, L18, L21, L22, L25 | period-1 orbit entered at step 0–17; identical `distance_to_exit` at steps 50 and 100 |
| **4** | **turn/move limit cycle** — period 4–16, zero net displacement | L1, L4, L19, L23 | `R·F·L·F` and the L23 `RRF×6 LLF×6` shuttle repeating 6–21× to the cap |
| **4** | **late-absorbing wander** — real motion for 30–60 steps, then absorbed | L9 (s36), L12 (s47), L11 (s63), L24 (s30) | orbit covers 37–70% of the tape; final distance 4–9 |
| **2** | **moved, lost the clock, approached then left** | L14, L20 | L14 distance 2 at step 50 → 6 at step 100; L20 stuck at 12 with an 81% `large` orbit |
| — | *parse / illegal action / default fallback* | **none** | 2340/2340 legal; `default_actions = 0` in 25/25 |
| — | *floor transition attempted and refused* | **none** | 0 climbs ever emitted (§2b) |

Cross-check against the numeric pre-pass: `steps_taken == max_steps == 100` in all 23 losses,
`current_floor == 1` in all 25, final `distance_to_exit` 1–15 (median 6).

## 4. Binding constraint — classified

* **HARNESS-BUG (dominant, and cheap).** Two distinct bugs.
  **(i) No feedback channel.** `tools/vlm.py:decide` is a pass-through, so no component of this
  harness can observe that an action did nothing — and in this engine "did nothing" is exactly
  "frame unchanged", a free, exact one-bit signal already sitting in `frame_data.image`. 80.4% of
  decisions are spent in orbits that this single bit dissolves. The three dead surfaces
  (`memory.md`, `tools/action.py`, `tools/frame.py`) show the author anticipated the mechanism and
  never wired it.
  **(ii) Prose describing a HUD that is not drawn.** `AGENT.md`'s perception section and `rules.md`'s
  climb gate were written against an older engine copy. In practice the gate is absolute: 0/2340.
  Both are plain-text edits and they unlock 12 of 25 levels.
* **VLM-CEILING (real, secondary — do not attack it with prose).** The model cannot tell "I moved"
  from "I was blocked" out of two near-identical first-person renders, and it cannot tell it is
  standing on a stair. Median 15 completion tokens means asking for step-by-step reasoning returns
  none. Both are *harness* jobs: measure the fact and hand it over as a stated observation. Round 1
  also priced obedience: after a harness refusal, a first re-ask changed the model's answer **2 times
  in 44**, a second **0 times in 42**.
* **TASK-GAP (latent).** Throughput and coverage — cells gained per decision, reaching a stair at
  all — only become binding once the orbits are gone. At 0.08 they are not yet measurable here; the
  chain hits them at 0.36.

## 5. Concrete changes to try (the gradient — ranked by leverage)

> This anchor is the chain root and rounds 1–4 have run. §5.1–§5.3 and §5.6 are **confirmed landed**
> — two children branched directly off this anchor and scored **0.760** and **0.880** against this
> 0.080. Reproduce them; do not re-derive them. §5.4, §5.5, §5.7 are the live frontier.

### 5.1 Give the harness a per-episode state object and an exact outcome bit *(CONFIRMED)*
*Files: `harness.py:build_agent` (build one state object per episode, close over it),
`tools/vlm.py:decide` (observe → inject → delegate → govern), a new `tools/<state>.py`.*
Hash `frame_data.image` each step; `prev == cur` ⇒ the last action was inert. Wrap the whole step in
`try/except` falling back to the pristine `baseline_step`, so the mechanism can never cost an
episode. **Use exact comparison, never a tolerance** — measured downstream: 83/83 refused
translations leave the frame byte-identical and 0/1077 real moves do, while a 0.2% `_NULL_FRAC`
manufactured 4 false "wall ahead" facts and lost L20 outright.
**Predicted effect:** dissolves the period-1 and period-4 orbits — rows 1–2 of §3, 17 losses. Price
on *inert-action rate* and *distinct views per episode*, not on `success` alone.

### 5.2 Delete the HUD from the prose; replace it with measurement *(free; CONFIRMED)*
*Files: `AGENT.md` §Visual Elements, `rules.md` `climb_up`/`climb_down`, `skills/navigation.md`
last bullet.*
State what the frame actually carries — flat-shaded brown depth-ring corridor, gold exit
`RGB(255,200,0)` with a white outline, cyan stair family `R == 0`, byte-identical frame ⇒ nothing
happened — and **drop the `[Q] UP` / `[E] DOWN` precondition entirely.**
**Predicted effect:** unblocks the 12 floor-2 levels (48% of the benchmark). Confirmed: the first
descendant to do this won 3/12; the 0.880 descendant wins 9/12.

### 5.3 Wire `memory.md` as the per-turn channel — appended, never spliced *(CONFIRMED)*
*Files: `tools/vlm.py` (an `_inject_memory`), `memory.md` placeholders.*
The system turn (`conversation_history[0]`) is the only per-turn text channel the engine has. Render
last-action outcome, heading, local position, proven-wall bearings and a stuck counter into it.
**Append to a stashed pristine base every step** — truncate-then-append deletes the JSON RESPONSE
FORMAT contract the parser depends on from every later call. `memory.md` is excluded from the
evolvable bundle by the adapter, so keep a `_FALLBACK_TEMPLATE` copy inside a `.py` that *is*
bundled.

### 5.6 Read the corridor exactly, not just the tiles *(was "structural, untried" — now the CONFIRMED jackpot: 0.080 → 0.880)*
*Files: a new `tools/plan.py` + `tools/action.py` (bearing algebra beside the key table).*
The renderer is a closed-form flat-shaded painter and it **inverts exactly**: front faces are
`rects[d] = (600/d, 400/d)` centred on (400,300) painted far-to-near, every face is
`darken(base, fog)` with `fog(d) = max(0.4, 1 − 0.08·d)` and integer truncation, so the exact packed
RGB of a 21×21 patch at the frame centre names **both the material and its distance**; the `d = 1`
side quads make columns 0–89 and 710–799 a pure one-cell readout of the left/right neighbour.
Scored against the engine's own data over 25 levels × both floors × every reachable cell × 4
headings: forward ray `(kind, distance)` **5624/5624 exact**, lateral cells **11248/11248 exact**.
Two consequences carry everything: the travel of a translation is known *before* it is spent (so a
refused move is refused pre-emptively, not learned by wasting a decision), and the pose becomes
exactly dead-reckonable, so each observation writes a whole ray plus two lateral cells into a
per-floor occupancy map. BFS on that map gives route-to-exit → route-to-frontier → route-to-stair.
Note the fog plateau from d ≥ 8 collapses the front-wall shade (a wall at 9/10/11 cells reads as 8,
48/5624) — recover `d` there from the painted run's half-width.
**Predicted/observed effect:** 0.080 → **0.880**. This strictly dominates §5.1's view-keyed ledger
(see §6).

### 5.4 Prefer the deepest unexplored bearing *(live frontier)*
Once the map exists, the remaining loss shape is *never saw a stair*: L8, L10, L16, L18, L19 ended
with no stair ever in view. A frontier preference (route to the reachable cell with the most
unsettled neighbours) is the lever that puts one there. Sizing from a descendant: 31% of decisions
re-propose a `(view, action)` already executed ≥2× from that view, 60% repeat one at all.

### 5.5 Do not walk off the stair you are standing on; make departure revocable *(live frontier)*
Measured downstream: **42 of 43 abandoned stair arrivals were the model's own `move_forward`**, and
only 17 of 43 arrivals were ever spent on a climb. Record where the on-stair latch set and route
back when the floor goes stale. **Do not simply loosen the gate** — letting the climb fire on the
arrival decision was replayed and **lost L13**, a start-floor win. Add a return budget first: 4
losses end upstairs having never come back. The clean proof that makes the transition free is
*no reachable frontier left and no exit found on this floor* — so settle the map before climbing,
and keep "floor proven ordinary" distinct from "unseen", since the renderer paints the exit only as
a front face and an exit cell *beside* the agent reads exactly like open floor.

### 5.7 Stride economics, stated as economics *(cheap, and a known regression trap)*
A stride costs one decision whatever its length and stops early at a wall, so it is never worse than
`small` — say *that* instead of recommending `small`. One descendant's rewrite collapsed the model to
94% `small` (1.02 vs 1.54 cells per forward) and ran out of clock on 5 losses with 32–54-cell routes.
Better: let the harness **supply** the stride from the measured clear run (§5.6 gives it for free),
clipped to land *on* an exit and *one cell short of* a stair (the engine walks straight through a
stair tile and the arrival latch needs it to have been one cell ahead). Price on cells-visited /
coverage, never on `success` alone.

## 6. Exhausted — do NOT repeat

* **JSON parsing, the parse fallback, the key table, granularity spelling.** 2340/2340 legal and
  `default_actions = 0` in 25/25 here, and the same at every point downstream. Zero headroom. Do not
  spend an iteration on the response contract.
* **Prompt-only rewording as the cure for the orbit.** `skills/navigation.md` *already* says "Avoid
  turning repeatedly in the same spot — if you have turned several times without moving, commit to a
  direction and move forward", and this run is 23/23 absorbed anyway. Downstream, adherence to an
  explicitly stated PROVEN-REFUSED line was 58–61% and did not move under a full rewrite.
  Mechanisms, not sentences.
* **Asking for reasoning.** This is an *Instruct* model: 0 of 2340 responses exceed 80 characters,
  median 15 completion tokens, max 19. Chain-of-thought prose buys nothing — budget it as a
  structured observation instead.
* **Re-asking the model after a refusal.** Priced: +2 changed answers in 44 on the first re-ask, 0 in
  42 on the second. One re-ask maximum; beyond that, substitute.
* **A frame-diff tolerance.** Refuted and priced: exact equality is correct; a 0.2% tolerance
  manufactures false walls and cost a level.
* **A VIEW-keyed ledger as the state key.** Refuted: the frame digest aliases badly — **257 distinct
  digests over 5624 distinct poses**. Key facts on a dead-reckoned *cell*, never on the view.
* **Frame identity as a climb-success test.** Refuted: **46 of 240 possible climbs (19.2%) leave the
  frame byte-identical even though they succeeded.** Commit a climb optimistically into a fresh map
  epoch and settle it afterwards by comparing per-pose digests across epochs.
* **Letting a grounded climb fire on the stair-arrival decision.** Replayed and refuted: −L13.
* **An unbounded "model casts the deciding vote" tiebreak.** Refuted: a degenerate always-`turn_right`
  proposer swings the target every step and spins in place (11/25 levels against 22/25). Bound the
  vote to the turn-minimal subset so it can never cost a decision.
* **The on-stair detector.** Solved exactly downstream (110 TP / 1 FP / 0 FN in 1880 decisions,
  15/15 grounded climbs effective). Extend it; do not re-derive or re-threshold it.
* **An exit-line-of-sight commit lever.** 0 exit-painted frames in the loss sets measured downstream,
  exactly one (adjacent) in each win. No opportunity in the loss set.
* **`distance_to_exit` as a progress signal.** `success` is the score. A harness that converts
  standing-still into real motion *raises* final `distance_to_exit` on some levels — the anchor's
  "1 cell away" on L3 and L5 is an artefact of never having moved. Never read it as a regression.
* **Stride-escalation as the story.** My own measurement says it is a minority pattern (5/23). The
  universal one is stride *freezing* (22/23 end on a single constant granularity, 13 on `small` from
  step 0). Don't build a mechanism aimed only at inflated strides.

### Dead weight to reuse or delete while you are in here
`tools/action.py` (`key_for`, `granularity_steps`) and `tools/frame.py` (`capture`) are referenced
only at their own `def` lines, and `memory.md` is read by no `.py` at all. They are exactly the three
surfaces §5.1–§5.3 need — wire them rather than carrying them as decoration.

### Rig traps (cost a round each, downstream)
* The evaluator resolves its level file **relative to CWD** — a pricing rig must run from the
  SpatialWorld root.
* `build_agent` purges `sys.modules["tools*"]`, so any lever override set *before* the call is thrown
  away and the "control" silently measures the shipped config. Set levers as **class attributes
  after `build_agent`**.
