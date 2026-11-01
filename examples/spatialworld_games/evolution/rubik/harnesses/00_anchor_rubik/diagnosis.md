# Diagnosis — rubik (spatial_world) / `staged_anchor` (`anchor_rubik`), arm `k4_claude_pv`

**Headline:** primary = **1/20 solved = 0.050**. 1901 steps / 1901 VLM calls / 1901 engine actions;
**0 parse failures, 0 default actions, 0 refused rotations, 0 animation timeouts, 0 warnings, 0 errors**
in all 20 session logs. The plumbing is perfect and the decision is empty: the single win is a level
whose one-move solution happened to be the model's step-0 constant, and it ended at **step 1**.

Binding constraint: **VLM-CEILING with a fully-available scaffold — the frozen agent emits a
PERIODIC WORD chosen in its first ~2 steps and never revises it, because nothing in the loop converts
the frame into a decision; meanwhile the frame the agent is already shown is an exact, axis-aligned
3×3 sticker grid that 9 pixel lookups decode, and 2–4 camera steps turn that into the cube's exact
configuration (99.0 % / 100.0 % of the radius-4 ball).** The budget is not the constraint: 100 steps
against a ≤4-move scramble is a **94 % surplus** once the state is read.

Three numbers carry this diagnosis:

1. **The frame is not in the decision path at all.** 8 of 20 episodes emit **one byte-identical
   61-character JSON answer for all 100 steps**, while the picture changes every step (the camera
   cycles through 8 distinct poses, or the cube is visibly turning). Median distinct answers per
   100-step episode = **2**. The policy is a constant function of the text, not a function of the image.
2. **65.9 % of every step taken could not reach a configuration the cube had not already stood on**
   — 800 state-preserving view rotations (42.1 %) + 452 face turns that returned to an
   already-visited configuration (23.8 %). And **750 of those 800 view turns (39.5 % of the entire
   budget) were `rotate_cube_z_*`, a ROLL about the viewing axis**: it changes neither the cube nor
   which face is visible. Nothing was learned and nothing moved.
3. **The scaffold is sitting in the frame, measured.** On **973 real rendered frames** I pulled out of
   this arm's own videos: 673 are an exact **240×240 axis-aligned face at origin (130,130)** = 3×3 of
   80 px stickers; **6201 probe reads land on exactly SIX RGB values (0.03 % off)**; and 4 probes 6 px
   outside that box separate square-on from off-axis with **0 false accepts in 300 off-axis frames**.
   Group theory then prices the read: **one face read pins 65.6 % of the radius-4 ball, F+R (one 90°
   camera turn = 2 steps) pins 99.0 %, F+R+B (4 steps) pins 100.0 %.**

---

## 1. What this harness does (mechanism)

The anchor is a deliberate no-op wrapper: it reproduces the prompt-only baseline while exposing the
rich-harness surface. Read in full, it is four hops and no policy.

| file | what it actually does |
|---|---|
| `harness.py:build_agent` | puts `HERE` on `sys.path`, purges any inherited `tools` module, sets `evaluator.current_prompt = compose_system_prompt()`, returns `async agent(frame_data) -> vlm.decide(...)`. |
| `harness.py:compose_system_prompt` | `AGENT.md` + `skills/*.md` (none exist) + `rules.md`, stripped and newline-joined. Verified byte-equal to the shipped `system_prompt.txt` modulo whitespace. |
| `tools/vlm.py:decide` | `step = evaluator.baseline_step or evaluator.openai_ai_model; return await step(frame_data)`. **That is the entire decision.** |
| `tools/action.py` | 70 lines of correct, **never-called** tables (`FACE_KEYS`, `VIEW_AXES`, `LEGAL_ACTIONS`, `is_legal`, `is_view_rotation`, `direction_const`). Dormant by design. |
| `tools/frame.py:capture` | `evaluator.capture_game_screenshot()` — **never called**; `baseline_step` captures its own frame. |
| `memory.md` | a scratchpad template that is **not injected** (and whose fields `last_pos` / `last_dir` / `stuck_counter` are maze leftovers, meaningless for a cube). |

So every step is the pristine SpatialWorld `RubikOpenAIEvaluator.openai_ai_model`:

```
capture_game_screenshot()                         # 500x500 PNG -> base64
build_conversation_history(..., history_window=29) # slot 0 = system prompt, then the last 29
                                                   # (user image + assistant answer) PAIRS
create_openai_response(...)                        # ONE call, Qwen3-VL-30B-A3B-Instruct
extract_action_info_from_response(...)             # JSON -> RUBIK_ACTIONS -> Action
self.execute_action(action)                        # <-- the EVALUATOR executes, inside this fn
return action
```

Three contract facts a proposer must not rediscover the hard way:

* **`openai_ai_model` executes the action itself** (`rubik_openai_evaluation.py:606`), then returns it.
  The engine does **not** re-execute: measured exactly **1901 engine actions for 1901 steps**. A
  candidate that *delegates* and then *also* executes will double-turn the cube; a candidate that
  builds its own `Action` without executing will turn nothing.
* **`current_prompt` is read on the first call only** (`build_full_prompt(is_first_call)`); thereafter
  slot 0 is carried forward from `previous_history[0]`. Per-step prose must be spliced into
  `evaluator.conversation_history[0]` in place.
* **Solved is free.** `state_callback` calls `engine.stop_evaluation()` the instant
  `cube_solved` is true (`:641-653`). The harness never has to *recognise* solved — only to *visit*
  it. L2 proves it live: `steps: 1`, `vlm_calls: 1`, `time_elapsed: 2.0 s`.

The composed prior is 971 chars: six face letters + colours, the two move families, and three
sentences of advice ("analyze the current state carefully", "identify which pieces are in the correct
positions", "plan your moves carefully"). No method, no notion of camera pose, no notion of distance.

## 2. What actually happened (causal chain, with evidence)

I replayed every executed move of all 20 episodes through an **independent 3×3×3 cube model built from
geometry alone** (`/tmp/rubdiag/cube.py`: 26 cubies as (position, 3×3 orientation matrix); a quarter
turn rotates every cubie in the layer). It validates against the quarter-turn ball sizes
**12 / 114 / 1068 / 10011**, `ord(F)=4`, `ord(RU)=63`, `ord(RU')=105` (my chirality is the mirror of
the engine's — a global relabel that leaves every coverage number below invariant).

### The chain

> step 0 is the only echo-free decision → it returns one of **3 names across 20 levels** → that answer
> and its own repeat enter the 29-turn history → the strongest pattern in the context becomes
> *"the assistant answers X"* → the frozen Instruct model (22–24 output tokens, `reasoning=None` on
> **1901/1901** calls) completes the pattern → the episode becomes a **fixed periodic word** →
> a periodic word either cycles in place (never advances) or marches monotonically out of the only
> neighbourhood that can contain the solution → 100 steps elapse.

**Step 0 is a constant.** Twenty different scrambles, twenty different pictures, **three answers**:
`rotate_front_cw` ×11, `rotate_cube_z_cw` ×8, `rotate_down_cw` ×1. The one outlier is the one win.
The whole 0.050 is this coin flip, not a mechanism.

**The lock is periodic, and the period is visible in six steps.** Tail-period of the last 40 picks:

```
L 1  z_cw z_cw z_cw z_cw z_cw z_cw …                       period 1   (100 identical answers)
L 6  z_cw z_acw z_cw z_acw …                               period 2   (camera +45°, -45°, forever:
                                                                       the agent sees TWO pictures)
L 5  z_cw z_acw y_cw y_acw …                               period 4   (camera returns home each cycle)
L 8  front_cw ×100                                         period 1   (F^4 = identity: 4 states total)
L16  F F F F U U U U L L L L R R R R D D D D B B B B …     period 24  (every block is the identity)
L18  front_cw down_cw front_cw down_cw …                   period 2   ((FD)^50 marches out)
L 4  F L R U D B F' L' R' U' D' B' …                       period 12  (reciting AVAILABLE ACTIONS in order)
L13  F U R F U' R' …                                       period 6
```

**17 of 20 episodes have a strictly periodic tail (period ≤ 14); 16 are periodic from step 0; none
ever breaks its period.** The three "aperiodic" ones (L10, L19, L20) are periodic over 8–14 steps with
occasional phase slips. This is not noise — it is a program, chosen immediately and executed to the
budget.

### Why a periodic word cannot win

A word `W` repeated forever visits only the cyclic structure `⟨W⟩`. Measured, the two branches:

* **Small order → cycles in place.** L8/L15 play `F` 100×: `F^4 = id`, so the episode stands on
  **4 configurations** in 100 steps (97 of which are revisits). L16's `X X X X` blocks are each the
  identity, so it returns home every 24 moves. L1/3/9/11/12 never touch the cube at all —
  **1 configuration, 100 steps.**
* **Large order → marches out.** L7/L13/L18 reach **101 distinct configurations** and spend
  **4 of 100 steps** inside the ≤4-move ball of their own start; they leave at step 4 and **never
  return**. A level's solution is by construction inside that ball.

Steps inside the ≤4 ball of the episode's own start: **1173/1901 = 61.7 %** — and that number is a
mirage, because 800 of those 1173 are the episodes that never moved the cube. The honest metric is
coverage, and the **median depth-1 shell coverage is 1 of 12**.

**The one episode that searched correctly still lost.** L16 executed a textbook 6-face `X X X X`
sweep: **12/12 of the depth-1 shell and 6/114 of the depth-2 shell, returning to the start every 24
moves** — the right blind policy, run for 100 steps, and 0. Its solution is therefore at depth ≥3.
That single episode bounds the whole blind-search axis (§4).

### The camera was rolled, not turned

`rotate_cube_z_*` maps to the FRONT face axis (`tools/action.py:VIEW_AXES`, `z -> axis_face "F"`): it
is a **roll about the viewing axis**. It changes neither the cube nor which face is visible. The
anchor spent **750 of 1901 steps (39.5 %) rolling**; only **50 steps (2.6 %)** were `y` turns, the only
axis it ever used that could bring a new face into view, and `x` was never used at all.

Worse, a view rotation is **45°**, so after an odd number of them the cube is off-axis and no face is
readable by anything: **400 of the 1901 frames the agent saw (21.0 %) were off-axis**, and in the 8
spin episodes exactly **50 %** of frames were. The harness paid for an observation it had just
destroyed.

### Independent proof the image is inert

The L1 log is 100 consecutive `View rotation executed: FRONT CLOCKWISE` lines, each preceded by the
byte-identical `{"action": "rotate_cube_z_cw", "face": "Z", "direction": "CW"}`. Eight z-rolls return
the camera to its home pose, so the model was shown **8 distinct pictures cycling 12.5 times** and
emitted the same 61 bytes 100 times. `prompt_tokens` climbs 975 → **9814** and plateaus (the 29-turn
window), i.e. by step 29 the context holds **29 images of a cube and 29 copies of the model's own last
answer**. Completion length never leaves 22–24 tokens.

## 3. Failure breakdown

| category | episodes | steps | note |
|---|---:|---:|---|
| `BLIND_ROLL` — 100 % view rotations, cube never touched, **1** configuration | **8/20** | 800 | L1,3,5,6,9,11,12,17 — guaranteed 0 before step 1. 750/800 were z-rolls. |
| `IDENTITY_CYCLE` — face turns whose word is the identity (`X^4` blocks) | 3/20 | 300 | L8, L15 (`F`×100 → 4 states), L16 (24-move identity sweep). |
| `OUTBOUND_MARCH` — periodic word of large order, leaves the ≤4 ball by step 4 and never returns | 6/20 | 600 | L7,13,18 (4/100 steps in-ball) + L10,14,20 (12–15 %). |
| `ACTION_LIST_RECITAL` — the word is the AVAILABLE ACTIONS list in order | 2/20 | 200 | L4 (period 12), L19 (period 6). |
| solved | 1/20 | 1 | L2 — the step-0 constant `rotate_down_cw` was that level's one-move solution. |
| parse failure / default action / illegal name | **0/1901** | 0 | `successful_actions == total_responses == 1901`, `default_actions: 0`, all 20 files. |
| engine-refused rotation / animation timeout / warning / error | **0** | 0 | grepped all 20 session logs. |

**Aggregate waste, by my cube replay:**

| | steps | share |
|---|---:|---:|
| state-preserving view rotations | 800 | 42.1 % |
| └ of which z-rolls (no cube change **and** no new face) | 750 | 39.5 % |
| face turns landing on an already-visited configuration | 452 | 23.8 % |
| **provably non-advancing** | **1252** | **65.9 %** |
| steps that reached a configuration not stood on before | 649 | 34.1 % |
| └ of which were *outside* the ≤4 ball (wrong direction) | 577 | 30.4 % |

Per-episode, from the candidate's own move stream only:

| lvl | steps | view | z-roll | face | distinct cube states | distinct answers | tail period | non-advancing |
|---|---|---|---|---|---|---|---|---|
| 1 | 100 | 100 | 100 | 0 | 1 | 1 | 1 | 100 % |
| 2 | 1 | 0 | 0 | 1 | 2 | 1 | — | 0 % |
| 3 | 100 | 100 | 100 | 0 | 1 | 1 | 1 | 100 % |
| 4 | 100 | 0 | 0 | 100 | 24 | 12 | 12 | 77 % |
| 5 | 100 | 100 | 50 | 0 | 1 | 4 | 4 | 100 % |
| 6 | 100 | 100 | 100 | 0 | 1 | 2 | 2 | 100 % |
| 7 | 100 | 0 | 0 | 100 | 101 | 5 | 5 | 0 % |
| 8 | 100 | 0 | 0 | 100 | 4 | 1 | 1 | 97 % |
| 9 | 100 | 100 | 100 | 0 | 1 | 1 | 1 | 100 % |
| 10 | 100 | 0 | 0 | 100 | 86 | 12 | 14 | 15 % |
| 11 | 100 | 100 | 100 | 0 | 1 | 1 | 1 | 100 % |
| 12 | 100 | 100 | 100 | 0 | 1 | 1 | 1 | 100 % |
| 13 | 100 | 0 | 0 | 100 | 101 | 5 | 6 | 0 % |
| 14 | 100 | 0 | 0 | 100 | 94 | 11 | 14 | 7 % |
| 15 | 100 | 0 | 0 | 100 | 4 | 1 | 1 | 97 % |
| 16 | 100 | 0 | 0 | 100 | 19 | 11 | (24-blocks) | 82 % |
| 17 | 100 | 100 | 100 | 0 | 1 | 2 | 2 | 100 % |
| 18 | 100 | 0 | 0 | 100 | 101 | 2 | 2 | 0 % |
| 19 | 100 | 0 | 0 | 100 | 36 | 9 | 6 | 65 % |
| 20 | 100 | 0 | 0 | 100 | 89 | 6 | 8 | 12 % |

Cross-check vs the numeric pre-pass: `steps == vlm_calls == total_frames` and
`used_rich_harness: true` in all 20 `eval_result.json`; `score_field: cube_solved`; model
`Qwen3-VL-30B-A3B-Instruct`; total wall clock **10 152 s = 2.82 h** at 5.3 s/call. Consistent.

## 4. Binding constraint — classify it

**Primary: VLM-CEILING, and the required scaffold is a PERCEPTION channel, not a prompt.**

The frozen model cannot play this game from a picture and a 971-char prior. The evidence is not
"it scored low" — it is that **the output is independent of the input**: 8 episodes with one
byte-identical answer across 100 visibly different frames, 3 distinct answers across 20 different
step-0 frames, 22–24 output tokens, `reasoning=None` on every call. Re-wording `AGENT.md` / `rules.md`
cannot reach a policy that never consults the observation. The fix must **pre-compute the decision**
and hand the model a narrow, grounded choice.

**Secondary: HARNESS-BUG, in the sense that the harness declines to be a harness.** `tools/frame.py`
and `tools/action.py` are never called; `memory.md` is never injected and still carries maze fields;
the only code that runs is a one-line delegation. There is no loop state, no camera ledger, no notion
of distance, so the 29-turn self-echo history is free to become the whole policy. This is cheap to
fix and is a precondition for the primary fix (§5.2, §5.5).

**The scaffold exists and I measured it on the agent's own observation.** From 973 rendered frames
pulled out of this arm's videos (lossy mpeg4, so purity figures below are *pessimistic*; the true PNG
is cleaner):

| fact | measurement |
|---|---|
| frame size / backdrop | 500×500, flat grey `(124,126,123)` |
| face in view, when square-on | exact axis-aligned **240×240 box at (130,130)** = 3×3 of **80 px** stickers |
| square-on vs off-axis | **673 / 300** of 973 frames; bimodal and unmistakable (off-axis bbox ≥ 280 px) |
| 9-sticker decode | probe at `(145+80·c, 145+80·r)` — 15 px into each cell, which dodges the centre glyph |
| palette | **6201 reads → 6 RGB values, 0.03 % off**: RED `(135,16,18)`=F, GREEN `(22,153,72)`=R, ORANGE `(253,83,34)`=B, WHITE `(251,253,250)`=D, YELLOW `(252,213,44)`=U, BLUE `(10,71,169)`=L |
| off-axis gate | 4 **mid-edge** probes 6 px outside the box — `(124,250) (375,250) (250,124) (250,375)`: accepts **664/673** square-on, **0/300** off-axis |
| the corner probes are the WEAK half | they accept **284/300** off-axis frames — and `(375,375)` lands inside the HUD panel. **Use the mid-edges, drop the corners.** |
| HUD panel | a separate 126×126 widget at x 354–479, y 374–499 (bottom-right). Keep every probe out of it. |
| decode is informative | **371 distinct 9-sticker face readings** over the sampled frames |

And group theory prices what a read is worth, over the radius-4 ball (11 206 states) using
"shallowest consistent state":

| reads | camera steps | ball identified |
|---|---:|---:|
| F only | 0 | **65.6 %** (5969 distinct front reads; class size mean 1.88, median 1, max 37) |
| F + R (one 90° turn = two 45° view rotations) | **2** | **99.0 %** |
| F + R + B (two 90° turns) | **4** | **100.0 %** |

So: **≤4 survey steps identify the configuration exactly, and the solution is then a proved word of
≤4 quarter turns — ≤8 steps against a 100-step budget.** The engine ends the episode by itself the
instant the cube is solved, so no step is spent recognising it.

**The blind-search axis is real but hard-capped, and that cap is measured.** 100 steps visit at most
100 configurations, so against a target uniform in the depth-d shell the information bound is
`P ≤ 100/|shell|`:

| scramble depth | shell | blind bound | achievable (r001's optimal router) |
|---|---:|---:|---:|
| 1 | 12 | 1.00 | **12/12 in 24 moves** (`X X X X` per face) |
| 2 | 114 | 0.88 | 42–49 / 114 ≈ **0.37–0.43** |
| 3 | 1068 | **0.094** | ≤ 0.094 |
| 4 | 10 011 | **0.010** | ≤ 0.010 |

A sibling candidate's rig reports this subset spans depths 1–4. Blind coverage therefore tops out
around **0.40–0.45** no matter how perfect the router — and L16 already demonstrated the cap live by
sweeping the depth-1 shell perfectly and losing. **Perception is the only axis that reaches 1.0.**

**Not the constraint, measured at zero cost:** JSON parsing, action spelling, the key/face/direction
metadata mapping, the fallback default, animation handling, the step budget, API retries, the
delegation rate, the composed-prompt order, and the slot-0 channel. Do not spend an iteration on any
of them.

## 5. Concrete changes to try (the gradient — ranked by leverage)

### 5.1 — Decode the face from the frame. (`tools/frame.py` — new mechanism; highest leverage in the repo.)
Replace the one-line `capture` passthrough with a decoder:

```
decode_face(png_b64) -> ("F-read" as 9 colour letters) | None
    - crop the fixed box: x 130..369, y 130..369
    - GATE first: the 4 MID-EDGE probes (124,250) (375,250) (250,124) (250,375) must all be
      backdrop (124,126,123) within L1 tolerance ~40.  0 false accepts in 300 off-axis frames.
      Do NOT use the 4 corner probes (284/300 false accepts; one is inside the HUD panel).
    - 9 probes at (145+80c, 145+80r); majority vote over a ±12 px patch; snap to the 6-colour
      palette above by nearest RGB (0.03 % of 6201 reads were >60 away from all six).
    - return None on any gate failure or any unsnappable cell.  A refusal must be loud, not a guess.
```
*Predicted:* this alone changes nothing (nothing consumes it yet) — it is the precondition for 5.2.
Ship it with a self-test so it degrades safely.

### 5.2 — Believe, survey, and plan. (`tools/action.py` — the policy.)
On top of 5.1:

1. Carry an exact cube state advanced **only by the move names this harness emitted** (the §2 model
   is ~40 module-level lines of pure geometry; it needs no engine source).
2. BFS the radius-4 ball **once at import** (11 206 states, 2 s in plain Python — measured).
3. **Survey route:** read F; intersect the ball against that read; if more than one shallowest
   candidate survives, spend **two** view rotations on the **`y` axis** (= one 90° turn, square-on
   again) and read the next face. Two reads ⇒ 99.0 %; four steps ⇒ 100.0 %.
4. Once the state is named, BFS the **proved shortest word home** (≤4 moves) and play it.
5. After every played move, re-read F and **check it against what the belief predicts**. A
   contradiction restarts the survey. That check is the only thing separating a correct belief from
   a confident wrong one.

*Predicted:* **0.050 → 0.85–1.00**, in **≤8 steps per level** (2–4 survey + ≤4 solve) against a
100-step cap. Causal reason: the task is ≤4 quarter turns deep, the state is exactly readable, the
shortest word is provable, and `state_callback` ends the episode on arrival — there is no residual
uncertainty left to lose to.
*Report:* solved count **and** steps-per-solve **and** the override rate (see the chain-integrity note).

### 5.3 — Keep a per-axis camera LEDGER, and never emit a lone view rotation. (`tools/action.py`.)
Two distinct hazards, both live in this arm:

* **45° parity.** One view rotation leaves the cube off-axis and the decoder blind (21 % of the
  anchor's own frames). **Emit view rotations only in pairs** (2 × 45° = 90°), and never when the
  belief is already resolved.
* **The gate cannot see a ROLL.** A multiple-of-90° roll about the viewing axis renders a perfectly
  square-on frame that passes the gate but shows a **rotated permutation** of the true face — and the
  centre glyph is a symmetric dark square, so it does not disambiguate the roll either. The cure is
  not a better pixel test: **track `(x,y,z)` view-turn counts per episode and derive which 9 slots the
  current pose shows.** Prefer the `y` axis for surveying (it brings new faces into view); treat `z`
  as forbidden (it is the roll — 750 wasted anchor steps) and `x` as ledger-tracked.

### 5.4 — Measure the three conventions that cannot be derived, behind an import-time self-test.
Chirality of the six quarter turns; whether a face turn is **cube-fixed** (does a camera pose rename
the faces?); and which 9 slots each surveyed pose shows. Get them wrong and the belief is confidently
wrong. Put all three behind a `self_test()` that **degrades to the 5.6 blind sweep** if it fails,
rather than shipping a silent desync.
*Live evidence on the survey-then-turn interleave:* the anchor **never mixed** view and face turns
(views per episode ∈ {0, 100}), so its 0 refusals prove nothing about mixing. The r001 run did mix:
**34 view→face and 33 face→view executed transitions, 0 refusals, 0 animation timeouts** — because
`execute_action` calls `_wait_for_animation()` after every face rotation. Mixing is safe on this
engine path, but n=67 is small; a survey-heavy candidate must re-assert `rotation failed == 0`.

### 5.5 — Break the self-echo history. (`tools/vlm.py` — one line, untried in this chain.)
`evaluator.history_window` is **29** (`rubik_config.py:20`), so from step 29 the context is 29 images
plus 29 copies of the model's own last answer — the mechanism that converts a step-0 coin flip into a
100-step fixed point (§2). The harness owns both `evaluator.history_window` and
`evaluator.conversation_history`; set the window to 1, or rewrite the history each turn to
`[slot0, current]`.
*Predicted:* breaks the 7 single-answer episodes and the periodic lock. **Necessary but not
sufficient** — the echo-free step-0 distribution is itself only 3 names wide, so this converts a
locked constant into a *resampled* constant. Ship it **with** 5.2, never alone. (Note 0 is falsy in
`truncate_history`, so `history_window = 0` silently means 29; use 1 or rewrite the list.)

### 5.6 — Fallback only: a layered near-ball sweep. (`tools/action.py`.)
If and only if 5.1's gate refuses (or 5.4's self-test fails), fall back to the explicit 6-face
`X X X X` sweep: 24 moves, covers **12/12** of the depth-1 shell, and returns to the start after each
block, then continue into the depth-2 shell by shallowest-unvisited. L16 ran exactly this pattern
live, so it is known to execute. **Cap your expectations at §4's table (~0.40–0.45)** — this is a
floor, not the goal, and the proposer must not mistake it for the mechanism.

### 5.7 — Retire the dead levers rather than re-wording them.
`tools/frame.py:capture` (never called), `tools/action.py`'s six dormant helpers (never called),
`memory.md`'s `last_pos` / `last_dir` / `stuck_counter` (maze fields, meaningless here), and the
"Important tips" paragraph in `rules.md` ("analyze the current state carefully") — which the traces
show the model does not act on. Replace `memory.md` with the brief 5.2 actually needs: **believed
configuration, distance to solved, the remaining word, and a ≤4-item menu.** Keep invariants
(no `z` rotations, view turns only in pairs) as **static unconditional prose** — r001 measured that
to be the one prose channel with an effect (a static z ban moved the model's z-naming rate
39.5 % → 0.4 %), while a per-turn 14-name ALLOWED list was inert (refusal ratio 1.09 vs chance).

### Must-keep plumbing (do not re-derive, do not break)
One engine action per step — `openai_ai_model` executes internally, so mute `evaluator.execute_action`
around the delegation if you build your own `Action`; slot-0-only prompt injection via
`conversation_history[0]`; rewrite `key` / `face` / `direction` / `action_type` **together** (the
model emits `face: "Z"` for `rotate_cube_z_cw`, which is the axis letter, not the engine's axis face —
r001 silently corrected 107 such disagreements); and the free termination at `state_callback`.

## 6. Exhausted — do NOT repeat

* **Prompt-only rewording of `AGENT.md` / `rules.md`.** The output is independent of the input
  (8 episodes, one byte-identical answer, 100 frames each). There is no wording that makes a policy
  consult an observation it is not given.
* **The raw 29-turn screenshot+JSON history as "working memory".** It is the *cause* of the lock, not
  a memory. 9814 prompt tokens buying a period-1 program.
* **Parser / fallback / action spelling / engine key-and-metadata mapping / animation handling /
  API retries / step budget.** 0 failures of any kind in 1901 steps across 20 logs. Measured clean.
* **Asking the frozen model for a move from the home pose and expecting variety.** Three distinct
  names across 20 different scrambles; 22–24 output tokens; `reasoning=None` always.
* **Asking it for a six-face colour grid.** It is shown at most one face (and 21 % of the time, none).
* **More model calls per episode.** The anchor already made 1901 and 1252 of them were provably
  non-advancing. The constraint is information, not compute.
* **Lone / `z`-axis view rotations.** `z` is a roll: no state change, no new face. 750 steps proved it.
* **A no-revisit / maximise-distinct-states objective (refuted live by r001).** Anti-correlated with
  winning; caps depth-1 shell coverage at **2/12** under an optimal router (vs 12/12 in 24 moves
  without it). Do not re-tune it, do not soften it to "don't undo the last move" — it is the wrong
  sign, not the wrong threshold.
* **A per-turn, state-dependent ALLOWED/REFUSED name list as an instruction channel (refuted live by
  r001).** 14 of 18 names, refusal ratio 1.09 vs chance, announced default named at 6.0 % vs 5.6 %
  chance. Static unconditional bans *do* work; per-turn lists do not. Narrow to **≤4 items**.
* **A macro/milestone/beginner-method prose skill as a step-0 lever (refuted live by r001).**
  +1150 prompt tokens bought a relabelled constant (3 names covered 18/20 levels).
* **`view-streak` as a run-length cap (refuted live by r001).** `run >= 2` licenses a stable
  `y, y, [override]` duty cycle forever — 66 of one episode's 100 steps. Cap the per-episode total.
* **An alphabetical substitution ring rotated by step count (refuted live by r001).** `len(played) % 12`
  is 0 at step 0 in every episode, so the first move is the constant `rotate_back_acw` — which *was*
  r001's entire 0.050.
* **`reduce_word` as a state key.** Sound but incomplete (it misses commuting relations: `R L R' L'`
  reduces to length 4 and is the identity). Use an exact cubie state (§2).
* **A still-frame digest as a retry signal.** Inert — `_retry` fired 0 times in 1901 steps, because
  every face turn and every view turn changes the picture.
* **Blind coverage as the *goal*.** Hard-capped at ~0.40–0.45 by §4's arithmetic and demonstrated
  live by L16 (perfect 12/12 depth-1 sweep, still 0). Keep it only as 5.6's fallback floor.

### Chain-integrity warning

Score in this chain moves inversely with how much the frozen model is used, so a candidate that
raises the primary by reducing the VLM to a rubber stamp has optimised the guard, not the task.
Report **two numbers together: the primary AND the override rate** (model pick vs committed move),
and put `model_pick` / `harness_override` / the fired rule code on the `Action` so the next
diagnosis is computable rather than reconstructed — r001 did this and it is why its trace was
readable. A plan-driven candidate will structurally override on the survey steps (step 0 has no
brief to comply with at all); say so, separate structural overrides from compliance failures, and
keep the model's accepted picks on the honoured path wherever they are consistent with the belief
(§5.2's menu is where it decides).

### Anti-cheat note on this diagnosis

I did not read a scramble table, a solver, cube internals, the solved flag or any level's ground
truth; the proposer sandbox hook blocks the game-engine/renderer source and I did not work around it
(one `grep` of the renderer was refused and abandoned). Everything above comes from three sources:
(1) the candidate's own files; (2) its traces — the recorded answer stream, `eval_result.json`,
and the session logs; (3) **the agent's own observation**, i.e. rendered frames read out of this
arm's videos with PIL/cv2, which is the sanctioned channel.

The cube facts are from an **independent 3×3×3 model built from geometry** (`/tmp/rubdiag/cube.py`),
validated only against convention-free invariants (ball sizes 12/114/1068/10011, `ord(F)=4`,
`ord(RU)=63`, `ord(RU')=105`), and every trajectory quantity is a displacement from **the episode's
own starting configuration** — strictly weaker information than the level's identity, and sufficient
because the solution of *every* level necessarily lies inside that ball. Per-level numbers in §3 are
the harness's own emitted move stream, not level data.

Every fix above keeps the frozen agent deciding from its observation plus state it could have derived
itself: 5.1 decodes the frame the agent is already shown; 5.2's belief is advanced only by move names
this harness emitted; 5.3's ledger counts the harness's own view turns; 5.5 edits the harness's own
conversation history. Nothing proposed opens `SpatialWorld/data/`, reads a scramble, or consults a
solver.
