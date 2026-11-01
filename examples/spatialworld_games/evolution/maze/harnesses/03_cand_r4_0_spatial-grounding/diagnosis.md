# Diagnosis — maze (SpatialWorld maze3d, 20 levels) / iter 0 · `anchor_maze` (k32_claude_pv staged anchor)

**Headline:** primary = **0.250 — 5/20 success (L03, L04, L05, L09, L19)**. Every one of the 15 losses
is a 100-call timeout. **0/1567 parse failures, 0 default actions, 0 illegal actions.**
**1012/1263 (80.1%) of all `move_forward` decisions moved ZERO cells**, and **960/1263 (76.0%) of them
re-pushed a `(cell, heading)` edge the episode had *already* observed to be a wall**. On **7/20 levels
the agent never leaves its start cell** (L01, L07, L08, L10, L13, L16, L20 — 1 cell visited of 7–71).

**Binding constraint: HARNESS-BUG.** `tools/vlm.py:decide` is a two-line open-loop pass-through. It
holds no state, so it discards the one free, oracle-less bit the observation hands it every step —
*did my last action change anything* — and it never computes the wall test that the frame answers
**exactly**. A single wall misread therefore becomes an **absorbing fixed point**: pose frozen ⇒ frame
frozen ⇒ the 29-turn history fills with identical (image, `move_forward`) pairs ⇒ the reflex policy
re-emits `move_forward` forever. 64.6% of the entire eval budget is spent pushing walls.

Priced offline on the exact engine transition (§7): **exact local wall occupancy + a visit set +
single-cell probes + a turn-to-look rule wins 20/20 inside budget (mean 28, max 87 of 100 calls).**
All three perception signals that mechanism needs are now **measured, not hypothesised** (§5.1).

> This independently reproduces the earlier `cand_r001_0` diagnosis of the same anchor (null forwards
> 1012 vs 1017, redundant pushes 960 vs 969 — a step-0 counting convention apart). **What is new here
> is §5.1 (the frame decoders, with censuses and exact constants), §5.2 (the invisible 4th side and
> the 8 levels whose only start opening is behind), §5.3 (three engine budget levers the anchor never
> touches), and §7 (the priced travel-compression trap).** Do not re-derive the shared parts.

---

## 1. What this harness does (mechanism)

```
eval_runner.py  (ANTICHEAT_GUARD on: _GuardedEvaluator + _DiskGuard + VLM-call counter)
  evaluator.baseline_step = evaluator.openai_ai_model        # pristine SpatialWorld per-step fn
  evaluator.openai_ai_model = harness.build_agent(_guarded_evaluator)
       │
harness.py:build_agent
  ├─ evaluator.current_prompt = compose_system_prompt()
  │        AGENT.md  +  skills/navigation.md  +  rules.md     (concatenated, in md-sorted order)
  └─ async agent(frame_data) -> tools.vlm.decide(evaluator, frame_data)
                                   └─ return await evaluator.baseline_step(frame_data)
```

* `tools/vlm.py:decide` is **two lines**. It fetches `baseline_step` and awaits it. Nothing else.
* `tools/frame.py:capture` and **all** of `tools/action.py` (`KEY_MAPPING`, `GRANULARITY_STEPS`,
  `key_for`, `granularity_steps`) are **dead code** — no importer on the decision path.
* `memory.md` is a template and is **deliberately not injected** (`NOTES.md` says so). The harness
  therefore keeps **zero** state across steps: no pose, no heading, no last-action outcome, no
  blocked-edge set, no visit count.
* Measured prompt shape (from `raw_response.usage`): 1 541 prompt tokens at frame 0, +~500/turn,
  saturating at **16 485 by step ≈30** — the documented 29-turn sliding image+text window.
* Measured completion shape: **8–19 tokens** per call and **`reasoning=None` on all 1567 calls**.
  The backbone is `Qwen3-VL-30B-A3B-**Instruct**`. It emits the bare JSON object and *no
  deliberation at all.* There is no chain of thought in these traces to repair, and no prompt can
  create one. The policy is a reflex.
* The per-turn user text is `"Current step: N. Continue based on previous conversation history."` —
  the step index is the **only** token that varies. **Nothing ever tells the agent what its last
  action did.**

So: a stateless reflex policy, told in prose to notice walls and notice being stuck, with no code
anywhere that checks whether it did either.

---

## 2. What actually happened (causal chain, with evidence)

### 2.0 The spawn pose is adversarial on 18/20 levels, and the harness walks straight into it

The agent spawns facing **NORTH**, and every level's `S` has **exactly one** open side:

| | L01 | L02 | L03 | L04 | L05 | L06 | L07 | L08 | L09 | L10 |
|---|---|---|---|---|---|---|---|---|---|---|
| start | (3,1) | (3,1) | (1,1) | (1,5) | (1,1) | (1,5) | (1,1) | (1,1) | (1,1) | (7,1) |
| the ONE opening | W | S | E | **N** | E | **N** | S | S | E | W |
| step-0 action | `fwd/s` | `fwd/s` | `fwd/s` | `fwd/m` | `turn_right` | `fwd/s` | `fwd/s` | `fwd/s` | `turn_right` | `fwd/s` |

(L11–L20 openings: E, E, S, E, E, S, S, S, E, S — **N is open on only L04 and L06**.)

**Step 0 is `move_forward` on 18/20 levels — a guaranteed no-op on 16 of them.** Step 0's correct
action is *uniquely determined* (rotate to the single opening) and the policy gets it wrong almost
every time. That is the cleanest single-frame measurement of the perception failure available: the
wall is **one cell away, filling the view**, and the model says "forward".

### 2.1 The absorbing fixed point — `maze_7_3l9a54wd` (L07), the purest case

```
step  0   at (1,1) facing N   move_forward small   -> 0 cells   (N is a wall)
step  1                       turn_right           -> facing E  (also a wall)
step  2..99                   move_forward small   x98, every one 0 cells
```
`S=(1,1)`; the **only** opening is SOUTH. The agent faced **N and E only** — it never once faced S or
W in 100 calls. Final: **1 cell visited of 17, 0 cells traversed**, BFS distance to exit unchanged at
8, timeout. Live-engine replay: **98/98 consecutive identical-pose pairs, 98/98 byte-identical view
digests, 1 distinct POS-HUD digest for the whole episode.** The agent stared at a constant image for
100 calls and pressed forward 99 times.

### 2.2 The orbit is a fixed point of the *frame → action* map, not of the maze

`maze_8_sdtjeplh` (L08, 7×7, exit distance 16) and `maze_16_dcn_pvqi` (L16, 11×11, exit distance 32)
are **different mazes with different frames** (frame-1 `prompt_tokens` 2059 vs 2055), yet their
`(action, granularity)` tapes are **byte-identical for all 100 steps**:

```
[ move_forward small , turn_right , move_forward small , move_forward medium ,
  move_forward large , turn_left ]   x 16.67      (period 6, from step 0)
```

`turn_right`/`turn_left` cancels, so the pose is period-6-invariant → the frame sequence is periodic →
the low-entropy policy reproduces the cycle forever. Both move **0 cells** and never face their only
opening (S). Note that the granularity ladder `small → small → medium → large` walks `rules.md`'s menu
**in listed order**: the model has concluded the *action* is right and is enumerating the *parameter*.

Eventual period of the action tape and the step it locks in:

| level | 1 | 2 | 6 | 7 | 8 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| period | 8 | 8 | 1 | 1 | 6 | 1 | 6 | 8 | 1 | 1 | 11 | 6 | 1 | 1 | 8 |
| locks in at | 9 | 50 | 19 | 2 | 0 | 12 | 4 | 48 | 6 | 88 | 49 | 0 | 82 | 25 | 16 |

**10 of 15 orbits lock in at or before step 27 — before the 29-turn window even saturates (step ≈30).
Window length is not the cause** (§6).

### 2.3 Progress, then a freeze two cells from the exit

* `maze_6_74redmv6` (L06): 8 cells traversed by step 12, **BFS distance 2 from the exit by step 8**.
  Step 13 `turn_left` → facing E at (5,5), where **N and W are open and E is a wall**. Steps 14–99:
  `move_forward` **×86, every one 0 cells** (`large` ×81, `medium` ×5). It burned 86% of its budget
  pushing one wall with two open sides beside it. Final distance 4.
* `maze_2_ymejypv2` (L02): same story — BFS 2 from step 7, 60 null forwards, final distance 2.
* `maze_18_0ro4a8kg` (L18): frozen at (11,1) facing S from step 25 — **75 consecutive null forwards**,
  N and E open the whole time.
* `maze_17_2gan2x1e` (L17): 18 null forwards from step 82 at (7,3) facing E, N and W open.

### 2.4 The escape capability exists; it is just never triggered — `maze_14_w8wb83bn` (L14)

L14 is the healthiest loss: 53 productive forwards, 23/49 cells, BFS 32 → 6. It contains
`turn_right ×3` at steps 29–31 and `turn_left ×3` at steps 43–45 — **the policy can emit a 3-turn
heading scan.** It just never does it when frozen (it still ends on 12 null forwards at (3,9) facing E
with N and W open). The behaviour is available and is not conditioned on anything, because **no signal
exists to condition it on.**

### 2.5 Healthy controls — everything except the wall judgement is sound

L03 wins in 7 calls, L05 in 7 (**0** null forwards), L09 in 10 (**0** nulls), L19 in 34.
Across all 20 episodes: `successful_actions = 1567/1567`, `default_actions = 0`, every
`response_text` parses to one of the three legal actions. Replaying the 20 recorded tapes through a
re-derived engine transition reproduces **`success` and final `distance_to_exit` for 20/20 episodes**
(§7), and replaying them through the **real headless engine** reproduces the final pose for 20/20.
Capture, prompt composition, the JSON contract, key mapping, granularity and exit commitment (all 5
wins terminate on arrival with 0 extra calls) are all healthy.

---

## 3. Failure breakdown

| category | rate | evidence |
|---|---:|---|
| success | **5/20 (25.0%)** | `eval_result.json` ×20 |
| budget timeout | **15/15 of losses** | every loss `steps_taken = 100` |
| **null forward** (0 cells moved) | **1012/1263 (80.1%)**; 64.6% of the whole 1567-call budget | exact transition replay, verified against the live engine |
| **redundant push** (re-pushed a `(cell,heading)` already measured null *in that episode*) | **960/1263 (76.0%)**; 94–97 of 97 on L07/L10/L13 | per-`(cell,heading)` ledger over the replay |
| cells requested but never travelled | **1519/1795 (84.6%)** | granularity vs actual displacement |
| never left the start cell | **7/20** — L01, L07, L08, L10, L13, L16, L20 | 1 unique cell, 0 cells traversed |
| …and never faced the single opening | **7/7** | all seven faced only {N, E}; openings were S or W |
| calls after the last productive move, in losses | **1034/1500 (68.9%)** | last `moved>0` index per episode |
| longest single null-forward run | 98 (L07), 94 (L13), 88 (L10), 86 (L06), 75 (L18) | consecutive-run scan |
| open cells ever visited, losses | median **11%**; L20 **1%**, L13/L16 **2%**, best L14 47% | BFS over the layout |
| reached within 2 cells of the exit and still lost | 2/15 (L02, L06) | BFS |
| parse / default / illegal action | **0/1567** | `evaluation_info`, every `response_text` |
| hidden reasoning to repair | **none** | `reasoning=None`, 8–19 completion tokens, all 1567 calls |
| wall-clock | 99.8 min for the 20-episode eval; 2.00–6.40 s/call (median 2.99) | `total_time` |

The loss is not malformed output, not engine execution, not budget size, and not exit commitment.
**It is the repeated selection of an action the observation had already proved to be a no-op.**

---

## 4. Binding constraint — classify it

**Primary: HARNESS-BUG — large and fully recoverable in harness code.** `tools/vlm.py:decide`
discards the transition result, and no module computes the wall test. Two consequences, both measured:
960 forward decisions re-litigate walls the episode already knew about, and 7 episodes never test the
one heading that works. The orbits of §2.2 are the *inevitable* behaviour of a stateless reflex over a
frozen observation — not a model defect.

**Secondary: VLM-CEILING on single-cell wall-ahead perception.** Step 0 is the clean experiment: the
wall is 1 cell ahead on all 20 levels and the model says `move_forward` on 18. `AGENT.md` already
describes walls as "brick-red / orange-brown blocks"; `skills/navigation.md` already says to turn when
blocked; `AGENT.md` already says to use POS to "detect when you are stuck". The prose is *correct and
present* and the reflex ignores it. **The fix must COMPUTE the wall test, not request it.**

**Not TASK-GAP.** Optimal routes are **4–32 cells** against a **100-call** budget (and a turn costs 1
call, or 1 call for any angle — §5.3). Entering the exit auto-commits. There is no goal-commitment or
budget failure to fix.

**Hard constraint on any fix (verified in `eval_runner.py`):** `ANTICHEAT_GUARD` is **ON** in this run
(`vlm_calls` is counted in every `eval_result.json`). A harness with **0** `baseline_step` delegations
in an episode is **VOIDED to 0.0** ("not a harness"). `evaluator.game.<attr>` raises and voids for
everything except `.screen`; harness reads under `SPATIALWORLD/data/` are blocked. So: keep the VLM in
the loop, and take the frame from `frame_data.image`, `evaluator.capture_game_screenshot()` or
`evaluator.game.screen` — all three are allowed.

---

## 5. Concrete changes to try (the gradient — ranked by leverage)

### 5.1 `tools/perception.py` — the frame answers the wall question EXACTLY. Constants measured.

**This is the highest-leverage change and it no longer needs calibrating — I measured it.** I drove
the real headless engine (`Maze3DGame` + `PygameInputSource`, 800×600, the same objects
`maze_openai_evaluation.py:run_evaluation` builds) over **every (cell, heading) pose of all 20 levels
= 2752 poses**, and compared frame statistics against the true layout.

`frame_data.image` is an `(600, 800, 3) uint8` RGB array. Let `view = image[40:570]` (below the 40-row
HUD bar, above the 30-row bottom bar) and
`wall = (R > G+12) & (R > B+12) & (R > 40)`  (the brick palette; fitted, then censused).
Define `ext(x0,x1) = wall[:, x0:x1].any(axis=1).sum()` — the number of view rows carrying wall ink in
that column band.

| signal | band | decode | census |
|---|---|---|---|
| **forward clear depth** `min(d,3)` | `x ∈ [380, 421)` | `ext == 401 → 0 (BLOCKED)`; `200 ≤ ext ≤ 399 → 1`; `132 ≤ ext ≤ 199 → 2`; `ext ≤ 131 → ≥3` | **2752/2752 exact, 0 errors** |
| **left side open?** | `x ∈ [60, 141)` | `ext == 455 → wall`; `ext ≤ 453 → open` | **2752/2752, 0 errors** |
| **right side open?** | `x ∈ [660, 741)` | identical thresholds | **2752/2752, 0 errors** |
| **heading** | red compass arrow `(255,50,50)` in `image[:40]` | bbox `(7,25,392,408)=N` `(15,33,392,408)=S` `(12,28,395,413)=E` `(12,28,387,405)=W` | 4 values, **level-invariant** (L01/09/13/20) |
| **heading (alt)** | `md5(exact-white mask of image[0:40, 560:800])` | 4 constants, one per heading | level-invariant, but **HUD-dependent** |
| **cell identity** | `md5(cyan (0,255,255) mask of image[0:40, 0:300])` | collision-free cell id | **134 digests ↔ 134 cells across 20 levels, 0 ambiguous** |
| **null-move / "did I move"** | `md5(image[40:570])` | digest unchanged ⟺ pose unchanged | **1016/1016 identical-pose pairs identical; 0/531 pose-changed pairs identical** |

Two cautions the earlier gradient raised, now **resolved**:
* *"the exit panel pulses, so a frame digest may not be a pure function of pose."* **It does not
  animate the capture.** 120 consecutive renders at one exit-visible pose (L13 (1,9) facing E, 3577
  gold px) produced **1 distinct view digest**; across 480 exit-visible poses, consecutive captures
  were byte-identical 480/480. `md5(image[40:570])` is a sound null detector.
* *"verify the HUD exists before building on it."* It does — in the engine the evaluator imports
  (`~/workspace/SpatialWorld/mllm_base_agent/environments/game/games/maze3d/`): a cyan `POS: x, y`
  top-left and a white `DIR: NORTH` top-right, legible in the pixel dump. **But there are TWO engine
  copies in this tree and the AutoSpatial one has both HUD texts removed.** So **prefer the compass
  arrow and the geometry bands (which work in both) over the HUD crops**, and treat the POS digest as
  a bonus cell id behind a presence check.

**Predicted effect:** this is the signal that turns the whole mechanism on; see §7 — 5/20 → 20/20
with it, 8/20 without a turn-to-look rule, ~12–13/20 with the did-I-move bit alone.

### 5.2 `tools/pose.py` — the ledger, and the ONE side the frame cannot see

Instantiate once in `harness.py:build_agent` (closure state ⇒ episode-lifetime) and thread it through
`decide`:

```
decide(evaluator, frame_data, state):
    state.sense(frame_data)                       # 5.1: cell id, heading, front/left/right occupancy
    state.settle()                                # attribute the PREVIOUS action's outcome
    action = await evaluator.baseline_step(frame_data)   # unchanged, 1 VLM call, keeps the run valid
    action = state.arbitrate(action)               # override only what is measured-impossible
    state.commit(action); return action
```

State (all self-derived; nothing privileged): `cell` (the POS digest or dead reckoning), `heading`
(compass bbox), `know: dict[(cell, heading) -> open|wall]`, `visited: set[cell]`.

**The trap, measured:** 5.1 gives you **three** sides. The **fourth (behind) is invisible** — and
**on 8 of 20 levels the start cell's ONLY opening is behind the spawn heading** (L02, L07, L08, L13,
L16, L17, L18, L20: facing N, opening S). A frontier policy that only ever considers *sensed* sides
has nothing to do at step 0 on those levels and **gives up: priced 8/20.** You need one of:

* **turn to look** — a turn is *perception*; rotating to an unsensed side measures it plus two more.
  With this rule the mechanism reaches **20/20** (§7); or
* **`move_backward`** — verified working (§5.3): `key="s"` moves 1 cell backward, heading unchanged,
  for one call. When all three visible sides are walls it **both tests and traverses** the only
  remaining side in a single call — a 1-call start escape on those 8 levels.

`settle` is the point: *if the last returned action was a forward and the view digest did not change,
record `(cell, heading) = wall`.* One line, and it is an exact measurement, not a guess.

**Then: never re-push a measured wall — but supply the substitute from the ledger, not from a reflex.**
The earlier gradient priced a bare veto that substitutes `turn_right` at **7/20** (it rescues L01/L02
and *loses* L19, because the tape runs out while the agent spins). The substitute must be the frontier
action: prefer an **unsensed** side, else an open side to an **unvisited** cell, else BFS over
`know`'s open edges to the nearest cell that still has one; order the relative choice
right → forward → left → back; rotate **monotonically** (never reverse a rotation before the pose has
changed — that is the L08/L16 `R…L` cancellation, 32 wasted turns each).

### 5.3 `tools/action.py` — three engine levers the anchor leaves on the table (all verified)

`tools/action.py` is currently dead code. The budget is **100 `ai_model_func` calls** — one engine
loop iteration each (`evaluation_engine.py: step_count += 1` per iteration), **not** game steps. Inside
one call, `PygameInputSource.execute_action` honours `metadata`:

| lever | verified behaviour | why it matters |
|---|---|---|
| `metadata["repeat"]=k` on a **turn** | `turn_right repeat=2` → N→S; `repeat=3` → N→W. Clamped 1..10. | **any heading change costs exactly 1 call**, so a 180° reversal is as cheap as a 90° turn |
| `metadata["repeat"]=k` on `w` with **no** `granularity` | moved **8 cells in one call** (the whole corridor) on L13 and L20 | up to **10 cells/call** vs 3 — but see the §7 trap |
| `granularity` on `w` | forces `repeat=1`; caps travel at 3 (`_action_uses_granularity`) | the prose menu caps you at 1/3 of the engine's reach |
| `key="s"` → `move_backward` | moves 1 cell backward, **heading unchanged**; `repeat` honoured | the 4th side, for 1 call and 0 turns (§5.2) |
| `action.metadata["response_text"]` | the model's **raw JSON**, attached by `openai_ai_model` | re-parse / re-map an action with **no second VLM call** |

`MAZE_KEY_MAPPING` has no `"s"`, so an unknown action name falls back to `"w"`. To let the *agent*
request a reverse, add `move_backward` to `rules.md` **and** re-parse `metadata["response_text"]` in
`tools/vlm.py`, rewriting `action.key`/`metadata`.

### 5.4 Write overrides back into the context — `tools/vlm.py`

`build_full_prompt(is_first_call=False)` returns `self.current_prompt` **unchanged**, so
`current_prompt` is dead after step 1. The only per-turn text channel is
**`evaluator.conversation_history[0]`** — `build_conversation_history` copies `previous_history[0]`
verbatim every turn. Two uses:

1. After an override, **replace the newest assistant entry** in `evaluator.conversation_history` with
   the action actually executed. Otherwise the 29-turn window teaches the model that its vetoed
   wall-push succeeded — exactly the self-reinforcing pattern that produced the byte-identical
   L08/L16 tapes.
2. Render `memory.md` (last outcome, measured front/left/right, blocked edges at this cell, visited
   count) into `conversation_history[0]` each turn. Cheap, and it is the only way to give a
   zero-reasoning reflex an error signal at all.

### 5.5 Keep the VLM, and give it the job it is actually good for

Do **not** cut the model out — 0 delegations **voids the episode** (§4). Also: a second VLM call per
step roughly doubles a run that already takes 99.8 min for 20 episodes at 2.0–6.4 s/call. So keep
`baseline_step` at **1 call/step, unchanged**, and use its proposal as the **tie-break** among the
headings the ledger has not ruled out. That is where vision genuinely helps and the mechanism cannot:
the exit is a glowing gold panel, visible down a corridor — measured present (`R>150, G>110, B<110,
R>B+60`) at **262/2752 poses, at exit distances 1–10**. Override only what is measured-impossible.

### 5.6 Prose riders — one paragraph, only alongside the above

`skills/navigation.md` / `AGENT.md`: "a forward that leaves POS unchanged **proves** that side is a
wall — never retry it"; "a dead-end needs two or three turns **in the same direction**; never reverse a
turn before you have moved"; "**you spawn facing a wall on 18 of 20 levels — your first action is a
turn**". Also de-emphasise `rules.md`'s granularity menu ordering: the L08/L16 orbit walks
`small → medium → large` in listed order, enumerating the parameter instead of questioning the action.
**Prose alone cannot fix this** — the current prompt already instructs POS comparison, and the model
answers in 8–19 tokens with `reasoning=None`.

---

## 6. Exhausted — do NOT repeat

* **Do not add "notice the wall / don't get stuck" prose as the fix.** It is already in `AGENT.md`
  *and* `skills/navigation.md`, verbatim, and the reflex does not act on it.
* **Do not tune granularity as the primary lever.** Every granularity is null when the first cell is a
  wall; L06 pushed `large` 81 times for 0 cells. (It is a real budget saving *after* 5.1–5.2 — but see
  the §7 trap before you reach for it.)
* **Do not spend an iteration on JSON format, examples, retries or the fallback.** 0/1567 parse
  failures, 0 defaults, 0 illegal actions.
* **Do not shrink or lengthen the 29-turn history window.** The window saturates at step ≈30 and
  **10 of 15 orbits lock in at or before step 27**. L08 and L16 produce identical 100-step tapes from
  *different* images — once the pose stops changing the policy is insensitive to visual content.
* **Do not ship a veto without a plan.** A bare `turn_right` substitute over the recorded tape prices
  at 7/20 and *loses* L19.
* **Do not treat the budget as the constraint.** Optimal routes are 4–32 cells against 100 calls; a
  turn costs 1 call for any angle. Every loss is a decision-quality loss.
* **Do not try to make the model reason.** `reasoning=None` and 8–19 completion tokens on all 1567
  calls — this backbone is `…-Instruct`, not `…-Thinking`.
* **Do not wire a second VLM call per step** without doing the wall-clock arithmetic (99.8 min / 20
  episodes already).
* **Do not build an exit-seeking planner before 5.1–5.2.** With exact local occupancy, plain frontier
  DFS wins 20/20; the planner is not where the points are.
* **Do not activate `tools/frame.py` without wiring its output into state.** Capture is healthy;
  capture that nothing reads is still 5/20.
* **Do not read `evaluator.game.<attr>`** (anything but `.screen`) **or any file under
  `SPATIALWORLD/data/`** from the harness — both are recorded and void the episode.

---

## 7. For the next proposer: price your mechanism offline, exactly — and mind the travel trap

The engine transition is re-derivable in ~40 lines and is **exact**: replaying the 20 recorded tapes
reproduces `success` and final `distance_to_exit` for **20/20** episodes, and replaying them through
the real headless engine reproduces the final pose for **20/20**.

```
cells            : data/maze3d/Level_NN.txt — '#' wall, 'S' start, 'E' exit, (x,y) = (col,row)
                   the START cell is re-enterable (verified against all 20 engine outcomes)
initial heading  : NORTH;  N is open on only L04 and L06  (so step 0 is a wall on 18/20)
turn_right       : N -> E -> S -> W -> N          turn_left: the inverse
move_forward     : small=1 / medium=2 / large=3 single-cell steps, STOPS EARLY at a wall
                   (a partial move is real: `large` with a wall 2 cells out moves 1)
terminal         : entering the exit cell ends the episode immediately, success=True
budget           : 100 ai_model_func CALLS (turns included; any-angle turn = 1 call with repeat)
```

Priced on that transition, with the policy observing **only** what §5.1 decodes exactly (cell
identity, heading, front/left/right occupancy) — no oracle anywhere in the policy:

| arm | wins / 20 | mean calls | max calls |
|---|---:|---:|---:|
| this anchor (recorded tapes) | **5** | 13 (over its 5 wins) | 34 — the other 15 hit the 100 cap |
| 3-side sensing, **no** turn-to-look / backward probe | **8** | — | — |
| + frontier DFS + visit set + **turn-to-look**, single-cell probes | **20** | **28** | **87** |
| …same, with any-angle turns (`repeat`) | **20** | 28 | 84 |
| …but **multi-cell travel** (granularity 3, or `repeat`=10) | **16** | 8 | 19 |
| …multi-cell travel, left-first ordering | 18 | 8 | 20 |
| …long runs only when **re-tracing already-sensed corridor** | 19 | 24 | 59 |

Read the last three rows carefully — **this is the trap.** Compressing travel is a 3.5× budget saving
and it **costs four levels**, because a multi-cell run blows past side openings without ever sensing
them, leaving holes in the map that make the explorer loop. The relative-turn ordering is worth ±2
once the map has holes, and it is worth 0 when it does not. **Ship the single-cell version first
(20/20 with margin 13 calls on the worst level, L-max 87), and treat travel compression as a separate,
separately-priced arm** — do not bundle it in and read the loss as "the ledger doesn't work".

Finally: keep §5.1's did-I-move bit as the **authority** when the pixel decode and the ledger disagree,
and make `perception` return `UNKNOWN` (→ fall back to single-cell probing) rather than guessing if its
agreement with that bit ever drops below ~99% on a live frame. The floor of the combined system is
then the ledger-only arm, never worse than this anchor.
