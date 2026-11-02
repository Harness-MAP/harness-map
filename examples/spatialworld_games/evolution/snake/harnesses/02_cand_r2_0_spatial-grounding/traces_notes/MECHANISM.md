# cand_r002_0 — module map, lever table, and what was measured

Parked here rather than in `NOTES.md` because the `traces` prefix is phi-skipped, so the detail is
free: it reaches the diagnostician and the next proposer without moving the descriptor.

## module map

| file | role now | state |
|---|---|---|
| `agent.py` | entrypoint + prior composer | UNCHANGED. `build_agent(evaluator)` still returns the per-step `agent(frame_data)`; `compose_system_prompt()` still rebuilds the prompt from `AGENT.md` + `skills/*.md` + `rules.md`, in that order, byte-equal to `system_prompt.txt`. |
| `tools/vlm.py` | the mechanism (now the main module: 5 of the 11 components) | REWRITTEN. `_function_globals`, `LatticeLedger`, `_ballot_card`, `_resolve_move`, `decide`. |
| `tools/action.py` | key / direction algebra | UNCHANGED. `KEY_MAPPING` and `DIRECTION_VECTORS` are imported by name, which adds no module edge. |
| `tools/frame.py` | screenshot bridge | UNCHANGED and unused — the evaluator captures its own frame. |
| `AGENT.md` | role + verified perception facts | three false claims removed (see below). |
| `rules.md` | contract + legal actions + anti-cheat | the phantom direction-letter HUD removed; the ballot channel documented. |
| `skills/strategy.md` | strategy prior | UNCHANGED. |
| `memory.md` | scratchpad template | UNCHANGED and still dormant. Its advertised wiring (`agent.py` filling it per step into `evaluator.current_prompt`) is a DEAD channel on this build: `current_prompt` is read on turn 1 only. The ballot is the live replacement. |

## the two seams, and why these two

The evaluator's `openai_ai_model` resolves `build_conversation_history` as a module global of its
own module and then stores what it gets back as `conversation_history`. Of the plausible injection
points, only two are live on this build:

| seam | status |
|---|---|
| `evaluator.current_prompt` mutated per step | dead after turn 1 (`build_full_prompt` reads it once) |
| `frame_data.image` replaced | dead (`baseline_step` re-captures the surface itself) |
| `conversation_history.append(...)` | dead (the appended message is dropped on rebuild) |
| `build_conversation_history` swapped in the step's globals | **LIVE, every turn** — used here |
| `Action.key` rewritten | **LIVE** — the only thing that decides what the engine presses |

## engine laws this relies on (documented, seed-independent, not oracle)

Read off `games/Snake/snake_adapter.py`:

* `reset_game`: `mid = 4 // 2`, `start_y = 4 - 1`, so the snake is `[(2,3,2), (1,3,2)]` and
  `last_move_dir = (1,0,0)` on EVERY seed. The anchor's "starts at the center" was false.
* `_try_move`: refuses when `last_move_dir + delta == 0` on all three axes — it returns before
  `_step`, so a refused 180 does not move the snake and does not even change `last_move_dir`.
* `_step`: out of `0..3` on any axis ends the episode; self-collision is tested against
  `snake[:-1]`, so the vacating tail cell is legal to enter.
* `_generate_food` / `score`: not read by anything here.

## lever table (what the next proposer can turn)

| lever | where | current value |
|---|---|---|
| occupancy window length | `LatticeLedger.__init__`, `deque(maxlen=4)` | 4 recent head cells. The true body is `len-1` cells and length is unobservable without pixels, so this OVER-forbids at length 2-3 and UNDER-forbids past length 5. Tier 2 of `menu` absorbs the over-forbidding; the under-forbidding is the residual risk. |
| menu tiering | `LatticeLedger.menu` | two tiers: cube ∧ window, else cube. Leaving the cube is always fatal, re-entering an over-counted window cell may not be, so the cube bound outranks the window. |
| projection rule | `_resolve_move` | keep if legal → first legal move on the asked axis → `fallback`. |
| last-resort ranking | `LatticeLedger.fallback` | least-asked this episode, ties to the roomiest arrival cell, then `_ORDER`. |
| one-ply room test | `LatticeLedger.room` | counts open neighbours of an arrival cell; only a tie-break today. |
| disclosure | `LatticeLedger.ballot` | reports last turn's asked-vs-pressed and the running tally. |
| stale-ballot strip | `_ballot_card` | strips `_MARK` from every `input_text` part, then appends one. |

## what was measured (offline rig, `traces_rig/`)

`traces_rig/stub.py` replicates `reset_game` / `_try_move` / `_step` and
`openai_utils.build_conversation_history` (part types `input_text` / `input_image`, system message
carried from `previous_history[0]`, `history_window=29`) and drives `tools/vlm.py:decide` directly.

**THE CAVEAT THAT BOUNDS EVERY NUMBER BELOW: a stub model cannot read prose.** The REQUEST side of
this mechanism — the ballot block itself — is therefore UNMEASURABLE offline, and nothing here
prices it. Only the key-side actuator and its ranking rules are priced. Food totals are also
incidental: a blind stub bumps into food, it does not steer towards it, so these are a floor on
survival, not a prediction of score.

Ladder, 20 seeds x 100 steps (`traces_rig/ladder.py`):

| stub model | parent (pass-through) | with the ballot |
|---|---|---|
| always `move_down` | 0 food, 20/20 wall deaths, 1.0 steps/ep | 2 food, 0 deaths, 100.0 steps/ep |
| always `move_forward` | 0 food, 20/20 wall deaths, 3.0 steps/ep | 4 food, 0 deaths, 100.0 |
| always `move_left` (the 180 at spawn) | 0 food, 0 deaths, **2000/2000 refusals** — total lock | 6 food, 0 deaths, 0 refusals |
| cycle `F,R,F,R` | 1 food, 20/20 wall deaths, 4.0 steps/ep | 17 food, 0 deaths, 100.0 |
| the measured live ask mix (fwd .386 / back .346 / right .127 / down .109 / up .030 / left .002) | 0 food, 20/20 wall deaths, 3.4 steps/ep | 27 food, 0 deaths, 100.0 |
| a garbage action name | 0 food, 20/20 wall deaths, 2.0 steps/ep | 2 food, 0 deaths, 100.0 |
| the live mix with `metadata=None` (parser-failure path) | — | 2 food, 0 deaths, 100.0 |

Invariants (`traces_rig/verify.py`), 20 seeds x 100 steps per arm:

* dead-reckoned head vs the engine's true head: **0 errors / 2000 steps** on each of three arms;
* presses that would leave the cube: **0**; deaths: **0**;
* the model's own answer was pressed unchanged on 38.0% (always-left), 46.0% (cycle), 51.9% (live
  ask mix) of decisions — the projection is a minority event, not a takeover.

Ablations (`traces_rig/ablate.py`), live-ask-mix stub unless noted:

| arm | food | deaths |
|---|---|---|
| full | 27 | 0 |
| − occupancy window (cube bound only) | 27 | 0 |
| − occupancy window, `cycle F,R` stub | **1** (full: 17) | 0 |
| − least-asked ranking (press `menu[0]` instead) | 18 | 0 |
| − axis preservation (always `fallback`) | 34 | **1** |
| − projection (ballot written, model's key pressed) | **0** | **20/20** |

Readings. The actuator is the load-bearing half: with the ballot written but the key left alone,
every episode still dies in ~2 steps. The occupancy window is worth nothing against a random asker
and almost everything against a cycling one (1 → 17), which is the behaviour it exists for. The
least-asked ranking is worth +9 food on the live ask mix. The `−axis` arm scoring HIGHER food (34)
is an artefact of the caveat above and should not be read as a win: a blind stub names no food, so
discarding its axis costs it nothing, whereas for the live model that axis is the only food
evidence in the loop — and the arm still took the only body death in the table.

## corrections to the priors

Three claims in the anchor's prose are false on this build and were removed:

1. `AGENT.md`, "Snake starts at the center" — it starts at `(2,3,2)` with its neck at `(1,3,2)`,
   on the `y = 3` floor, already heading `+X`; a `+Y` step at the opening frame leaves the cube.
2. `AGENT.md`, "Maximum possible score: 64 (filling the entire grid)" — replaced with what scoring
   actually is (a point per food, the snake grows by one).
3. `rules.md`, six `Look for the "X" letter on screen` clauses plus "Indicators only show for moves
   that won't cause immediate collision" — this build renders no direction letters and no legality
   affordance of any kind, so the agent was being told to read a HUD that is not there. The ballot
   now supplies that affordance instead, and `rules.md` says so.

`system_prompt.txt` (and its `system_prompt.md` mirror) were regenerated so they stay byte-equal to
`compose_system_prompt()`.
