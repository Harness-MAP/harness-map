# Diagnosis — maze3d_pro iter 1 / cand_r001_1_action-planning

**Headline:** primary `success` **7/25 = 0.280** (parent anchor `anchor_maze3d_pro` 2/25 = 0.080;
sibling `cand_r001_0_spatial-grounding` 8/25 = 0.320, i.e. statistically tied with this one at
temp 1.0). Wins: **L1, L3, L6, L7, L15, L17, L20** — a strict superset of the parent's {L6, L17},
no regressions. All 18 losses terminate at exactly 100/100 steps with `distance_to_exit` 3–15;
1943/1943 decisions parsed to a legal action and `default_actions = 0` in all 25 episodes.

**Binding constraint (one sentence):** the harness **never completes a single floor transition**
(0 in 25 episodes; 114 of 114 `climb_up` executions refused), and **12 of the 25 levels put the
exit on floor 2**, so 48% of the benchmark was unwinnable by construction and 7/13 is the real
score on the reachable half — the cause is a **HARNESS-BUG in the three prose priors this iter
deliberately froze**, which gate `climb_up` on a `[Q] UP` HUD prompt that the eval engine does not
draw, compounded by a **VLM-CEILING on "am I standing on a stair"** for which the frame carries an
exact signal the harness does not read.

---

## 1. What this harness does (mechanism)

`orchestrator.py:build_agent(evaluator)` returns the per-step `agent(frame_data)` the extended
evaluator installs as `ai_model_func`. It builds ONE `tools.plan.ActionPlanner` per episode and
threads it through `tools.vlm.decide`, whose loop is

    observe(frame) -> inject working memory into conversation_history[0] -> evaluator.baseline_step
                   -> planner.apply(proposal) -> engine Action

* `tools/plan.py:ActionPlanner.observe` fingerprints `frame_data.image` (md5 of every 2nd pixel)
  and scores the PREVIOUS decision: a translation that left the screen unchanged (identical
  fingerprint, or < 0.2% of sampled pixels differing) is filed into `refused[view]`.
* `govern(name)` vetoes exactly three things — a translation already in `refused[view]`, a climb
  equal to `climb_lock` (the inverse of the last executed climb), and a turn proposed while
  `since_move >= 4` — and re-aims onto `_reaim()`, a right-hand wall follower (bearings h+1, h,
  h-1, h+2: one right turn on arrival, left turns after).
* `tools/action.py:retarget` rebuilds the Action for the re-aimed name, stripping
  `granularity`/`repeat`/`json_data` first.
* `render_memory()` renders the ledger from `memory.md`'s template; `tools/vlm.py:_inject_memory`
  stashes the whole first-call system turn and appends the block to `conversation_history[0]`
  every step (the one channel `build_conversation_history` re-reads).
* `AGENT.md` + `skills/navigation.md` + `rules.md` are concatenated by `compose_system_prompt()`
  into `evaluator.current_prompt`. **This iter froze all three unchanged** — NOTES.md §axis: "the
  prose priors are deliberately held fixed so this iter measures the mechanism alone." That
  decision is the single largest source of the residual loss (§4).

---

## 2. What actually happened (causal chain, with evidence)

### 2a. The two new mechanisms both verifiably work

**The per-turn memory channel lands.** Frame-0 `prompt_tokens` is **1986** in both parent and
candidate (correct — `_inject_memory` is a no-op while the history is empty). Frame 1 is
**2493–2504** in the parent and **2757–2778** in the candidate: a stable **+264…+280 token** block,
and the delta persists at frames 2–4 (3007→3276, 3514→3788). The stash-and-append on
`conversation_history[0]` is sound; the JSON RESPONSE FORMAT survives (0 parse failures, 0 defaults).

**The governor is load-bearing.** I rebuilt an exact forward model of the engine from the level
files and the action contract (see §5.6) and calibrated it on the parent, which is a pure
pass-through: replaying the parent's recorded proposals reproduces **25/25 `success` and 25/25
exact `distance_to_exit`**. Against that instrument:

| | parent (ungoverned) | this candidate |
|---|---|---|
| decisions | 2340 | 1943 |
| `move_forward` share of proposals | 2090 (89.3%) | 1359 (69.9%) |
| longest same-action run per episode | mean **62.5**, max **100** (four episodes of 100/100 `move_forward`) | mean **10.8** |
| translations refused | **1904 of 2090 = 91.1%** | 240 of 1008 = **23.8%** |
| successful moves | **186** | **768** |
| override rate | — | **519/1943 = 26.7%** (413 on a proven-refused translation, 106 on a stalled turn, 0 on the climb lock; 36 free agreements) |

**Four of the seven wins were produced by the re-aim, not by the model.** Executing the recorded
proposal tape verbatim in the exact simulator ends L1 at distance 4, L15 at 5, L17 at 12 and L20 at
7 — all losses. Only L3, L6, L7 are in the model's own tape. The override is doing real work.

### 2b. Residual failure 1 — the refused climb is an ABSORBING state (L11, L13)

L13's tape (`action_taken`, 100 decisions, `F`=forward `L/R`=turn `U`=climb_up):

    RFFFRFFLFFRFFFLFFF UUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU

18 productive decisions, then **82 consecutive `climb_up`**. L11 is the same shape (68 productive,
then 32 `climb_up`). Final `current_floor = 1` in both, so all 114 climbs were refused no-ops:
**5.9% of the candidate's entire decision budget, and 100% of two episodes.**

I reproduced the state machine in isolation against the real `tools/plan.py`. After one refused
`climb_up` (frame byte-identical) the harness tells the model:

    - last decision: `climb_up` was executed as you asked.
    - PROVEN REFUSED from this exact screen: nothing proven yet
    - stair lock: `climb_down` is locked: you just changed floor, explore this level before going back
    - mode: FOLLOWING YOU - your proposal is executed unless it is already proven useless.

and after **8** identical refused climbs: `overrides = 0`, `refused[view] = None`. Three
independent code paths each contribute one false statement:

1. `plan.py:observe` scores outcomes only for `name in _act.MOVEMENT_ACTIONS` — a refused climb is
   never filed, so `blocked_here` stays "nothing proven yet" forever.
2. `plan.py:_last_line` appends the "and the screen …" outcome clause only for
   `MOVEMENT_ACTIONS`, so a refused climb is reported as **"was executed as you asked."** —
   affirmatively false, 82 times.
3. `plan.py:_commit` sets `climb_lock = inverse_climb(name)` for *any* executed climb, refused
   included, so `climb_line` asserts **"you just changed floor"** when nothing happened, and bans
   the only other floor action for 8 decisions.

And nothing vetoes it: `govern`'s climb branch fires only on `name == self.climb_lock` (the
*inverse*), and the stall branch is `self.sweep and name in _act.TURN_ACTIONS` — **climbs are
outside the escape sweep entirely**. The first climb at each of these two levels (L11 step 68,
L13 step 18) was invited by `fields()["untried_here"]`, which enumerates all six legal actions and
therefore prints `` `climb_up` `` as "never tried from this exact screen" at every fresh view
(2 of 2 first-climbs). The parent, with no memory block, proposed **zero** climbs in 2340 decisions.

### 2c. Residual failure 2 — the agent walks over stairs and never climbs (the real ceiling)

The exit is on floor 2 for **L2, L4, L5, L8, L9, L10, L11, L12, L14, L16, L18, L19** (12 levels)
and on the start floor for the other 13. **Every win is an F1-exit level; every F2-exit level is a
loss.** Across parent + both candidates (75 episodes) not one episode ended above floor 1 — yet
tree-wide 65 of 845 maze3d_pro episodes on 2026-10-04 (same pure-visual engine) *did* reach floor 2,
including L2 and L4 wins. The floor transition is reachable; this arm never finds it.

A cyan-family pixel census over the frames the agent was actually shown (per-episode video,
800×600, BGR filter `B>60 ∧ G>60 ∧ R<60 ∧ (B+G)−2R>80`) shows exactly why. **L23**, decision index
: action : cyan px —

      0:R:0      1:F:0      2:F:0      3:F:1753   4:F:15314  5:F:27133  6:F:61082  7:F:242044
      8:L:0      9:F:0     10:F:0     11:R:0     ...
     21:F:1753  22:F:15310 23:F:27131 24:F:61087 25:F:242038 26:L:0
     37..41 -> 1753/15314/27133/61082/242043  42:L:0
     53..57, 69..73, 85..89 -> the same five values, then 0

Six byte-identical laps of a period-16 orbit. Each lap the agent walks *up to* the stair
(1.7k → 15.3k → 27.1k → 61.1k → **242k** ≈ 50% of the frame when the stair is the adjacent cell
dead ahead), steps forward **onto** it (the next frame reads **0** cyan — the tile underfoot is
never drawn), then **turns left and walks away**. L21 does the same thing **ten** times; the L20
win crossed a stair once at step 11 and kept going. Across these three episodes **17 of 17**
saturated-cyan frames were followed by a forward move and a 0-cyan frame: the agent stood on a
stair 17 times and proposed `climb_up` on none of them. Replay-derived totals: the candidate stood
on a stair tile for ≈121 decisions across the 25 episodes, including at least once on **7 of the 12
F2-exit levels** — each of those was one correct keystroke from the other floor.

The reason it never climbs is written into the priors the iter froze:

* `AGENT.md`: "**Top HUD (Top-Left)**: Displays your current `POS` …, `FLOOR` …, and `DIR` …
  **Rely on this to map your progress.**" and "**Stair Prompts (Bottom-Center HUD)**: This text
  ONLY appears when you are standing exactly ON a stair tile. If it shows `[Q] UP` …"
* `rules.md`: "`climb_up`: … **CRITICAL: ONLY use this action if you are standing on a stair AND
  see "[Q] UP" in the bottom HUD.**" (same for `climb_down`)
* `skills/navigation.md`: "Use `climb_up` or `climb_down` when standing on a stair tile and the
  corresponding prompt (`[Q] UP` or `[E] DOWN`) is shown at the bottom."

**None of those UI elements exists.** Measured on the frames the agent saw, 229 frames / 3
episodes: **0 near-white pixels in rows 0–59** (max channel in that strip 52–76) — no POS/FLOOR/DIR
readout at all; and across the **174** frames whose total cyan is 0 (which includes all 17 on-stair
frames) there are **0 bright-cyan pixels in the bottom text band (rows 520–579)** — the `[Q] UP`
prompt never appears. This independently re-confirms the 2026-10-03 direct-render measurement (the
engine draws no text anywhere). The harness therefore gates its only floor-changing action on an
unsatisfiable precondition, and the frozen model obeys: the parent climbed 0 times in 2340
decisions, and the candidate's 114 climbs all fired *off*-stair, i.e. exactly where the rule did
not bind — which is itself the proof that the model has no way to tell when it is on a stair.

### 2d. Residual failure 3 — the PRODUCTIVE orbit the governor cannot see

L12: `RFFFRFFRFRRFLFFLFF` then **`LLFF` × 21**. Overrides in the exact replay: **0**. Every `F`
succeeds, so `observe` resets `since_move`, `sweep` never turns on, and nothing in `govern` has an
opinion about a legal, non-stalling, non-progressing 4-cycle. L14 is `FRFFF` × 19 (19 overrides, 57
successful moves, distance 7). L16 `FFLFFFFF` × 10, L22 `FLFRFF` × 10, L18 period 9 × 9, L21 period
8 × 5, L23 period 16 × 6, L24 period 9 × ~5, L4 period ~3.

Replay-derived coverage over the 18 losses: the agent visits **5–28 of 15–129 open cells** (median
≈13%) and **72–95% of its decisions re-enter a cell it has already stood in**. The ledger is a
*refusal* ledger; it has no concept of "I have been in this view before and already left it this
way." `plan.py` already keeps `tried[view]` and already renders `view_visits` into the prompt — but
`govern` reads **neither**.

---

## 3. Failure breakdown (18 losses)

| class | levels | n | decisions burned | mechanism |
|---|---|---|---|---|
| **A — refused-climb absorption** | L11, L13 | 2 | 114 (82 + 32) | `observe`/`_last_line`/`_commit`/`govern` are all blind to a no-op climb (§2b) |
| **B — productive short orbit** (period 3–16, 5–21 reps) | L4, L12, L14, L16, L18, L21, L22, L23, L24 | 9 | ~700 | every translation succeeds, so the refusal ledger and the sweep never engage (§2d) |
| **C — wander, no short cycle** | L2, L5, L8, L9, L10, L19, L25 | 7 | ~700 | 6 of the 7 are F2-exit levels; no exploration frontier, no floor change |
| **gating all of B/C on 12 levels** | L2,4,5,8,9,10,11,12,14,16,18,19 | 12 | — | 0 floor transitions ⇒ unwinnable (§2c) |

Cross-checks against the numeric pre-pass: `steps_taken = 100` and `vlm_calls = 100` on every loss
(exactly one VLM call per decision — `baseline_step` is called every step, so the guard's call
counter can never void the run); `distance_to_exit` 3–15; `successful_actions = total_responses` and
`default_actions = 0` in all 25 response files. **There is no parse, fallback, legality or
budget-accounting defect anywhere in this run.**

---

## 4. Binding constraint — classify it

**HARNESS-BUG (dominant, 12 levels gated).** `AGENT.md`, `rules.md` and `skills/navigation.md`
describe a HUD and a `[Q] UP`/`[E] DOWN` stair prompt that the eval engine does not render, and
`rules.md` makes both climbs *conditional* on the missing prompt. The precondition is unsatisfiable,
so the only floor-changing action is effectively disabled and 48% of the benchmark is out of reach.
This is fixable in the harness, in three prose files.

**VLM-CEILING (what makes the prose fix insufficient).** Correcting the prose is **necessary but
not sufficient**, and this is already measured: the sibling `cand_r001_0_spatial-grounding`
rewrote exactly these files — "*Never refuse to climb merely because you saw no prompt; the prompt
does not exist*" — and went from 0 to **179** climb proposals (118 up + 61 down) with **still 0
floor transitions in 25 episodes**. The frame genuinely does not show the agent that it is standing
on a stair: the tile underfoot is never drawn, so an on-stair frame carries **0** cyan, identical to
"no stair anywhere in view". No re-wording can recover information the pixels do not contain. The
fix must **scaffold the decision**: the harness must compute the on-stair event from the frame
sequence (the saturate→0 transition of §2c) and act on it.

**TASK-GAP (the second-largest residue, 9 losses).** Inside a floor the agent moves legally and
productively — 768 successful moves, 23.8% refusal rate — and still orbits, covering ~13% of the
open cells with 72–95% revisits. Nothing in the loop rewards an unvisited frontier. A refusal
ledger cannot fix this; it needs a visit/transition ledger and a commitment rule.

---

## 5. Concrete changes to try (ranked by leverage)

### 5.1 Correct the three prose priors AND add a frame-measured on-stair detector (ship together)
*Files: `AGENT.md`, `rules.md`, `skills/navigation.md`, new `tools/stair.py`, `tools/plan.py:govern`/`observe`.*

(a) `AGENT.md` — delete the "Top HUD (Top-Left) … Rely on this to map your progress" bullet and the
whole "Stair Prompts (Bottom-Center HUD)" bullet. State the measured facts instead: the render is
800×600 and carries **no text anywhere** (0 near-white px in rows 0–59 over 229 measured frames);
walls are the brown family; the exit is gold with a white outline; a stair is a cyan/teal tile
**that is only visible while it is ahead of you — standing on it paints zero cyan**.
(b) `rules.md` — strike "ONLY use this action if you are standing on a stair AND see '[Q] UP' in
the bottom HUD" from both climbs. Replace with: a climb only works while you are standing on a
stair tile; otherwise it is a silent no-op that leaves the frame byte-identical. Add explicitly:
*the `[Q] UP` / `[E] DOWN` prompt does not exist on this build; never refuse to climb because you
did not see one.*
(c) `skills/navigation.md` — same correction in the "The exit can be on any floor" bullet.
(d) NEW `tools/stair.py` — a cyan-family census of `frame_data.image`, returning
`(total_cyan_px, centre_band_cyan_px)`. **Re-measure the thresholds on the live frame** (the ladder
below is off lossy video, so treat the *shape* as the finding, not the constants): 0 = no stair in
view *or* standing on one; ~1.7k / 15.3k / 27.1k / 61.1k at receding distance; **≈242k (≈50% of the
frame) when the stair is the adjacent cell dead ahead**; an off-axis adjacent stair reads ≈46k, so
restrict the saturation test to the central band. Wire into `ActionPlanner`: set an `on_stair` latch
when (the previous frame was saturated) ∧ (the executed action was a translation) ∧ (the frame
changed). While the latch is set, `govern` forces `climb_up` once, and `climb_down` once if the
up-climb left the frame identical — then releases the latch.
**Critical guard:** a *successful* climb is frame-invisible in 22/96 measured cases, so never treat
an identical frame as proof a climb failed. Bound the latch with a **budget** (one up + one down per
latch), not with a frame-identity verdict, and never file a view-keyed "not a stair" fact ("am I on
a stair" disagrees in 57 of 486 view groups — it is not view-invariant).
**Predicted effect:** unlocks the 12 F2-exit levels; 7 of them already had the agent standing on a
stair within the 100-step budget, and L21/L23 walk across one 10 and 6 times respectively. This is
the only change on this list that can move `success` above the 13/25 structural ceiling.

### 5.2 Make a refused climb visible to the loop (certain, ~20 lines, recovers 2 episodes)
*File: `tools/plan.py`.*
* `observe`: add a `consec_null` counter — "the previous action left the frame unchanged" is an
  exact outcome signal (0 counterexamples in 1584 calibrated poses) — and increment it for climbs
  too, **not** as a view-keyed refusal.
* `govern`: add a branch re-aiming a climb when `consec_null_climbs >= 1`, and widen the stall
  branch from `_act.TURN_ACTIONS` to `_act.TURN_ACTIONS | _act.CLIMB_ACTIONS` so the escape sweep
  can break a climb loop.
* `_last_line`: append the outcome clause for climbs. A refused climb must read "*and the screen did
  NOT change at all, so there is no stair under you*", never "was executed as you asked".
* `_commit`: set `climb_lock` only when the climb actually changed the frame; reword `climb_line` so
  it never asserts "you just changed floor" for an unverified climb.
* `fields()["untried_here"]`: restrict to `MOVEMENT_ACTIONS | TURN_ACTIONS`, listing `climb_*` only
  when the §5.1 detector says a stair is underfoot. This removes the invitation that fired the first
  climb in both absorbed episodes (2/2).
**Predicted effect:** recovers the 114 absorbed decisions. L13 is an F1-exit level that lost 82 of
100 decisions to this one bug — a direct win candidate independent of §5.1.

### 5.3 Turn the refusal ledger into a VISIT/TRANSITION ledger (9 losses)
*File: `tools/plan.py:govern` + a new `transitions` dict.*
`tried[view]` and `views[view]` already exist and `govern` reads neither. Record
`transitions[(view, executed_action)] += 1`; in `govern`, re-aim a proposal that would repeat a
transition already taken **≥2** times from this view onto the lowest-count bearing from this view,
and surface a per-view "you have left this screen heading X twice already" line in the memory block.
Keying on the view (not a pose) is sound for the bearing question: "is the cell ahead a wall" is
constant within all 486 calibrated view groups (0 inconsistent), and the fact is level-poolable
(0 disagreements in 172 groups at 9.21:1 aliasing).
**Predicted effect:** directly attacks L4/L12/L14/L16/L18/L21/L22/L23/L24 — nine losses whose tapes
are literally a period-3-to-16 cycle repeated to the cap, and where the current governor fires 0–19
times. Price it on coverage (cells visited / open cells, currently a median 13%) and revisit rate
(currently 72–95%), not on `success` alone — those two move first and are far less noisy.

### 5.4 Fix the sweep's latching and its self-description (free, correctness)
*File: `tools/plan.py`.* `_reaim` sets `self.sweep = True` and increments `sweep_turns` **before**
`govern` discovers the model already agrees, so the docstring's "agreement costs nothing" is false
and a single climb-lock or refusal veto latches ESCAPING mode until a translation succeeds — the
harness was in that mode for **649 of 1943 decisions (33.4%)**. Move the side effects behind the
agreement check. Also: `_mode_line` tells the model the escape "always rotat[es] the same
direction" while `_reaim` does right-once-then-left-forever; make the prose match the mechanism.
Minor: the sealed-cell guard `sweep_turns > 4` spends 5 rotations (450°) before trying
`move_backward`; `>= 4` is correct. And `observe` files `move_backward` into the view-keyed
`refused` ledger, which is **unsound** (backward disagrees in 65 of 486 view groups) — dormant today
(0 `move_backward` in 1943 decisions) but it will bite the moment the sealed-cell branch fires.

### 5.5 Seed the sampler so the next gradient is readable (measurement, not score)
`create_openai_response` sends `temperature=1.0, top_p=0.9` and **no seed**. Content-address a
per-request `seed` on `evaluator.client` — `client.chat.completions` is a cached property so
shadowing `.create` sticks (openai 2.42.0), and the eval guard forwards the `evaluator.client` read.
Prior calibration: paired repeats went identical-trace 0/3 → 1/3 and final distance reproduced
0/3 → 3/3. 7/25 vs the sibling's 8/25 is noise; without this, a ±1-episode change cannot be
attributed to a mechanism.

### 5.6 A calibrated offline rig exists — price changes before burning a live eval
The engine's movement/climb semantics are exactly reproducible in ~30 lines, which lets the next
iter sweep thresholds and veto rules for free. Level files:
`/home/azureuser/workspace/SpatialWorld/data/game/maze3d_pro_merge/Level_NN.txt` — 5×5 to 13×13,
`#` wall, `S` start, `E` exit, `T` stair, floors separated by a `===` line. Start heading NORTH =
−row; `turn_right` = +col; `move_forward` granularity small/medium/large = 1/2/3 cells, stopping
early at a wall; `climb_up` requires `T` underfoot and a non-wall cell at the same (x,y) one floor
up. Replaying the parent's recorded proposals through that model reproduced **25/25 `success` and
25/25 exact `distance_to_exit`** — it is exact, and it also proves the anchor is a pure pass-through.
**Anti-cheat:** the rig is for OFFLINE pricing only. Nothing derived from a level file may enter the
shipped harness; the shipped policy must keep deciding from `frame_data.image` alone, exactly as it
does today (verified: this candidate touches no `evaluator.game` attribute and opens nothing under
the data tree).

---

## 6. Exhausted — do NOT repeat

* **JSON parsing, the parse fallback, the key table, granularity spelling.** 1943/1943 legal parsed
  actions, `default_actions = 0` in all 25 episodes, in both parent and candidate. Zero headroom.
* **The view-keyed `move_forward` refusal ledger + the right-handed escape sweep.** SHIPPED and
  working: refusal 91.1% → 23.8%, successful moves 186 → 768, longest same-action run 62.5 → 10.8,
  `success` 2/25 → 7/25, and 4 of the 7 wins are not in the model's own proposal tape. Extend it;
  do not re-derive it.
* **The per-turn memory channel.** `_inject_memory`'s stash-and-append on
  `conversation_history[0]` is verified live (+264…+280 prompt tokens every turn after the first,
  JSON contract intact). Do not re-plumb it, and do not splice the base turn.
* **Correcting the missing-HUD prose ALONE.** The sibling candidate did exactly that and reached 179
  climb proposals with still **0** floor transitions. It must ship with the frame-measured on-stair
  detector (§5.1) or it buys nothing.
* **An exit-line-of-sight commit lever as the next axis.** Gold-in-the-central-band is an exact
  exit-LoS detector (0 false positives in 1776 calibrated poses), but replay shows **0 LoS frames in
  all 18 losses** — it fires only in 4 of the 7 episodes already being won. It is a closer with no
  opportunity in the loss set; rank it below §5.3.
* **View-keyed facts for `move_backward` or for "I am / am not on a stair".** Unsound: those
  predicates disagree in 65/486 and 57/486 view groups respectively, against 0/486 for
  "is the cell ahead a wall". Use consecutive-frame no-op facts for those instead.
* **Generic "avoid loops / look carefully" prose and shrinking the history window.**
  `skills/navigation.md` already says "Avoid turning repeatedly in the same spot" and the memory
  block already reports `view_visits`; the period-4-to-16 orbits of §2d survived both.
