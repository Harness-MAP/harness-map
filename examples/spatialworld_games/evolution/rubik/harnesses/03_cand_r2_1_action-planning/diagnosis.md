# Diagnosis — rubik (spatial_world) iter r001 / `cand_r001_0_spatial-grounding` (k4_claude_pv)

**Headline:** primary = **1/20 solved = 0.050** — **flat** against the `staged_anchor` parent (0.050).
1901 steps / 1901 VLM calls; **0 parse failures, 0 default actions, 0 engine refusals, exactly one
engine action per step in 1901/1901**. The mechanism is live and does exactly what it was designed to
do: overrides fired on 370/1901 steps (19.5 %), the parent's `BLIND_SPIN` is gone (committed view
rotations 800 → **67**), and the parent's provably-wasted budget collapsed from **80.1 % → 3.5 %**.

Binding constraint: **HARNESS-BUG — the `revisit` rule has the wrong sign.** It maximises *distinct
configurations visited* (median 4 → **101**, exactly as the parent gradient asked), but the solution
of every level is **one specific configuration ≤4 quarter-turns from the START**. Refusing to re-stand
on a visited configuration turns the trajectory into a self-avoiding walk that leaves that
neighbourhood at step 4–12 and **never returns**: steps spent inside the ≤4-move ball of the start
fell from the parent's **1173/1901 (61.7 %)** to **168/1901 (8.8 %)**.

Two numbers carry this diagnosis:

1. **The rule caps the attainable score, independent of the model.** I ran an *optimal* ball-covering
   router under the shipped `RULES` table: with `revisit` active it reaches **2 of the 12** depth-1
   states and **2 of the 114** depth-2 states in 100 moves. With `revisit` removed, the same router
   reaches **8/12 and 49/114**; an explicit 6-face `X X X X` sweep reaches **12/12 in 24 moves**. The
   live run measured **1–2 of 12** per episode — i.e. **the harness is already performing at the
   ceiling its own rule imposes.** No prompt, skill or model change can move the primary while that
   predicate is in the table.
2. **The whole 0.050 is the harness's own substitution constant, not the mechanism.** `_offset` is
   `len(played) % 12`, so at step 0 it is always 0 and `survivors()[0]` is always
   `sorted(FACE_ROTATION_ACTIONS)[0] = rotate_back_acw` (B′). Five episodes had their step-0 pick
   refused (`rotate_cube_z_cw`) and **all five were therefore played B′**; L5 is the one level B′
   solves, and it ended in 1 step. Meanwhile **L2 — which the parent solved in 1 step** with
   `rotate_down_cw` — got the same constant B′, missed, and burned 100 steps. The candidate traded one
   coin flip for another. (`traces_notes/MECHANISM.md` predicted this exact L5 win on the stub rig:
   "the first move of the preference ring, which on that level happens to be the one-move solution".)

---

## 1. What this harness does (mechanism)

`orchestrator.py:build_agent` installs `tools/vlm.py:decide`, a six-statement turn. Verified live, not
assumed — I re-ran the shipped decision table against the recorded pick stream of all 20 episodes and
reproduced the committed move **1901/1901 with zero mismatches**, so every claim below is exact.

| stage | code | measured behaviour |
|---|---|---|
| inform | `vlm.py:_inform` → `_splice` | appends the HARNESS STATE brief to `conversation_history[0]`. Confirmed in the payload: `prompt_tokens` 975 → **2109** at step 0 (the +1150 is `skills/macro_commitment.md`) and plateaus at **~11 400** vs the parent's 9 800 (the +400 is the brief re-spliced each turn, not accumulating). |
| mute | `vlm.py:_arm` | swaps `evaluator.execute_action` for a recording sink so `baseline_step`'s own `execute_action` call (`rubik_openai_evaluation.py:606`) cannot pre-empt the veto. **Works: exactly one `… rotation executed:` line per step in 1901/1901, `default_actions: 0`.** |
| decide | `await step(frame_data)` | the one pristine model call. `Qwen3-VL-30B-A3B-**Instruct**`, `reasoning=None`, completions a flat **22–28 tokens** on all 1901 calls. |
| commit | `vlm.py:_commit` → `action.py:survivors` | honours the pick unless a rule fires, else plays `survivors(book)[0]`; rewrites `key` + `face`/`direction`/`action_type`/`function_called` together from `SPEC`. |

`tools/action.py:RULES` is the policy. Firings over the eval: **`revisit` 330, `view-streak` 33,
`blind-axis` 7.** `|refused|` is **3 of 18 on 1565/1901 steps**, so the brief's ALLOWED line is a
**14-name comma list** — rendered, it is the last line of an ~11 400-token prompt whose preceding 29
turns are the model's own repeated answers.

## 2. What actually happened (causal chain, with evidence)

I built an **independent 20-cubie cube model** (`/tmp/.../cube.py`, no engine source, no scramble
data — see Anti-cheat) and replayed the executed move stream of every episode. It reproduces the BFS
ball sizes **12 / 114 / 1068 / 10011** and `ord(F)=4`, `ord(RU)=63`, `ord(RU')=105` (the parent's
validated model reports the `RU`/`RU'` pair mirrored — my chirality convention is the mirror image,
a global relabel that leaves every coverage metric below invariant).

**Chain:** the model names a move → the rule table refuses the 4th repeat → the harness substitutes
the next element of an alphabetical ring → **the model adopts the substitution** → it repeats that
move three times → the net word grows by one letter every three steps → the walk marches monotonically
out of the neighbourhood where the solution lives → 100 steps elapse far from solved.

**The ratchet, step by step (L20, `rubik_20__adrz7vq`):**

```
stp  MODEL PICK          COMMITTED           OVR  rule      reduced word
0    rotate_up_cw        rotate_up_cw                       (start)
1    rotate_up_cw        rotate_up_cw                       U
2    rotate_up_cw        rotate_up_cw                       U2
3    rotate_up_cw        rotate_down_cw      <<<  revisit   U'
4    rotate_down_cw      rotate_down_cw                     U' D     <- model ADOPTS the substitution
5    rotate_down_cw      rotate_down_cw                     U' D2
6    rotate_down_cw      rotate_left_acw     <<<  revisit   U' D'
7    rotate_left_acw     rotate_left_acw                    U' D' L' <- adopts again
...  period 3 forever ->  U' D' L R' B D' L R ...   (33 overrides in 100 steps)
```

**Adoption is measured, not inferred: after an override, P(next model pick == what the harness
played) = 236/361 = 0.654**, against a 1/18 = 0.056 chance baseline — an **11.8× enrichment**. The
substitution ring is therefore not a fallback, it *is* the policy; the model is a copy function on
top of it (it also repeats its own previous pick **938/1901 = 49.3 %** of the time).

**The outbound march, measured.** Distance from the episode's own start, per step:

| | parent | candidate |
|---|---:|---:|
| steps inside the ≤4-move ball of the start | 1173 / 1901 = **61.7 %** | 168 / 1901 = **8.8 %** |
| first step outside that ball | — (11/20 never left) | **step 4–12, in 19/19 capped episodes** |
| returns to the ball after leaving | — | **0** |
| distinct cube states / episode (median) | 4 | **101** |
| final reduced-word length (median) | 0 | **50** |
| depth-1 shell coverage / episode | 0–12 | **1–2 of 12** |
| depth-2 shell coverage / episode | 0–6 | **1–3 of 114** |

The parent gradient predicted "distinct states from a median of 4 to ≥40 and primary into the
0.15–0.35 band on cycle-break alone." Distinct states hit **101** and the primary did not move,
because **distinct-state count is not the quantity that correlates with winning** — near-ball
coverage is, and that is the one metric the candidate made *worse*.

**Why `revisit` cannot be rescued by a better policy.** `new_book()` seeds `seen = {()}`, so the
**start configuration is refused from step 0 onwards**. Fanning out across the depth-1 shell requires
passing back through the start (`X` then `X'`, or `X X X X`), and every such return is exactly what
the rule refuses. Measured ceiling, with an optimal greedy ball-covering router and a 100-move budget:

| | d=1 shell | d=2 shell |
|---|---:|---:|
| **with** the shipped `revisit` rule | **2 / 12 (17 %)** | **2 / 114 (2 %)** |
| without it (leash only) | 8 / 12 (67 %) | 49 / 114 (43 %) |
| explicit 6-face `X X X X` sweep, 24 moves | **12 / 12 (100 %)** | 6 / 114 |
| …then 76 more leashed moves | 12 / 12 | 34 / 114 (30 %) |

The live run's 1–2 of 12 is **at** the 2/12 cap. The budget is irrelevant; the rule is the constraint.

**The brief is read, but only for static bans.** Two channels, two opposite outcomes:

- **Static, unconditional prohibition — WORKS.** `blind-axis` refuses `rotate_cube_z_*` on every step.
  The parent named a z rotation **750/1901 times (39.5 %)**; the candidate named one **7/1901 (0.4 %)**
  — a 100× suppression, and the parent's eight pure-z `BLIND_SPIN` episodes are gone.
- **Per-turn, state-dependent ALLOWED list — INERT, and slightly anti-correlated.**
  `P(pick is refused) = 0.195` against a uniform-chance `0.178` → **ratio 1.09**. The model names the
  announced default (`survivors()[0]`, printed first in the ALLOWED line) on **6.0 %** of steps
  against a 5.6 % chance baseline. Per action, for the five names it actually favours, it is
  **4–28× _more_ likely to name them when they are REFUSED than when they are allowed**:

  | action | P(named \| refused) | P(named \| allowed) | ratio |
  |---|---:|---:|---:|
  | `rotate_cube_y_cw` | 1.000 (33/33) | 0.036 | **27.9** |
  | `rotate_down_cw` | 0.662 (102/154) | 0.104 | 6.4 |
  | `rotate_back_acw` | 0.410 (50/122) | 0.065 | 6.3 |
  | `rotate_right_cw` | 0.593 (51/86) | 0.095 | 6.2 |
  | `rotate_left_acw` | 0.392 (49/125) | 0.083 | 4.7 |

  This is not perversity, it is the same phenomenon seen from both sides: `revisit` fires *precisely*
  when the model is mid-run on a name, and the model's 49.3 % stickiness dominates the 14-name list.

**`view-streak` caps the run length but not the total — L2 (`rubik_2_vqwtjvio`).** The model named
`rotate_cube_y_cw` on **99 consecutive steps**. The rule lets two through and refuses the third, so
the harness itself authors a stable period-3 pattern `y, y, [substituted face move]` forever:
**66 of that episode's 100 steps were committed view rotations** (66 of the eval's 67), while the
substituted face moves walked the cube to a 33-letter word. This is the level the parent solved in one
step.

**The step-0 prior is still a constant.** The echo-free first decision across 20 levels used only
**5 distinct names**, three of them covering 18/20: `rotate_cube_z_cw` ×6, `rotate_front_cw` ×6,
`rotate_up_cw` ×6. The parent's was `rotate_front_cw` ×11 / `rotate_cube_z_cw` ×8. The 4.7 kB macro
skill bought a **relabelled constant, not a reading of the frame**.

## 3. Failure breakdown

| category | episodes | steps | note |
|---|---:|---:|---|
| `OUTBOUND_MARCH` — walk exits the ≤4 ball by step ≤12 and never returns | **19/19 capped** | 1733 | the dominant mode; `revisit` forces it |
| └ of which `RATCHET` (model triples the harness's substitution; ov ≈ 33/100) | 9/20 | 900 | L4, 9, 12, 14, 16, 18, 20 (+L2, L3 in view form) |
| └ of which model-driven 2-move alternation (0 overrides) | 4/20 | 400 | L7, L8, L13, L17 — `U R U2 R U2 …`, word grows ~2/3 per step |
| `VIEW_DUTY_CYCLE` — `view-streak` permits a 2-in-3 spin forever | 1/20 | 66 | L2 only; 66 % of that episode state-preserving |
| solved | 1/20 | 1 | L5, step-0 substitution constant B′ — see Headline #2 |
| parse failure / default action / illegal name | **0/1901** | 0 | `successful_actions == total_responses == 1901` in all 20 |
| engine-refused move (`rotation failed`) | **0** | 0 | the `_retry` animation path never fired |
| provable waste by the parent's own metric (view + X⁴ + immediate undo) | — | **67 (3.5 %)** | was **1523 (80.1 %)**: X⁴ 723→**0**, undos 0→**0**, views 800→67 |

Cross-check vs the numeric pre-pass: `steps == vlm_calls == total_frames` and
`used_rich_harness: true` in all 20 `eval_result.json`; `default_actions: 0` in all 20 — consistent.

**Credit where it is due — four things this candidate fixed and the next proposer must keep:**

1. **The veto is correctly placed.** `_arm`/`_disarm` around the delegation is the only way to beat
   `baseline_step`'s own `execute_action` call. 1 engine action per step, 1901/1901. Keep verbatim.
2. **`SPEC` closes the parent's §5.5 latent hazard.** The model emits `{"action":"rotate_cube_y_cw",
   "face":"Y",…}` — `face` is the *axis letter*, not the engine's axis face. `_rewrite` sets `key`,
   `face`, `direction`, `action_type` together from `SPEC`, silently correcting **107** such
   disagreements. Without it the engine and the log diverge. Keep verbatim.
3. **The slot-0 splice is sound.** Re-spliced onto a pristine copy each turn, after the engine's
   AVAILABLE ACTIONS + RESPONSE FORMAT tail: 0 parse failures, 0 defaults, and the token plateau
   proves it reached the payload. Keep the plumbing; change the content.
4. **`blind-axis` is the one measured-effective prose/veto channel** (750 → 7). Keep it, and note
   *why* it worked: it is **static and unconditional**.

## 4. Binding constraint — classify it

**Primary: HARNESS-BUG — a sign error in the search objective, and it is fully fixable here.**
`tools/action.py:RULES` optimises "never re-stand on a configuration". The task is "visit the one
configuration that is ≤4 quarter-turns from the start". Those objectives are anti-correlated, and the
measurement is unambiguous: with the rule in place an *optimal* router reaches 2/12 of the depth-1
shell; without it, 12/12 in 24 moves. The harness is at its own ceiling, so **this is not a tuning
question** — the predicate must be replaced, not weakened. Worth, by the coverage arithmetic, every
depth-1 level outright plus ~30 % of each depth-2 level.

**Secondary: VLM-CEILING on per-turn state — now measured, with the working channel identified.**
The frozen `Instruct` model emits 22–28 tokens, has no reasoning channel, repeats its previous answer
49.3 % of the time, copies the harness's last played move 65.4 % of the time, and shows **no
measurable response to a 14-name per-turn ALLOWED list** (refusal ratio 1.09, default named at
chance). It *does* obey a **static, unconditional ban** (z: 39.5 % → 0.4 %). So the scaffolding
prescription is specific: **state invariants statically in prose, and shrink the per-turn channel from
a 14-name list to a ≤4-item menu.** Re-wording the macro skill cannot reach it — 1150 prompt tokens
of milestone ladder and macro vocabulary produced a relabelled step-0 constant.

**Not the constraint, measured at zero cost:** parsing, action spelling, the engine key/metadata
mapping, animation handling, the step budget, the delegation rate, or the slot-0 channel.

**A free lunch the next candidate must exploit:** `rubik_openai_evaluation.py:646–654`
(`state_callback`) calls `engine.stop_evaluation()` the moment `cube_solved` is true. **The harness
never has to *recognise* solved** — it only has to *visit* it. Coverage of the near-ball is therefore
the complete objective, and no perception budget need be spent on a solved-detector.

## 5. Concrete changes to try (the gradient — ranked by leverage)

**5.1 — Replace `revisit` with a LEASH. (`tools/action.py:RULES`; highest leverage in the file.)**
Add an exact cube state to the ledger and refuse a face turn that takes the cube **more than R
quarter-turns from where the episode started** (start with `R = 2`, widen on a step schedule, e.g.
`R = 2 + step//40`). Backtracking through already-visited states must be **allowed** — that is the
whole difference from `revisit`, and it is what lets the walk fan out across the shell.
*Implementation, anti-cheat clean:* `reduce_word` is sound but incomplete (it misses commuting
relations — `R L R' L'` reduces to length 4 yet is the identity). Replace it with a 20-cubie
(position, orientation) state advanced by the harness's **own emitted moves** — ~30 module-level lines,
pure group theory, no engine source and no scramble data, exactly the same provenance as the current
word. Build the ≤R ball by BFS once per episode (126 states at R=2, 1194 at R=3 — instant).
*Predicted:* steps inside the ≤4 ball **8.8 % → ~100 %**; depth-1 shell coverage **1–2/12 → 12/12**;
primary **0.050 → 0.25–0.40** (every depth-1 level becomes a certainty within 24 steps, each depth-2
level gets ~30 % of its shell).
*Why:* the measured cap. 2/12 with the rule, 12/12 without it, at identical budget.

**5.2 — Make the substitution a ROUTER, not an alphabetical ring.
(`tools/action.py:preference` / `survivors`.)**
Replace `_FACE_RING` rotated by `len(played) % 12` with: BFS inside the ≤R ball to the nearest
**unvisited** in-ball state, return the first move along that path. Delete `_offset` — its only
observable effect today is that step 0 is the constant `rotate_back_acw` in every episode, which is
the entire 0.050 primary (Headline #2). Seed the episode with the explicit 6-face `X X X X` sweep
(24 moves, covers all 12 depth-1 states, returns to the start each cycle) before handing control back.
*Predicted:* this is the lever that the 0.654 adoption rate makes compound — whatever the harness
plays, the model names next 65 % of the time, so a ball-aware substitution steers the *model's* stream
too, not just the harness's.

**5.3 — Narrow the per-turn channel to a ≤4-item menu; do NOT dictate. (`memory.md` + `vlm.py:brief`.)**
A 14-name ALLOWED list is measurably inert (ratio 1.09). Print **at most four** candidates, each with
the one fact that distinguishes it (`rotate_left_cw — 1 turn from the start, not yet tried`), and
honour any menu pick.
*Hazard to respect:* I replayed the recorded pick stream under an R=2 leash and the override rate goes
to **79.3 %** (1507/1901). That is the degenerate "optimised the guard, not the task" path the parent
diagnosis warns about. A menu keeps the frozen VLM deciding *and* keeps every decision in-ball; a bare
leash reduces it to a rubber stamp. Target an override rate **below ~35 %**, and report it.

**5.4 — Move invariants from the per-turn list into STATIC prose. (`skills/macro_commitment.md`.)**
The one channel with a measured effect was an unconditional ban (z: 750 → 7). State the leash the same
way — "never play a move that takes the cube more than two quarter-turns from where it started; the
harness will refuse it and spend your step" — instead of expecting the model to diff a 14-name list.

**5.5 — Fix `view-streak`: cap the TOTAL, not the run. (`tools/action.py:RULES`.)**
`_view_run(b) >= 2` permits a stable `y, y, [override]` duty cycle forever — 66 of L2's 100 steps.
Replace with a per-episode view budget (e.g. refuse every view rotation once 6 have been committed), or
drop view rotations from the vocabulary entirely until a perception consumer exists: **nothing in this
harness reads the frame**, so a state-preserving move is currently a pure loss of a step.

**5.6 — Retire the dead and contradictory brief fields. (`memory.md`.)**
`screen` (the still-frame digest) is structurally near-constant — every face turn and every view turn
changes the picture, and `_retry` never fired in 1901 steps. Worse, the skill tells the agent "*A long
word means the cube has been walked away from the start rather than toward the solution*" — which is
**true**, and the harness then forces exactly that. The prose and the mechanism contradict each other.
Replace `word`/`stood`/`screen` with `quarter-turns from the start: 2 (budget 2)` and `in-budget moves
not yet tried: …`.

**5.7 — Decide the fate of the macro skill on its measurement.** 4.7 kB / ~1150 prompt tokens bought a
relabelled step-0 constant (3 names cover 18/20 levels) and no compliance with any of its contracts.
The milestone ladder and relative-face compiler presume the agent can read the case from the frame; it
sees one face of six. Either delete it and spend the tokens on 5.3's menu, or gate it behind a real
observation channel (parent §5.3) — but do not re-word it.

## 6. Exhausted — do NOT repeat

Carried forward from `staged_anchor/diagnosis.md` (prompt-only rewording; long raw screenshot+JSON
history as working memory; parser/fallback/spelling/engine-mapping/animation work; blind search over
the Cayley ball with `plan_tour`/`BALL_RADIUS` tuning; asking the frozen model for a move from the home
pose; asking for a six-face colour grid from a ≤2-face frame; more model calls per episode; a
piece-based potential), plus **newly refuted in this run**:

- **A no-revisit / distinct-states objective is REFUTED, and it is actively harmful.** It is
  anti-correlated with winning and caps depth-1 shell coverage at **2/12** under an optimal policy.
  Do not re-tune it, do not soften it to "don't undo the last move" — replace it with a leash (5.1).
- **A per-turn, state-dependent ALLOWED/REFUSED list is REFUTED as an instruction channel at this
  width.** 14 of 18 names, refusal ratio **1.09** vs chance, announced default named at **6.0 %** vs
  5.6 % chance. A *static unconditional* ban does work (750 → 7) — that is the channel to use.
- **The macro/milestone prose skill is REFUTED as a step-0 lever.** Still a constant; 3 names cover
  18/20 levels. +1150 prompt tokens for a relabel.
- **`view-streak` as a run-length cap is REFUTED** — it licenses a 2-in-3 spin indefinitely (66 steps).
- **The alphabetical substitution ring rotated by step count is REFUTED** — `_offset` is 0 at step 0 in
  every episode, so the harness's first move is the constant B′, which is the whole 0.050.
- **The still-frame digest (`screen`) is inert** — `_retry` fired 0 times in 1901 steps.
- **`reduce_word` is sound but incomplete** — it misses commuting relations (`R L R' L'` → length 4,
  actually the identity), so it is not a usable state key. Use an exact cubie state instead.
- **Do not re-verify the plumbing.** 0 parse failures, 0 defaults, 0 engine refusals, 1 engine action
  per step, 107 `face`-field corrections, brief confirmed in the payload. It is correct; keep it.

### Chain-integrity warning, carried forward and sharpened

Score in this chain still moves inversely with how much the frozen model is used. The measurement that
matters for 5.1–5.3 is therefore **two numbers reported together: the primary AND the override rate.**
My leash replay shows a bare R=2 leash drives the override rate to 79.3 % — that would likely raise
the primary while reducing the VLM to a rubber stamp, which is optimising the guard, not the task. A
candidate that reports `primary 0.35 @ override 30 %` has done the work; one that reports
`primary 0.40 @ override 80 %` has not. Put `model_pick` / `harness_override` / fired rule codes on the
Action as this candidate already does — that is what made this diagnosis computable rather than
reconstructed.

### Anti-cheat note on this diagnosis

I did **not** read the scramble table, a solver, cube internals or the score — the proposer sandbox
hook blocks the engine/renderer source, and I did not work around it. Instead I built an **independent
20-cubie group model** from first principles (validated against the BFS ball sizes 12/114/1068/10011
and three known move orders) and measured only **scramble-free** quantities: the walk's displacement
from **its own starting configuration** and its coverage of the ball around that start. That is
strictly weaker information than the parent diagnosis used and strictly stronger evidence, because the
solution of *every* level necessarily lies inside that ball.

Every fix above keeps the agent deciding from its observation plus state it could have derived itself:
5.1's leash and 5.2's router read only the **harness's own emitted move word**, exactly like the
shipped `reduce_word`; 5.3 narrows the choice set the model picks from; 5.6 uses the free termination at
`state_callback`, which requires no perception at all. Nothing proposed opens `SpatialWorld/data/`,
reads a scramble or consults a solver.
