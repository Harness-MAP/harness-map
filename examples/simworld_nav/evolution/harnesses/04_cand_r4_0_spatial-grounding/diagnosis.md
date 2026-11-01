# Diagnosis — SimWorld-MMNav (hard) iter 1 / `cand_r1_0_spatial-grounding` ("commit geometry")

**Headline:** SR **0/50** · SubtaskSR **0.285** (anchor 0.36, **−0.075**) ·
per_position_SR **[1.00, 0.02, 0.022, 0.00]** vs anchor **[1.00, 0.24, 0.111, 0.00]** ·
static_collision_mean **5.12** (anchor 1.78) · cost 97 k/ep (anchor 81 k).
**Binding constraint: HARNESS-BUG — the rewritten arrival-judge system prompt
(`completion_checker.py:695`) destroyed the only working commit signal.** Its true-positive rate
inside the ROAD pass window collapsed from **9/43 (21 %) → 1/58 (1.7 %)** while its far-field
false-positive rate did **not** move (1.6 % → 1.5 %). The judge is now a near-constant `0` with a
~2 % noise floor, so the agent walks *through* the scoring window in 40/50 episodes and dies on the
env's per-leg substep deadline.

**The candidate bundled four edits. The evidence separates them cleanly:**

| edit | file | verdict | evidence |
|---|---|---|---|
| **E1** approach latch before the relative-turn commit | `active_perception.py:124,488-533` | **CORRECT — ship it** | On the anchor's own traces **7/7** TURNING failures are position-box shortfalls of **78–980 cm**, all closable by 1–2 forwards. The one time it fired here (`ep_20_9_task_dist_21_0_1`) it converted a 432 cm shortfall into a **TURNING pass**. |
| **E2** `commit_burst` FINISH pose prefix | `completion_checker.py:568-605` | **CORRECT — keep, unexercised** | Verified against all **181/181** four-leg hard task_configs: side=`right` ⇒ required yaw delta +90 (=`Rotate_Right`) in 103/103; side=`left` ⇒ 270 (=`Rotate_Left`) in 78/78. FINISH was reached 0/50 times, so it scored nothing. |
| **E3** xcheck cap `True`→`False` + verdict memo | `completion_checker.py:147,905-935` | **HARMFUL — revert** | `XCHECK_BLOCK` 18 → **117**. 4 of 18 in-window planner commits were refused **for free** (cap spent, no VLM call). Produces 35-long `XCHECK_BLOCK; EMPTY_FALLBACK` lock-in loops. |
| **E4** arrival-judge prompt rewrite | `completion_checker.py:695-728` | **CATASTROPHIC — revert verbatim** | In-window TPR 21 % → 1.7 %. Commit precision 57 % → 14 %. Reason-string entropy collapsed: 97 distinct reasons/348 calls → **33 distinct/420 calls, 81 % verbatim echoes of the prompt's own disqualifier**. |

> **E4 was an explicitly exhausted axis.** The parent diagnosis §6 states verbatim: *"Do not re-word
> or re-shape the arrival question. Parent §6 priced six reformulations plus graded confidence
> offline against the live model on the exact frames; all lost."* This run is the seventh
> reformulation and the first priced on 50 episodes. It is now the most expensive refutation on record.

---

## 1. What this harness does (mechanism)

`orchestrator.py:run_episode`, one outer iteration per **decision** (not per env step). Seven modules:

| # | call site | role in the outcome |
|---|---|---|
| 1 | `active.should_emit` (`orchestrator.py:305`) | **Short-circuits perceive+plan entirely** when it returns a list. Owns (a) the ORIENTATION cardinal burst `[rot…, -1]` — 50/50 pass, 0 VLM calls; (a″) the relative-turn burst, now with **E1**'s 2-forward approach; (a′) red-signal idle; (b) anti-spin; (b′) wall-stuck recovery; (d) cap-2 forward momentum. |
| 2–3 | `perceiver.perceive` → `planner.plan` | 2 VLM calls/decision → `{Summary, Match, Actions}`. |
| 4 | `completion.verify` (`orchestrator.py:406`) | honours a planner `-1` only if `planner_match == 1 and forward_streak >= 1`. |
| 5 | `cross_check_planner_commit` (`orchestrator.py:420`) | 3rd VLM call gating a planner `-1`. **E3** changed its cap semantics. |
| 6 | `proactive_commit_check` (`orchestrator.py:449`) | at `forward_streak ≥ 4`, every 5 forwards, ≤12/leg: independent binary arrival call; `Match==1` ⇒ inject the commit burst. **This is where 8/8 of this run's commits came from.** |
| 7 | `request_side_scan` / `scan_reach` (`orchestrator.py:477`) | physical yaw toward the named side, re-ask, yaw back on miss. Fired 4× (all `SIDE_SCAN_MISS`). |

Stages 5 and 6 both call **the same** `_binary_reach_vlm_call` (`completion_checker.py:646`).
**E4 rewrote that one system prompt, so it hit every commit path in the harness simultaneously.**
That is why a "prompt tweak" moved the primary metric by −0.075.

Env grading (`traffic_world.py:397-455`, read for analysis only):
ORIENTATION/TURNING `(target − yaw) % 360 < 45`; ROAD/FINISH **L1 < 2000**; TURNING additionally
requires the agent inside `correct_robot_position_area`. Per-leg budget
(`:640-649`): ORIENTATION 10, TURNING 50, ROAD/FINISH `int(L1/100)` substeps — charged on **every**
action (`:686`), and `substeps_count >= subtask_max_steps` silently terminates the episode (`:384`).

---

## 2. What actually happened (causal chain, with evidence)

### 2.1 The decisive frame — `ep_20_0_task_dist_11_0_1`, step 9, anchor vs candidate, same position

Both arms take the identical opening: `[Rotate_Right ×2, -1]` (ORIENTATION pass), then 6 forwards to
`x = 3152` (`env_step_log.jsonl` steps 3–8 in both). ROAD waypoint is `[2200, −38300]`, so L1 = **952
< 2000 — a pass is available right there.** At step 9 both arms fire `proactive_commit_check` on the
same frame:

```
ANCHOR    {"Match": 1, "Reason": "The same building with a curved glass facade and adjacent
                                  tall structure is visible in both views."}
          -> REACH_COMMIT -> subtask 1 PASS (L1 952)  -> episode ends at 12 steps, [T,T,F,F]

CANDIDATE {"Match": 0, "Reason": "reference structure is small and centered ahead with road
                                  between"}
          -> no commit
```

The candidate then walks **straight through the window** — `x` = 2652 (L1 452), **2152 (L1 48)**,
1652, 1152, 652, 152 — refusing at every one, out the far side to `x = −3348`, into a facade at step
22, and then emits **20 consecutive blocked forwards** (`moving_distance` frozen at 10196.26,
`building` collision counter 1→20) until the leg deadline killed it at step 42.

`int(L1(spawn, waypoint)/100) = int(3952/100) = 39`; the leg started at step 3; **39 + 3 = 42 = the
observed step count, and `notes` is empty** — the env terminated, the harness never noticed.

### 2.2 The refusal is prompt echo, not perception — measured on 420 calls

The candidate's judge does not answer the image; it recites the prompt's own disqualifier clause
("*still DOWN THE STREET — small, near the vanishing point, centred ahead with road between you and
it*"):

| | anchor prompt | candidate prompt (E4) |
|---|---|---|
| arrival calls | 348 | 420 |
| **distinct `Reason` strings** | **97** | **33** |
| top reason share | 17 % | **52 %** (`"reference building is small and centered ahead with road between"`) |
| top-5 share | 45 % | **81 %** (all five are the same sentence) |

It emits that exact string at **74 cm** from the ROAD waypoint (`ep_20_17` step 11), 342 cm
(`ep_20_9` step 30), 210 cm (`ep_20_15` step 32) and 280 cm (`ep_20_14` step 49).

### 2.3 The judge lost all discriminative power — the ROC, by true L1 distance on the subtask-1 leg

Bucketed by ground-truth L1 to the ROAD waypoint at the frame each call was made on (oracle used for
**analysis only**; no fix below reads it):

| region | anchor `Match=1` | candidate `Match=1` |
|---|---|---|
| **IN window** (L1 < 2000 — a commit scores) | **9 / 43 = 21 %** | **1 / 58 = 1.7 %** |
| NEAR (2000–3500) | 0 / 33 | 1 / 34 |
| FAR (> 3500 — a commit is fatal) | 4 / 258 = **1.6 %** | 5 / 327 = **1.5 %** |
| **TPR − FPR** | **+19.4 pts** | **+0.2 pts** |

E4's stated goal was to suppress far-field false positives. **It suppressed none of them** (1.6 % →
1.5 %, noise) and removed 92 % of the true positives. Downstream, commit *precision* got worse too:

```
subtask-1 commits attempted / correct      anchor 21 / 12 (57%)      candidate 7 / 1 (14%)
commit L1 error, anchor:    317 337 342 578 856 952 980 1006 1127 1192 1442 1480 | 2022 2888 3074 3537 6932 21134 22988 32586 58789
commit L1 error, candidate: 932 | 2442 24488 25586 27516 45289 45527
```

The anchor's commits cluster on the target; the candidate's are noise fired from wherever the agent
happened to be after over-walking — **five of its seven commits are 24 000–45 000 cm off**.

### 2.4 E3 turns "budget spent" into a permanent refusal and locks the agent into a forward-mash

`cross_check_planner_commit` now returns `False` when `fires >= MAX_XCHECK_FIRES_PER_SUBTASK`
(`completion_checker.py:925-935`), and memoises `Match=0` per `forward_streak // 2` bucket
(`:905-920`). Because the candidate never commits, the planner re-raises `-1` for the rest of the
leg — **118 planner `-1` claims on subtask-1 legs (anchor: 29)** — and after the 3rd VLM fire every
later claim is refused *for free*, forever, with no VLM call. The trace signature:

```
notes: XCHECK_BLOCK; EMPTY_FALLBACK; XCHECK_BLOCK; EMPTY_FALLBACK; … (×35 in one episode)
```
`XCHECK_BLOCK` 18 → **117**; `EMPTY_FALLBACK` 21 → **118**. Of the 18 in-window planner commits with
`planner_match == 1`, **4 were free refusals** (`ep_20_21` steps 99 / 102 / 105 — step 102 is
**L1 149 cm** from the waypoint, 15 arrival calls already spent) and the other 14 were killed by E4's
judge. **Net: 18/18 in-window planner commits blocked, 0 allowed.**

### 2.5 Two inherited bugs that E4's over-walking converted from latent to dominant

**(a) `SpatialMemory.log_step` never reads the collision channel — rule (b′) is dead code.**
`spatial_memory.py:66` sets `coll = 0` and passes it straight into
`_StepRecord.static_collisions_cum`; the documented `info["agent_collision"]` read was **never
written**. `read_audit.json` confirms: `info.agent_collision` appears in **0** of the 25 089 reads.
So `static_collision_delta()` is identically `0`, and both consumers are unreachable:
`active_perception.py:620-627` (wall-stuck recovery, requires `wall_coll_delta > 0`) and the iter_008
momentum veto. That is why `ep_20_0` mashed a facade 20× with no recovery.
**Note: `agent_collision` is a forbidden read (parent §6) — the conjunct cannot be fixed, it must be
deleted** and the rule re-gated on odometry alone.

Cost this run: **281 blocked forwards (9 % of all 3118 forwards; anchor 4 %) in 22 runs, 20 of them
≥2 steps long, 9 of them 19 steps long — 259 substeps wasted**, and `static_collision_mean` 1.78 →
**5.12** with `static_collision_rate` 0.14 → 0.30.

**(b) The STUCK detector cannot fire.** `orchestrator.py:280` computes
`forward_count = sum(1 for a in chosen_actions if a == 0)` where `chosen_actions` is the **previous
decision's burst**, not the leg's history — on the `EMPTY_FALLBACK` path it is `[0]`, so
`forward_count == 1` and `forward_count > 3` is never true. `SpatialMemory` already exposes the
correct accessor (`forward_count_since_boundary()`, `spatial_memory.py:101`) and it is unused.

---

## 3. Failure breakdown (50 episodes, subtask-1 leg)

| class | rate | root cause |
|---|---|---|
| **Reach-the-window, refuse to commit** | **40 / 50** (anchor 28/50) — 41/50 physically entered the L1<2000 window, **1** committed there | E4 (§2.2–2.3) primary; E3 (§2.4) on the 4 free-refusal cases |
| of which **walked back out** of the window | **38 / 40** (anchor 26/28) | no over-walk detector; §2.5(b) |
| **Died on the env per-leg substep deadline** | 27/49 failed legs have empty `notes` (≥13 land within ±1 substep of `int(L1/100)`) | harness has no model of the deadline |
| **Facade mash** (≥5 consecutive blocked forwards) | **17 / 50** (anchor 8/50) | §2.5(a) dead rule (b′) + §2.5(b) dead STUCK |
| **Fatal far-field commit** | 6 / 50 | E4 noise floor (1.5 % FPR, unchanged from anchor) |
| ORIENTATION | 50/50 pass, 0 VLM calls | cardinal burst — **do not touch** |
| TURNING | reached 1×, **passed 1/1** (E1 fired, §4) | upstream ROAD death |
| FINISH | reached 0× | upstream |
| parse-fail / crash / socket / illegal action | 0 | — |
| `read_audit` | **PASS** — 0 forbidden, 0 undeclared | — |

---

## 4. Binding constraint — classified

**Primary: HARNESS-BUG (E4), fully recoverable by reverting one string literal.**
`_binary_reach_vlm_call`'s system prompt is a *single shared* gate on every commit path in the
harness. The rewrite replaced a criterion the frozen model can answer ("is this the same place",
21 % TPR / 1.6 % FPR) with one it cannot ("am I standing at it"), and wrote the negative case so
vividly that the model reproduces it verbatim in 81 % of replies (§2.2). The frozen VLM has **no
metric-depth grounding** — it cannot distinguish "this building is 900 cm away" from "this building
is 9000 cm away" from RGB — so any criterion phrased as a *distance* judgement degenerates to a
constant. The anchor's criterion works precisely because it is an *identity* judgement, which the
model can do.

The env's rubric makes this decisive: **the ROAD criterion is L1 < 2000 cm, i.e. up to four forward
steps of slack, and 181/181 tasks place the waypoint 500 cm inside the TURNING box's near edge.**
"Arrival" in the *rubric's* sense happens while the landmark is still visibly down the street. E4
encoded a criterion **stricter than the thing being graded**.

**Secondary: HARNESS-BUG (E3).** A cost cap is not evidence. Converting cap-exhaustion into a
standing veto (`:925`) means the gate gets *tighter* the longer a leg runs — exactly backwards, since
a long leg is one where the agent has walked *past* the waypoint. Combined with E4 it produced
permanent commit lock-out.

**Tertiary: TASK-GAP.** The harness still has no notion of "I have walked far enough that I am past
the waypoint" and no model of the per-leg substep deadline. With a working recognizer this costs
little; with a broken one it costs the episode. 38/40 window entries were walked straight out of.

**Not the constraint:** locomotion (41/50 entered the window), ORIENTATION (50/50), parsing (0
failures), the module graph, or cost.

---

## 5. Concrete changes to try (the gradient — ranked by leverage)

### A — Revert the arrival-judge system prompt to the anchor's, byte for byte. `modules/completion_checker.py:695-728` (and the user-turn line at `:735-738`)

```python
sys_prompt = (
    "You compare two egocentric outdoor images and judge whether "
    "the navigator has REACHED the landmark depicted in the "
    "reference image. Answer with a single integer Match field: "
    "1 if the current view shows the same landmark / place as "
    "the reference (the same building, intersection, or scene "
    "is clearly visible and prominent in the current view); "
    "0 otherwise. Be conservative — output 1 only when the "
    "match is unambiguous. Respond strictly with JSON: "
    '{"Match": 0 or 1, "Reason": "<one short clause>"}.'
)
```
and restore the user turn's final text to `'Output strict JSON: {"Match": 0 or 1, "Reason": "…"}'`.

- **Why:** §2.3. This is the *only* measured operating point with signal: 21 % TPR / 1.6 % FPR
  in-window vs far-field, 57 % commit precision, 12/50 subtask-1 passes.
- **Predicted:** per_position_SR[1] **0.02 → ~0.24**, SubtaskSR **0.285 → ~0.36**, and
  `static_collision_mean` back to ~1.8 (the collisions are a *consequence* of over-walking).
- **This is a revert, not a new idea. Do not attempt a compromise wording** — see §6.

### B — Keep E1 (the approach latch) exactly as shipped, and raise `_TURN_APPROACH_FORWARDS` only if C lands. `modules/active_perception.py:124`

Measured on **the anchor's own traces**, all 7 TURNING failures among its 12 ROAD passes:

| episode | ROAD-commit coord | TURNING box | shortfall | forwards needed |
|---|---|---|---|---|
| ep_20_2_21  | 17222 | [17300, 19300] | **78** | 1 |
| ep_20_13_11 | −16944 | [−22700, −17300] | **356** | 1 |
| ep_20_0_11  | 3152 | [−2700, 2700] | **452** | 1 |
| ep_20_8_21  | 16794 | [17300, 22700] | **506** | 2 |
| ep_20_11_21 | 3327 | [−2700, 2700] | **627** | 2 |
| ep_20_1_11  | 3392 | [−2700, 2700] | **692** | 2 |
| ep_20_23_21 | −56320 | [−59300, −57300] | **980** | 2 |

All 5 anchor TURNING *passes* had shortfall 0. **`_TURN_APPROACH_FORWARDS = 2` (1000 cm) covers 7/7.**

- **Structural reason, verified 181/181 task_configs:** the ROAD waypoint sits **exactly 500 cm**
  inside the TURNING box's near edge, while the ROAD criterion tolerates **2000 cm** of undershoot.
  Any undershoot > 500 cm scores ROAD and forfeits TURNING. The recognizer commits early by
  construction, so this is the *modal* TURNING failure, not a tail case.
- **Live confirmation, 1/1:** `ep_20_9_task_dist_21_0_1` — ROAD committed at `x = −16868`, box
  `[−19300, −17300]`, i.e. **432 cm outside**. The latch emitted `[0]`, `[0]` → `x = −17868`
  (inside), then `[Rotate_Left, -1]` → **TURNING PASS**. The anchor fails this exact geometry.
- **Predicted (once A lands):** per_position_SR[2] **0.11 → ~0.27**, SubtaskSR **+0.035**.
- **Bounded risk:** overshoot tolerance past the waypoint is 4900 cm in 136/181 tasks but only
  **1500 cm in 45/181**. Worst case (max 2000 cm ROAD overshoot + 1000 cm approach = 3000) exits the
  box on those 45. Not observed — commits are undershoots in 12/12 anchor cases — but **do not raise
  `_TURN_APPROACH_FORWARDS` above 2** without re-measuring.

### C — Revert E3's cap semantics; keep the memo. `modules/completion_checker.py:925-935`

```python
if fires >= MAX_XCHECK_FIRES_PER_SUBTASK:
    return True          # cost cap == "cannot verify" == defer to verify(), as in the anchor
```
Keep `MAX_XCHECK_FIRES_PER_SUBTASK = 3` and the `_xcheck_verdict_cache` (`:905-920`) — the memo is
genuinely free and correct. **Only the `False` is wrong.**

- **Why:** §2.4. A spent budget is a statement about *our* cost policy, not about the world. The
  `False` makes the gate strictly tighter the longer a leg runs, which is precisely inverted — a long
  leg is one where the agent is *past* the waypoint. It killed 4 in-window commits outright, one at
  **149 cm**, and generated the 35-deep `XCHECK_BLOCK; EMPTY_FALLBACK` lock-in loops.
- **Predicted:** +2–4 subtask-1 passes on top of A, and removes the lock-in note pattern entirely.

### D — Make the wall-stuck recovery reachable: delete the collision conjunct, gate on odometry. `modules/active_perception.py:610-627` + `modules/spatial_memory.py:66,127`

```python
# rule (b'): the forward stalled. `info["agent_collision"]` is a FORBIDDEN read
# (validate_harness), so the collision conjunct can never be satisfied — drop it and
# use the sanctioned odometry channel, requiring 2 consecutive stalls to avoid
# reacting to a single transient block (e.g. a pedestrian crossing in front).
if len(wall_recent) >= 2 and all(a == _FORWARD for a in wall_recent):
    if float(md_delta_fn(window=2)) < _WALL_STUCK_MD_THRESHOLD_CM:
        self._consecutive_momentum = 0
        return [_WALL_RECOVERY_ROTATION]
```
Then delete `static_collisions_cum` / `static_collision_delta` and the iter_008 momentum veto that
reads it, or mark them dead — they are unreachable by construction, and leaving live-looking dead
code is what let this survive three iters.

- **Why:** §2.5(a). The rule has been dead since iter_008; `read_audit.json` proves the channel is
  never read (0/25 089), and the channel is *forbidden*, so it can never be revived.
- **Measured operating point:** 22 blocked-forward runs this arm, **20 of length ≥ 2**, only 2
  singletons. A 2-stall trigger catches 20/22 (91 %) and never fires on a transient. It reclaims
  **259 wasted substeps** and should drop `static_collision_mean` 5.12 → ~2.
- **Compose with:** fixing `orchestrator.py:280` to
  `forward_count = memory.forward_count_since_boundary()` (the accessor already exists,
  `spatial_memory.py:101`, and is unused) so STUCK becomes a real backstop rather than dead code.

### E — Keep E2 (`commit_burst`) unchanged; fix its docstring rationale. `modules/completion_checker.py:296-317,568-605`

**The code is right — verified exhaustively on 181/181 four-leg hard task_configs:**
`(FINISH.correct_robot_orientation − TURNING.correct_robot_orientation) % 360` is **+90 for all 103
`on your right` tasks** and **270 (= −90) for all 78 `on your left` tasks**, matching
`side=="right" → Rotate_Right(+90)` / `side=="left" → Rotate_Left(−90)`. 181/181 FINISH legs classify
as FINISH; 181/181 ROAD legs do not.

**But conjunct 2's stated reason is false and is a latent trap:** ROAD templates *do* say "on your
left/right" — 181/181 of them (`"Continue straight and stop at the intersection with the … building
…, on your left."`). The only thing preventing a quarter-turn prefix on ROAD commits is
`_INTERSECTION_RE` and the absence of a `face` clause. Rewrite the docstring to say so, and keep
`_INTERSECTION_RE` as load-bearing, not belt-and-braces. A FINISH-prefix leak onto a ROAD commit
would corrupt the heading the E1 turn burst rotates from.

### F — (after A–D) Over-walk backstop for the TASK-GAP. `modules/completion_checker.py` + `orchestrator.py`

The recognizer's residual miss rate still walks 38/40 window entries straight out. An honest,
observation-only backstop: track `forward_streak` since the leg began and, once it exceeds the
largest streak at which any arrival call in this leg returned a *near*-positive, stop emitting
forwards and switch to the side-scan path rather than continuing down the street. Do **not** read
`env.subtask_max_steps` (parent §6: it is laundered `target_location`). Lowest confidence of the
set — land A–D first and re-measure, since A alone changes the miss distribution.

---

## 6. Exhausted — do NOT repeat

- **Do NOT re-word, re-shape, or "compromise" the arrival question.** Parent §6 priced six
  reformulations offline; **this run is the seventh, priced on 50 episodes, and it cost −0.075
  SubtaskSR and −92 % of in-window true positives.** The failure mode is now characterised: the
  frozen model has no metric-depth grounding, so any *distance/arrival* phrasing degenerates to a
  constant `0` and the model parrots the prompt's negative clause (81 % verbatim echo, §2.2).
  Only *identity* phrasing ("is this the same place") has signal. Treat `sys_prompt` in
  `_binary_reach_vlm_call` as **frozen**.
- **Do not add an illumination-invariance clause.** Already refuted by the parent; note that E4
  *included* one ("Do NOT answer 0 merely because lighting, colour, weather…") and it changed
  nothing — the refusals simply migrated from "lighting" to "down the street".
- **Do not treat a cost cap as evidence.** §2.4 / §5-C.
- **Do not touch the ORIENTATION cardinal burst.** 50/50, 0 VLM calls, three iters running.
- **Do not tune `K_REACH_FORWARD_STREAK` / `K_REACH_INTERVAL` / `MAX_REACH_FIRES_PER_SUBTASK`.** Run
  to death iters 006–010. Coverage is not the binding term: 58 in-window arrival calls fired this
  arm and 57 said no. The *answer* is the term.
- **Do not read** `agent_collision`, `agent_location`, `target_location`, `destination`,
  `correct_robot_location`, `correct_robot_position_area`, `ideal_path`, `gt.json`, or
  `env.subtask_max_steps`. Note specifically that the dead rule (b′) in §5-D cannot be "fixed" by
  wiring `info["agent_collision"]` — that read is forbidden.
- **Avoid numeric literals ≥ 2000** in candidate code (lint flags them as map-tuned). Express
  distances as forward-action counts, as E1 correctly does.
- **Do not bundle a revert-risk edit with an unproven one again.** E1 is worth ~+0.035 SubtaskSR and
  it scored **nothing** this iter because E4 killed the leg upstream of it. Ship A+C (reverts) with
  B+E (already correct) and keep D separable.

---

## 7. Provenance

- Candidate: `$OUT_DIR` = `…/prop_r001_df84f4/cand_r001_0_spatial-grounding` — `orchestrator.py` +
  7 modules; `git diff` vs anchor = 3 files (`orchestrator.py`, `modules/active_perception.py`,
  `modules/completion_checker.py`).
- Eval: `cand_r1_0_spatial-grounding_r001/score.json`, subset `search_hard_50`, 50/50 episodes
  complete, `read_audit.json` verdict **PASS** (0 forbidden / 0 undeclared reads, 25 089 reads).
- Traces: `$TRACE_DIR` (50 episodes). Baseline: `…/arm_hard5010_adaptive_k4c1/anchor_nav/traces`
  (50 episodes, same subset, same 50 task ids — compared pairwise; 12 episodes regressed, 0 improved).
- Per-frame measurements from `vlm_trajectory.jsonl` (420 candidate / 348 anchor arrival calls),
  `actions.jsonl`, `env_step_log.jsonl`.
- Template audit over **199** `single_agent_world/hard/map_road_20_*/task_dist_*/task_config.json`
  (181 with the four-leg ORIENTATION/ROAD/TURNING/FINISH shape).
- Env semantics quoted from `SimWorldGym/simworld_gym/envs/traffic_world.py:384, 397-455, 640-649, 686`.
- **Anti-cheat:** ground-truth positions (`gt.json`, `env_step_log` coordinates,
  `correct_robot_position_area`) were used **only** to measure this diagnosis's rates. Every fix in
  §5 decides from `obs['rgb']`, `obs['object_mask']`, `info['moving_distance']`,
  `info['agent']['agent_rotation']` and the static `task_spec` instruction text — the same channels
  the candidate already declares.
