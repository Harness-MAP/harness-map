# Diagnosis — SimWorld-MMNav (`hard`, search_hard_50) / candidate `cand_r001_0_spatial-grounding` ("commit geometry"), iter 1, parent `anchor_nav`

**Headline:** SR **0/50** · SubtaskSR **0.285** (parent 0.36) · per-leg pass **[1.00, 0.02, 0.022, 0.00]**
(parent [1.00, 0.24, 0.111, 0.00]) · static_collision 1.78 → **5.12** · cost 80.7k → **97.2k** tok/ep.
This is a **regression, not noise**: leg-1 passes fell 12/50 → **1/50**, Fisher two-sided **p = 0.0018**.

**Binding constraint (one sentence):** change (4) — the replacement of the REACH prompt's
"clearly visible and prominent" criterion with a frame-geometry "are you STANDING AT it" criterion —
**did not change what the recogniser discriminates, only which canned clause it emits**: in-window
recall collapsed 9/44 (0.205) → **1/58 (0.017)** while far-field false positives stayed *exactly*
6/138 → 6/139 (0.043), inverting the likelihood ratio from 4.8 to **0.40** (the call is now
*more* likely to say "arrived" 15 000 cm away than at the waypoint); and change (3) — the
cross-check cap returning `False` — converted that recogniser's refusals into an **unbounded, free,
self-displacing veto** (XCHECK_BLOCK 18 → **117**, each one injecting a `Move_Forward`).

> **Confidence.** Every number below is a per-call or per-frame measurement over all 50 episodes of
> *both* runs (candidate: 420 REACH calls / 1 123 plan calls / 3 118 forwards; parent: 348 / 918 /
> 2 812), A/B'd against the parent's own traces at `anchor_nav/traces/`. `gt.json` positions were read
> **for this analysis only**; no recommendation routes them into a policy path.
> Evaluator taint audit: **PASS, forbidden=0, undeclared=0** on all 50 episodes.

---

## 0. What this candidate changed (4 edits, same 7-module graph)

| # | edit | file | verdict |
|---|---|---|---|
| 1 | TURN burst spends 2 approach forwards before `[rotate, -1]`, odometry stall-guarded | `active_perception.py:504-532` | **correct, but starved** — 1 opportunity in 50 eps; it worked on that one |
| 2 | `commit_burst()` composes the terminal pose; FINISH gets `[Rotate_side, -1]` | `completion_checker.py:568-604` + `harness.py:459,515-523` | predicate **100 % correct (199/199 FINISH, 0/561 false)** — **but double-applied at runtime → 180°, not 90°, 6/6** |
| 3 | xcheck cap branch `True` → `False`; verdict memo per `(subtask, streak//2)`; cap 2→3 | `completion_checker.py:920-935, 954` | **harmful** — see §2.3 |
| 4 | REACH system prompt: visibility criterion → frame-geometry position criterion | `completion_checker.py:695-725` | **dominant cause of the regression** — see §2.2 |

---

## 1. What this harness does (mechanism)

`harness.py:run_episode`, one iteration per *decision* (not per env step). Unchanged from the parent:

| # | call site | authority over `-1` |
|---|---|---|
| 1 | `active.should_emit` (`harness.py:~305` → `active_perception.py:420`) | short-circuits perceive+plan. Owns the ORIENTATION cardinal burst, **(a″) the relative-TURN burst (now latched, edit 1)**, red idle, anti-spin, and the cap-2 momentum cadence → 3 forwards per VLM pair |
| 2 | `perceiver.perceive` | 1 call, **4 images** (RGB, robot_view hint, object_mask, JET depth) |
| 3 | `planner.plan` | 1 **text-only** call over that description → `{Summary, Match, Actions}` |
| 4 | `completion.verify` (`:398`) | honours a planner `-1` iff `planner_match == 1 ∧ forward_streak >= 1` |
| 5 | `cross_check_planner_commit` (`:420`) | **2-image** binary REACH call; **cap 3/leg, memoised, and on exhaustion now VETOES** |
| 6 | `proactive_commit_check` (`:442`) | same 2-image call, fires at streak ≥ 4 every 5 forwards, ≤12/leg |
| 7 | `request_side_scan` / `scan_reach` (`:477`) | depth-gated yaw + re-ask (fired 4×, committed 0) |
| — | `commit_burst` (`:459`, `:493`, `:518`) | **new**: composes the pose every `-1` goes out with |

Stages 5 and 6 are the *same function*, `_binary_reach_vlm_call` (`completion_checker.py:646`) — 2 images,
no mask, no depth, no description. **It holds the veto over the richest channel in the stack.** Edit (4)
changed only its system prompt; edit (3) changed only what happens when its budget is spent.

---

## 2. What actually happened (causal chain, with evidence)

### 2.1 The walker is unchanged. The stopper was destroyed.

| | parent | candidate |
|---|---|---|
| physically entered the leg-1 pass window (min L1 < 2 000) | 40/50 | **41/50** |
| …and committed there (leg 1 = pass) | **12** | **1** |
| …and walked straight through it | 28 | **40** |
| never got near the window | 10 | 9 |
| episodes ended by a *fatal* out-of-window commit | 20 | 7 |
| total commits | 87 | 59 |
| verdict notes | `REACH_COMMIT 17 · XCHECK_BLOCK 18 · EMPTY_FALLBACK 21 · STUCK 3` | `REACH_COMMIT 8 · **XCHECK_BLOCK 117** · **EMPTY_FALLBACK 118** · STUCK 6 · SIDE_SCAN_MISS 4` |

Motion is byte-identical in character (ORIENTATION 50/50 both runs). **Nothing about walking changed;
the harness simply stopped being able to stop.** The parent's 12 leg-1 passes decompose exactly as
**9 proactive REACH true positives + 3 planner claims that committed on cap exhaustion**; the
candidate's 1 decomposes as **1 REACH TP + 0 planner**. Edit (4) accounts for the −8, edit (3) for the −3.

### 2.2 Edit (4): the new criterion moved the operating point and *destroyed* the discrimination

348 (parent) / 420 (candidate) `_binary_reach_vlm_call` firings, bucketed by the **true** L1 gap to the
then-current leg's criterion:

| L1 gap | parent fires | `Match=1` | rate | cand fires | `Match=1` | rate |
|---|---|---|---|---|---|---|
| **< 2 000 (would PASS)** | 44 | **9** | **0.205** | 58 | **1** | **0.017** |
| 2 000 – 5 000 | 55 | 0 | 0.000 | 74 | 1 | 0.014 |
| 5 000 – 15 000 | 111 | 2 | 0.018 | 149 | 0 | 0.000 |
| **> 15 000 (fatal if believed)** | 138 | **6** | **0.043** | 139 | **6** | **0.043** |
| precision of a `Match=1` | | **9/17 = 0.53** | | | **1/8 = 0.125** | |
| LR = P(yes\|in-window)/P(yes\|far) | | **4.8** | | | **0.40** | |

In-window recall 9/44 → 1/58, **p = 0.0021**. The far-field tail did not move by a single call.

**The model is quoting the prompt, not reading the frame.** All 57 in-window refusals are the *new
disqualifier clause*, verbatim:

```
 31x  "reference building is small and centered ahead with road between"
 10x  "the reference building is small and centered ahead with road between"
  8x  "reference structure is small and centered ahead with road between"
  5x  "...down the street and small in the distance"
```
against the prompt's `"…still DOWN THE STREET — small, near the vanishing point, centred ahead with
road between you and it"` (`completion_checker.py:~713`). And all 6 far-field false positives are the
*new accept clause*, verbatim:

```
 ep_20_10_21 s37 gap 45 289  "the building facade fills the frame and is directly in front of the robot"
 ep_20_18_11 s13 gap 27 516  "the building facade fills the frame and is directly in front of the robot"
 ep_20_21_11 s17 gap 24 488  "the building facade fills the near field and runs off the edge of the frame"
 ep_20_4_21  s32 gap 45 527  "the building facade fills the frame and the ground is immediately in front"
 ep_20_5_11  s52 gap 25 586  "the building facade fills the frame and the entrance awning is directly in front"
 ep_20_9_21  s31 gap 43 552  "the building facade fills the frame and the trash bins are at the base"
```
against `"its facade fills the near field and/or runs off the edge of the view"` (`:~706`). This is the
**identical pathology the parent diagnosis identified** ("all 8 far-field FPs quote the *clearly visible*
clause back verbatim") — the candidate swapped which clause gets parroted and nothing else.

**Why the new feature cannot work here — measured.** The criterion asks "does the facade subtend a
large part of the frame". Building-class fraction of `obs['object_mask']` (the `(128,0,128)` class), at
the frames of each decision class:

| frame class | n | mean | median | p10 | p90 |
|---|---|---|---|---|---|
| in-window, REFUSED | 57 | 0.276 | 0.276 | 0.185 | 0.367 |
| in-window, ACCEPTED | 1 | 0.153 | — | — | — |
| far-field, FALSE POSITIVE | 6 | **0.309** | 0.312 | 0.275 | 0.339 |
| far-field, refused | 133 | 0.293 | 0.304 | 0.190 | 0.379 |

**Flat — and slightly *higher* at the far-field FPs than in-window.** In a dense procedural city every
forward frame is ~28 % building, so "a facade fills the frame" carries **zero** arrival information, and
it drops the *identity* constraint the old wording at least gestured at. The criterion asks the model
for a feature that does not vary. Faced with an uninformative question, the model falls back to
restating whichever branch of the prompt it lands on.

### 2.3 Edit (3): the cap flip turned refusals into a free, unbounded, self-displacing veto

`completion_checker.py:925-935` now returns `False` when `fires >= MAX_XCHECK_FIRES_PER_SUBTASK` (3),
and `:920-922` returns a memoised `False` for a repeat claim in the same `streak//2` bucket. **Both
paths cost nothing**, so a leg can now veto an unlimited number of claims with no grounding at all.
Downstream, `harness.py` line 518 strips the `-1`, `chosen_actions` goes empty, and `:548-550`
`EMPTY_FALLBACK` injects a **`Move_Forward`** — the veto physically carries the agent out of the window
it is standing in, 500 cm per action, 1 500 cm per cycle.

Measured: **118 planner `-1` claims → 117 XCHECK_BLOCK → 118 EMPTY_FALLBACK forwards → 0 commits.**
**62 of the 118 (53 %) were blocked with no VLM call at all.** The parent, with `return True`, had
18 blocks and committed the third claim.

`ep_20_9_task_dist_11_0_1` is the pure case — 35 consecutive `XCHECK_BLOCK; EMPTY_FALLBACK` pairs
(leg-1 target `(38300, −22200)`, threshold 2 000):

```
 s27  gap  658   plan(M=0)      REACH {"Match":0,"Reason":"reference building is small and centered ahead..."}
 s28  gap  158   <-- INSIDE the pass window, 158 cm from the criterion
 s29  gap  342   <-- still inside
 s30  gap  842   plan(M=1, Actions=[-1])   REACH {"Match":0, same clause}   -> XCHECK_BLOCK -> forward
 s33  gap 2342   plan(M=1, Actions=[-1])   REACH {"Match":0, same clause}   -> XCHECK_BLOCK -> forward
 s36  gap 3842   plan(M=1, Actions=[-1])   REACH {"Match":0, same clause}   -> XCHECK_BLOCK -> forward
 s39  gap 5342   plan(M=1, Actions=[-1])   cap spent, NO VLM CALL           -> XCHECK_BLOCK -> forward
 ...  (32 more identical cycles)
 s134 episode ends ~50 000 cm past the waypoint
```
The parent, on this same episode, committed at step 30 on a REACH `Match=1` at gap **342**
(`"same intersection with prominent building and canopy"`) and **passed leg 1**.

**Population-level:** in **9 distinct episodes** the planner emitted `-1` while **inside** the leg-1
pass window — 16 claims, gaps **58 / 220 / 220 / 296 / 480 / 574 / 649 / 710 / 790 / 842 / 851 / 1 020 /
1 204 / 1 558 / 1 720 / 1 796 cm**. **All 16 were vetoed; 0 committed** (12 by a VLM refusal, 4 for free
on cap/memo exhaustion). In the parent, 10 such claims produced 3 commits.

**Collateral:** the 118 injected fallback forwards are unsteered and the STUCK detector is still a
no-op (`harness.py:273-285`, carried-forward defect). Stalled forwards (Δposition < 1 cm) rose
**118/2 812 (4.2 %) → 281/3 118 (9.0 %)**; `static_collision_mean` 1.78 → **5.12**; cost/ep 80.7k →
**97.2k**. The collision and cost regressions are entirely secondary to this loop.

### 2.4 Edit (2): the FINISH pose is **double-applied** — 180°, not ±90° — 6/6. **Live bug.**

The predicate is perfect: run over all 760 instruction strings in
`envs/setting/single_agent_world/hard/*/*/task_config.json`, `_is_finish_instruction` is
**True on 199/199 FINISH legs and False on 561/561 ORIENTATION/ROAD/TURNING legs.** The direction map
(`left → Rotate_Left`, `right → Rotate_Right`) matches the parent diagnosis's measured 50/50 mapping.

But the REACH-commit path composes the burst **twice**:

```python
# harness.py:459  (proactive REACH path)
planner_actions = completion.commit_burst(instruction)     # -> [Rotate_Left, -1]
...
# harness.py:515-523  (the trim loop, run on EVERY commit)
trimmed = []
for a in planner_actions:              # a = Rotate_Left, then a = -1
    if a == -1:
        trimmed.extend(completion.commit_burst(instruction,
                       already_facing_side=facing_side_already))   # -> [Rotate_Left, -1]  AGAIN
        break
    if a in allowed:
        trimmed.append(a)              # <-- the FIRST rotation survives here
chosen_actions = trimmed               # -> [Rotate_Left, Rotate_Left, -1]   = 180 deg
```

Observed on **6/6** FINISH commits in the run (action context, last 4 steps before the `-1`):

```
 ep_20_10_21 s39  [0, 1, 1, -1]      ep_20_4_21  s34  [0, 2, 2, -1]
 ep_20_18_11 s15  [0, 1, 1, -1]      ep_20_5_11  s54  [0, 1, 1, -1]
 ep_20_21_11 s19  [0, 1, 1, -1]      ep_20_9_21  s33  [0, 1, 1, -1]
```
(parent, for contrast: all 7 FINISH commits are `[…, 0, -1]` — no rotation at all.)

FINISH requires `|correct_orientation − yaw| < 45°` and the required delta is ±90° from the travel
heading. The parent emits 0° → 90° error. This candidate emits 180° → **also 90° error, mirrored**,
i.e. **facing directly away from the building**. The fix is exactly neutralised by the double-apply.
It scored nothing here only because all 6 FINISH commits were far-field FPs (gaps 24 488–45 527 cm)
and would have failed on distance regardless. The planner path (`planner_actions = [-1]` → trim →
`[1, -1]`) is correct; the side-scan path (`already_facing_side=True`) is correct. **Only line 459 is
wrong**, and the NOTES' fake-env dry run (`…0,0,0,1,-1` — a single rotation) exercised the planner
path, not the REACH path, which is why it passed.

### 2.5 Edit (1): correct, and validated on its one opportunity

Only **1** TURNING commit occurred in 50 episodes (parent: 12), because leg 1 almost never passed. That
one — `ep_20_9_task_dist_21_0_1`, the run's only [True, True, True, False] episode — is a clean
confirmation:

```
 s20  -1   at (-16868, -21700)   ROAD target (-17800,-21700), L1 = 932  -> PASS
 s21  0 ---+ the two approach forwards (_TURN_APPROACH_FORWARDS = 2)
 s22  0 ---+
 s23  1    Rotate_Left
 s24  -1   at (-17868, -21700)   TURNING box x in [-19300,-17300]  -> PASS
```
Without the latch the turn would have committed at x = **−16 868**, i.e. **432 cm outside** the box's
`x ≤ −17 300` edge — exactly the parent's 78–980 cm outward-miss failure mode. **Keep this edit.**
It is not responsible for any part of the regression, and it is the only one of the four with a
positive data point.

### 2.6 The planner channel is, again, the better arrival detector — and it was silenced

Taking the **first** planner `-1` per (episode, leg) — the unbiased claim distribution, visible here
precisely *because* nothing ever committed:

```
 CAND (n=12, all ROAD):  220  220  480  574  790  842  1558 | 4796  7613  19796  21712  27851
                         \__________ 7 IN-WINDOW __________/  \______ 5 out ______/     in-window = 0.58
 PARENT (n=11):          388  478  574 | 2785 2817 3078 5006 7537 18139 18634 32516     in-window = 0.27
```
**The first planner `-1` of a leg is a 0.58-precision arrival detector.** The REACH call that vetoes it
has 0.017 in-window recall and a likelihood ratio of 0.40. Two of the five out-of-window first claims
arrive at `forward_streak` = **0** and **3** (gaps 21 712 and 7 613); all seven in-window claims arrive at
streak ≥ **9**. A floor already present in the code (`K_REACH_FORWARD_STREAK = 4`, currently applied only
to the *proactive* path) separates them for free → **7/10**.

---

## 3. Failure breakdown (50 episodes)

| category | eps | Δ vs parent | evidence |
|---|---|---|---|
| **Walked through the leg-1 window, never committed** | **40** | 28 → 40 | §2.1; median overshoot 18 357 cm |
| — of which an *in-window* planner `-1` was vetoed | **9** | 3 → 9 | §2.3, 16 claims at 58–1 796 cm, 0 committed |
| — of which the REACH call fired in-window and refused | **~24** | — | §2.2, 57/58 in-window refusals |
| Fatal false commit, out of window | 7 | 20 → 7 | all 6 FINISH FPs at 24 488–45 527 cm + 1 ROAD |
| Wedged / never approached the window | 9 | 10 → 9 | STUCK 6, stalled-forward rate 4.2 % → 9.0 % |
| Turn committed short of the box | 0 | 7 → 0 | only 1 TURNING commit occurred at all |
| **Reached FINISH with a correct facing** | **0** | 0 → 0 | §2.4 — pose emitted, but 180° |

Failure is now **monolithic**: the harness cannot terminate a leg. 59 commits over 50 episodes, 50 of
which are the free ORIENTATION cardinal burst.

---

## 4. Binding constraint — classified

**Primary: HARNESS-BUG (self-inflicted, three of them), all deterministic and all in this iter's diff.**

1. `completion_checker.py:695-725` — the replaced criterion is **anti-correlated with arrival**
   (LR 0.40). The prompt's clauses are being *recited*, not applied; the geometric feature it asks for
   (facade frame share) is measured **flat at 0.28 everywhere** (§2.2). Cost: **−8 leg-1 passes**.
2. `completion_checker.py:925-935` + `:920-922` + `harness.py:548-550` — a free, unbounded veto whose
   only side effect is to walk the agent away from the window it is standing in (§2.3).
   Cost: **−3 leg-1 passes**, +97 blocks, +97 unsteered forwards, ×2.9 collisions, +20 % cost.
3. `harness.py:459` — `commit_burst` applied twice on the REACH path → 180° terminal pose (§2.4).
   Cost: 0 points *here* (nothing reached FINISH), but it silently negates edit (2), which is the
   only mechanism that can ever make leg 3 non-zero.

**Secondary: VLM-CEILING, and now sharply bounded.** §2.2 + §2.6 together refute the framing that the
arrival decision is a *prompt-wording* problem in either direction. Two different criteria, on the same
2 images, with the same frozen model, produced the **same far-field FP rate (0.043 both)** and moved
only the yes-rate. The 2-image channel has essentially **no** arrival discrimination at any operating
point; re-wording moves it along a flat ROC. Meanwhile the 4-image perceive+plan channel is a
**0.58-precision** detector on its first claim. The ceiling is in the **evidence set**, exactly as the
parent diagnosis §5-D(1) said — and edit (4) shipped §5-D(2) *without* §5-D(1), i.e. the half the
parent explicitly labelled "lowest confidence" and told the proposer to ship as a separate candidate.

**Tertiary: TASK-GAP, unchanged.** Still no "I have passed it" state: 40 episodes walked a median of
18 357 cm beyond the window with no mechanism able to notice.

---

## 5. Concrete changes to try (the gradient — ranked by leverage)

### A — **Revert the REACH system prompt to the parent's wording.** `modules/completion_checker.py:695-725`
Restore the parent's `_binary_reach_vlm_call` system prompt byte-for-byte. Not a re-word — a **revert**.
- **Why:** §2.2. The new criterion costs 8/12 of the lost leg-1 passes on its own and buys *nothing*:
  the far-field FP rate is identical to 3 decimal places (6/138 vs 6/139). This is the single
  highest-confidence, lowest-risk item in the report.
- **Predicted:** leg 1 **0.02 → ~0.18-0.24**, SubtaskSR **0.285 → ~0.35**. Confidence: very high
  (direct A/B on the same 50 tasks, p = 0.0021).
- **Do not replace it with a third wording.** §6.

### B — **Strip the cross-check's veto authority over the FIRST planner claim of a leg, and make any veto position-preserving.** `modules/completion_checker.py:925-935`, `harness.py:548-550`
Three coupled edits:
1. `cross_check_planner_commit` — gate **only the first claim per `cur_idx`**. If the first claim of a
   leg is refused, record it; on the **second** claim of the same leg, honour the planner
   (`return True`). The cap/memo then bounds cost without ever becoming a standing veto. Neither
   `True` (parent: rubber-stamps the 3rd claim at whatever position the vetoes walked to) nor `False`
   (this iter: unbounded free veto) is right — **the cap should not be the thing that decides.**
2. `harness.py:548-550` — a blocked `-1` must **not** inject `Move_Forward`. Emit a net-zero probe
   (`[Rotate_Left, Rotate_Right]`, in-place per `rules.md`) or simply re-perceive. §2.3: the fallback
   forward is what turns a 842 cm claim into a 2 342 cm one and then a 3 842 cm one.
3. Delete `_xcheck_verdict_cache` (`:555`, `:920-922`, `:954`) or make it cache **only `True`**. A
   memoised `False` at a position the agent has not left is a self-fulfilling veto.
- **Why:** §2.3 — 16 in-window planner claims, 0 commits, 62/118 blocks with no evidence behind them.
- **Predicted:** recovers the 9-episode in-window-planner-veto population; leg 1 **+0.06-0.14** on top
  of A, and removes the +20 % cost / ×2.9 collision tax. Confidence high — deterministic, 9 traced instances.

### C — **Fix the double-apply.** `harness.py:459` — one line
```python
-   planner_actions = completion.commit_burst(instruction)
+   planner_actions = [-1]     # the trim loop at :515-523 composes the pose
```
- **Why:** §2.4, 6/6. The trim loop is the single composer by design (that is edit 2's stated goal); line
  459 pre-composes and the loop then composes again, doubling the rotation to 180°.
- **Predicted:** 0 points until something reaches FINISH in-window, but leg 3 is arithmetically pinned at
  0.00 while this stands. **Add a regression test that drives the REACH path, not just the planner path** —
  the shipped dry run missed this because it only exercised `planner_actions = [-1]`.

### D — **Keep edit (1) exactly as it is.** `modules/active_perception.py:504-532`
§2.5: its one opportunity converted a 432 cm outward miss into a pass. It is untested at n=1 but it is
not implicated in the regression and the parent priced it offline at +6 net over 50 episodes. Leave it.

### E — **Give the arrival call the evidence, not another criterion.** `modules/completion_checker.py:646-760`
This is the parent's §5-D(**1**), still unshipped and now much better motivated. Attach
`obs['object_mask']` and `obs['depth']` with the captions `prompt_assembler.py:130-137` already carries,
and thread the current perceiver `Description` in as text.
- **Why:** §2.2 shows the 2-image channel's ROC is flat — moving along it is worthless, so the evidence
  set is the only remaining axis. §2.6 shows the 4-image channel, on the same model and the same steps,
  is a 0.58-precision detector. Ship it **alone**, on top of A/B/C, so attribution stays clean.
- **Cost:** ~+1 300 input tok per REACH call; well inside budget even at the restored 97k/ep.

### F — **Promote the planner's first in-window claim to a commit path, floored on motion.** `harness.py:398-431`
§2.6: `first planner -1 per leg` is in-window 7/12 = 0.58, and a `forward_streak >= K_REACH_FORWARD_STREAK`
(=4) floor on the *planner* path — the constant already exists and is already applied to the proactive
path — removes 2 of the 5 out-of-window claims (streaks 0 and 3) and **0** of the 7 in-window ones
(min streak 9) → **7/10**. Derived only from the action history; no privileged state.
- **Risk:** a false commit is fatal, so this trades 3 fatal terminations for ~7 leg passes. Priced on
  this run that is net positive, but it is a *fitted* streak threshold on one run — price it against A+B
  before shipping it as a standalone commit path, and prefer it as a **fallback** after the cross-check
  declines twice (i.e. fold it into B.1) rather than as a new parallel channel.

### G — Carried forward, still unshipped and still free
`modules/planner.py:120-133` — the shared `reasoning_template` Output Format omits `Match`, so
`verify()`'s `planner_match == 1` silently fails on ~12 % of replies. With F/B making the planner channel
load-bearing this stops being latent. Also `spatial_memory.py:66` (`coll = 0` hard-coded) and
`harness.py:273-285` (STUCK counts the current burst, not history) — both still dead, and the stalled-forward
rate **doubled to 9.0 %** this iter, so they now cost more than they did in the parent.

---

## 6. Exhausted — do NOT repeat

**New, established by this run — do not re-derive:**
- **Do not re-word the REACH arrival criterion again, in either direction.** This is now measured at two
  far-apart operating points on the same 50 tasks: *visibility* wording → 0.205 in-window / 0.043 far;
  *frame-geometry position* wording → 0.017 / 0.043. **The far-field FP rate is invariant to the wording
  (6 calls, both runs);** only the yes-rate moves. The model **recites** whichever clause matches its
  prior — 57/57 in-window refusals and 6/6 far-field accepts quote the new prompt verbatim. The 2-image
  channel has no arrival discrimination at any threshold. The parent's exhausted-list entry ("do not
  re-word for illumination invariance") now generalises: **do not re-word this prompt at all.** Change the
  **evidence set** (§5-E) or change **which channel decides** (§5-B/F).
- **Do not use "how much of the frame the facade fills" as an arrival feature.** Measured flat:
  building-class fraction is 0.276 in-window-refused / 0.309 far-field-FP / 0.293 far-field-refused.
  Every frame in this city is ~28 % building. (Distinct from the parent's refuted `yel_band`/`pur_sides`
  guards — same conclusion, different feature.)
- **`return False` on cross-check cap exhaustion is refuted.** Combined with a memo it produces an
  unbounded free veto (117 blocks, 62 with zero evidence) and, via `EMPTY_FALLBACK`, physically removes
  the agent from the window. **Neither `True` nor `False` is the answer** — the cap must not be the
  decider (§5-B.1).
- **A blocked `-1` must never be replaced by `Move_Forward`.** Traced end to end on ep_20_9_11:
  842 → 2 342 → 3 842 → … → 50 000 cm, 35 cycles.
- **Verify a commit-composition change by driving the REACH path**, not only the planner path. The
  double-apply at `harness.py:459` survived a full fake-env dry run for exactly this reason.
- **`_is_finish_instruction` is correct and does not need re-deriving** — 199/199 FINISH, 0/561 others,
  over every `task_config.json` in `hard/`. Keep it verbatim if the composer is rewritten.

**Carried forward from the parent gradient, re-confirmed here:**
- **Do not touch the ORIENTATION path.** 50/50 again, 0 VLM calls. The only thing that works end to end.
- **Do not tune `K_REACH_FORWARD_STREAK` / `K_REACH_INTERVAL` / `MAX_REACH_FIRES_PER_SUBTASK`.** Coverage
  was never the binding term: 58 in-window firings this run (up from 44) produced **1** yes.
- **Do not re-run `N_REACH_CONFIRMATIONS_REQUIRED = 2`** (iter 8, regressed 0.30 → 0.25).
- **Do not ship a bare depth-polarity flip**, and do not re-enable `_strongly_far` / `_looming_state` as
  gates. `request_side_scan` remains dead as gated: 4 fires / 0 commits in 50 episodes.
- **Do not read** `agent_location`, `target_location`, `destination`, `correct_robot_location`,
  `ideal_path`, `gt.json`, `current_instruction`, `env.unwrapped._agent_rotation`,
  `env.subtask_max_steps`, or `info['agent_collision']`.
- **Avoid numeric literals ≥ 2000** in candidate code (lint WARNs them as map-tuned); express distances
  as forward-action counts.
- **`bootstrap/*/diagnosis.md` is one file copied 7×** documenting VOID rounds; do not mine its `0.0`s.

---

## 7. Provenance

- Candidate: `$OUT_DIR` = `…/prop_r001_df84f4/cand_r001_0_spatial-grounding` (`harness.py` 644 L +
  7 modules, 3 019 L). Score: `…/arm_hard5010_adaptive_k4c1/cand_r1_0_spatial-grounding_r001/score.json`
  — SR 0.0, SubtaskSR 0.285, per_position_SR [1.0, 0.02, 0.0222, 0.0], distance_progress 0.1256,
  static_collision 5.12, 97 152 tok / 857.8 s per episode, 50/50 episodes complete.
- Candidate traces: `…/cand_r1_0_spatial-grounding/traces/` — 50 episodes; 420 REACH calls,
  1 123 plan calls, 3 118 forwards.
- Parent (A/B baseline): `…/anchor_nav_r000/score.json` + `…/anchor_nav/traces/` — SubtaskSR 0.36,
  per_position_SR [1.0, 0.24, 0.1111, 0.0]; 348 REACH calls, 918 plan calls, 2 812 forwards.
  REACH calls are identified in `vlm_trajectory.jsonl` as `kind="perceive"` with **2** images
  (perceive proper carries 4); the count reproduces the parent diagnosis's 348 exactly.
- Instruction-predicate check (§2.4): all 760 instruction strings in
  `SimWorldGym/simworld_gym/envs/setting/single_agent_world/hard/*/*/task_config.json`, run through the
  candidate's own `_is_finish_instruction` / `_named_side`.
- Mask measurements (§2.2): `(128,0,128)` building-class fraction over the `object_mask` image of the
  4-image perceive call at the same step as each REACH call (197 frames).
- `read_audit.json`: **0 forbidden / 0 escalated / 0 undeclared** on all 50 episodes; evaluator taint
  audit PASS. `gt.json` / `trajectory.json` positions were used **only** for this analysis; no
  recommendation in §5 routes them into a policy path.
