# Diagnosis — SimWorld-MMNav (easy) / `anchor_nav` + depth-attach fix, arm `smoke_easy_k4`

**Headline:** SR **0/2** · SubtaskSR **0.25 on both episodes** · per-leg pass **[2/2, 0/2, –, –]**
(legs 2/3 never reached). Binding constraint: **the `-1` commit is decided by one binary
landmark-recognition VLM call and by nothing else, and on this pair it saturated at both rails** —
`Match=1` **2/2** on ep_11 at **2617 cm short** (irreversible, ended the episode at step 9), and
`Match=0` **19/19** on ep_21 across the entire 31 500 cm walk, so that episode never committed and
died on an env deadline the harness does not know exists.

> **Sample size.** 2 episodes × 2 runs each (the trace dir holds two independent passes over the same
> tasks — `call_idx` resets at line 47/48 of each `vlm_trajectory.jsonl`; `actions.jsonl` /
> `env_step_log.jsonl` hold only the last). Treat the **scalar as uninformative** (0/2 leg-1 passes vs
> the pooled-anchor 5/20 is p≈0.56 — not a regression signal). Everything ranked below is instead a
> **per-frame measurement** that replicates across both runs and, where stated, across both episodes.

---

## 0. What this candidate changed

`git diff` vs the anchor is **one hunk, `modules/prompt_assembler.py:115-130`**: the aux-image loop
now tries `("depth_map", "depth")` in order instead of only `"depth_map"`. This is the **first half**
of the parent diagnosis's §5-C. Nothing else moved (`harness.py` and the other six modules are
byte-identical to the anchor).

**It works, verified:** `read_audit.json` records `obs.depth` 9× (ep_11) / 57× (ep_21) and
`obs.depth_map` 6× / 38× — i.e. `depth_map` misses and `depth` hits on every call. All **44/44**
perceive calls now log **4 images** (was 3/3 on 200/200 in the parent census). Cost ≈ **+340 input
tokens per perceive call** (derived: the 2-image REACH call is 1026 in-tokens, the 4-image perceive
call 2409–2506).

**The second half of §5-C was NOT shipped** — the prompt still tells the model the depth image is
usable for distance and still ships the wrong seg legend. §2.3/§2.4 below show why that is the
expensive half.

---

## 1. What this harness does (mechanism)

`harness.py:run_episode`, one iteration per decision (not per env step):

| order | call site | role in the outcome |
|---|---|---|
| 1 | `active.should_emit` (`harness.py:305`) | **short-circuits perceive+plan entirely** when it returns a list. Owns the ORIENTATION cardinal burst `[rot…, -1]` and the cap-2 **forward-momentum** cadence (PP → m1 → m2 → …, i.e. **3 forwards per VLM pair**). |
| 2 | `perceiver.perceive` | 1 VLM call, 4 images (RGB, hint, seg mask, jet depth) → free-text `Description`. |
| 3 | `planner.plan` | 1 text-only VLM call → `{Summary, Match, Actions}`. |
| 4 | `completion.verify` (`:390`) | honours a planner `-1` only if `planner_match == 1 and forward_streak >= 1` (`completion_checker.py:457`). |
| 5 | `cross_check_planner_commit` (`:412`) | 2nd binary VLM call, gates a planner `-1`. |
| 6 | `proactive_commit_check` (`:429`) | **the only commit path that actually fires here.** At `forward_streak ≥ 4`, every 5 forwards, ≤12/subtask: one binary REACH VLM call `(live RGB, hint) → {Match}`; `Match==1` and not `_strongly_far` ⇒ inject `[-1]`. |
| 7 | `request_side_scan` / `scan_reach` (`:458`) | depth-gated physical yaw toward the instruction's named side, re-ask REACH, yaw back on miss. |

**Measured on this run: stages 4 and 5 never executed.** The planner emitted `-1` in **0/44** plan
calls (both episodes, both runs). `verify()` returned False on every call at `not actions or -1 not in
actions`. So `K_MIN_FORWARD_BEFORE_PLANNER_COMMIT`, the `planner_match == 1` conjunct, and the whole
`cross_check_planner_commit` cap logic (the parent's §5-A target) are **unexercised on this subset**.
Stage 6 is the entire commit policy.

---

## 2. What actually happened (causal chain, with evidence)

### 2.1 ep_11 — the recognizer said YES 2617 cm early, and that ended the episode

`gt`: spawn `(-21700, -32183)`, ROAD waypoint `(-21700, -37800)`, `dist_threshold` 2000 (L1).

```
step 0,1  Rotate_Right ×2      (active_perception cardinal burst)
step 2    -1                   ORIENTATION  -> PASS, current_subtask 0->1
step 3-8  Move_Forward ×6      y: -32183 -> -35183   (moving_distance 3000)
step 9    -1                   ROAD  L1 = 2617 > 2000  -> FAIL, episode over
```

The `-1` at step 9 is `notes: "REACH_COMMIT"` — stage 6. The decisive VLM call is
`vlm_trajectory.jsonl` idx 6 (run 0) / idx 13 (run 1), `step=9`, 2 images:

```
prompt : "…Subtask instruction: Continue straight and stop at the intersection where you see
          the mainly white high residential building with a modern glass facade and metal
          framing ON YOUR LEFT."
run 0  : {"Match": 1, "Reason": "same building with modern glass facade and metal framing"}
run 1  : {"Match": 1, "Reason": "The same tall building with a modern glass facade is visible
          on the left in both views."}
```

**Both runs, 2/2.** The claim is checkable and it is false. Class census of the **left** quarter of
that exact frame (`frames/5a67f77ae795.png`, rows 0.30–0.62·H, cols 0–0.25·W, 34 560 px):

| step | y | LEFT building px | LEFT sky px |
|---|---|---|---|
| 3 | −32 183 | 23 749 (0.687) | 0 |
| 6 | −33 683 | 22 789 (0.659) | 804 (0.023) |
| **9 (commit)** | **−35 183** | **0 (0.000)** | **21 600 (0.625)** |

**There is no building on the left at the frame the model committed on — the band is 62.5 % sky.**
The model matched a facade that is ahead/right against a hint captured facing left. Six more forwards
(`y = -38183`, L1 = 383) would have passed, and the leg's env budget had 49 of 56 substeps left.

A wrong `-1` is fatal by construction — `traffic_world.py:632-654`:
`self._evaluation_of_subtask = self._get_evaluation_of_subtask(); … else: terminated = True`.

### 2.2 ep_21 — the recognizer said NO 19/19, and the env's *per-subtask* deadline killed it

`gt`: spawn `(11500, -41700)`, ROAD waypoint `(17800, -41700)` ⇒ pass window `x ∈ [15800, 19800]`.

```
step 0    Rotate_Right ;  step 1  -1  ORIENTATION -> PASS
step 2-64 Move_Forward ×63   x: 11500 -> 43000    (31 500 cm, never a single turn, never a -1)
verdict   steps 65, notes ""   (no STUCK, no MAX_STEPS, no CRASH)
```

The agent was **inside the ROAD pass window at steps 11–18** (x = 16000…19800) and walked straight
through it. Every REACH fire in that window returned 0:

```
step 14  {"Match": 0, "Reason": "Different building features and lighting conditions"}
step 15  {"Match": 0, "Reason": "Different building facades and lighting conditions"}
step 20  {"Match": 0, "Reason": "Different buildings and lighting conditions"}
```

Pooled over both runs: **19 REACH fires, 19 × `Match=0`, 14/19 citing lighting/brightness.**
Measured domain gap: hint frame mean luma **91.2** (ep_11's hint: **158.0**) vs live frames **43–85**
(ep_11: **36–62**). The frozen model is being asked a cross-illumination appearance match and refuses.

**Why the episode ended at exactly step 65 — this is new, and the harness is blind to it.**
`traffic_world.py:384` terminates on `substeps_count >= subtask_max_steps`, and `:641-649` sets that
budget **at each successful commit**: ORIENTATION → 10, TURNING → 50, ROAD/FINISH →
`int(L1(current_pos, correct_robot_location) / 100)`. After the ORIENTATION commit at step 1:
`int(|17800 − 11500| / 100) = 63`. `:686` then does `substeps_count += 1` on *every* action.
**63 forwards later the deadline fires — exactly the observed 63 forwards, exactly the observed 65
total actions, and exactly the empty `notes` field** (the harness never appended a note because it
never noticed; `env.step` simply returned `terminated=True`).

Consequences the proposer must internalise:

- A ROAD/FINISH leg allows `dist/100` steps but needs `dist/500` forwards — **the budget is 5× the
  required forwards**. That is generous, but it is **hard**, and it is spent by *every* action:
  momentum forwards, the `[2,1]` traffic idle, and each `request_side_scan` turn-and-restore pair.
- With a recognizer at recall 0, the agent is **guaranteed** to overshoot ~5× and then die. ep_21
  ended 21 300 cm past the corner, off the end of the corridor.
- `env.subtask_max_steps` is **off-limits to read** (parent §6: it is laundered `target_location`).
  The lever is to stop wasting the budget and to commit on evidence the model can actually produce —
  not to read the deadline.

### 2.3 Every depth predicate in `CompletionChecker` is reading the depth channel **backwards** — measured, 14/14 frames

`obs['depth']` (the channel this candidate just started shipping) is **JET-colormapped**. Measured on
the delivered frames: 239 unique colours, centre-column ramp
`(128,0,0) → (212,0,0) → (86,255,170) → (0,224,255) → (0,44,255)`. Ground truth for the far end:
the **sky** mask class occupies **exactly** the `(128,0,0)` pixel set — `IoU(green_mask, depth==(128,0,0))
= 1.000` on **14/14** sampled frames across both episodes. So **far = dark red, near = blue, mid = bright
cyan**; both extremes are dark.

`completion_checker.py:255 _depth_gray` reduces that with `arr.mean(axis=2)` and the module asserts
*"LOWER value == NEARER"*. The parent proved the LUT is a symmetric tent; here is what it does to the
**actual** frames:

```
frame           p10    p90   near_thr   SKY gray   -> class   BUILDING gray -> class
ep11 step 3     42.7  170.3     81.0      42.7     -> NEAR        118.1     -> far
ep11 step 6     42.7  170.3     81.0      42.7     -> NEAR        114.0     -> far
ep11 step 9     42.7  170.3     81.0      42.7     -> NEAR         91.2     -> far
ep21 step 2..27 42.7–70.7 …              42.7     -> NEAR (14/14)  101–127  -> far (14/14)
```

**The sky — the farthest thing in every frame — is classified NEAR on 14/14 frames; buildings are
classified far on 14/14.** The polarity is inverted, not merely folded. Downstream:

- `_strongly_far` (`:335`), the accept-validator that is supposed to veto a geometrically impossible
  `Match=1`: its `near_frac` is driven by sky and measured **0.145–0.436** on 14/14 frames, never
  below the `0.05` veto threshold. **It never vetoes.** At ep_11 step 9 it computed `near_frac =
  0.436` and waved the fatal commit through (`harness.py:429` → `:611` → `:438`).
- `_side_band_near` (`:376`), the trigger for the physical side-scan: fires when the named-side band
  median reads "near", i.e. when that band is **sky**. Measured False on 13/14 frames and **True
  exactly once — ep_11 step 9, `median = 42.7` = pure sky on the left.** Its single firing
  opportunity in this pair was the one frame where there is provably *no* building on that side. (It
  was pre-empted by the REACH commit, so it cost nothing here; it will not be pre-empted next time.)
- `_looming_state` (`:275`) returned "near" at ep_11 steps 6/9 and ep_21 steps 17/20 — also sky.

**A JET inverse LUT does recover the correct monotone scale** (nearest-neighbour match to
`matplotlib jet`, max residual 22.7 RGB units): sky → **255.0** on 14/14, building → 200–226,
ground plane → 68–78. All three orderings correct.

**But a bare polarity flip is a trap.** With the existing `_CONE_ROW=(0.20,0.58)` band (rows 120–348,
mostly above the horizon) the decoded `near_frac` is **0.000 on 14/14 frames** ⇒ `_strongly_far`
would flip from *never vetoes* to **always vetoes**, i.e. every episode becomes ep_21. Any
polarity fix must re-derive the band and threshold on real frames before it ships.

### 2.4 The seg legend the harness feeds the model is wrong — and one class is wrong in a new way

`prompt_assembler.py:116-118` sends: *"green=trees, purple=buildings, yellow=sidewalks/crosswalks,
black=driveways"*. Measured palette, every mask frame: exactly four classes —
`(128,0,128)` buildings, `(255,255,0)` ground/road, `(0,255,0)`, `(0,127,255)` (≈1.5 %). **There is
no black class at all.**

New this pass, **correcting the parent diagnosis's palette table**: `(0,255,0)` is **SKY, not trees.**
Proof: `IoU(green, depth-max) = 1.000` on 14/14 frames (§2.3); green occupies rows 0–299 only; its
mean RGB luma is 130–193 (bright overcast sky) against the yellow ground's 9–38. 12–20 % of every
frame is labelled for the model as "trees" when it is sky. The perceiver cites seg colours in
**33/38** responses, so it is consuming this.

The depth caption is worse now that the image actually arrives: `perception_template(depth=True)`
(`prompt_assembler.py:64`) tells the model *"Judge the distance by the depth image to infer how many
forward steps can be taken or how close intersections are"*, and `:119` labels it *"use it to judge
distance"* — with **no jet legend**. The model obliges: **42/44** perceive responses reason about
depth, e.g. ep_11 step 9 (the fatal frame): *"The depth map indicates that the building is relatively
close…"*. `AGENT.md` additionally asserts *"nearer = brighter"*, which is **false for jet** (both ends
are dark; the brightest band is mid-range).

### 2.5 Latent: the planner drops `Match` on 39 % of replies

`planner.py:120` uses the baseline `reasoning_template`, whose system block ends
*"You must ALWAYS return a JSON **exactly like**: {"Summary": …, "Actions": […]}"* — **no `Match`
field**. `planner.py:133` then asks for `{"Summary", "Match", "Actions"}` in the user turn. The system
block wins: **15/38** plan replies on ep_21 omit `Match`, so `plan_out["match"]` is `None`
(`planner.py:163`). `verify()` (`:457`) requires `planner_match == 1`, so **any** planner-driven
commit is silently vetoed ~40 % of the time. Currently masked (the planner emitted `-1` 0/44 times);
it will bite the moment anything makes the planner willing to commit.

### 2.6 Other confirmations (no new action needed)

- **ORIENTATION is solved.** 2/2 via the closed-form cardinal burst, 0 VLM calls. Do not touch it.
- **STUCK never fires** (`harness.py:283`): the predicate needs `>3` forwards *inside one burst*;
  40/44 bursts are the bare `[0]`. Consistent with the parent's 0/20.
- **Cadence:** `active_perception` momentum gives 3 forwards per perceive+plan pair, as designed.
  Per run: ep_11 = 3 plan calls / 6 forwards, ep_21 = 19 plan calls / 63 forwards (3.3 per pair).

---

## 3. Failure breakdown

| class | eps | rate | root cause | evidence |
|---|---|---|---|---|
| **Fatal early `-1`** (recognizer FP) | ep_11 | 1/2 (2/2 runs) | stage-6 REACH `Match=1` at 2617 cm; `_strongly_far` veto inert (§2.3) and no named-side precondition | REACH idx 6/13 step 9; left band 0 building px |
| **Never commits → env subtask deadline** (recognizer FN) | ep_21 | 1/2 (2/2 runs) | 19/19 `Match=0`, 14/19 blaming lighting; hint 91–158 luma vs live 43–85 | `substeps_count` 63 = `int(6300/100)`, exact |
| ORIENTATION | both | 2/2 pass | cardinal burst | — |
| TURNING / FINISH | both | never reached | upstream leg-1 death | — |
| parse-fail / illegal action / crash / socket | both | 0 | — | `notes` = "REACH_COMMIT" and "" |

Both failures are the **same defect at opposite rails**: the commit is a single binary appearance
match, with no positional, geometric, or side-consistency evidence attached to it.

---

## 4. Binding constraint — classified

**Primary: HARNESS-BUG.** Two defects sit directly on the irreversible action and both are verified in
the current source and in the current frames:

1. **All three depth predicates read the channel with inverted polarity** (§2.3, 14/14 frames). The
   one guard designed to stop a far-field false `Match=1` (`_strongly_far`) computed `near_frac =
   0.436` **on the frame that killed ep_11** and passed it. The side-scan trigger fires on sky.
2. **The commit has no named-side precondition**, although the instruction names a side on both
   episodes and `_named_side` (`:244`) already parses it exactly. At the fatal frame the named side
   held **zero** building pixels.

**Second: VLM-CEILING** on the recognizer itself, reconfirmed. 19/19 refusals with a 2–4× luma gap.
The parent already priced six reformulations plus graded confidence offline and all lost; 14/19
lighting citations here are the same rationalization it already refuted. **Do not re-word the
question.** The fix must attach independent evidence to the decision.

**Third: TASK-GAP** — the harness has no model of the env's per-leg step deadline (§2.2) and no
notion of "I have walked far enough that I am past the waypoint". It walked 21 300 cm past a corner
and never changed tactic.

**Honest budget for this pair.** ep_11 is recoverable by a veto (it had 49 spare substeps and needed
6 forwards). ep_21 is recoverable only by a commit trigger the recognizer is not part of. Fixing (1)
and (2) plausibly turns ep_11 into a leg-1 pass; ep_21 needs §5-B or §5-C.

---

## 5. Concrete changes to try (the gradient — ranked by leverage)

### A — Named-side building precondition on `Match=1`. `modules/completion_checker.py:611` (+ a new predicate near `:376`)

```python
def _named_side_has_structure(obs, instruction) -> bool:
    """True unless the instruction names a side and that side is EMPTY of buildings.
    Fails OPEN (True) on: no named side, missing mask, unexpected palette."""
    side = _named_side(instruction)
    if side is None: return True
    m = obs.get("object_mask") if isinstance(obs, Mapping) else None
    if m is None: return True
    a = np.asarray(m)
    if a.ndim != 3 or a.shape[-1] < 3: return True
    h, w = a.shape[:2]
    r0, r1 = int(0.30 * h), int(0.62 * h)
    sw = max(1, int(0.25 * w))
    band = a[r0:r1, 0:sw] if side == "left" else a[r0:r1, (w - sw):w]
    bld = np.all(band[..., :3] == np.array([128, 0, 128]), axis=-1)
    return float(bld.mean()) > 0.02
```
Call it where `_strongly_far` is called today, in **both** `proactive_commit_check:611` and
`cross_check_planner_commit:675`: `if not _named_side_has_structure(obs, instruction): return False`.

- **Why:** §2.1. The instruction says the landmark is on a specific side; a frame with **no building
  on that side** cannot be the arrival frame, whatever the recognizer says.
- **Measured on every frame in this run (32 frames, both episodes, both runs):**
  `left_building_fraction` = **0.000** at ep_11's fatal frame (the only frame it would veto) and
  **0.595–0.719** on **29/29** ep_21 frames — including all four frames inside the ROAD pass window
  (0.595, 0.629, 0.632, 0.634). **1/1 true veto, 0/29 false vetoes.**
- **Not the refuted `pur_sides` guard** (parent §5.X1): that pooled *both* side-thirds as a graded
  threshold on `Match=1` and its TP/FP values interleaved. This is the **named** side only and the
  operating point is a **hard zero**, not a fitted threshold. The `0.02` epsilon exists only to
  absorb mask speckle; `> 0` gives the identical verdict on all 32 frames.
- **Replication:** the parent census lists its 6 recognizer false positives at L1 =
  {26734, 22874, 29234, **2616**, **2616**, 37331}. The two 2616s are this same task in its two arms;
  ep_11's commit here is L1 **2617** in both runs. **The same false commit has now fired in 4
  independent runs** — this is not sampling noise.
- **Predicted:** kills the ep_11 class outright. The parent's other four FPs are far-field
  (22 000–37 000 cm) — that is the case §5-B's *corrected* `_strongly_far` is for, not this guard.
  Downside is bounded by construction: a false veto costs continued walking, a false commit costs
  the whole episode.
- **Risk to state:** at a *true* intersection the near block face ends. Measured here it does **not**
  empty the band — the opposite corner still fills 0.595–0.634 — but that is 4 in-window frames on
  1 task. Price on ≥8 distinct in-window positives (parent §7 recipe) before trusting it alone.

### B — Fix the depth polarity by decoding jet, then **re-derive** the band. `modules/completion_checker.py:255`

```python
_JET = (matplotlib.colormaps['jet'](np.linspace(0, 1, 256))[:, :3] * 255.0)

def _depth_gray(obs):            # returns a value MONOTONE in true depth; LOW == NEAR
    ...                           # nearest-LUT-neighbour per pixel -> float index 0..255
```
Measured: max NN residual **22.7** RGB units; sky → **255.0** (far extreme) on 14/14; buildings
200–226; ground 68–78. The module's stated "LOWER == NEARER" convention becomes **true**.

- **Ship this ISOLATED.** The parent (§5-D) records that a jet-inverse decode **regressed** in a
  sibling lineage, and §2.3 shows a bare flip makes `_strongly_far` veto **100 %** of frames
  (`near_frac = 0.000` on 14/14 with the current `_CONE_ROW`). Either
  (i) ship the decode **and** delete `_strongly_far` / `_looming_state`, keeping only
  `_side_band_near` re-derived as *"the nearest **building-class** pixel in the named-side band"*
  (mask-restricted percentile, excluding sky and ground), or
  (ii) ship §5-A alone first and leave the depth predicates dead.
- Preferred form of the re-derived side predicate (measured below):
  `L_bld_p10 = percentile(decoded[rows 0.30–0.62·H, named-side quarter][mask == buildings], 10)`.
- **Offline-priced on ep_21's 29 frames:** `L_bld_p10` = **215 / 213 / 204** at x = 17500 / 18000 /
  19000 — all three inside the ROAD pass window — versus **165–191** on the other 26 frames
  (max outside = 191). A threshold near 200 gives 3/3 precision here. It also reads `nan` (< 50
  building px) at ep_11's fatal frame, agreeing with §5-A. **Thin: 1 task, 3 positives.** It is a
  *fitted* threshold, unlike §5-A. Price it to the ≥8-positive bar before it gates anything.
- **Mechanism (not map-tuning):** at an intersection the near block face ends, so the nearest
  building on that side jumps to the far side of the cross street. No coordinate, no cm literal.

### C — Tell the model the truth about the two images it now gets. `modules/prompt_assembler.py:64, 116-120` (+ `AGENT.md`, `rules.md`, `harness.py` docstring)

This candidate shipped the attach; ship the caption:

```python
(("object_mask",), "The object segmentation mask of the current view "
 "(purple=buildings, yellow=ground/road, green=SKY, blue=other). "
 "There is no driveway class."),
(("depth_map", "depth"), "The depth map of the current view: a JET colormap, "
 "normalized per frame. BLUE is nearest, CYAN/GREEN is mid-range, RED is "
 "farthest, and BOTH extremes render dark — do NOT read brightness as distance."),
```

- **Why:** §2.4. `green=trees` is wrong — measured `IoU(green, depth-max) = 1.000`, it is sky;
  `yellow` is the road, not the sidewalk; `black=driveways` does not exist. And the model now reasons
  over the jet image in **42/44** perceive calls with no legend, including on the fatal frame.
- Also correct `AGENT.md`'s *"normalized uint8 … nearer = brighter"* — false for jet.
- **Predicted:** small alone, but it is free and it is the precondition for every distance claim the
  perceiver makes. If the truthful caption does not help, the honest alternative is
  `perception_template(depth=False)` + dropping the 4th image (recovers ≈340 tokens/call) — the one
  thing not to keep is the current state: a promised, delivered, undecodable channel.

### D — Give the planner a `Match` field it is allowed to emit. `modules/planner.py:120-133`

Append one line to the system prompt (or post-process): the system block's *"exactly like
{"Summary", "Actions"}"* currently overrides the user turn's ask, dropping `Match` on **15/38**
replies (§2.5). Either add `Match` to the system Output Format, or treat `match is None` as
"unknown" rather than folding it into the `== 1` test at `:457`.

- **Why:** it is a silent 40 % veto on a gate the chain believes is live.
- **Predicted:** zero on this subset (planner `-1` rate is 0/44); it removes a latent trap for any
  candidate that makes the planner commit.

### E — Do not waste the per-leg step deadline. `modules/active_perception.py`, `harness.py:458-478`

Now that §2.2 has pinned the deadline as a real terminator: every action charges it, including the
`[2,1]` traffic idle (net-zero displacement, 2 substeps) and each `request_side_scan` turn+restore
(2 substeps, plus a VLM call). A ROAD leg gets `dist/100` substeps and needs `dist/500` forwards, so
the slack is 4/5 — comfortable, but `MAX_SIDE_SCANS_PER_SUBTASK` and
`_MAX_TRAFFIC_IDLES_PER_SUBTASK = 4` (up to 8 substeps) both eat it. Keep the caps small and prefer
position-preserving probes that are also **substep-free** (there are none — every `env.step` counts).
**Do not read `env.subtask_max_steps`** — parent §6, it is laundered `target_location`.

---

## 6. Exhausted — do NOT repeat

- **Do not touch the ORIENTATION path.** 2/2 here, 20/20 in the parent census. The only thing working.
- **Do not re-word or re-shape the arrival question.** Parent §6 priced six reformulations plus
  graded confidence offline against the live model on the exact frames; all lost. The 14/19
  lighting citations here are the same rationalization the byte-identical-prompt arm already refuted
  — **an illumination-invariance clause is a known dead end.**
- **Do not use `yel_band` or `pur_sides` as a guard on `Match=1`** (parent §5.X1, measured on 2× data,
  TP/FP interleaved). §5-A is a *different* feature (named side only, hard-zero operating point) —
  keep them distinguished when reporting.
- **Do not wire a raw segmentation conjunct as the arrival TRIGGER.** Re-measured this pass: the
  lateral yellow (ground) fraction in the left/right bands is **0.51–0.66 at every position** on
  ep_21's 29 frames with no intersection signature — the parent's "no lateral-opening cue" holds for
  the *raw* mask. (The §5-B signal is depth-decoded and building-restricted; that is not the same
  feature, and it is still under-powered.)
- **Do not tune `K_REACH_FORWARD_STREAK` / `K_REACH_INTERVAL` / `MAX_REACH_FIRES_PER_SUBTASK`.** Run
  to death in both directions across iters 006–010. On ep_21 the schedule fired 19 times across the
  entire walk including 3 fires inside the pass window — coverage is not the binding term, the
  answer is.
- **Do not read `agent_collision`, `agent_location`, `target_location`, `destination`,
  `correct_robot_location`, `ideal_path`, `gt.json`, or `env.subtask_max_steps`.** `validate_harness.py`
  hard-fails the first and the last is `int(L1_to_target/100)`.
- **Avoid numeric literals ≥ 2000** in candidate code (lint WARNs them as map-tuned). Express
  distances as action counts.
- **Do not ship a bare depth-polarity flip.** §2.3: with the current `_CONE_ROW`, correct polarity
  makes `_strongly_far` veto 14/14 frames — it converts every episode into the ep_21 failure.
- **`bootstrap/*/diagnosis.md` in this tree is one file copied 7×**, and it documents that rounds
  r1–r6 of both archived runs were VOID (zero traces). Do not mine those `0.0`s as refutations.

---

## 7. Provenance

- Candidate: `$OUT_DIR` = `simworld_gym/simworld_config/harnesses/nav` (anchor + the
  `prompt_assembler.py` depth-key hunk; `git diff` = 1 file, +14/−5).
- Traces: `runs_adaptive/nav_easy__smoke_easy_k4/arm_smoke_easy_k4/anchor_nav/traces/` —
  `ep_20_0_task_dist_11_0_1` (10 steps, 7 VLM calls/run) and `ep_20_0_task_dist_21_0_1`
  (65 steps, 48 VLM calls/run), **two runs each** in `vlm_trajectory.jsonl`.
- Env semantics quoted from `SimWorldGym/simworld_gym/envs/traffic_world.py:370-386, 631-690`
  (identical in `simple_world.py`).
- Pixel measurements: 32 mask frames / 14 depth frames decoded from `traces/*/frames/`; jet LUT =
  `matplotlib.colormaps['jet']`, nearest-neighbour inverse, max residual 22.7.
- `read_audit.json`: **0 forbidden reads, 0 undeclared reads** on both episodes. Every measurement
  above that informs a *recommendation* uses only `obs['object_mask']`, `obs['depth']`,
  `obs['rgb']`, `info['moving_distance']` and the instruction text; `gt.json` / `env_step_log`
  positions were used **only for this analysis**, never in a proposed code path.
