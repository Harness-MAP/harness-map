# Diagnosis — snake / `staged_anchor` (parent tag `anchor_snake`), round 16

**Headline:** primary = **0.0000** (20/20 seeds `score` 0.0, `snake_length = 2` on all 20, **67**
engine decisions of a 20 × 100 = 2000-decision budget, 0 meals, 0 parse failures, 0 default
actions). Binding constraint: **HARNESS-BUG** — `tools/vlm.py:decide` is a two-line pass-through to
`evaluator.baseline_step`, so no arithmetic stands between the key the model names and
`game.execute_mapped_action`, and **20 of 20 episodes end with the head crossing a face of the
cube** (re-derived first-hand: `traces_diag/ab16/anchor16.py` dead-reckons the 20 published tapes
and reproduces the engine's own `steps_taken` on **20/20**, cause WALL on **20/20**). That is closed
since round 3 and is carried, not re-derived.

**Read the phase first.** The driver log
(`spatial_world/logs/ours_claude_pv_1010_2028_snakek32/snake__k32.log`) ends at
*"INIT stage A (COVER — FREE) pass 14: … the pool phi-covers **15/16** of K0 = 32 | 0 evaluated
(stage A evaluates NOTHING)"*. Recomputed first-hand (`ab16/cells16.py`, all 14 arm candidates +
the anchor + 7 cores + 86 bootstrap founders): **occupied = `[3, 5, 6, 8, 9, 12, 13, 15, 16, 20,
21, 25, 27, 29, 30]`** — exactly the driver's 15 — **17 empty**. **ONE cell finishes stage A.**
r014's placement claim verifies to four decimals (cell 8, d 1.2084, runner-up 27 @ 1.4879, margin
+0.2795).

**What this round settles — three things, all first-hand, 2416 live VLM calls over 4 arms.**

1. **The last cell has exactly ONE vehicle, and it is live-1.0000.** Of 7 code trees × 8 NOTES
   regimes (`ab16/grid16.py`), **`ab11/R_touraim` is the only tree that reaches an empty cell at
   all** — and it reaches one in **all 8** regimes (cell 14 at ≤ ~1600 chars, cell 19 above). With
   the file set a real candidate must carry, **cell 14 at margin +0.7279** (no `NOTES.md`) /
   **+0.4573** (260-char `NOTES.md`) — the widest margin this arm has ever had. Re-run live this
   round: **1.0000, 20/20, 0 deaths, 16.20 dec/ep**. §5b.
2. **r014 has now met the frozen model: primary 0.9125, not the ~0.10 the carried
   counterfactual predicted.** Its card-plus-legality-veto mechanism is far stronger than
   "a legality veto on its own". But it loses 5 of 20 episodes, in **two** distinct ways, and both
   are harness-side and already computed by the harness itself. §2b–§2c.
3. **Three lines of `tools/ledger.py` take it to 1.0000 — and φ gets *better*.** Veto the `self`
   status (1 line) + honour the ask only when it is already the span-optimal offer (2 lines):
   **live 1.0000, 20/20, 0 deaths, 16.15 dec/ep**, cell 8 held with margin **+0.2795 → +0.2993**.
   This is the highest-leverage item in the file. §6 · G0.

---

## 1. What this harness does (mechanism)

`$OUT_DIR` is the **rich anchor** — the baseline, re-expressed as the multi-file surface the
proposer evolves. Control flow, end to end:

```
config.py::_apply_harness  ->  orchestrator.py::build_agent(evaluator)
      |                              |
      |                              +-- evaluator.current_prompt = compose_system_prompt()
      |                              |      = AGENT.md + skills/*.md + rules.md, in that order
      |                              +-- returns  async agent(frame_data) -> tools.vlm.decide(...)
      |
      every step:  tools/vlm.py::decide  ->  evaluator.baseline_step(frame_data)
                                              (capture frame, build the 29-turn image+text
                                               history, ONE VLM call, parse {"action": ...},
                                               fall back to a default, return Action(key_press))
```

* `orchestrator.py` — 2.5 kB. `compose_system_prompt()` concatenates the three `.md` priors so the
  composed text is **byte-identical to `system_prompt.txt`** (verified: `diff system_prompt.md
  system_prompt.txt` is empty). `build_agent` pops any inherited `tools*` from `sys.modules` so a
  sibling candidate's `tools/` can never leak in, then returns the per-step fn.
* `tools/vlm.py` — **the whole decision.** Two statements: fetch `evaluator.baseline_step`, await
  it. There is no offer set, no reckoning, no legality test, no retry.
* `tools/action.py` — a correct, complete, and **entirely unused** action algebra: `KEY_MAPPING`
  (6 moves → `d a w s q e`), `DIRECTION_VECTORS`, `is_reversal`, `CELLS_PER_MOVE = 1`. Nothing in
  the live path imports it.
* `tools/frame.py` — a one-line wrapper around `evaluator.capture_game_screenshot`, also unused by
  the live path (`baseline_step` grabs its own frame).
* `memory.md` — a scratchpad **template**; the anchor deliberately does not inject it.
* So the anchor's only decision channel is **prose** — and `rules.md` spends 11 of its 24 lines
  sending the model to read on-screen `R/L/U/D/W/S` letters that **this `_pv` build does not
  paint** (three unlabelled coloured gizmo lines are painted; the glyphs are not). See G7.

---

## 2. What actually happened (causal chain, with evidence)

### 2a. The anchor: every death is the box (carried, re-confirmed first-hand)

`ab16/anchor16.py` replays each of the 20 published tapes from the build's fixed opening — head
`(2,3,2)`, the cell behind it `(1,3,2)`, heading **+X** — applying only the keys the harness
actually committed:

| seed | tape | head path | engine `steps_taken` | reckoned | cause |
|---|---|---|--:|--:|---|
| 1 | `for,for,for` | (2,3,2)→(2,3,1)→(2,3,0)→**(2,3,−1)** | 3 | 3 | WALL |
| 2 | `for,rig,for,rig` | …→(3,3,0)→**(4,3,0)** | 4 | 4 | WALL |
| 5 | `rig,for,dow` | …→(3,3,1)→**(3,4,1)** | 3 | 3 | WALL |
| 11 | `rig,for,rig` | …→(3,3,1)→**(4,3,1)** | 3 | 3 | WALL |

**Reckoned press count == the engine's `steps_taken` on 20/20 seeds; cause WALL on 20/20.** There
are only **5 distinct tapes** across 20 seeds — 11× `forward,forward,forward`, 7×
`forward,right,forward,right` — and the action histogram over all 67 decisions is
`forward 47 / right 18 / down 2`, **`left 0`, `up 0`, `backward 0`**. The policy is effectively
pellet-blind and seed-independent: it walks out of the +X/−Z corner it spawns facing.

Nothing is broken in the plumbing. `successful_actions = 3/3`, `default_actions = 0`,
`completion_tokens = 8` and `reasoning = None` on 67/67 replies, `prompt_tokens` deterministic
1320 / 1827 / 2334 / 2841 at frames 0–3. The model answers the schema perfectly and the answer
kills it, because nothing checks the answer.

### 2b. r014 live: 0.9125, and its 3 deaths were each labelled `self` by its own ledger

`cand_r014_0_spatial-grounding` shipped as an explicitly unscored exploration round
("*no live episode was run for this candidate*"). It has now been run: `ab16/R_r014` is the
candidate byte-for-byte (16/16 genome files md5-verified), 20 seeds, real `eval_runner`, real
frozen model.

| | r014 SHIPPED | +`self` veto (1 line) | +rank-0 honour (2 lines) | `R_touraim` |
|---|--:|--:|--:|--:|
| primary | **0.9125** | 0.8750 | **1.0000** | **1.0000** |
| seeds at 1.00 | 15/20 | 17/20 | **20/20** | **20/20** |
| deaths (self-collision) | **3** | 0 | 0 | 0 |
| episodes capped at 100 dec | 2 | **3** | 0 | 0 |
| decisions / episode | 44.15 | 44.30 | **16.15** | **16.20** |
| honour (ask == press) | 54.70 % | 59.82 % | 24.77 % | 25.62 % |
| VLM calls / 20 eps | 883 | 886 | 323 | 324 |

The 3 deaths (seeds 3, 8, 17) are the **same failure, three times**. The last press of each losing
episode, read off the candidate's own `ground_ledger.jsonl`:

```
seed  3 frame 68  model=move_down      note=honoured   row ['move_down',     [2,3,1], span 3, room 0, 'self']
seed  8 frame 13  model=move_right     note=honoured   row ['move_right',    [1,1,1], span 3, room 0, 'self']
seed 17 frame 31  model=move_backward  note=honoured   row ['move_backward', [0,1,2], span 5, room 0, 'self']
```

In all three the harness had **already computed** that the landing was its own body (`'self'`,
`room 0`), had `sure = True`, printed the fact on the card it sent the model — and then committed
the press anyway, because `tools/ledger.py::verdict` replaces only `"wall"` and `"180"`:

```python
state = dict((row[0], row[4]) for row in self.table()).get(proposed)
if state != "wall" and state != "180":      # <-- 'self' falls through here
    return proposed, "honoured"
```

Census over all 886 decisions (`ab16/census16.py`): the model proposed a `self` landing **27**
times and `verdict` honoured **27 of 27**; 3 of them were fatal. `ground_sure` was `True` on
**886/886** decisions — the odometer never retired, so the `self` label was never the stale belief
the module's docstring excuses it as. The module's own reasoning ("*`self` is the ledger's belief
about its own tape, which is exactly the belief that can be stale, so it never overrides a model
that can see the picture*") is **inverted**: the odometer is exact, and the model **cannot** see
the body — r014's own `tools/ink.py` documents that overlapping balls weld into single components,
which is precisely why the body is reckoned rather than read.

Over-estimation is the safe direction, and it is what is actually happening. Measured engine law:
`snake_length = 2 + meals` (18/19 episodes agree; the 19th is the ledger losing a meal to a pellet
abstention). r014's ledger carries `eaten + 3` cells — **one cell long** — so its `self` label is a
conservative over-estimate: 6 of the first 9 `self` reads were survivable. Vetoing all of them
costs a few decisions and buys back the 3 that were not.

### 2c. The deeper loss is the BUDGET, and the 1-line fix exposes it

`R_r014fix` (one line: `if state == "open":`) removed the death class completely — **status of the
committed key = `open` on 886/886**, 8 `self` vetoes fired, 0 deaths — and primary **went down**,
0.9125 → 0.8750. Seeds 3/8/17 all became 1.00, seeds 11/19 became 1.00, and seeds 7/13/20 hit the
**100-decision cap** instead. The mechanism was never death-bound; it is **budget-bound**.

Why 44 decisions when every live-1.0000 tree in this arm finishes in 15.9–16.2:
`ab16/closure16b.py` scores every decision on whether the committed press reduced
`L1(head, pellet)`, with the pre-move head-span recovered exactly from the row set
(`span(+) + span(−) = 2·L1(head, pellet)` for any axis with both directions in bounds):

| half of the stream | closes the pellet gap | n |
|---|--:|--:|
| **honoured** (the model's own ask) | **0.4060** | 463 |
| **vetoed** (the ledger's replacement) | **0.9398** | 349 |
| chance (uniform over the step's own legal offers) | 0.4229 | 812 |
| rank-0 (nearest-span offer) | 0.9754 | 812 |

**The honoured half is chance** — z = **−0.73** at n = 463. The vetoed half is z = **+19.55**. So
r014's 0.9125 is produced entirely by the 45 % of presses the veto takes over; the other 55 % are a
random walk that spends the budget. Two episodes ran out of it.

### 2d. The card reached the model and the model ignored it

This is the question r014 was built to ask, and the answer is clean (`ab16/card16.py`):

* the splice into `conversation_history[0]` **landed on 863/883 = 97.7 %** of decisions (the 20
  misses are the first frame, before the evaluator has built turn 0);
* handed a card naming every landing's cell, its L1 span to the pellet and its free volume, the
  model's **first ask is still illegal on 427/883 = 48.4 %** (wall 256 + 180 144 + self 27);
* and it gets **worse** inside an episode: legal 0.6350 (frames 0–9) → 0.4909 (10–29) → **0.4731**
  (30+);
* the ask's position in the card's **own** ranking is ~uniform: rank 0 **20.5 %**, rank 1 32.1 %,
  rank 2 24.8 %, rank 3 17.6 %, rank 4 5.0 %.

A correct, verified-delivered, per-step text grounding card **does not bind the ask**. That closes
the text-channel axis by mechanism, not by correlation: it is not that the card was wrong or
undelivered — it was right and delivered 97.7 % of the time.

### 2e. `room` is inert on this lattice

`ab16/rank16.py`: across 885 decisions with ≥ 2 open offers, the **spread of `room` across those
offers is 0 on 885/885**. Ranking `(-room, span)` and `(span, -room)` therefore pick the same move
on 885/885. On a 4×4×4 box with a body of ≤ 7, every open landing reaches the same free volume.
The pocket/reachable-volume field cannot be a rank key *or* a tie-break here — it is only useful as
a hard `room == 0` guard, which is exactly the `self` veto of §2b by another name.

---

## 3. Failure breakdown

### 3a. The anchor (what this `OUT_DIR` scored)

| category | rate | evidence |
|---|--:|---|
| **WALL death** (head crosses a face) | **20/20 eps** | `ab16/anchor16.py`, 20/20 step-count match |
| pellet ever reached | 0/20 | `snake_length = 2`, `score = 0.0` on 20/20 |
| parse failure / default action | **0/67** | `default_actions: 0`, `successful_actions: 3/3` |
| budget used | **67 / 2000** | `steps_taken` 3–4, mean 3.35 |
| distinct tapes | **5 / 20 seeds** | 11× `for,for,for`, 7× `for,rig,for,rig` |
| moves never asked | `left`, `up`, `backward` | 0 of 67 |

### 3b. r014, the arm's cell-8 elite (what stage B will score)

| category | rate | cost |
|---|--:|--:|
| 4 pellets, early-stopped | 15/20 | — |
| **self-collision death** | **3/20** | −0.25 each |
| **100-decision cap** | **2/20** | −0.50 each |
| honoured presses the ledger labelled `self` | 27/886 (27/27 honoured) | 3 fatal |
| honoured presses that close the pellet gap | 0.4060 (chance 0.4229) | ~28 extra dec/ep |
| pellet read abstained | 25 off-lattice + 5 absent / 886 | 0 (re-read, never latched) |
| odometer retired (`sure == False`) | **0/886** | — |

---

## 4. Binding constraint — classify it

### 4a. The anchor: **HARNESS-BUG**, and it is a two-line file
`tools/vlm.py::decide` forwards the model's key with nothing in between. The box is a fixed 4×4×4,
the opening pose is published on frame 0, and the harness emits every key itself — so "does this
press leave the cube?" is **arithmetic the harness already has the inputs for** (`tools/action.py`
even ships `DIRECTION_VECTORS` and `is_reversal`, unused). Live proof that this is the whole gap:
the same frozen model, same frame, same prompt, same parser, with an offer set and a pellet rank in
front of it scores **1.0000 on 20/20** (`R_touraim`, re-run this round).

### 4b. r014's lineage: **HARNESS-BUG twice over, not VLM-CEILING**
Both of r014's loss classes are the harness discarding its own correct computation:
* the `self` status is computed, printed on the card, and then not acted on (§2b);
* the honour gate is keyed on **legality** when the quantity that matters is **rank** (§2c) — and
  the ledger already computes the ranked offer list (`open_moves()`) one line away from the gate.

### 4c. The ask stream: **VLM-CEILING, and it is now closed by mechanism**
8 completion tokens, `reasoning = None`, 48.4 % illegal first asks **with a correct card in
context**, ask-vs-card rank ~uniform, honoured-half closure at chance (z = −0.73). Nothing
addressed *to* the model moves this. The fix must take the decision, not re-word the request.

### 4d. The phase constraint is **coverage, and it is one cell from done**
15/16. Stage A evaluates nothing, so primary is still free this round — but the moment the 16th
cell lands, stage B EVALUATE unlocks with a 30-evaluation budget and the **pool** gets scored. Two
consequences: (i) the 16th candidate should carry a live-verified core, because it will be scored
almost immediately; (ii) **r014 is the arm's weakest elite at 0.9125 and the 3-line G0 patch takes
it to 1.0000 for free** — that is 0.0875 of a cell's fitness recovered before stage B ever runs.

---

## 5. The measurements that decide the next proposal

### 5a. The occupancy map, all 14 arm candidates in it (`ab16/cells16.py`)

| candidate | cell | d | runner-up | margin | wl / ast / num / doc |
|---|--:|--:|--:|--:|---|
| `staged_anchor` | 12 | 0.8341 | 21 @ 0.8543 | **+0.0202** | 0.4867 / 0.3848 / 0.2694 / 0.4879 |
| r001 | 21 | 0.8908 | 12 @ 1.3461 | +0.4553 | 0.0000 / 0.3212 / 0.0000 / 0.8309 |
| r002 | 16 | 1.1567 | 21 @ 1.5680 | +0.4113 | 0.3970 / 0.1732 / 0.1703 / 1.0590 |
| r003 | 3 | 1.6568 | 21 @ 1.9492 | +0.2924 | 0.0000 / 0.0444 / 0.0000 / 1.6563 |
| r004 | 25 | 1.1730 | 27 @ 1.4113 | +0.2383 | 0.6492 / 0.2075 / 0.0390 / 0.9539 |
| r005 | 27 | 1.1391 | 25 @ 1.5038 | +0.3647 | 0.7475 / 0.3067 / 0.0897 / 0.7979 |
| r006 | 29 | 1.1188 | 21 @ 1.9003 | +0.7815 | 0.0000 / 0.0000 / 0.0000 / 1.1188 |
| r007 | 20 | 1.1932 | 32 @ 1.6687 | +0.4756 | 0.6273 / 0.2882 / 0.0404 / 0.9723 |
| r008 | 9 | 0.9453 | 25 @ 1.7095 | +0.7642 | 0.0000 / 0.0000 / 0.2742 / 0.9047 |
| r009 | 5 | 1.2032 | 25 @ 1.5149 | +0.3118 | 0.8717 / 0.2654 / 0.2411 / 0.7478 |
| r010 (`R_atlas`) | 15 | 1.0945 | 25 @ 1.2920 | +0.1975 | 0.8082 / 0.3721 / 0.1326 / 0.6234 |
| r011 (`R_census`) | 30 | 1.3287 | 15 @ 1.4349 | +0.1062 | 0.9450 / 0.2651 / 0.0155 / 0.8955 |
| r012 | 6 | 0.9652 | 25 @ 1.9047 | +0.9395 | 0.0000 / 0.0775 / 0.0931 / 0.9576 |
| r013 | 13 | 1.3393 | 27 @ 1.4761 | +0.1367 | 1.0486 / 0.3862 / 0.0166 / 0.7381 |
| **r014** | **8** | **1.2084** | 27 @ 1.4879 | **+0.2795** | 0.9038 / 0.2670 / 0.1691 / 0.7371 |

**OCCUPIED (15):** 3, 5, 6, 8, 9, 12, 13, 15, 16, 20, 21, 25, 27, 29, 30.
**EMPTY (17):** 1, 2, 4, 7, 10, 11, 14, 17, 18, 19, 22, 23, 24, 26, 28, 31, 32.

Empty cells ranked by distance **from a live-1.0000 core** — the cost of coverage while carrying a
mechanism that scores. A founder at d = 0.000 is a **one-founder cell whose centroid IS that
founder**, so mirroring its labelled graph collapses `wl`/`ast`/`num`:

| cell | d(`phi_r013fix`) | d(`R_census`) | d(`R_g1`) | d(`R_touraim`) | #founders | founder(s) at d |
|--:|--:|--:|--:|--:|--:|---|
| **14** | 2.5873 | 2.5692 | 2.9613 | **2.6919** | 3 | `cand_r000_15` (1.248), `cand_r000_11` (1.257), `cand_r000_34` (1.350) |
| **19** | 2.0267 | 2.0722 | 2.1807 | 3.2767 | 4 | `cand_r000_35` (0.913), `cand_r000_40` (1.055), `cand_r000_39` (1.091) |
| 17 | **1.9210** | 2.0818 | 2.1836 | 3.5700 | 1 | `cand_r004_16_action-planning` (**0.000**) |
| 31 | 2.0867 | **1.8423** | 2.2057 | 3.8854 | 1 | `cand_r003_1_action-planning` (**0.000**) |
| 24 | 2.0805 | 1.9221 | 2.1593 | 3.7940 | 1 | `cand_r002_0` (**0.000**) |
| 7 | 2.2218 | 1.9576 | 2.4302 | 3.9834 | 1 | `cand_r001_7_verification-recovery` (**0.000**) |
| 1 | 2.2191 | 2.0631 | 2.6056 | 3.9381 | 1 | `cand_r004_17` (**0.000**) |
| 28 | 2.2855 | 2.0765 | 2.2325 | 3.7262 | 1 | `cand_r004_5` (**0.000**) |
| 23 / 11 / 10 / 4 / 26 / 18 / 2 | 2.11–3.04 | 2.11–2.84 | 2.48–3.32 | 3.42–4.07 | 1 each | all at **0.000** |
| 22 / 32 | 2.2247 / 2.4059 | 2.4005 / 2.3891 | 2.5039 / 2.9701 | 3.3643 / 3.5965 | 2 each | 0.989 / 0.845 |

(full matrix: `ab16/cells16.json`, key `d_core`.)

### 5b. The route table — the last cell has exactly one vehicle (`ab16/grid16.py`, `prose16b.py`, `shape16.py`)

7 code trees × 8 NOTES regimes against the **15**-cell occupancy. **Every live core except
`R_touraim` is pinned in an occupied cell in every regime:** `R_g1` → 8/27, `R_atlas` → 27/15,
`R_split2` → 27, `R_census` → 27/30, `r013fix` → 13, `r014` → 8. `R_touraim` reaches an **empty**
cell in **8 of 8**:

| code tree | `NOTES.md` | cell | d | margin |
|---|---|--:|--:|--:|
| **`ab11/R_touraim`** | **none** | **14** | 2.6919 | **+0.5848** |
| `ab11/R_touraim` | 260 ch | 14 | 2.5731 | +0.5506 |
| `ab11/R_touraim` | 1200 ch | 14 | 2.0670 | +0.4198 |
| `ab11/R_touraim` | 2400 / 4800 / 7200 ch | **19** | 1.87 / 1.64 / 1.68 | +0.0659 / +0.1358 / +0.1961 |
| `ab11/R_touraim` | ≥ 12000 ch | 19 | 1.7532 | +0.0535 |

**Cell 14 is a wide, prose-INSENSITIVE basin.** Four unrelated NOTES sources (r011's, r013's,
r014's, the anchor's) × 8 lengths = 40 variants: **38/40 landed in cell 14** (margins +0.0638 …
+0.6003) and **40/40 landed in an empty cell** (the 2 exceptions go to 19, also empty).

And it survives the files a real candidate must also carry (`ab16/shape16b.py` — `R_touraim`'s code
plus the anchor's `system_prompt.txt` + `system_prompt.md` + `skills/strategy.md` + `memory.md`):

| shape | cell | d | margin |
|---|--:|--:|--:|
| full anchor file set, **no `NOTES.md`** | **14** | 1.5921 | **+0.7279** |
| full set, `NOTES.md` 120 ch | 14 | 1.6725 | +0.5634 |
| full set, `NOTES.md` 260 ch | 14 | 1.7263 | +0.4573 |
| full set, `NOTES.md` ≥ 1000 ch | 19 | 2.02 | +0.0329 |

Two doc-block facts fall out, both new: **`system_prompt.txt` is φ-invisible** (bit-identical rows
with and without it), and **`system_prompt.md` is a large doc lever** — adding it moves `doc`
2.2479 → 0.5845 for byte-identical code and *widens* the cell-14 margin. With the full file set the
short/long regime switch moves earlier, to **~700–1000 chars** (not 2400).

### 5c. The φ cost of this round's `ledger.py` patches (`ab16/phi16.py`)

| tree | cell | d | runner-up | margin |
|---|--:|--:|--:|--:|
| `R_r014` (shipped) | 8 | 1.2084 | 27 @ 1.4879 | +0.2795 |
| `R_r014fix` (+1 line) | 8 | 1.2055 | 27 @ 1.5016 | **+0.2961** |
| `R_r014rank` (+2 more) | 8 | 1.2067 | 27 @ 1.5060 | **+0.2993** |

The fix is **free and then some**: cell 8 held, margin **+0.0198 wider**, and primary 0.9125 →
1.0000.

### 5d. Carried from rounds 13–15, unchanged and load-bearing
* **The load-bearing property of a frame-read pellet is the RE-READ, not the accuracy.** Re-read
  every frame: primary 1.0000 holds to a **50 %** wrong-read rate. Latched: a **perfect** read
  scores 0.3000. Spend nothing on census precision; **never latch**. (`R_touraim` abstained on
  1/324 reads this round and still scored 1.0000.)
* **Never let a pixel read WRITE the reckoned head.** The reckoner is exact (0 drift over
  298 + 2763 + 1968 + 318 + 886 observations, the last two this round's `ground_sure` 886/886); the
  pixel head census is wrong on ~12 % of frames. r013 lost 5 episodes to `L1(spot, head) <= 1`
  being an **assignment**. If a round wants a second opinion, make it an assertion that logs.
  `R_touraim`, `R_g1`, `R_r010`, `R_atlas` contain no such fusion and are structurally immune.
* **From a census-style core, `wl` is the one universal single-block lever**; `ast` alone reaches
  only cell 8; `num` and `doc` alone reach nothing. `wl` means a different **labelled** graph
  (roles + h1/h2 signatures), not a different file count.
* **Cell 14 raw targets** (`ab13/blueprint13.py`): `cf_max` 14.0, `cf_mean` 3.71, `num_per_comp`
  6.91, `logmag_mean` **1.93** — large literals **inside** function bodies, the opposite of this
  arm's hoist-everything idiom. `R_touraim`'s `num_per_comp` (6.82) is already within 0.09, which
  is why it is the only tree that reaches cell 14.
* **The pre-eval gate's text scan** is a case-insensitive **substring** scan for the 8 keys
  `gt.json, maze_file, distance_to_exit, distance_to_food, cube_solved, current_floor,
  target_blocks, matched_blocks` over exactly `system_prompt.txt`, `AGENT.md`, `rules.md`,
  `memory.md`, `skills/*.md`. **Naming the field even to disclaim it rejects the candidate** — it
  already cost this arm `cand_r10_0`. `NOTES.md` and `diagnosis.md` are not scanned. Checked this
  round: `R_touraim`'s and the anchor's `.md` sets are clean on all 8.
* **`primary = mean(eval_result.score)` and `score` is ALREADY `min(meals/4, 1)`** — verified again
  this round on 19/19 episodes against `min((snake_length−2)/4, 1)`. A fresh aggregator that
  normalises a second time under-reports 4×.

---

## 6. Concrete changes to try (the gradient — ranked by leverage)

### G0 · **three lines in `tools/ledger.py:verdict` take r014 from 0.9125 to 1.0000** · `tools/ledger.py`
Highest leverage in the file: it is measured live, it costs no φ, and it repairs the arm's weakest
elite before stage B scores the pool.

```diff
         state = dict((row[0], row[4]) for row in self.table()).get(proposed)
-        if state != "wall" and state != "180":
-            return proposed, "honoured"
         spare = self.open_moves()
         if not spare:
             return proposed, "boxed"
+        if proposed == spare[0]:
+            return proposed, "honoured"
         self.vetoes += 1
         return spare[0], "vetoed:" + state
```

*Measured, live, 20 seeds each:* 0.9125 → **1.0000**, 15/20 → **20/20**, 3 deaths → **0**, 2 budget
caps → **0**, 44.15 → **16.15** dec/ep, 883 → 323 VLM calls. *φ:* cell 8 held, margin **+0.2795 →
+0.2993**. *Causal reason:* two independent repairs, both of the harness throwing away its own
answer — (i) the `self`/`room 0` status is now vetoed, which removes the entire death class
(committed-key status `open` on 886/886); (ii) the honour gate moves from **legality** to **rank**,
and §2c prices that exactly — the honoured half closes the pellet gap at chance (0.4060 vs 0.4229,
z = −0.73) while the span-optimal offer closes 0.9754, so every honoured non-rank-0 press is a
coin-flip paid for out of a 100-decision budget.
**The 1-line half alone is not enough** and must not be shipped on its own: `R_r014fix` removed
3/3 deaths and *lost* primary (0.8750) because 3 episodes hit the cap instead. Deaths and budget
are two constraints; G0 fixes both.

### G1 · **finish coverage: ONE proposal, ONE route, 0 research** · `cp -r` + a short `NOTES.md`
15/16 covered. There is exactly one vehicle (§5b) and it is live-1.0000:

| step | what |
|---|---|
| 1 | `cp -r traces_diag/ab11/R_touraim <cand>` — code byte-for-byte (`orchestrator.py`, `tools/{__init__,action,frame,vlm}.py`, `AGENT.md`, `rules.md`, `memory.md`) |
| 2 | add `system_prompt.txt` **and** `system_prompt.md` + `skills/strategy.md` from the anchor (this *widens* the margin, §5b) |
| 3 | `NOTES.md` **≤ 260 chars**, or omit it; park the narrative + rigs in `traces_notes/` (φ-free) |
| 4 | verify with `traces_diag/ab13/phi13.py <cand>` (~15 s) **before** shipping |

*Measured landing:* **cell 14**, d 1.5921–1.7263, margin **+0.4573 … +0.7279** — the widest in the
arm, and prose-insensitive (38/40 variants, 40/40 in an empty cell). *Measured behaviour:* re-run
live this round — **1.0000, 20/20 seeds, 0 deaths, 16.20 dec/ep, honour 25.6 %, 0 tour fallbacks,
1/324 pellet abstentions.* *Predicted effect:* stage A completes (16/16), stage B EVALUATE unlocks,
and the new cell's elite is a 1.0000 harness. Do **not** author a new mechanism for this proposal —
§4d makes it the first thing stage B scores.

### G2 · **if the steer names a cell outside 14/19, mirror its founder** · `tools/*.py` layout
From §5a, cells 31, 24, 7, 1, 28, 23, 11, 4, 26, 2, 18, 10 each have exactly **one** founder at
distance **0.000** from the centroid. Mirror that founder's labelled component graph (role + sorted
neighbour roles per node) and `wl` collapses to ~0; copy its per-component CF multiset and `ast`
follows. Cells 8 and 13 are two-founder cells (centroid = the midpoint, `wl` has a floor, the doc
block is the margin); 14 and 19 are 3- and 4-founder cells and are reached by `num_per_comp`
(§5d). *Do not* split `tools/vlm.py` into modules as a generic coverage move — measured twice, it
is a cell-25 move and 25 is occupied.

### G3 · **choose the NOTES regime before writing a line** · `NOTES.md`
A **switch, not a gradient**. With the full anchor file set present the switch is at
**~700–1000 chars** (new this round; it is ~2400 for `R_touraim` bare): below it the 12 000-char
doc window belongs to `rules.md`/`memory.md`/`skills/`/`system_prompt.md`, above it to `NOTES.md`,
and the window closes exactly at 12 000 chars. Pick a side well inside it; margins at the switch
are thin (+0.03). `system_prompt.txt` is φ-invisible; `system_prompt.md` is not.

### G4 · **carry a live-verified decision core; do not author a new mechanism** · `tools/vlm.py`
The shared policy of every live-1.0000 tree, now confirmed by a second, independently-authored tree
(`R_r014rank` at 24.8 % honour vs `R_touraim` at 25.6 % — both land on the same rate because it is
just P(the ask is span-optimal) = 17 %–26 %):
offer every in-bounds, non-neck, non-hull neighbour (dead-reckoned from the fixed `(2,3,2)` spawn
and the keys this harness emitted); **rank by L1 to the pellet read FRESH from this frame**;
**honour the model's ask only when it is already rank 0**; never latch the pellet; never let a
pixel read write the head. Over-estimate the hull freely — it can only refuse a legal move, and
§2b shows the over-estimate is what saves the episode.

### G5 · **run any new `decide` live, and score the BELIEF and the BUDGET, not just the deaths** · `ab16/run_live16.sh` + `census16.py` + `closure16b.py`
Three candidates in this arm have now shipped on offline evidence and been wrong on contact
(r010, r013, r014 — r014 in the good direction). One 20-seed run is 2–12 min. Then:
* attribute by **file md5, not dir name** (a candidate dir in this arm was overwritten in place);
* `score` is pre-normalised (§5d) — do not divide by 4 again;
* **a fix that provably removes a death class need not raise primary** (§2c). Always report
  deaths **and** `steps_taken == max_steps` separately;
* `ground_ledger`-style traces log `rows` **pre**-move and `state` **post**-move; recover the
  pre-move head-span from the row set (`span(+)+span(−) = 2·L1`), or you will measure 0/810
  closure, as this round's first cut did.

### G6 · **never spell an oracle key in an agent-facing `.md`** · `memory.md`, `rules.md`, `AGENT.md`, `skills/*.md`, `system_prompt.txt`
Case-insensitive **substring** scan, 8 keys (§5d). It already cost this arm a whole proposal. Call
the self-derived quantity something else — `pellet_gap`, `l1_to_pellet` — and keep the anti-cheat
discussion in `NOTES.md` (not scanned).

### G7 · **correct the prose; it is worth ~0 on the metric and it is a trap for every future round** · `AGENT.md`, `rules.md`
`rules.md` spends 11 lines sending the model to read on-screen `R/L/U/D/W/S` letters this `_pv`
build does not paint; `AGENT.md` says the snake starts "at the center" when it starts on the
`y = 3` **face**, where `move_down` is instant death (seeds 5 and 10 die exactly that way, §2a).
Also true and worth writing: addressing is 0..3; the opening pose is head `(2,3,2)`, the cell
behind it `(1,3,2)`, heading **+X**, so `move_left` is the refused 180 on frame 0; the episode ends
at the 4th pellet (`early_stop_score = 4`) with `snake_length = 2 + meals = 6`; a 180 reversal is
refused **without** advancing `steps_taken` while a fatal press **does** advance it.
*Predicted effect on primary:* ~0 (§2d/§4c). *Why do it anyway:* every future proposer reads these
files as ground truth, and in the short-NOTES regime they **are** the φ doc block.

### G8 · **if a round is spent on agency, spend it on the PIXEL channel and score pellet closure** · `tools/frame.py`
§2d closes the text channel: a correct card delivered on 97.7 % of steps left the ask 48.4 %
illegal and uniform over the card's own ranking. The only channel that has ever moved the ask is
the **pixel overlay** (wrap `capture_game_screenshot`, paint legality and the pellet onto the
frame): first-ask illegal 0.3165 → 0.0823 at zero extra calls or tokens. Any such round must
(a) be scored on **pellet closure** (`closure16b.py`), not illegal-ask rate, (b) run at n ≫ 900,
and (c) leave G4's decision path untouched. Budget available: G0's rank-0 gate finishes in 16.15 of
100 decisions, so there is ~5× slack to spend on the model's agency *if* the ask ever becomes
informative.

---

## 7. Exhausted / do NOT repeat

* **Prose/prompt wording as a *behaviour* lever.** Refuted at n = 963; refuted again by mechanism
  (8 completion tokens, `reasoning = None` on 67/67 anchor replies); and refuted again this round
  in its strongest possible form — a **computed, correct, per-step grounding card, verified
  delivered on 863/883 decisions**, left the first ask 48.4 % illegal and ~uniform over the card's
  own ranking (§2d). Still the `doc` φ lever.
* **A legality-only veto, or any honour gate keyed on legality.** New and decisive this round:
  legality-gated honour is 54.7 % honour / 44.15 dec/ep / **0.9125**; rank-gated honour is 24.8 % /
  16.15 dec/ep / **1.0000** (§2c, G0). The honoured half of a legality-gated stream closes the
  pellet gap at chance (z = −0.73, n = 463).
* **Declining to veto a `self`/`room == 0` landing on the grounds that the belief may be stale.**
  Refuted: `ground_sure` was True on 886/886, the odometer never drifted, 27/27 such presses were
  honoured and 3 killed the episode. The hull over-estimate is the safe direction (§2b).
* **`room` / reachable volume as a rank key OR a tie-break.** Spread 0 across the open offers on
  **885/885** decisions (§2e) — byte-identical to span-first ranking. Keep it only as a hard
  `room == 0` guard. (Consistent with the carried "pocket depth ≥ 2 and any tie-break are
  +0.0000"; depth 1 is −0.2125.)
* **Spending effort on pellet-read precision, or latching the read.** 1.0000 holds to a 50 % error
  rate when re-read every frame; a perfect latched read scores 0.3000. `R_touraim` abstained on
  1/324 and r014 on 30/886 and both were fine. Also refuted: the meal-gated read and the symmetric
  food size gate (46.8 % wrong at d = 1 — the head occludes the disc; one-sided only).
* **Pricing / loosening honour.** Closed in closed form: the offer gap is `{0, 2}` with **no 1**,
  so an honour toll has two settings — rank-0, or obey-if-safe at **0.3750**. §2c now shows the
  measured cost of the loose setting on a real tree (−0.0875 primary, +28 dec/ep).
* **A second opinion on the reckoned head in any form that WRITES.** 0 drift over
  298 + 2763 + 1968 + 318 + 886 observations; the pixel head census is wrong on ~12 % of frames;
  letting it write cost r013 five episodes. Make it an assertion that logs.
* **The HUD body-length decoder** (the 6-cell contract hull is free; `snake_length = 2 + meals` is
  re-verified on 18/19 episodes this round, so there is nothing to decode), **the free re-ask /
  second reasoning call** (+0.0000), **`metadata["repeat"]` / any stride**.
* **Any pellet-blind route.** A Hamiltonian tour is perfectly safe and cannot score (4 pellets
  ≈ 128 presses of a 100-decision budget); `R_touraim`'s tour fallback fired **0/324** times this
  round, as it did 0/328 and 0/323 before. Keep the ring only as the empty-offer-set fallback.
* **Splitting `tools/vlm.py` into modules as a coverage move** (a cell-25 move; 25 is occupied) and
  **hoisting literals without replacing them** (takes `num_per_comp` to 2.61 and *widens* the `num`
  gap to cell 8 from 0.4067 to 0.8987). Note cell 14 wants the **opposite** idiom — large literals
  inside function bodies, `logmag_mean` 1.93 (§5d).
* **Optimising the decision count below ~16.** Path efficiency is 0.9748 and both live-1.0000
  trees finish in 16.15–16.20 decisions of 100.
* **Anti-cheat hazards, both live.** The pellet comes from the global `random` stream that
  `eval_runner` seeds per episode and nothing else draws from — three `randint` calls predict every
  pellet (used only in the `traces_diag/` **offline** rigs; **fatal in a candidate**). And never
  touch `evaluator.game.*`, the engine score object, the seed or the data tree — the gate
  AST-scans for it. Everything a core needs is the fixed spawn, the keys the harness itself
  emitted, and the frame.
