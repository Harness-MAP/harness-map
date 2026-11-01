# maze3d_pro rich harness — module map & levers

Rich harness for the SpatialWorld multi-floor 3D maze (`maze3d_pro`). The frozen agent
(`Qwen3-VL-30B-A3B-Instruct`) and the SpatialWorld engine are untouched; only this directory is
optimised. `build_agent(evaluator)` is the entrypoint the extended evaluator installs as
`ai_model_func`; the engine `Action` contract, the legal action set, the JSON response format and the
maze `success` scoring all stay byte-compatible with the prompt-only baseline.

The parent is the **staged anchor** (`anchor_maze3d_pro`, 2/25 = 0.0800), a deliberate pass-through:
`orchestrator.py` composed the prompt from the `.md` priors and `tools/vlm.py:decide` forwarded every step
to `evaluator.baseline_step` unchanged. It held **no state at all**, so the one bit that separates a
move that happened from a move that was refused never reached anything, and **1829 of its 2340
decisions (78.2%) re-pressed a wall that same episode had already proven at the same cell and the same
bearing.** `tools/action.py`, `tools/frame.py` and `memory.md` were in the genome and referenced by
nothing. **0 of 2340 decisions were a climb** and **0 were `move_backward`**, both legal and both
documented.

## axis

**code / action-planning — EVERY GOAL IS PRICED IN DECISIONS, INCLUDING THE STAIRCASE, AND THE MODEL
IS OBEYED WHEREVER ITS ANSWER IS FREE.**

An episode is budgeted in exactly one resource: 100 calls to the frozen model. A translation costs one
decision whether it carries one cell or three and it stops itself at a wall; a 90-degree turn costs
one decision and carries nothing. **Distance is nearly free and turning is the expensive thing** — so
every edge of the search in `tools/plan.py:edges` costs ONE DECISION, and the number goals are
compared on is the number the budget is actually denominated in.

The part that is new against every ancestor in this lineage is **what gets priced**. Those ancestors
treated a floor change as a decision of a different kind — something to gate on a latch, a stair
budget, or a "have I explored enough of this floor" test. This harness does not have that mechanism at
all. **A staircase IS a question** — the question *what is on the other floor* — and it is ranked
against every other open question by route cost and nothing else. The departure premium measures to
**zero**: `STAIR_TOLL` of 0, 1, 2 and 3 all score 125 of 125 on the stub panel, 8 and 20 score 120, so
every premium is a cost and none of them buys a level. That is the upstream departure-gate refutation
taken one step further: not only is a gate wrong, so is a price.

**The asymmetry is where the work is.** The way *back* down the staircase just climbed carries
`RETURN_TOLL` = 50, half the budget, which keeps it permanently on the board as a plannable route
while making it the last thing elected. That premium is worth **+80 of 125**: at zero the agent rides
one staircase up and down forever. "Commit on the way out, keep the way back" becomes an asymmetric
price rather than a latch.

The frozen model is asked **every step** and obeyed **wherever its answer is free**: the governor
re-plans from the pose the proposal would leave the body in and keeps the proposal when the remaining
route is no longer than the harness's own first decision would have left
(`decisions_after <= decisions_now − 1`). That bound is the whole discipline. An unbounded vote is
refuted upstream (an always-`turn_right` proposer swings the target every step and spins in place); a
harness that never defers is a hand-coded solver with a model-shaped ornament. Free-or-cheaper keeps
both out, and it is worth **+119 of 125 and +33 of 75**.

## module map

| file | role | this iteration |
|------|------|----------------|
| `orchestrator.py` | ENTRYPOINT. `build_agent(evaluator)` returns the async per-step fn and opens ONE `plan.Pilot` for the episode; `compose_system_prompt()` recomposes the agent's prompt from `AGENT.md` + `skills/*.md` + `rules.md`. | rewired to open the Pilot and read `max_steps` |
| `tools/sight.py` | **NEW.** ONE FRAME READ AS GEOMETRY: the corridor's length and kind down the centre column, the FLANK LADDER beside it, and the gate that refuses to decode a frame this renderer did not draw. | new |
| `tools/chart.py` | **NEW.** What the frames have PROVEN, per floor: three kinds of open kept strictly apart, the dead-reckoned pose, the `(cell, bearing)` refusal ledger, the frontier, the clash-wipe, and the EPOCH that settles a floor change with no floor readout to read. | new |
| `tools/plan.py` | **NEW — the axis.** `edges`/`search` (the decision-priced route), `candidates`/`elect` (goals priced in decisions), `sweep` (the forced escape), `Pilot` (observe → aim → want → govern → speak). | new |
| `tools/action.py` | the key table, the engine `Action` metadata envelope, and the bearing arithmetic the search needs (`heading_after`, `advance`, `turn_toward`, `is_undo`, `read_proposal`, `retarget`). | extended from a dormant table into the planner's arithmetic |
| `tools/vlm.py` | the per-step loop around the evaluator's pristine VLM call, plus the message-0 injection. | rewired from a two-line pass-through |
| `tools/frame.py` | frame capture via the evaluator's own screenshot. | unchanged |
| `AGENT.md` | role framing + the perception facts this build ACTUALLY has, and what each working-memory line means. | rewritten |
| `rules.md` | the legal action set, the granularity contract, the REAL climb precondition, and what a decision costs. | rewritten |
| `skills/navigation.md` | the navigation priority order, re-pointed at the measured block. | rewritten |
| `memory.md` | the per-step scratchpad TEMPLATE, rendered live by `Pilot.render_memory`. | wired (it was read by nothing) |
| `system_prompt.txt` / `.md` | redundant references. `compose_system_prompt` never reads them. | untouched — NOT a lever |

## change

Six things, all inside one mechanism.

**1. `tools/sight.py` — one frame, read as geometry.** The distance ahead is the HEIGHT of the paint
down column 400: the run that is neither ceiling `(15,15,35)` nor ground `(50,60,65)` spans
`FACE_SPAN / d` rows at `d` cells (about 401 / 201 / 133 / 101 / 81 / 67 px at 1..6), and the bands are
disjoint, so rounding to the nearest rung is exact rather than fitted. What is at that distance is the
COLOUR of that run, voted over the whole run rather than one sampled pixel: `red == 0` is the teal
stair, `blue == 0` the gold exit, anything else a brown wall. The FLANK LADDER reads the two
neighbours of the corridor cell `k` ahead out of the band that ends `SIDE_SPAN // (k+1)` off the centre
line and starts `SIDE_SPAN // k` off it, sampled at rows 298/300/302 and required to be unanimous; the
occlusion gate is `k <= depth − 1`. `Sight.blocked()` is therefore a **pre-step** fact: the anchor had
to spend a decision to learn "something is in the next cell", 1829 times.

**2. `tools/sight.py` — and it refuses to decode a frame it was not measured on.** Every constant
above is an equality against this renderer's palette, which makes a frame from anywhere else decode to
confident nonsense. `Sight.grounded` requires at least an eighth of the frame to be exactly one of the
two background colours before the reading is allowed to speak (measured on this rig's views: 0.488 to
0.946, so no real view can fail it; an all-black frame and a wrongly-shaped one both fail it). On an
unreadable frame `plan.py:honours` stands the governor down to the pass-through, so the floor of this
candidate's behaviour **is** the parent's score — verified at exactly 6/125 on both degenerate frames,
never below it.

**3. `tools/chart.py` — three kinds of open, kept apart.** A cell the ray walked THROUGH is `PLAIN`; a
cell a flank band showed empty is `PASSABLE`; `STAIR` and `EXIT` are the painted tiles. Collapsing
`PASSABLE` into `PLAIN` would destroy the frontier — one look down a corridor would mark the whole
neighbourhood explored — so the two are never merged, and `PASSABLE` is where the questions live. The
pose is reckoned, never read: a turn folds unconditionally, a translation folds by
`min(requested, pre-step reach)` and only when the frame changed (an unchanged frame files a wall —
that is the refusal bit), and a climb folds optimistically into a new epoch. A reading that contradicts
the chart at the cell straight ahead WIPES this floor's grid and re-seeds it at the believed pose; the
pose and the ledger survive, because they are the harder facts.

**4. `tools/chart.py` — the EPOCH, which settles a floor change with no floor readout.** Nothing on
screen says which floor the body is on, and a climb pressed on the top floor is simply ignored by the
engine — so a chart that counted floors optimistically opened a fresh grid for a floor that does not
exist and then re-elected the same staircase from it (measured before the fix: `climbs=4, epoch=4` on a
two-floor level). Every climb now opens an EPOCH, and on a fresh epoch's **first look only** the digest
at the very pose the climb was pressed from is compared with that pose's digest in the epoch the press
left. Byte-equal means the body never moved: the epochs are the same floor and they are merged, with
nothing lost. Restricting it to the first look is load-bearing — allowing the match at any later pose
conflates two real floors into one grid and traded 2 levels for 4.

**5. `tools/plan.py` — the route, the goals, the look-around.** `search` is a breadth-first walk over
`(cell, bearing)` at one decision per edge, carrying down the OPENING (the single action that starts
the cheapest route) so one search answers both *what does this goal cost* and *what do I press now*.
Goals are the exit (price 0 — an exit on this floor ends the episode and no floor change can improve on
it), every open question (price 0), and every unspent staircase link (price 0 out, `RETURN_TOLL` back).
`LOOK_AROUND` = 3 is the one remaining number and it is **not a tuning**: a body that has just walked
onto a stair has faced exactly one of four bearings from that cell, and three more decisions of
rotation face the other three, each of them filing a corridor the chart did not have. Only then is the
floor change priced against a frontier that is actually complete. Swept at 1/2/3/4: **105 / 115 / 125 /
99** of 125. It peaks at the compass and falls off the moment a fourth decision buys a bearing already
seen.

**6. `tools/vlm.py` — the loop and the channel.** `observe` settles the previous decision and files the
look; `inject_memory` rewrites `conversation_history[0]["content"][0]["text"]` in place as
`stashed_base + block`; `baseline_step` is called unchanged; `apply` governs the proposal. Nothing is
injected on the first call, so turn 1 is byte-identical to the baseline's.

## why (the gradient this answers)

`diagnosis.md` names a compound HARNESS-BUG and ranks the fixes. This iteration takes §5.1 (state +
ledger + forced escape), §5.2 (prose that describes the build that exists), §5.3 (ground the stair and
commit the climb), §5.4 (state the measured facts through message 0) and §5.5 (admit and ration
`move_backward`) as **one mechanism**, because on this parent they are not separable: a ledger with no
router still spins, a router with no chart has nothing to route over, and a climb with no chart has no
precondition it can check. §5.6 (clamp the stride to the measured free run) is in and is **not** behind
a switch — `Sight.stride` is the clamp, and it is what makes a `large` down a two-cell corridor
impossible to emit.

The diagnosis's own round table is why the climb is in scope at all: r001 broke the orbit and landed on
**exactly the 8-level no-stair set**, not one level more, because its 293 climb proposals changed floor
0 times in 50 episodes; the jump to 19–22/25 arrives in r004, the first round where floor changes
actually land. A harness that cannot leave floor 0 is capped at **8/25 = 0.32** before a single
navigation decision is made, because 12 levels have the exit upstairs and 5 more have floor 0
disconnected from an exit that is on it — **17 of 25 need the staircase.** This harness reproduces that
ceiling exactly: with the stair goal removed it scores **8/25 on every one of eight proposer arms.**

The STEER asked to move toward the region exemplified by `cand_r002_1_action-planning` and
`cand_r001_0_spatial-grounding` and named the productive difference as the **token-level** half of φ.
Measured against the frozen k=2 partition in `cvt_v3_k2/`, that region's centroid differs from this
parent's cell almost entirely in the TOKEN block (‖Δstructure‖ 0.620 vs ‖Δtoken‖ 1.884), and inside the
token block it is **`num` that moves** (‖Δ‖ 1.689 of 1.884: `logmag_mean` 0.58→2.06, `logmag_std`
0.61→1.96, `num_per_comp` 1.10→2.45) — a mechanism written with many constants of widely varying
magnitude, which is exactly what a pixel ruler and a decision price are made of. The anchor sat at
`num_per_comp` 0.286; this candidate sits at 5.84 with `logmag_std` 1.43, and it lands in the target
cell at **d 4.740 with margin 0.473**, against the two exemplars' own margins of 0.061 and 0.098.

## measured, on an offline rig (not on one live run)

The rig lives outside this directory (never in the genome), runs from the benchmark root, parses the 25
level files, and SYNTHESISES a frame that satisfies the decode `tools/sight.py` documents —
centre-column run height `FACE_SPAN / d` in the face colour, flank bands at
`SIDE_SPAN // k .. SIDE_SPAN // (k+1)` off the centre line sampled at rows 298/300/302. It drives the
real `Pilot` with **eight context-free stub proposers**: five degenerate ones —
`move_forward/small` forever (the anchor's own dominant failure: 4 of its episodes emitted it 100 times
byte-identically), `move_forward/large` forever, `turn_right` forever, a fixed 4-cycle, a seeded
pseudo-random proposer over all six actions — and three that get honoured — a greedy walker that reads
the stated `ahead` line, and two that answer the harness's own best decision 50% and 85% of the time.
Sampling live is `temperature=1.0, top_p=0.9` with no seed, so a panel transcribed from the losing
streams is the only way to price a stuck cure at all; a panel with no stuck arm cannot.

**The five degenerate arms are a WORST CASE.** Every proposal on them is garbage, so the harness carries
100% of the load and `overrides` runs at or near 100%. The three honoured arms are the realistic end.

| configuration | 5 degenerate arms / 125 | 3 honoured arms / 75 |
|---|---|---|
| **parent pass-through (the anchor)** | **6** | — |
| **SHIPPED** | **125** | **75** |
| governor off — always honour the model | 6 | 42 |
| no climb goal at all | 40 (= 8/25 on every arm) | 24 |
| `RETURN_TOLL` = 0 (the return competes as a fresh link) | 45 | 27 |
| `RETURN_TOLL` = 8 / 50 / 100 | 125 / 125 / 125 | 75 / 75 / 75 |
| `STAIR_TOLL` = 0 / 1 / 2 / 3 | 125 / 125 / 125 / 125 | 75 / 75 / 75 / 75 |
| `STAIR_TOLL` = 8 / 20 | 120 / 120 | 72 / 72 |
| `LOOK_AROUND` = 1 / 2 / **3** / 4 / 5 | 102 / 120 / **125** / 99 / 92 | 61 / 72 / **75** / 60 / 55 |
| no reversal (`move_backward` edge removed) | 120 | 72 |
| stride over flank-glimpsed ground allowed | 125 | 75 |
| no epoch settlement | 100 | 60 |
| epoch merge allowed at ANY pose, not just the first ‡ | 95 | — |
| HOLD the elected goal across decisions ‡ | 100 | 60 |
| HOLD it, re-fitted on top of the shipped tier ◊ | 10 | 6 |
| `sweep` ranked openness-first † | 120 | 72 |
| **frame undecodable (all-black, or wrong shape)** | **6 — exactly the parent, never below** | — |

Every row without a dagger was re-run against the SHIPPED configuration, so it is directly comparable
to the 125/75 line above. † was measured against the intermediate configuration (`STAIR_TOLL` 8,
`LOOK_AROUND` 3; baseline 120/72), before the departure premium was swept to 0. ‡ was measured earlier
still, at `LOOK_AROUND` 1 / `STAIR_TOLL` 8: the HOLD row against that config's re-electing baseline of
**105/63** — the held variant came out one level worse on every one of eight arms, and the single level
it loses is L24 — and the ANY-POSE epoch row against that config's own held-goal baseline of **100**,
so first-look-only is +5 of 125 over it. Each ablation is reported against the baseline it was run
beside; nothing in this table is a cross-configuration subtraction.

◊ is NOT a clean ablation and is listed because the failure is informative: holding the goal on top of
`LOOK_AROUND` **deadlocks**. A held stair goal is satisfied geometrically the moment the body arrives,
so the route has no opening left to return, and the three look-around decisions never advance the state
that would release the climb. Holding and the look-around are mutually exclusive mechanisms; if a future
iteration wants commitment back it has to make the look-around part of the GOAL (four poses at the stair
cell, one per bearing) rather than a delay on the press.

Per level at the shipped configuration, on the 4-cycle arm: **25/25**, in 3, 14, 5, 9, 18, 4, 37, 21,
19, 14, 40, 46, 64, 28, 14, 20, 8, 35, 36, 77, 34, 46, 42, 83, 98 decisions — mean **32.6**, max **98**
against a budget of 100, and **18 of 25 episodes end on a floor they did not start on.** `wipes = 0` on
every level.

### Read this before trusting the number
A 25/25 on a SYNTHETIC frame is not a 25/25 on the engine's frame, and the gap between them is the one
risk this candidate carries. The rig's frame is a function of the ray reading and the flank kinds
alone; the real renderer also fogs by distance and draws outlines, so (a) genuine digest collisions
after a climb are **rarer** live than here, which can only help, and (b) if any palette equality in
`tools/sight.py` is off for this build, `Sight.grounded` refuses the frame and the harness scores the
parent's 0.08 rather than anything worse. **The decode is the thing to check first in the next
diagnosis**: count how many steps report `ok = False`, and read `merges` and `wipes` off the trace
line `Pilot.notes()` emits.

## evolvable lever table

| lever | where | kind | status |
|---|---|---|---|
| `LOOK_AROUND` (3) | `tools/plan.py` | code / constant | **swept, sharp peak at the compass.** 1/2/3/4/5 → 102/120/125/99/92. Do not re-sweep; find out WHY 4 is worse (it is buying a bearing already seen, and the 3 decisions are doing real charting). |
| `STAIR_TOLL` (0) | `tools/plan.py` | code / constant | **swept, plateau 0–3.** A staircase is a question. Any premium is a cost. |
| `RETURN_TOLL` (50) | `tools/plan.py` | code / constant | **swept, plateau 8–100, cliff at 0.** The asymmetry, not the value, is what matters. |
| `Sight.GROUNDED` (0.125) | `tools/sight.py` | code / constant | the refuse-to-decode gate. Measured headroom is 4x (real views 0.488–0.946). |
| the epoch settlement | `chart.py:settle_epoch` | code / mechanism | **the next lever.** Worth +25 of 125 / +15 of 75 over having no settlement at all, and +5 of 125 over settling at any pose. A SECOND confirmation — take one more decision and test the merged grid's prediction at the new pose — is the open move. |
| the free-or-cheaper bound | `plan.py:honours` | code / mechanism | the model's vote. Widening it to "no worse" or narrowing it to "on-plan only" are both unmeasured. |
| goal tiers | `plan.py:candidates` | code / mechanism | exit / question / staircase, all at price 0. A fourth (a *re-look* at a stale `PLAIN` cell) is unexplored. |
| the frontier definition | `chart.py:questions` | code / mechanism | `PASSABLE`, or `PLAIN` with an unknown neighbour. This is what sets exploration ORDER, and order is what L25 spends 98 of its 100 decisions on — the one level with no headroom left. |
| the stride-over-glimpsed rule | `plan.py:edges` | code / mechanism | a stride may LAND on `PASSABLE` but not continue through it. Measured **win-neutral** at the shipped config (125/125 either way) after being worth levels at an earlier one — the rule is now redundant with the frontier that elects those cells as goals anyway. A candidate for removal, which would simplify `edges`. |
| `RUNGS` (6) | `tools/sight.py` | code / constant | neighbour pairs read per look. The 6th band is 7 px wide; a 7th is not readable. |
| `sweep` ranking | `tools/plan.py` | code / mechanism | novelty, then openness, then constant handedness, never the undo. Openness-first measured **win-neutral** (120/125 either way) — but the escape only fires when NO goal is reachable, which with a working chart is rare, so this lever is untested rather than flat. |
| the working-memory wording | `memory.md` | prose | five lines, rendered live. **The rig cannot price prose** — its proposers ignore text — so this needs a live eval to move. |
| the priority ladder | `skills/navigation.md` | prose | re-pointed at the measured block. Same caveat. |
| the stride algebra | `tools/action.py` | code / mechanism | `turn_toward` returns the arc length and is currently used only for its handedness. A rotation carried by `metadata["repeat"]` is NOT attempted here — that is the sibling's axis this round, deliberately left alone. |

## exhausted or refuted — do NOT spend an iteration here

* **The response contract.** 2340/2340 legal and `default_actions = 0` in 25/25 at the anchor, and
  unchanged downstream. `tools/vlm.py` still delegates the call and the parse to `baseline_step`.
* **Prompt-only re-wording as the cure for the orbit.** `skills/navigation.md` already said *"avoid
  turning repeatedly in the same spot"* and 23/23 losses were absorbed anyway; adherence to an
  explicitly stated proven-refused line measures 58–61% upstream and did not move under a full
  rewrite. Mechanisms, not sentences.
* **Asking for reasoning or a scratchpad field.** This is an *Instruct* build: `reasoning=None` in
  2340/2340 responses, median 15 and max 19 completion tokens.
* **A frame-diff tolerance.** Exact equality is correct; 0.2% manufactures false walls.
* **A VIEW-keyed ledger.** 257 distinct digests over 5624 distinct poses. The ledger is keyed on the
  dead-reckoned CELL, which is a function of the maze and so reconverges after a divergence.
* **Frame identity as a general climb-success test.** 46 of 240 successful climbs (19.2%) are
  frame-identical. The epoch settlement uses the equality in ONE place only — the first look, at the
  pose the press was made from — and never as proof that a climb failed in general.
* **Any "how much of this floor is explored" departure gate** — and now, any departure PRICE either.
  The gate is refuted upstream as a cliff (caps of 1–2 win L13+L20 and cost 26–29 episodes elsewhere,
  ≥5 is inert, no interior value exists); the price is refuted here (0–3 are all 125/125, 8 and 20 are
  120/125).
* **A latch, a stair budget, or a patience counter before a climb.** None of them exist in this
  harness. `LOOK_AROUND` is not a patience gate — it is the three decisions that finish the compass.
* **Re-offering a traversed stair link.** Keyed out: one up and one down per staircase, ever, keyed on
  the CELL and never on a believed floor number.
* **An unbounded "model casts the deciding vote" tiebreak.** An always-`turn_right` proposer spins in
  place (11/25 vs 22/25 upstream). `honours` bounds the vote to the free-or-cheaper subset.
* **HOLDING a goal across decisions.** Built, measured, removed: one level worse on every one of eight
  arms. Price is self-stabilising in a way a bearing is not — moving toward the cheapest goal makes
  that goal cheaper — and a held goal cannot abandon a staircase that a look from halfway there has
  already made pointless.
* **`move_backward` as a dead-end FALLBACK.** The pair backward-then-forward is itself a two-cycle and
  it was worth a quarter of the episodes it ran in upstream. `sweep` is always a TURN. The reversal is
  a planned route edge over ray-proven ground only, and removing that edge is −5 of 125 / −3 of 75.
* **An exit-line-of-sight commit lever.** 2 gold-painted frames in 4236 at the anchor, both adjacent to
  its one real win. The rung is in `want()` because it is free, not because it is worth levels.
* **`distance_to_*`-style progress metrics.** `success` is the score. The anchor's "1 cell away" on two
  levels is an artefact of never having moved off a start cell that happens to sit near the exit.

## traps that have each cost a round somewhere in this tree

* **The pre-eval gate text-scans the agent-facing priors case-insensitively** (`system_prompt.txt`,
  `AGENT.md`, `rules.md`, `memory.md`, `skills/*.md`) and AST-scans `orchestrator.py` + `tools/*.py`, for a
  list of privileged names used as an attribute or a string subscript. Never let those spellings into
  prose or code — not even in a comment discussing them. `NOTES.md` and `diagnosis.md` are
  proposer-facing and are not scanned.
* **Decide from `frame_data.image` only.** Under `ANTICHEAT_GUARD=1` the evaluator's game object allows
  `.screen` and nothing else; any other read or write — even one swallowed by `getattr(..., default)`
  or `hasattr` — voids the score to 0.0. Nothing in this harness touches it; the only evaluator
  attributes read are `baseline_step`, `openai_ai_model`, `conversation_history`, `current_prompt` and
  `max_steps`.
* **A harness with 0 `baseline_step` calls is VOIDED** as a hand-coded solver. `tools/vlm.py:decide`
  calls it on every step, unconditionally, before the governor runs.
* **`build_agent` purges `sys.modules["tools*"]`**, so any lever override set *before* the call is
  thrown away. An offline rig must set levers as module/class attributes AFTER `build_agent`.
* **`evaluator.game` does not exist yet inside `build_agent`** — `run_evaluation` builds it after the
  call returns. The Pilot is constructed from `max_steps` and the `memory.md` text and nothing else.
* **`build_full_prompt` returns `current_prompt` unchanged on every non-first call** and
  `build_conversation_history` reuses `previous_history[0]` **by reference**, so re-rendering
  `evaluator.current_prompt` after step 1 reaches nobody. Message 0 is the only per-turn channel, it is
  written as `stashed_base + block`, and it is never truncated at a marker — cutting it deletes the JSON
  response contract from every later turn.
* **A re-targeted TURN must have `repeat` stripped.** The input source forces `repeat = 1` only when a
  `granularity` is present, so a stale `repeat: 3` on a turn fires 270 degrees. `action.retarget` drops
  `granularity`, `repeat` and `json_data` before rebuilding, and `read_proposal` reads the granularity
  from `metadata` first and the parsed `json_data` second, in the input source's own order.
* **`memory.md` is run through `str.format`.** A stray `{` or `}` anywhere in it makes the render throw;
  `plan.clean_template` falls back to the built-in template when the placeholders are gone, and
  `render_memory` swallows a format error into an empty block rather than costing the step.
* **The evaluator resolves its level file relative to CWD**, so an offline rig must run from the
  benchmark root and must live outside the candidate directory.

## files

`orchestrator.py`, `tools/__init__.py`, `tools/sight.py` (new), `tools/chart.py` (new), `tools/plan.py`
(new), `tools/action.py`, `tools/vlm.py`, `AGENT.md`, `rules.md`, `skills/navigation.md`, `memory.md`,
`NOTES.md`. Untouched: `tools/frame.py`, `system_prompt.txt`, `system_prompt.md`.
