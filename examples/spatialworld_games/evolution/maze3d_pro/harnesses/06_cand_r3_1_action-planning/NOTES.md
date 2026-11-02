# cand_r003_1_action-planning — the frame as GEOMETRY: a rangefinder, a flank strip, and a supplied stride

## axis

**code / perception → `tools/sight.py` (`Scene`), consumed by `tools/plan.py` (`ActionPlanner`).**
One new reading of the same pixels — the renderer inverted into *how far can I walk* and *which way
is open* — and the governor rewired to spend decisions against it. `build_agent(evaluator)`,
`run_episode`'s contract, the legal action set and the response format are untouched. The three
prose priors are edited as the measured co-requisite of the stride (see §change, item 5), not as the
axis.

## change

1. **A rangefinder.** `Scene.rangefind` reads the contiguous non-background run through the frame's
   centre column. The renderer draws the front face of the cell *d* cells ahead as a rect of size
   (600/d, 400/d) centred on the frame, far-to-near, and no flank polygon ever reaches the centre
   column — so that run *is* one front face, its vertical extent is `2*floor((400/d)/2)+1` and its
   colour names the family. The ladder is **generated from the frame's own shape**, not carried as a
   table of pixel counts.
2. **A flank strip.** `Scene.flank_strip` reads one pixel per side band: the side face of the cell
   *k* ahead occupies `x ∈ (cx - 300/k, cx - 300/(k+1))`, which nothing can overdraw while the front
   face stands farther than *k*. `k = 0` is the agent's **own left and right neighbour** — the two
   bearings a turn would actually take it to.
3. **The governor spends decisions against those two facts.** `blocked_ahead()` (`reach == 0`) vetoes
   a forward move *before* it is spent instead of after the frame comes back identical; `reaim()` is
   no longer a probe-then-rotate wall follower but turns straight to a side the frame shows open (a
   dead end therefore costs two rotations and the came-from bearing is left for last *by
   construction*, since the body arrived facing forward); `screen_changed` is now **exact** (the
   0.2 % tolerance that manufactured L20's absorbing state is deleted); and the climb price in
   `StairSense` applies to the **model's** climb as well as the harness's, which it did not before.
4. **The harness supplies the stride.** `supply()`/`granularity()` replace the old one-sided
   `_trim`: the granularity of every `move_forward` is the measured clear run, capped to land **on**
   the exit, to stop **one cell short** of a stair (the arrival latch needs the stair to have been
   one cell ahead, and the engine walks straight through a stair tile), and to stop **at** a side
   branch rather than past it. The action NAME is always still the model's.
5. **Prose co-requisite (the parent's §5.1).** The four *use `small`* recommendations the previous
   iteration introduced — measured as the cause of 94 % `small` forward proposals against the
   grandparent's 64 %, and of 1.02 cells per forward against 1.54 — are deleted, and the two exact
   lines (`DEAD AHEAD`, `OPEN BEARINGS`) are stated instead, with the stride declared as the
   harness's to choose. `memory.md` gains those two placeholders.
6. **Two climb clauses the LIVE runs forced** (found by paired live repeats, §live below, not by
   the scripted panel — which is neutral on both, 90 → 89 of 175):
   * the **two climbs are now priced differently on purpose.** The harness's unasked climb still
     needs the floor to be both PAID (`floor_age ≥ FLOOR_PATIENCE`) and STALE; the model's own climb
     needs only the clock (`StairSense.patience_owed`). A faster agent re-sees fewer views, so
     `stale` reaches 1 less often — and with both climbs gated on it an episode can stand on the
     only stair of an upstairs-exit level and never leave.
   * **`holding_stair()`: do not step off a stair the floor has already paid for.** A translation is
     refused and `reaim` turns in place instead, so the body stays on the tile while the remaining
     bearings finish exhausting the floor and the grounded climb fires on the next decision. Both
     `refusal` and `reaim` read the same predicate — vetoing the model's step is useless if the
     re-aim then steps off anyway, which is the hole the first version of this clause had.
7. **Token-level implementation (the search-space steer).** Every module-level helper is folded into
   the class that owned its state, and every constant table with it: `sight.py` 9 components → 2,
   `plan.py` 8 → 1, so the harness goes 29 → 15 components of much denser control flow. See §steer.

## why

The parent's `diagnosis.md` classifies the binding constraint as a **TASK-GAP in travel throughput**:
all 16 losses end at exactly 100/100 decisions, the win set spends 2.3–5.7 decisions per
shortest-path cell, 5 losses need routes of 32–54 cells, 36 % of decisions are turns and 83
translations were spent discovering by execution what the picture already showed. Its two highest-
leverage items (§5.1 the stride, §5.2 the exact outcome test) and its structural fallback (§5.6
*read the corridor, not just the tiles*) are all the same lever seen from three sides: **the frame
carries the local geometry and the harness was not reading it.** So read it, and let the harness pay
for distance with the decisions it has.

## verified

### the instrument
A measurement rig (`scratchpad/rig/`, outside the candidate) drives the **real**
`Maze3DProOpenAIEvaluator` with a scripted policy and scores the candidate's **own** `Scene`
against the engine's `tower_data`, 1820 decisions over all 25 levels:

| reading | result |
|---|---|
| front-face family from the centre pixel | **1820 / 1820** (1732 wall, 72 stair, 16 exit) |
| front-face distance from the extent | **1820 / 1820**, 0 off-by-one, 0 unreadable |
| `reach` never exceeds the true walkable run | **0 violations / 1820** |
| `reach == 0` ⇔ a forward move cannot move the body | **1820 / 1820** |
| left/right neighbour (k = 0) | **3640 / 3640** |
| side cells at k = 1, 2, 3 while in sight | **6178 / 6178** |
| the engine traverses exactly the supplied stride | **1820 / 1820** (1076 of 1076 cells) |
| stair landings, all from one cell away | **18 / 18** |

The ladder (extent → cells) is 401·201·133·101·81·67·57·51·45 → 1…9, with a **+2** on a stair or
exit face from its 3 px border; those nine values are the only extents that occur and each belongs
to one distance. Wall faces corroborate by fog colour: (73,36,27)/(67,33,25)/(60,30,22)/(54,27,20)/
(48,24,18) at 1…5 cells. Flank colours at k=0: wall (147,78,55), open = ground (50,60,65), stair
(0,92,110).

### the stride, priced cap by cap
Mean cells per forward over the same frames: `min(3, reach)` **2.03** → with the stair/exit cap
**1.97** → with the branch cap **1.90**. The parent's live tape gained **1.02**. So the two safety
caps cost 0.13 cells and the lever is worth ≈ +0.9 cells per forward decision.

### the panel — 7 arms × 25 levels, parent vs candidate
The replay rig is **vacuous for this change** (once the executed stream diverges the model's
recorded answers no longer belong to the frames), so the model is held **constant and scripted**
instead and the harness is the only difference. Arm `replay` is the parent's calibration point and
reproduces its live 9/25 with the identical win set. Every arm is deterministic and was confirmed
reproducible (two runs of the `mix` arm, different processes, identical 11/25 and identical win
set — the first version of that arm seeded itself from `hash()`, which is per-process randomised,
and its numbers were discarded).

| arm (the stub model) | parent | candidate | Δ win set |
|---|---|---|---|
| `small` — the live modal answer (94 %) | 17 | 17 | +L5 / −L21 |
| `large` | 12 | **17** | +5 / −0 |
| `zigzag` F,F,R,F,F,L | 4 | **6** | +2 / −0 |
| `roundrobin` over all six actions | 5 | **6** | +1 / −0 |
| `climber` — a climb every third decision | 14 | **17** | +4 / −L16 |
| `mix` — the live per-decision action mix | 4 | **11** | +7 / −0 |
| `replay` — the model's own recorded answers | 9 | **16** | +7 / −0 |
| **total of 175 episodes** | **65** | **90** | **+27 / −2** |

The two climb clauses added afterwards (§change 6) re-score the same panel at **89** (`mix` 11 → 10),
i.e. panel-neutral; they were adopted on the live evidence below, not on this table.

Candidate ≥ parent on **every** arm. Its win set is *identical* (the same 17) across `small`,
`large` and `climber`: the harness has absorbed the granularity and the climb, so what the model
asks for on those two axes no longer changes the outcome.

Mechanism metrics over the 6 scripted arms (150 episodes): decisions spent **11639 → 9857**; cells
asked per decision **0.569 → 0.894**; executed `small` share **79.2 % → 34.5 %**; climb presses
**874 → 90**; episodes reaching floor 2 **54 → 61**; distinct new cells per decision
**0.177 → 0.175** (flat — see residue). Split by where the exit is (13 levels start-floor, 12
upstairs): upstairs levels reach floor 2 **34/72 → 42/72** and are won **23/72 → 34/72**;
start-floor levels are won **33/78 → 40/78** while *leaving* the exit floor slightly less often
(**20/78 → 19/78**), which is the climb price working.

### ablations (arms `small` + `mix` + `replay`, 75 episodes each)

| configuration | small | mix | replay | total |
|---|---|---|---|---|
| candidate, everything on | 17 | 11 | 16 | **44** |
| − the supplied stride (`supply` → "") | 18 | 8 | 10 | 36 |
| − the pre-step wall veto and the flank-aimed sweep (`reaim` restored to probe-then-rotate) | 2 | 9 | 6 | 17 |
| − the whole geometry chain (`rangefind`/`flank_strip` blinded; this also removes the stair latch, whose premise now reads off the rangefinder) | 4 | 5 | 5 | 14 |
| − the climb price on the model's own climb | 17 | 13 | 16 | 46 |
| parent | 17 | 4 | 9 | 30 |

Both halves of the decode carry weight; the stride supply is worth ≈ +8 of the +14 over the parent
on these arms; and the climb price is **win-neutral to −2** while removing ~90 % of the presses —
which is what the parent's diagnosis measured for its §5.3, and is the price of the one `climber`
regression below.

### live, through the real evaluator against the real Qwen3-VL-30B
Three single episodes first (`eval_runner.py`, `ANTICHEAT_GUARD=1`): **L20 won in 55 decisions**
(score 1.0, `vlm_calls == steps == 55`, `default_actions = 0`, `used_rich_harness`, nothing voided)
— the level the parent lost twice and whose loss the diagnosis traced to the deleted 0.2 %
tolerance — while L5 and L13 lost at 100/100. Because one episode of a temperature-1.0 model proves
nothing, those two were then run as **3 paired repeats per harness**, and the result was the most
useful measurement of this iteration:

| live, 3 repeats | parent | candidate, as first built |
|---|---|---|
| L5 (exit upstairs) | WIN@92, WIN@35, WIN@43 | lose, lose, lose — **0 floor changes in 3 of 3** |
| L13 (exit on the start floor) | WIN@37, lose, WIN@74 | lose, lose, lose |

An instrumented live episode (the planner patched at the class, engine state read rig-side only)
names the cause exactly: on L5 **the latch was set on exactly ONE decision of 100** — step 24, with
`floor_age = 24` so the patience was already paid, but `stale = 0` because the view was new — and
the supplied stride carried the body off the tile, after which the stair was never in sight again.
The parent, stepping one cell at a time, stood on that same stair on **14 of its 73 decisions** and
climbed at the third opportunity. **The throughput lever broke the floor gate by making the agent
stop loitering**, and the climb price I had just made symmetric removed the model's own climb as the
escape hatch.

Both clauses in §change item 6 follow from that tape, and the re-check says plainly how far they
get. With the climb price fixed (and the hold still defeated by `reaim`): **L5 r3 WIN@55 — 5
decisions on the stair, 1 climb; L13 r2 WIN@17; L5 r1 lost — 1 decision on the stair, 0 climbs.**
The win/loss split tracks **stair occupancy** exactly. With the hold completed and wired into
`reaim`: **L13 r2 WIN@17** (vs the parent's 37 and 74), but **L5 r1 and r3 still lose, with
`holds = 0`** — and that zero is the finding:

> `holding_stair()` can only protect a stair the body reaches **after** the clock price is paid.
> A faster agent reaches L5's only stair EARLY, at `floor_age` well under `FLOOR_PATIENCE = 24`,
> walks off it with the price unpaid, and never sees it again. Holding it unconditionally is not the
> answer either — the body would sit turning on a start-floor level's stair and then leave the exit
> floor the moment the clock ran out.

So the live L5 regression is **diagnosed and only half closed**, and the half that is missing is
exactly the mechanism the parent's diagnosis §5.5/§6 already names: *remember the view the latch was
set at and go back to it when the floor goes stale* — a **revocable departure**, not a better
threshold. That is this iteration's hand-off, with the instrument to price it (`scratchpad/rig/`,
`stub.py live` arm: one instrumented live episode, planner patched at the class, engine state
rig-side only). The full 25-level live eval is the outer loop's to run, and its L5/L8/L10/L16/L18/L19
column is where this residue will show.

### contract
`pre_eval_gate` replica returns **None** (no privileged mention in any agent-facing prior, no
privileged attribute or subscript in any `.py`). No module reads `evaluator.game` at all — only
`frame_data.image`, `evaluator.baseline_step`, `conversation_history[0]` and `current_prompt`; the
only occurrences of `tower_data`/`.game` in the candidate are in docstrings. `run_episode`'s
factory, `EpisodeResult` scoring and the six legal actions are unchanged; `baseline_step` is still
called exactly once per decision. Decode cost ≈ 1.5 ms/decision against a ~2000 ms VLM call.

## steer (search-space, token-level implementation)

The directive asked to move toward the region of `cand_r001_1_action-planning` and
`cand_r002_0_spatial-grounding`, whose productive difference is **token-level**: both are 14
components of dense control flow where this lineage had drifted to 29 thin ones. Folding each
module's helpers and constant tables into the class that owns their state moves it there:

| | parent | **this** | ex1 | ex2 |
|---|---|---|---|---|
| components | 29 | **15** | 14 | 14 |
| `cf_mean` | 4.90 | **12.47** | 6.14 | 8.79 |
| `cf_max` | 45 | **76** | 52 | 89 |
| `cf_std` | 8.11 | **23.32** | 12.94 | 22.38 |
| `num_per_comp` | 3.45 | **11.47** | 3.00 | 5.14 |
| `logmag_mean` | 0.643 | **0.706** | 0.917 | 0.756 |

## watch / residue

* **Two regressions in 175 episodes, and they are different faults.**
  * arm `climber` L16: the parent pressed **14** climbs and stumbled upstairs to the win; the
    candidate pressed **0** and never left floor 0. This is the climb price refusing exactly the
    behaviour it was built to refuse, and losing a lottery ticket by it. `− climb price` in the
    ablation table recovers it (and costs nothing else), so if the price is ever relaxed, relax it
    for a floor whose stairs are the **only** unexplored thing left, not by lowering the constant.
  * arm `small` L21: the candidate asks for **90** cells of travel against the parent's 42 yet
    visits **18** distinct cells against 32, and ends at BFS 5 instead of 0. On a twisty level a
    three-cell stride buys ground without buying *new* ground: it is the coverage gap below, not
    the stride being illegal (every stride is traversed in full, 1820/1820).
* **Coverage is flat.** Distinct new cells visited per decision 0.177 → 0.175: this change buys route
  progress, not exploration. The parent's §5.4 **transition ledger** (`transitions[(view, action)]`,
  re-aiming a transition already taken ≥ 2 times from a view; 60 % of decisions repeat one) is
  untouched and is now better licensed than ever, because the exact outcome test leaves the
  view-keyed ledger 0-wrong.
* **THE NEXT AXIS, now measured live rather than inferred: return to the abandoned stair.**
  `holding_stair()` stops the body *leaving* a stair whose clock price is paid — the cheap half, and
  the half that fires. The expensive half is open and is what still loses L5 live: a stair reached
  BEFORE the price is paid is walked off and forgotten (the parent's diagnosis counts 42 of 43
  abandoned arrivals; the live tapes here show occupancy of 0–1 decisions where the parent had 14).
  Record the view id at which the latch was set, and when the floor goes stale re-aim the sweep back
  toward it — the view-keyed ledger is 0-wrong after the exact outcome test, so the key is sound.
  This is the **revocable departure** the diagnosis §6 names, and it is the only thing that can
  settle the start-floor/upstairs trade-off, which no value of `FLOOR_PATIENCE` can.
* **The stair gate is now the whole residue.** Two clauses of it are measured exact (the latch, the
  arrival) and two are heuristics with no frame evidence behind them (`FLOOR_PATIENCE = 24`,
  `STALE_FLOOR = 1`). The live tape shows the second one is **coupled to travel speed** — the faster
  the agent, the rarer `stale ≥ 1` — so any further throughput gain will press on it again. Price it
  on stair-occupancy decisions, not on the constant.

## module map + lever table

| file | what it is | kind | state |
|---|---|---|---|
| `agent.py` | entrypoint. `compose_system_prompt()` rebuilds the prior from `AGENT.md` + `skills/*.md` + `rules.md`; `build_agent(evaluator)` opens ONE `ActionPlanner` per episode and returns `agent(frame_data)`. | code | stable — do not change the factory |
| `tools/sight.py` → `Scene` | the frame decoded: colour census + **rangefinder** (centre column) + **flank strip** (side bands) + `stride()`/`bearings()`/`junction`. | code | **this iteration.** Exact; extend, do not re-derive |
| `tools/sight.py` → `StairSense` | the stair-underfoot latch, its one-press budget, and the floor price (`STALE_FLOOR`, `FLOOR_PATIENCE`). | code | latch exact (110 TP / 1 FP / 0 FN live); the **price** is the open question (see watch) |
| `tools/plan.py` → `ActionPlanner` | the governor: view fingerprint + ledgers, the veto chain (incl. `holding_stair`), the flank-aimed sweep, the supplied stride, the working-memory block. | code | **this iteration.** Next: the transition ledger (§5.4) and the return-to-stair (§5.5) |
| `tools/action.py` | the key table + `retarget()` (strips `granularity`/`repeat`/`json_data` before re-aiming). | code | stable; a stale `repeat` on a re-aimed turn fires three turns |
| `tools/vlm.py` | `decide()`: observe → inject the block into `conversation_history[0]` → `baseline_step` → govern. | code | stable — never splice the base system turn |
| `tools/frame.py` | the evaluator's own screenshot. | code | stable |
| `AGENT.md` | role + the measured perception facts (no HUD, the two exact lines, who picks the stride). | prose | rewritten this iteration |
| `rules.md` | legal actions, the climb rule, granularity (now declared harness-chosen). | prose | rewritten this iteration |
| `skills/navigation.md` | the per-step priority order. | prose | rewritten this iteration |
| `memory.md` | the runtime block template; placeholders must match `ActionPlanner.fields()`. | prose | gained `{ahead_line}` / `{open_line}` |
| `system_prompt.txt` | **NOT a lever** — `compose_system_prompt` never reads it. | — | leave alone |

## files

`tools/sight.py` (rewritten), `tools/plan.py` (rewritten), `agent.py` (docstring),
`AGENT.md`, `rules.md`, `skills/navigation.md`, `memory.md`, `NOTES.md`.
