# block3d rich harness — module map & levers (r001_0 · spatial-grounding)

Parent: `anchor_block3d` (the rich replica of the prompt-only baseline). The frozen agent
(Qwen3-VL-30B-A3B-Instruct) and the SpatialWorld engine are untouched; only this harness is
optimized. `system_prompt.txt` is kept as the prompt-only fallback and is NOT a lever —
`compose_system_prompt` never reads it. The full measurement record sits in
`traces_notes/measurements.md`, which is outside the descriptor's genome.

## module map
| file | role |
|------|------|
| harness.py | ENTRYPOINT. `build_agent(evaluator)` composes the priors and returns the async per-step fn backed by ONE episode-scoped `Arbiter`. |
| AGENT.md | role framing + the perception facts verified against this renderer (the panels are a complete specification, scoring is exact-match, one highlighted cube, a wall move still costs a decision). |
| rules.md | the legal action set (6 cursor steps + the place/remove toggle) + control mapping + observation tips. |
| skills/building.md | the build-strategy prior (view analysis / error correction). |
| **skills/grounding.md** | **NEW.** how to use the grounded-state block: never re-derive a stated coordinate, build exactly the listed cells, the toggle is its own inverse, never step in a dead direction, commit on a listed cell, watch the unexplained-squares meter. |
| memory.md | the working-state schema — now LIVE, maintained in code (`Ledger`) and rendered by `Arbiter.brief`; the file documents the schema. |
| **tools/lattice.py** | **NEW.** FITS the three panel rings off the frame — square side, left edge, the three tops, cell pitch — as measurements, and refuses rather than guessing. |
| **tools/panels.py** | **NEW.** reads the silhouettes back square by square, builds the admitted volume, and runs the three self-consistency tests that let the read refuse itself. |
| **tools/reckon.py** | **NEW.** the cursor, kept twice — booked from this harness's own returned actions and sighted off the highlight hexagon — plus the placed ledger, the dead-step test and the route. |
| **tools/debt.py** | **NEW.** the residual projection debt, the forced set, the ranking and the goal cell, re-derived every decision. |
| **tools/keys.py** | the action/key algebra, now TWO-WAY: the inverse map is what lets the arbiter read the model's own choice back into a name before comparing it with the route. |
| tools/vlm.py | the decide step — REWRITTEN into `Arbiter`: brief → one `baseline_step` call → veto what the picture proves wasted → book → stamp. |
| tools/frame.py | frame capture — reuses `evaluator.capture_game_screenshot`, and reports a failed grab as a refusal rather than raising. |

## evolvable levers (axes)
- prose: AGENT.md / rules.md / skills/*.md / memory.md — what the agent is told.
- code: harness.py / tools/*.py — the mechanism (fit, read, reckon, rank, arbitrate).
- the ranking key in `debt.py:rank_cells` is the single highest-leverage lever in the tree:
  forced-first and the layer tier are both load-bearing, the rim is measured from the admitted
  volume's own centroid rather than the grid's, and the walk term is the tie-break inside a tier.
- the arbitration policy in `vlm.py:choose_key` is the second: which proposals are honoured and
  which are replaced, and on which cells the model's commit vote is taken as it stands.
- **open next:** rank several minimal sub-structures when the admitted volume over-approximates
  (two of the twenty levels); spend leftover budget on the pruning phase rather than idling; let
  the model choose between two equally-ranked goal cells instead of only voting on the one.

## anti-cheat
Decide from the observation only. The only engine surface this harness touches is the rendered
screenshot the frozen model is also shown: the structure is read from the panels in that picture,
the cursor from the highlight in that picture, and the placed set from the harness's own returned
actions. No layout file, no engine attribute, no privileged field is read anywhere, and exactly
one frozen-model call is made per decision. The adapter's `pre_eval_gate` was run against this
dir: **PASS**.

---

## axis
**Observation grounding / goal selection (code lever).** The anchor asked the frozen model for one
keystroke per turn and executed it blind. This candidate reads the specification and the cursor
back out of the picture both parties already share, and re-derives the cell to work toward every
single decision from the RESIDUAL PROJECTION DEBT — the lit panel squares the blocks placed so far
do not yet explain. The component graph keeps the anchor's shape; what changed is the content
inside the components: a pass-through decision became a measured one, a dormant action table
became a live two-way algebra, and a dormant memory template became a rendered per-turn brief.

## change
`tools/lattice.py` measures the three panel rings instead of asserting a pixel table, reading only
the ring borders that the painted squares cannot occlude, and refusing when the strip does not
resolve into three congruent rings. `tools/panels.py` votes each cell's inset interior against the
exact fill ink and then tests the reading three ways — no panel empty, the axis each pair of panels
shares in agreement, the admitted volume re-projecting exactly — so a garbled read stands the whole
wrapper down to the pristine pass-through. `tools/reckon.py` keeps the cursor twice, booked through
the engine's clamping rule and sighted as the unique cell whose projected hexagon contains every
lit highlight pixel; the picture wins when it speaks, the book carries the state when occlusion
makes the cover ambiguous, and the placed set is booked rather than counted off the self-occluding
scene. `tools/debt.py` ranks the admitted-but-unplaced cells forced-first, then still-owed, then by
layer, then by rim distance from the admitted volume's own centroid, then by the shortest walk from
where the cursor actually is; a placed cell the panels exclude is a stray and outranks everything,
and once the volume is complete but unaccepted the goal becomes a cell whose removal keeps all
three silhouettes intact, never offered again once lifted. `tools/vlm.py:Arbiter` still makes
exactly one model call per decision, appends the brief to the system turn the engine composed (from
a stash, never a splice, so the response contract survives), leaves the first decision untouched,
and keeps the model's answer unless the picture proves it wasted.

## why
`diagnosis.md` classifies the constraint as a VLM-CEILING on cursor localisation and routing in
front of a decision path two lines long, and every leak it measures is execution rather than
perception: 42.5% of the parent's moves clamped against uniform random's 15.8%, 14.0 of 216 cells
visited against random's 24.4, 32.2% of decisions after the last new state, 85.3% of standing
blocks on cells no panel asks for, 21.0% of toggles cancelling an earlier one — with 0 parse
failures, 0 default actions and 0 illegal actions in 951 completions, and an optimal tape for the
hardest level of 33 of the 50 decisions the wallet allows. Those are exactly the jobs this
candidate takes off the model, while leaving it the one it does well: it commits at 2.25 times its
own base rate when it is already standing on a cell the structure wants. Measured offline against
the real evaluation adapter with the model held at its worst — a scripted emitter returning the
same action every turn — the harness wins 20/20 inside the wallet on seven such arms, with zero
null moves and zero self-cancelling toggles; the decoders read 20/20 on the lattice fit and the
projections, 18/20 on volume-equals-structure, and the cursor refuses rather than guessing on the
~5% of mid-build turns where placed cubes occlude the highlight. Full tables, per-level decision
counts and the seven-way ranking-key ablation are in `traces_notes/measurements.md`. The honest
risk is that the whole gain rides on one clean panel read: if the fit or the cross-check fails the
wrapper stands down and the harness is the anchor again, which is the floor rather than a loss, and
it is why the lattice is fitted rather than tabulated.

## files
`tools/lattice.py` (new) · `tools/panels.py` (new) · `tools/reckon.py` (new) · `tools/debt.py`
(new) · `tools/keys.py` (algebra, two-way) · `tools/vlm.py` (rewritten) · `tools/frame.py`
(refusal-safe) · `harness.py` (wires the arbiter) · `skills/grounding.md` (new prior) ·
`memory.md` (schema now live) · `AGENT.md` (verified perception facts) ·
`traces_notes/measurements.md` (new, outside the genome) · `NOTES.md`
