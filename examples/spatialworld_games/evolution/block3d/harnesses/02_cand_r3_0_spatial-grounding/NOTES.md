# block3d harness — layer-factored edge covers + a boustrophedon press tape

## axis
**code / planner.** A new `tools/layers.py` reads the three target panels as a per-HEIGHT bipartite
problem and solves each height exactly; a new `tools/serpent.py` turns the result into a press tape; a new
`tools/press.py` owns the press algebra; a new `tools/ledger.py` keeps a booked replay of every press; and
`tools/vlm.py` is rewritten so the model is called only when its answer could change the emitted key. The
parent's entire decision path was `return await evaluator.baseline_step(frame_data)`.

## the mechanism, in one paragraph
The three panels FACTOR BY HEIGHT. At a fixed height `z`, FRONT fixes exactly which rows that layer must
light and SIDE exactly which columns; TOP is the only constraint that couples one height to another. So a
layer, read on its own, is a bipartite graph — columns on one side, rows on the other, an edge for each
admitted TOP square — and "light every row and every column" is precisely an EDGE COVER of it. The
smallest one is `|xs| + |ys| - (maximum matching)`, computed by Kuhn augmenting paths. Taking the union
over heights settles FRONT and SIDE completely and leaves one residue: a TOP square no layer happened to
use, settled at its lowest admissible height. That union is the SMALLEST structure the panels admit, and
its size is a hard lower bound on how many blocks the level can need; the admitted hull is the largest.
The tape walks the hull in `(forced, height)` tiers with a boustrophedon sweep inside each tier, so it is
monotone: the engine re-tests set equality after every executed action, which makes each press a free test
of one more candidate structure. `tools/vlm.py` spends the presses, hands the model every choice the plan
is indifferent about, and refuses the two presses the ledger can prove are waste.

## why (the gradient this answers)
The diagnosis classified the binding constraint as a VLM ceiling on cursor localisation and routing behind
a two-line decision path, and gave three measurements: movement conditioned on the image is *worse than
not looking* (every direction is emitted more often exactly when it is blocked, up to 4.1x), committing is
informative (2.5x the base rate when standing on a wanted cell), and the wallet is not the constraint
(every level's optimal tape is at most 33 of 50 presses). The implied split is: own the route, keep the
model on the choices it is good at. This candidate takes that split and adds the part the gradient did not
specify — HOW to choose which cells to build. A hull descent searches a 3-D set; the height factorisation
turns it into six independent exact matchings, which is both cheap and a certificate.

## honest limits
The panels are a lossy channel: on a level whose hull over-approximates, the win depends on the target
being a prefix of the tape (14/20 on an adversarial random bank). The ink read answers only when the
footprint is whole; after that the reckoning carries the episode, and it is exact only while every key
returned is the key executed. `support` is quadratic in the hull size — microseconds at the hull sizes a
level reaches, but not free. The lane sweep costs 10-15% more travel than a nearest-first walk on randomly
scattered targets, within 1% on the structured bank, and it is what makes the rim key unnecessary. No part
of this reads engine state: the only inputs are `frame_data.image` (with `capture_game_screenshot` as the
fallback channel) and the keys this harness itself returned.

## files
`orchestrator.py`, `AGENT.md`, `rules.md`, `memory.md`, `skills/building.md`, `skills/sweeping.md` (new),
`tools/__init__.py`, `tools/frame.py`, `tools/panels.py`, `tools/layers.py` (new), `tools/serpent.py`
(new), `tools/press.py` (new, replaces `tools/action.py`), `tools/ledger.py` (new), `tools/vlm.py`,
`traces_notes/MEASUREMENTS.md` (new, the full offline log). `tools/action.py` removed — `tools/press.py`
supersedes it and is on the live path.
## evolvable levers (axes)
- **prose** — `AGENT.md` / `rules.md` / `skills/*.md` / `memory.md`, plus the brief's field list in
  `tools/vlm.py:_brief`, which is the live half of this axis.
- **code** — the planner (`layers.py`), the schedule (`serpent.py`), the decode (`panels.py`,
  `frame.py`), the arbitration and ask policy (`vlm.py`), the audits (`ledger.py`).
- **not a lever** — `system_prompt.txt` / `system_prompt.md` are never read by `compose_system_prompt`.

## module map
| file | role |
|------|------|
| `orchestrator.py` | ENTRYPOINT. `build_agent(evaluator)` returns the async per-step fn; `compose_system_prompt()` rebuilds the prompt from the `.md` priors. The `sys.modules["tools*"]` purge is load-bearing and stays. |
| `AGENT.md` | role framing + the perception facts verified off the frame (6x6x6, absorbing faces, spawn, the three panels, the self-occluding view, the ~50-press episode). |
| `rules.md` | the seven legal actions, the one-action JSON contract, toggle semantics, and that a press into a face is legal and useless. |
| `skills/building.md` | reading a panel square into a statement about the structure; ALLOWED vs REQUIRED. |
| `skills/sweeping.md` | the press economy: L1 travel, interchangeable axes, the lane sweep, finishing over tidying. |
| `memory.md` | the per-turn scratchpad TEMPLATE; `tools/vlm.py` renders the live lines above it into the DIALOGUE tail. |
| `tools/frame.py` | the frame array, the viewport digest (a one-bit clamp proof), the cursor from the highlight ink. |
| `tools/panels.py` | the panels, read by FITTING the lattice the strip draws; the 3-way marginal self-check; the hull; the per-height census. |
| `tools/layers.py` | THE PLANNER: spans, admitted pairs, maximum matching, minimum edge cover, the minimal shape, graded support, forced cells, phantoms. |
| `tools/serpent.py` | THE SCHEDULE: lanes, the boustrophedon sweep, the `(forced, height)` tier key, the tape, what is still owed. |
| `tools/press.py` | the clamp, L1, the closing (free) axes, the key/name map, the two-engine-steps-per-press billing, the tape's toll. |
| `tools/ledger.py` | the dead-reckoned cursor and placed set, audited against the digest and the ink; the press account. |
| `tools/vlm.py` | plan once, audit, arbitrate, ask only when it matters, publish the plan, journal every decision. |

## what the measurements say (full log: `traces_notes/MEASUREMENTS.md`)
- **The decode is a measurement, not a constant table.** The masks reproduce the true three projections on
  121/121 synthetic levels; a blank strip refuses; an all-lit strip still fits. Move the strip (left edge
  40 -> 70 -> 100), rescale the pitch (15 -> 12 -> 20 -> 25) or change the stride (150 -> 100 -> 170) and
  the fit follows and the decode stays exact. The fit must be occlusion-proof: a lit square on a panel
  border paints over the outline, so the per-ROW census can lose a panel's top row. Two readings are
  taken — outline runs down the tallest column, and the row census as an arithmetic progression of three
  congruent panels. With the row census alone, 2 of 44 levels failed; with both, 0 of 121.
- **The cursor read is exact under its own precondition.** Inverting the highlight centroid to the nearest
  lattice point is 216/216 on an empty grid and never wrong; once cubes clip the footprint it REFUSES and
  the dead-reckoned ledger carries the episode.
- **Forced-first is load-bearing; the rim key is refuted.** Over 60 levels, `(forced, height)` makes the
  true target a prefix of the tape 53/60 times. Demote forced below height and that falls to 42/60 — and
  on the hull-over bank from 14/20 to 3/20. Adding a rim-from-hull-centroid third key is prefix-neutral
  and costs 8% more travel; adding minimum-cover membership is prefix-neutral and costs 4% more. Under a
  boustrophedon the lane order already encodes position, so both only add tier boundaries to cross.
- **The affordability switch is refuted.** Aiming at the minimal shape when the hull tape does not fit the
  wallet wins 15/15/11 against the hull tape's 19/15/11, and "either" is also 19/15/11. It never adds a
  level; recorded so the next round does not spend itself on it.
- **End to end, driving the real `orchestrator.py`:** 19/20 structured, 15/20 hull-closed, 11/20 hull-over —
  and 19/20 structured with the model replying at RANDOM, with no highlight ink at all, and with cubes
  occluding the cursor. **Clamped presses 0, cancelled work 0, stand-downs 0** in every bank. The parent,
  on the real levels, won 1/20 with 42.5% of its moves clamped and 21.0% of its toggles cancelling its own
  work. The five hull-closed losses are budget losses on random 11-to-18-cell targets scattered through
  the box, an adversarial shape family for a lane sweep.
- **The spawn gift is taken by PLAN.** On a level whose only target cell is `(2,2,0)` the tape's first cell
  is the spawn, so the first decision emits `place_block` and wins at press 1 — measured even with the
  model scripted to say `move_left`. The parent took that gift on 4 of the 9 levels that offer it.
- **The ask policy.** Ask only when the answer could change the key: the first decision is always the
  pristine uninjected pass-through, then the model is asked when two or more axes close the gap equally or
  two or more surplus blocks are equally droppable. Standing on the objective the key is the toggle and
  nothing could change it, so nothing is asked. Measured: 5 model calls per episode instead of 50, same
  wins.

