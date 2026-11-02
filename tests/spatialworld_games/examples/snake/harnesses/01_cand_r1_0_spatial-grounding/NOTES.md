# Snake candidate — dead-reckoned legality audit over the frozen model's pick

## axis:
Code lever inside `tools/vlm.py:decide` — a self-derived head/heading/trail/visit ledger that
vetoes a provably fatal key and substitutes the least-visited legal direction. Token-level by
construction: the control flow and every constant moved INSIDE the existing components and the
component graph is byte-frozen (same nodes, same edges, no new module and no new import).

## change:
`decide` still delegates to `evaluator.baseline_step` once per step, but its answer is no longer
pressed unreviewed. The head cell is re-derived from the fixed spawn plus the keys this harness
itself emitted, and `act.key` is rewritten when the action the model named would leave the 4x4x4
box, reverse the heading into an engine refusal, or re-enter the body. A one-ply room test drops
arrival cells with no continuation. The geometry sits at module scope beside the tables it reads
rather than in new helpers, which is what holds the graph fixed.

## why:
The parent scored 0.0000 with 20/20 episodes walking out through a face by step 4, and that
predicate needs no pixels and no oracle. Its prose also asserted three measurably false facts — a
centre spawn, a 64-point objective and a direction-letter HUD this build does not render — so the
priors now say what the frame actually draws.

## measured:
Against a replica of `snake_adapter._try_move`/`_step`, 20 seeds x 100 steps: primary 0.0000 ->
0.7250 on the diagnostician's fitted stand-in and 0.0250 -> 0.5125 on the uniform control, no wall
or body deaths, model honoured on 38-56% of decisions. Rig, ablations, descriptor aim and the
refuted variants: `traces_phi/LEVERS.md`.

## files:
- `tools/vlm.py` — the ledger, the module-scope geometry and the audit.
- `AGENT.md`, `rules.md`, `skills/spatial_grounding.md`, `memory.md` — the measured facts.
