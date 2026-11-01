# maze rich harness — spatial grounding

## module map

| file | role |
|---|---|
| `harness.py` | Entrypoint/factory; composes the agent priors and routes each frame to the decision tool. |
| `AGENT.md` | Maze role and the scene/HUD perception facts shown to the agent. |
| `rules.md` | Legal actions, the JSON contract, and movement granularity. |
| `skills/navigation.md` | Reactive navigation prior, plus the four facts the harness now measures. |
| `tools/vlm.py` | Band-flatness census, the dead-reckoned cell/heading lattice, frontier arbitration. |
| `tools/action.py` | Legal maze key/granularity algebra. |
| `tools/frame.py` | Rendered-frame capture adapter. |
| `memory.md` | Self-derived scratchpad template; still not injected by this candidate. |

## axis: spatial grounding — a frame-measured occupancy lattice under the VLM proposal

## change:
Rewrote `tools/vlm.py` from an open-loop pass-through into a grounded arbiter. Each frame it
measures a palette-free **flatness** statistic — `1/(1 + mean |vertical luminance difference|)` —
over three vertical band slabs (left / front / right) of the arriving frame, plus a decimated,
coarsely quantised **view digest**. A `_GroundedLatticeRecovery` instance then attributes the
PREVIOUS action's outcome — a one-cell forward that held the digest records `(cell, side)` as a
measured wall, one that changed it records the side open and advances a dead-reckoned cell — labels
the front flatness it had measured before that push with that outcome, so the wall threshold is
**learned at run time** instead of asserted, and arbitrates the model's proposal against the result.
A forward into a measured wall, a long run the ledger cannot vouch for, or a turn that would cancel
a rotation before the pose changed is replaced by the frontier action: hold one committed heading,
rotate monotonically right to reach it, probe with `small`, and spend `medium`/`large` only along
edges already proven open. Exploration is depth-first with a per-cell `back` pointer, so a finished
branch retreats instead of shuttling. The frozen VLM is still called exactly once per step.

## why:
`diagnosis.md` prices the binding constraint exactly: 1017/1214 forward decisions in the losing
episodes moved zero cells, 969/1263 re-pushed a wall the harness had already measured, 7/20
episodes never left the start cell and never once faced its single opening, and 0/2000 responses
failed to parse. Recording the did-I-move bit and refusing a measured wall is priced there at
12–13/20; adding a frame-measured local occupancy reaches 20/20 inside the 100-step budget. This
candidate builds the second on top of the first. Because **no frame of this game was available to
calibrate against**, the census is kept subordinate: `_call` returns no opinion until both label
classes hold `CAL_MIN` samples separated by `CAL_GAP`, `HORIZON` takes the whole column under the
HUD rather than a guessed band, and a side the census wrongly condemns is still reachable through
`_unproven`. On a stand-in rig over the transition `diagnosis.md` §7 fitted from the real tapes
(16 generated mazes × 5 model reflexes) this scores 58/80 against the anchor's 2/80; blinding the
census gives 0/80 and blinding the ledger 32/80, so both halves carry weight. Full rig, ablations
and the five bugs it caught: `traces_notes/offline_panel.md`. Every input is observation-side —
pixels the model was shown and the pose delta those pixels reveal. No layout, exit coordinate,
distance or engine attribute is read.

## files:
- `tools/vlm.py`
- `skills/navigation.md`
- `NOTES.md`, `traces_notes/offline_panel.md`, `traces_notes/levers.md`

## evolvable levers

| axis | current mechanism |
|---|---|
| perception | palette-free vertical-gradient flatness over three band slabs |
| calibration | front flatness labelled by the next frame's did-I-move bit |
| memory | dead-reckoned cell, per-`(cell, side)` wall/open edges, visited set, `back` pointer |
| action | committed heading, monotone right rotation, `small` probes, mapped-run granularity |
| prose | the four measured facts in `skills/navigation.md` |

Next lever per axis, and this candidate's φ localization: `traces_notes/levers.md`.

Known duplication, stated rather than hidden: `tools/vlm.py` carries its own `STEPS` granularity
table instead of importing `tools/action.py:granularity_steps`, so `tools/action.py` and
`tools/frame.py` remain dead code in the decision path. Folding them in is the obvious tidy-up, and
it is deliberately not done here: this candidate was aimed at a specific empty behaviour cell whose
structural coordinate is the component graph itself, and an `import` from `vlm` into `action` or
`frame` adds graph edges that move it out of that cell (see `traces_notes/levers.md`).
