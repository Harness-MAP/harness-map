# cand_r033_0_spatial-grounding

## axis:
Decision locus (code): the model still picks every move, but a decoded map now GROUNDS the prompt and VETS the pick.

## change:
`tools/grid.py` decodes the `map:` block at the cursor into the eight neighbouring squares (wall, lava, creature, stairs down, boulder, open, dark) plus the prose `lava adjacent` and creature lines. `harness.py:GroundedAgent` keeps a visit count per square and a set of bumped (square, direction) pairs. Each turn it prints the grounded ring and the filled `memory.md` trail into the prompt. Then it projects the reply onto the legal set with whole-word matching (`attack east` becomes `east`), and it swaps a move that enters lava, re-bumps, or paces back onto a worn square for the best open neighbour.

## why:
The diagnosis: period-2 orbits, wall jams, lava deaths and illegal verbs coerced to `north`.

## files:
harness.py, tools/grid.py (new; prompting.py and action_io.py folded in), AGENT.md, rules.md, memory.md, skills/unstick.md.
