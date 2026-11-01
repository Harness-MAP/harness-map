## axis:
Decision locus (code): an in-episode value learner picks moves; the model breaks ties.
## change:
One-module `harness.py`; optimistic (square, move) values, bump penalty, novelty reward, eligibility trace, reflexes, lava veto, tie consult. Detail in DESIGN.txt.
## why:
Iter-0: orbits, bumps, lava, `[yn]` freezes from a memoryless policy.
## files:
`harness.py`, `tools/` + `skills/` removed, `NOTES.md`, `DESIGN.txt`.
