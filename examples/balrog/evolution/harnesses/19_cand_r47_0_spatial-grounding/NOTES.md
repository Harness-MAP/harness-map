# cand_r047_0 — prose-noun neighbour ledger + code arbitration (single grounded call)

## module map
| file | role |
|---|---|
| harness.py | `GroundedHarness.act`: perceive (ledger) → modal reflex → one model call → arbitrate; writes the deciding rule to `response.reasoning` |
| tools/ledger.py | NEW. `Ledger`: per-level visits, combat-aware failed (square, dir) moves, previous square; `adjacent_nouns`/`classify` turn "<noun> adjacent <dirs>" prose into deadly/blocked/creature/open per neighbour |
| tools/action_io.py | `legal_actions`, whole-word `mentions`, `arbitrate` (legal projection → deadly/blocked/failed veto → orbit-break → ranked fallback) |
| tools/prompting.py | naive transcript + priors + `render_table` (glyph, game-text noun, visits, failed, just-left) |
| AGENT.md / memory.md | short priors explaining the table and "moving into a monster attacks it" |

## axis:
Code: spatial grounding + output arbitration, moving toward the r029 region (naive single call, grounded input,
filtered output) from the naive anchor. It varies r029's wiring in three places its diagnosis showed were broken:
where failure is booked, where the pre-move veto comes from, and whether visit counts are binding.

## change:
Parent is the naive anchor. I added a new `tools/ledger.py` perception/memory module. It parses the cursor, the
`map:` block (row = y−1), `Dlvl` and the adjacent-noun prose, and it grounds each of the eight neighbours by glyph AND
by the noun the game names there. The noun outranks the glyph: bars, walls and stone are BLOCKED before the first
bump, and lava or water is DEADLY. A no-move books a failed (square, dir) only when the frame has no combat message,
so a swing at a surviving rat is not vetoed. That was r029's 0/35 repeat-swing bug.

`tools/action_io.arbitrate` projects the reply onto the legal names (`attack east` → `east`) and vetoes
deadly/blocked/failed moves. It also makes visit counts binding: a move back to the square just left, with no creature
adjacent, is replaced by a strictly less-visited open direction when one exists. That targets the A→B→A orbit, which was
38–48% of steps in r029.

`harness.modal_reflex` answers `[yn]` with `y` and `--More--` with `more` when those verbs are legal. Each step's
deciding rule (model / veto / orbit-break / fallback / modal) goes to `response.reasoning`.

## why:
Anchor diagnosis, modes A–E: orbits, wall pins, lava named adjacent, `[yn]` freezes, and no swings. r029 (the steer
exemplar, 0.150) fixed lava and illegal verbs with a ledger. Its diagnosis named three remaining causes:
- the swing booked as a failed move, which cost 10/10 monster episodes;
- advisory visit counts that the model ignores (orbits unchanged);
- blocking prose nouns (bars, walls) that were never used before the first bump.

This candidate fixes all three and stays inside the same single-call design.

## files:
harness.py, tools/ledger.py (new), tools/action_io.py, tools/prompting.py, AGENT.md, memory.md, NOTES.md
