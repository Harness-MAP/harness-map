# cand_r029_0 — neighbour ledger (grounded single call)

## module map
| file | role |
|---|---|
| orchestrator.py | `Ledger` (per-level visits + failed (square, direction) moves, parsed from cursor/map/prose) and `LedgerHarness` (one model call per step, same as naive) |
| tools/prompting.py | naive transcript + priors + `render_ledger`: an 8-row neighbour table (glyph, visits, move-failed, DEADLY) |
| tools/action_io.py | `legal_actions` from the system prompt; `extract_action` = whole-word projection onto the legal set with a veto list and a fallback |
| AGENT.md / memory.md | two short priors explaining the ledger columns |

## axis:
Code: where the grounding sits. The model still decides every step with one call. What reaches it is
a code-kept, per-direction fact table, and what leaves it is projected onto the legal set and vetoed.
Earlier designs either take the decision away from the model (field, tape, BT, planner) or put
state into a free-text memory. This one keeps the naive decision locus and grounds only its input
and output.

## change:
Added a `Ledger` to orchestrator.py. Each step it reads the cursor `(x, y)`, the `map:` block (row = y−1),
`Dlvl` and the adjacent lava/water prose. It counts visits per square and records a (square,
direction) pair as failed when the last move left the cursor unmoved. It resets when the level
changes. prompting.py renders the eight compass neighbours as a table after the transcript.
action_io.py parses the legal names, takes the earliest whole-word legal mention that is not vetoed
(failed here, or DEADLY), and otherwise falls back to the least-visited open, non-wall direction. It no
longer lets an illegal reply be coerced to `north`.

## why:
The anchor diagnosis found 35% period-2 orbit steps, 28% no-move steps (wall bumps, and illegal
verbs coerced to `north`) and lava deaths where the hazard was named adjacent. Each of these comes
from a stateless, ungrounded call. The ledger addresses the orbits (visit counts), the bumps (failed
moves) and the lava deaths (DEADLY veto), and it keeps the model as the decision maker.

## files:
orchestrator.py, tools/prompting.py, tools/action_io.py, AGENT.md, memory.md, NOTES.md
