# cand_r053_0 — code-ranked move menu (grounded single call, r029 region)

## module map
| file | role |
|---|---|
| harness.py | `Ledger`: per-level visits, failed (square, dir) moves, prev square, prose-adjacent deadly (lava/water) and blocked (wall/bars) sets. `Ledger.menu` ranks the open legal compass moves. `MenuHarness`: one model call per step, plus the orbit override |
| tools/prompting.py | naive transcript + priors + `render_menu`: a numbered best-first MOVE MENU |
| tools/action_io.py | `legal_actions`, and whole-word legal projection with a veto and a fallback (from r029) |
| AGENT.md / memory.md | short priors: `>` is the goal, keep swinging, take menu option 1, what the menu tags mean |

## axis:
Code, following the r029 steer: the ledger that grounds a single call is restructured from an advisory fact TABLE into a
code-RANKED MENU that code partly enforces. The ranking moves out of the model and into code, and the model still
makes the call.

## change:
`Ledger.menu` ranks each legal compass move by (stairs `>` < adjacent creature < other; wall-ish glyph last;
the square just left last; fewest visits; fixed index). It drops moves into lava/water, moves into walls or bars
that the prose names adjacent (vetoed before the first bump, not after), and moves that already failed from this
square. prompting.py renders the menu best-first. Three fixes to r029's mechanism: (1) a no-move books `failed` only
when it was NOT a fight (target glyph not a letter, no hit/miss/bites text), so swings can repeat; (2) the wall/bars
prose veto fires before the move; (3) orbit override: if the reply walks back to the square just left while menu
option 1 is a never-stood non-creature square, emit option 1 instead. The fallback for no legal or vetoed mention is
menu option 1.

## why:
r029's diagnosis: the model ignores an unranked table (A→B→A orbits 38–48%, 73–90% revisits). The swing-as-failed
bug gave rats free bites (0/35 repeat swings, 10/10 melee deaths). 250+ first-contact bumps on walls/bars were
named adjacent in the prose beforehand. Menu order is the known lever for a model that takes the first option it
sees. The override breaks the 2-cycle deterministically without taking every decision away from the model.

## files:
harness.py, tools/prompting.py, tools/action_io.py (copied from r029), AGENT.md, memory.md, NOTES.md
