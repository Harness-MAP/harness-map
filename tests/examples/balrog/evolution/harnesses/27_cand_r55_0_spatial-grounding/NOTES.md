# cand_r055_0 — middleware stage pipeline whose movement locus is a CHART + BFS route

## module map
| file | role | lever |
|---|---|---|
| agent.py | `PipelineHarness.act`: context → `pre` hooks outer→inner (first non-None short-circuits, NO model call) → else one model call → `snap` → `post` hooks inner→outer; deciding stage written to `reasoning` | code |
| tools/context.py | NEW. `StepContext` from the observation: cursor (row = y−1), Dlvl, map rows, message-only combat flag, `<noun> adjacent <dirs>` prose, `creature_dirs()` (letter glyph in the ring) | code |
| tools/chart.py | NEW. Per-level map memory (all drawn glyphs, visits, bump-closed edges booked only without a combat message, prose-noun doors/walls/lava outranking glyphs) + `route()`: BFS to a drawn `>`, else the nearest unvisited square bordering undrawn space; distance then compass order, re-elected every step; no diagonals through doors | code |
| tools/stages.py | NEW. Legal (audit) · Modal (`[yn]`→y, `--More--`→more) · Hazard (post veto onto `}`/lava/water) · Route (owns the chart; short-circuits with the BFS hop unless a creature is adjacent) · Bump (post veto of refused/wall moves → least-visited passable) | code |
| tools/action_io.py | legal-set parse + whole-word `snap` (earliest mention, longest name) | code |
| tools/prompting.py | naive transcript + priors + "Checks on this frame" notes from stages | code |
| AGENT.md | tells the model it is consulted only for fights, prompts and no-route levels | prose |

## axis:
Component graph / decision locus (code). Moves toward the r018 (middleware stack) + r047 (combat-aware ledger,
code arbitration) region, but puts the stage stack around a new multi-hop CHART stage that can pre-empt the model.

## change:
The parent (r029) was a single model call plus an advisory one-hop ledger table. I replaced it with an ordered
middleware pipeline (r018 wiring). The stages share one StepContext. A new RouteStage keeps a persistent per-level
chart and, in its `pre` hook, answers with the BFS first hop toward a drawn `>` or the nearest frontier, so the model
is not called on that step. The model is consulted only when a creature is adjacent (moving into it attacks), when a
modal prompt is not answered in code, or when no route exists (Boxoban pushes, sealed or lit fully charted rooms).
Its reply is snapped to the legal set, then Bump and Hazard `post` hooks veto refused, wall or lethal targets.
Failed moves are booked only on a no-move WITHOUT a combat message (r029's melee-veto bug). Lava parsing also
uses the `}` glyph (r029's dead hazard code).

## why:
The r029 diagnosis names the binding constraint: an advisory table does not steer the VLM (it picks the least-visited
square 33% of the time), orbits are untouched, and a one-hop ledger cannot break a corridor orbit. The r018 and r047
diagnoses both say the missing piece in their pipelines was "no goal and no route". This candidate keeps their
structure but gives the pipeline a route-owning stage. Offline stub rig (the model always answers junk, 0 LLM calls on
routed steps), 40 episodes: MazeWalk-9x9 5/5, MazeWalk-15x15 4/5, Corridor-R3 2/5, others 0 → 0.275 with a null model,
versus 0.150 for r029 with the real model. The real model now only handles fights, Boxoban and Quest, and token cost
drops sharply on the maze tasks.

## files:
agent.py (rewritten), tools/context.py (new), tools/chart.py (new), tools/stages.py (new), tools/action_io.py,
tools/prompting.py, AGENT.md, memory.md (emptied), NOTES.md
