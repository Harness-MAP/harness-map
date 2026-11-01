# cand_r045_0_spatial-grounding — middleware stack over a map-memory Chart

## module map / levers
| file | role | lever |
|---|---|---|
| harness.py | `PipelineHarness`: Chart.observe → StepContext → stage `pre` hooks → ONE model call → legal projection → stage `post` hooks (inner→outer). Resets the chart when the system prompt changes (the agent is reused across tasks). Writes `route=… model:… route:/swing:/veto:… picked=…` to `reasoning` | code |
| tools/chart.py | NEW. Per-level map memory: glyphs, visits, prose-named wall/bars SQUARES, terrain-refused (square, dir) edges; BFS `plan()` to a charted `>` or the nearest frontier (never stood on AND borders uncharted) | code |
| tools/stages.py | NEW. `VetoStage` (lava/water, refused edge, named wall/bars), `SwingStage` (an adjacent creature letter is a fight target, never vetoed or overridden), `RouteStage` (ROUTE note; post-hoc override of a move onto an already-visited square with the BFS hop) | code |
| tools/action_io.py | legal set from the system prompt; whole-word projection with veto + fallback | code |
| tools/prompting.py | priors + stage notes + neighbour ledger (`#` = "corridor or iron bars") | code |
| AGENT.md | explains ROUTE / ledger / keep attacking | prose |

## axis:
Wiring / decision locus (code). Following the steer toward r018 (onion middleware stack) and r039
(map memory + BFS route), the parent's single linear ledger→model→veto path becomes a staged
pipeline over a persistent Chart. Code, not the prompt table, now owns exploration ranking.

## change:
Replaced the neighbour-only `Ledger` with `tools/chart.Chart`. It remembers every square seen and
runs a BFS route. The frontier uses the AND definition (r036's fix for r039's 2-cycle). A no-move
is booked as a blocked edge only when the message is not combat and the target is not a creature:
the refusal message is enough, and so is a silent no-move onto a non-creature. This fixes r029's
bug where every swing was booked as a failure (0/35 repeat swings, 10/10 monster deaths).
Prose `wall/bars/corner … adjacent <dirs>` now convicts those specific SQUARES before the first
bump. Glyphs are never convicted, since `#` is bars in Boxoban but corridor elsewhere.
New `tools/stages.py` holds the Veto, Swing and Route stages, which communicate through a
`StepContext`. RouteStage moves the visit ranking into code: if the model picks a move onto a visited
square, or a vetoed one, while the route leads elsewhere, the route hop replaces it. SwingStage
exempts any attack on an adjacent creature from that override. The `run_episode`/`build_agent`
entry, the legal action set and the one-call-per-step contract are unchanged.

## why:
The r029 diagnosis names three things. (1) The swing booking is a HARNESS-BUG. (2) Period-2 orbits
make up 38–48% of steps, and the model ignores the advisory visit column, which is a VLM ceiling:
"the ranking has to be done in code". (3) The wall/bars prose is not vetoed before the first bump.
r039 showed the model copies a ROUTE hop but used a broken frontier; r018 showed the
pre/post stage wiring. This candidate combines the two and adds a code-owned route override.

## files:
harness.py, tools/chart.py (new), tools/stages.py (new), tools/prompting.py, AGENT.md, NOTES.md
