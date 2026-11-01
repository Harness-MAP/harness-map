# Rubik candidate — walk the neighbourhood of the start instead of fleeing it

## axis: code lever — `tools/action.py`'s decision objective, rewritten at token level as plain `def`s

## change:
The parent refused any face turn that returned the cube to a configuration already stood on. That
predicate has the wrong sign — the solution of every level is one specific configuration a few
quarter turns from the episode's own start, so maximising distinct configurations is a
self-avoiding walk *away* from the only neighbourhood that can contain it, and the r001 diagnosis
measured an optimal router capped at 2 of the 12 one-turn configurations under that rule. This
candidate replaces it with the opposite objective. `tools/action.py` now carries an exact cube
state — 20 cubies, built from geometry and advanced only by the move names this harness itself
emitted and the engine reported executed — enumerates the ball of radius 2 around the episode's
start once at import (shells 12 / 114, validated against the real group's 12 / 114 / 1068 and
`order(R)=4`, `order(RU)=105`), and honours a pick exactly when it **shortens the walk's distance
to the shallowest configuration of that ball not yet stood on**. Targets are depth-stratified, so
the twelve one-turn configurations are finished before the hundred-and-fourteen two-turn ones are
sampled, and a tie-break that continues the face already turning makes the emergent tour `X X X X`
per face: the cheapest tour of the near shell, and the one a frozen agent that copies its own last
answer follows for free. Returning through the start is now the normal case. The per-turn channel
shrinks from the parent's inert 14-name ALLOWED list to a **menu of median 2 names**, each carrying
the one fact that distinguishes it, while the invariants move into static prose
(`skills/near_ball_sweep.md`, replacing the refuted macro ladder) because the one channel r001
measured as effective was an unconditional ban. A refused pick is deferred rather than discarded:
it becomes the tie-break as soon as it is the best move on the board. Step 0 — the one echo-free
decision — is never overridden for a legal face turn, which removes the substitution constant the
diagnosis showed *was* the parent's whole 0.050. The one-call decision path, the mute/unmute veto
placement, the four-field `SPEC` rewrite, the parser and the legal action set are unchanged; a
guarded per-step journal now writes the harness's own bookkeeping beside the trace so the override
rate and the coverage are read from the run rather than reconstructed.

## why:
The gradient named one binding constraint and measured its ceiling: with `revisit` active an
optimal ball-covering router reaches 2/12 of the one-turn shell in 100 moves, without it 12/12 in
24, and the live run was already *at* its own 2/12 cap — so no prompt, skill or model change could
move the primary while that predicate stood. The replacement has a floor that does not depend on
the model: because no step can be spent on anything but a distance-reducing move, all twelve
one-turn configurations are stood on **by step 23 under every reply stream tested, adversarial
constants included**, and the evaluator ends the episode the instant the cube is solved, so the
harness never has to recognise the finish — only walk over it. Five of the twenty levels are one
quarter turn from solved. Paired against the parent on the real evaluator and renderer with
stubbed replies, 72 episodes / 36 paired cells: **candidate 17 solved, parent 2**. All four
one-turn levels fall on 3/3 arms at a 30-step cap — including the arm that never names a legal
face turn — and at the real 100-step cap L20 falls on 3/3 while L6 and L8 stay outside the covered
subset of the two-turn shell on every arm, which is the honest shape of the ~42/114 ceiling: that
subset is nearly the same each episode, so a two-turn level is won or lost close to
deterministically rather than at an independent 37 %. Read the override rate with that floor in mind — it is now a compliance measurement, not a
harness choice: 0 % for a stream that follows the menu, 26-31 % for one that follows the three
static rules, 73-100 % for streams that ignore both, with a ~50 % geometric floor in the sampling
phase where exactly one move stays inside the ball. The long version, the two rejected orderings
and the rig tables are in `traces_notes/MECHANISM.md`.

## module map

| file | role |
|---|---|
| `orchestrator.py` | Unchanged evaluator-facing `build_agent(evaluator)` factory and sorted Markdown composer. |
| `AGENT.md` | Unchanged Rubik role and fixed face-centre facts. |
| `skills/near_ball_sweep.md` | The static search contract: no view turns, four turns of one face, the inverse at two turns out, how to read the menu. |
| `rules.md` | Unchanged legal face and view move taxonomy and baseline guidance. |
| `memory.md` | The brief template: position, shell coverage, last turns, what the previous pick cost, the menu. |
| `tools/vlm.py` | The turn: inform, delegate one model call, commit through the router, journal. |
| `tools/action.py` | The 18-action vocabulary, the exact cube state, the radius-2 ball, the frontier router. |
| `tools/frame.py` | Frame capture and the digest used as a render between two execution attempts. |

## validation
`py_compile` on all four modules; `build_agent` imports and composes (4 250 chars, down from
5 500). Cube model checked against the group invariants above. Offline policy simulator over seven
reply models x 100 steps: one-turn shell 12/12 at step 23 in all seven, two-turn shell 42-45/114.
Four tie-break orderings and a wider honoured set measured and rejected (the wider set pushes the
sweep from step 23 to 41-69). The REAL `eval_runner.run_one` over the REAL renderer with
`create_openai_response` stubbed, 72 episodes at 30- and 100-step caps: `used_rich_harness: true`,
`steps == vlm_calls == total_frames`, `Default actions: 0`, one `rotation executed` per step, 0
state desyncs, journal written. The frozen model was not called, as proposer scope requires.

## anti-cheat
Every quantity is a displacement from the episode's **own** start, derived from the action names
this harness emitted. Nothing opens `SpatialWorld/data/`, a scramble table, a solver, cube
internals, the solved flag or the score; the cube model is built from geometry in `tools/action.py`
and validated against invariants that carry no level information. The agent still decides from the
rendered frame plus state it could have derived itself, and the harness narrows the choice set
rather than reading the answer.

## files:
- `tools/action.py` — the exact cube state, the radius-2 ball, the frontier router, the refusal table.
- `tools/vlm.py` — the turn, the menu brief, the commit path, the per-step journal.
- `tools/frame.py` — the capture/digest pair, now documented as a render rather than a signal.
- `memory.md` — the brief template; the dead `word`/`stood`/`screen` fields are gone.
- `skills/near_ball_sweep.md` — new; replaces `skills/macro_commitment.md` (deleted, refuted).
- `orchestrator.py` — docstrings retargeted to the router; the factory and composer are unchanged.
- `traces_notes/MECHANISM.md` — the implementation record; outside the agent's prompt.
