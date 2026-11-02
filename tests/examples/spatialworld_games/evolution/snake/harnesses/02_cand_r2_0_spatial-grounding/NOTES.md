# Snake candidate — the lattice ballot

## axis:
Code lever, token-level. `tools/vlm.py` becomes the harness's main module and interposes on the two
surfaces either side of the one pristine model call: the REQUEST
(`build_conversation_history` swapped in the step function's own module globals for the duration of
that call) and the KEY (`Action.key` projected onto the moves a reckoned lattice says are
pressable). The component graph keeps its shape — five top-level components in `tools/vlm.py`, each
reached only from `decide`, no new module and no new import, with `agent.py`, `tools/action.py`
and `tools/frame.py` untouched — so what moved is control flow and constants INSIDE components.

## change:
`decide` still awaits `evaluator.baseline_step` exactly once per step and still returns the engine
`Action` the evaluator built. Around that call, `LatticeLedger` integrates the head cell, the
heading and a short occupancy window from nothing but the keys this harness itself pressed, plus
two documented engine laws (a 4×4×4 cube, and a 180 the engine refuses outright and which therefore
moves nothing). `_ballot_card` strips every stale ballot from the rebuilt request and appends a
fresh one beside the newest screenshot: the reckoned head, the per-axis face distances, what was
pressed LAST turn and why, and the moves that are pressable now. `_resolve_move` then projects the
model's named action onto that list — kept as-is when legal, else the first legal move on the axis
it named, else the legal move it has asked for LEAST this episode.

## why:
Three things are true of this build at once. Prose alone cannot hold the snake inside the cube: a
maximally explicit per-turn directive was obeyed about a quarter of the time, so the request channel
needs an actuator behind it. A blind substitution is the opposite error — it discards the only food
evidence in the loop, the model's own look at the frame, since the harness has no pixel access and
cannot see the food at all. And the model's answers collapse onto one axis: it almost never names
`move_left` even when the food lies on the −X side, so when its named axis is blocked end to end,
the legal move it has asked for least is precisely the one its own habit is suppressing. Projection
keeps the model's axis wherever the axis survives and de-biases only where it does not. Disclosing
the previous press closes a loop the anchor leaves open: after an override the stored assistant turn
records what the model asked for, not what the engine did, so its own history contradicts the frame
it is looking at.

## measured:
Offline rig in `traces_rig/` (a replica of `_try_move`/`_step` and of the real
`build_conversation_history`), 20 seeds × 100 steps. **A stub model cannot read prose, so the ballot
itself is unmeasurable offline and only the key-side actuator is priced here; food totals are a
survival floor, not a score prediction.** Six individually fatal stub askers — including the
measured live ask mix, a garbage action name and a `metadata=None` parser failure — go from 20/20
deaths at 1–4 steps per episode (or, for always-`move_left`, 2000/2000 refused presses and a total
lock) to **0 deaths and 100/100 steps in every arm**. Dead reckoning agrees with the engine's true
head on **2000/2000** steps and **0** presses would leave the cube; the model's own answer is pressed
unchanged on 38–52% of decisions. Ablations: dropping the projection and keeping only the ballot
returns to 20/20 deaths; dropping the occupancy window is neutral on a random asker but 17 → 1 food
on a cycling one; replacing least-asked with menu order is 27 → 18. Full tables, the module map, the
lever table and the live/dead channel map: `traces_notes/MECHANISM.md`.

## files:
- `tools/vlm.py` — the ledger, the request-side ballot, the key-side projection.
- `AGENT.md`, `rules.md`, `system_prompt.txt` (+ its `.md` mirror) — three measurably false
  perception claims removed: the centre spawn, the 64-point objective, and the direction-letter HUD
  this build never renders. The composed prompt is still byte-equal to `compose_system_prompt()`.
- `NOTES.md` — this note; `traces_notes/MECHANISM.md` — the detail.
