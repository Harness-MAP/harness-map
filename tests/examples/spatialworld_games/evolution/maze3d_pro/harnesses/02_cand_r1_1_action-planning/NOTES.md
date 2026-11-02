# maze3d_pro rich harness — module map & levers

Parent: `anchor_maze3d_pro` (the rich anchor, which delegates every decision to the evaluator's
pristine VLM step and keeps no state). Iter 1, axis **action-planning**. The frozen agent
(Qwen3-VL-30B-A3B-Instruct) and the SpatialWorld engine are untouched; only this harness changed.

## module map
| file | role |
|------|------|
| agent.py | ENTRYPOINT. `build_agent(evaluator)` returns the async per-step fn and now builds ONE `tools.plan.ActionPlanner` per episode and threads it through `tools.vlm.decide`; `compose_system_prompt()` recomposes the prompt from the .md priors (unchanged, byte-identical output). |
| **tools/plan.py** | **NEW — the mechanism.** `ActionPlanner`: measures each action's outcome by comparing consecutive `frame_data.image` frames, keeps a refused-translation ledger keyed on the frame fingerprint, governs the model's proposal, and renders the per-turn working-memory block. |
| tools/vlm.py | the perceive->VLM->parse->GOVERN step. Still calls `evaluator.baseline_step` every step (parsing/fallback/legal set untouched); adds observe -> inject-memory -> govern around it. Owns `_inject_memory`. |
| tools/action.py | the action/key algebra, now ACTIVE: `name_for(key)` (the inverse table — the engine dispatches on `action.key`), `inverse_climb`, and `retarget(...)` which builds a re-aimed Action from a proposal's metadata envelope. |
| memory.md | the RUNTIME working-memory template rendered into the agent's system turn each step (was an inert placeholder). `plan.DEFAULT_TEMPLATE` stands in if it is missing or its placeholders do not match. |
| AGENT.md / rules.md / skills/navigation.md | **unchanged** — the prose priors are deliberately held fixed so this iter measures the mechanism alone. |
| tools/frame.py | frame capture — reuses `evaluator.capture_game_screenshot` (unchanged). |

## evolvable levers (axes)
- prose: AGENT.md / rules.md / skills/*.md — what the agent is told (untouched this iter).
- code:  agent.py / tools/*.py — the mechanism. Within tools/plan.py the live sub-levers are the
  outcome test (`_SAMPLE` / `_DIFF_EPS` / `_NULL_FRAC`), the ledger key (currently the frame
  fingerprint), the veto set in `govern`, the escape rule in `_reaim`, `stall_limit`,
  `climb_lock_span`, and `turn_sense` (handedness).
- memory: memory.md — the wording of the injected working-memory block.

## anti-cheat
Decide from the observation only. The planner's ONLY input is `frame_data.image`, the frame the
agent is shown; no `evaluator.game` attribute is touched, no layout/coordinate/distance/floor field
is read, nothing under the data tree is opened. Verified: the `pre_eval_gate` replica returns None
(no privileged mention in any agent-facing prior, no privileged attribute/subscript in any .py), and
an AST walk of all five modules finds no engine-state attribute at all.

## axis
**code / action-planning** — `tools/plan.py` (new): a closed-loop action-outcome governor around
the frozen model's proposal, plus the per-turn memory channel that makes its verdicts legible to
the model. One module, one mechanism; no prose prior was reworded.

## change
The anchor's 23 losses all burn the full 100-step budget, and the traces say why: nothing in the
loop can tell a move that happened from a move that did not. Levels 2/5/18 emit `move_forward` on
97-100 of 100 decisions from a single pose; level 1 is a 25x `turn_right, move_forward, turn_left,
move_forward` orbit; the lineage diagnosis measures 1647/1741 translations (94.6%) refused and four
episodes spending 87-92 decisions on an up/down 2-cycle at one stair.

`ActionPlanner` closes that loop. Each step it fingerprints the frame the agent is shown (every 2nd
pixel, md5) and scores the PREVIOUS decision: a translation that left the screen unchanged did not
happen, so the cell ahead is a wall — and because the engine stops a multi-cell stride early at a
wall, "zero cells moved" convicts the first cell whatever granularity was asked for. That verdict is
filed against the fingerprint, i.e. against the agent's whole first-person view, so it is exactly
re-usable whenever that view returns (including after a four-turn circuit of the same cell) with no
pose, no map and no coordinates. Equality is deliberately tolerant — identical fingerprints mean
unchanged, and a changed fingerprint still counts as unchanged below 0.2% of sampled pixels — so a
HUD tick cannot fake a successful move while a real step or turn, which repaints most of a
first-person view, always registers (measured both ways).

The frozen model is then asked on EVERY step and obeyed unless its proposal is provably useless:
a translation this exact view already refused, a climb that would undo the floor change just made
(locked until a translation succeeds, with an 8-decision release for a genuinely trapped landing),
or a turn chosen after 4 decisions without moving. In those three cases the escape step is computed
first and the model is still obeyed whenever it already agrees, so agreement is free. The escape is
a right-hand wall follower: probe ahead, and if that is refused rotate to the next bearing in the
order right, straight, left, back — one right turn on arrival, left turns after it, which leaves the
came-from bearing for last. That ordering is the load-bearing detail: the first version rotated
always the same way, tried the came-from bearing SECOND, and walked back out of every dead end, a
2-cell orbit the rig caught at 2 cells covered in 100 decisions. A bearing already in the refused
ledger is rotated past without spending a probe, and the sweep releases the instant a translation
succeeds, handing the model back the wheel.

A veto the model cannot see would just be re-proposed, so `render_memory()` states the ledger back
to it in plain English — last proposal and whether the screen moved, what this exact view has proven
refused, what it has never tried, the stair lock, the mode — appended to
`conversation_history[0]`, the one channel the engine re-reads every turn (`current_prompt` is
consumed by `build_full_prompt` on the first call only). The base system turn is stashed whole and
the block appended after it, never spliced, because truncating it would delete the JSON RESPONSE
FORMAT contract from every later turn; the block itself closes with the JSON-only reminder. Both
`retarget` and the whole governor are failure-contained: granularity, repeat AND json_data are
stripped before a re-aimed turn (a stale `repeat: 3` would fire three 90-degree turns, since the
input source only forces repeat=1 when a granularity is present), and any exception anywhere in the
governor falls back to the model's own Action.

Offline rig (real `tools/*` driven against a stub maze + stub models replaying the anchor's measured
pathologies, 100 decisions each) — cells covered, ungoverned -> governed: forward-spam 1 -> 23,
forward-spam/medium 1 -> 23, turn-forward-turn-back orbit 2 -> 23, pure spinning 1 -> 15,
forward-spam from a mid-maze pose 7 -> 23, stair up/down 2-cycle 2 cells/no progress -> solved in 10
decisions. Refused translations fall from 94-100 to 5-20 per 100. Overrides are 18-52 of 100 against
stubs that propose a provably refused action nearly every step, and the no-op control matters more:
replayed over an already-solving script the governed and ungoverned executed-action tapes are
byte-identical with ZERO overrides, so an episode the anchor was already winning is untouched.
Overhead is 1.9 ms/step at 1024x768 against a ~2000 ms VLM call, and `baseline_step` is still called
every single step (so the guard's VLM-call counter can never void the run).

## why
The gradient is the parent's own 2/25 with 23 episodes at exactly 100/100 steps and
`distance_to_exit` residues of 1-15 — a clock loss, not a routing loss — plus the lineage
`diagnosis.md` for this same anchor, which classifies the binding constraint as VLM-CEILING
(harness-fixable) and ranks "add an observation-derived action-outcome governor" first, with the
one-way stair lock second and the compact outcome memory third. Those three are one mechanism, so
they ship together as one module. The hypothesis is narrow: the frozen model's perception and syntax
are already sound (2479/2479 legal parsed actions, 0 defaults, 364/364 legal climbs), so the
recoverable loss is the ~95% of decisions spent on actions whose uselessness is visible in the
frames the agent was already shown. Ruled out as exhausted/low-value by the diagnosis and
deliberately not touched: JSON parsing, the fallback, the key table, granularity spelling, generic
"look carefully / avoid loops" prose, and shrinking the history window alone.

## files
- `tools/plan.py` — NEW: `ActionPlanner` (outcome measurement, refused/tried/view ledgers, the
  governor, the right-hand-wall-follower escape, the stair lock, `render_memory`).
- `tools/vlm.py` — `decide()` now observes -> injects memory -> delegates to `baseline_step` ->
  governs; added `_inject_memory` (stash-and-append on `conversation_history[0]`).
- `tools/action.py` — added `name_for`, `inverse_climb`, `retarget`, `TURN_ACTIONS`,
  `CLIMB_ACTIONS`, `LEGAL_ACTIONS`, `REVERSE_KEY_MAPPING`.
- `agent.py` — `build_agent` builds the per-episode planner and threads it into `vlm.decide`;
  added `_int_or`.
- `memory.md` — now the live runtime template for the injected block.
- unchanged: `AGENT.md`, `rules.md`, `skills/navigation.md`, `tools/frame.py`, `tools/__init__.py`,
  `system_prompt.txt`.
