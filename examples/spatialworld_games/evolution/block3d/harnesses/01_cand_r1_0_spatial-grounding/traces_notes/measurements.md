# r001_0 · spatial-grounding — full offline measurement record

Proposer/diagnostician-facing detail that does not need to sit in NOTES.md. `traces_notes/` is
outside the descriptor's genome (the descriptor skips every path part beginning with `traces`), so
this file is a free place to keep the numbers at full length.

## 1. decoder accuracy, all 20 shipped levels, rendered by the real evaluation adapter

| measurement | result |
|---|---|
| lattice fit resolves three congruent rings | 20/20 — every level reads (left 40, top 30, side 90, pitch 15) |
| decoded silhouettes == the true projections | 20/20 on all three panels |
| admitted volume == the structure exactly | 18/20 (over-approximates on L17: 12 vs 10, and L19: 19 vs 15) |
| hexagon cursor readout == the engine cursor at spawn | 20/20 |
| hexagon cursor readout mid-build | 286/303 and 294/309 over two full sweeps (~95%); every miss is a REFUSAL (several cells cover the occluded highlight), never a wrong cell |

## 2. end-to-end wins against the real adapter, 50-decision wallet

Scripted emitter: the same action returned on every single turn, i.e. the model's contribution held
at its worst. The harness owns routing and the commit, so the only thing the script can change is
whether a proposal is honoured or overridden.

| scripted proposal | wins | decisions | overrides | nulls | cancels |
|---|---:|---:|---:|---:|---:|
| `place_block` | 20/20 | 303 | 187 | 0 | 0 |
| `move_backward` | 20/20 | 309 | 302 | 0 | 0 |
| `move_left` | 20/20 | 309 | 266 | 0 | 0 |
| `move_right` | 20/20 | 309 | 254 | 0 | 0 |
| `move_up` | 20/20 | 309 | 281 | 0 | 0 |
| `move_forward` | 20/20 | 309 | 285 | 0 | 0 |
| `move_down` | 20/20 | 309 | 273 | 0 | 0 |

Per-level decision counts on the pure-route arm: L01 1, L02 5, L03 3, L04 3, L05 6, L06 8, L07 9,
L08 8, L09 10, L10 8, L11 10, L12 15, L13 10, L14 15, L15 20, L16 23, L17 33, L18 26, L19 43,
L20 45. The two over-approximated levels (17, 19) are won as a PREFIX of the build order, before
the admitted volume is complete.

## 3. the ranking key — why it is what it is

Seven orderings were run end-to-end against the real adapter at the 50-decision wallet:

| ranking key | wins |
|---|---:|
| forced · layer · rim-from-HULL-centroid · walk | **20/20** |
| forced · still-owed · layer · rim-from-hull-centroid · walk (shipped) | **20/20** |
| forced · layer · walk · rim-from-grid-centre | 19/20 (loses L19) |
| forced · still-owed · layer · walk · rim-from-grid-centre | 19/20 (loses L19) |
| forced · worth-descending · layer · walk | 19/20 (loses L19) |
| worth-descending · forced · layer · rim · walk | 19/20 (loses L19) |
| forced · layer-DESCENDING · walk · rim | 18/20 (loses L17 and L19) |

Two facts fall out. The rim must be measured from the ADMITTED VOLUME's own centroid, not from the
grid centre: that one change is the whole of L19. And the layer tier must ascend — reversing the
growth direction costs two levels. The still-owed key is win-neutral on this set but is what makes
the endgame well-defined: after a stray has been lifted, the cells whose squares are already
explained must not be re-offered ahead of the ones that still owe something.

## 4. what the parent did, for comparison (from diagnosis.md)

1/20 won; 42.5% of its moves clamped against a uniform-random policy's 15.8%; 14.0 of 216 cells
visited per episode against random's 24.4; 32.2% of decisions after the last new state; 10/20
episodes ending inside a detected periodic orbit; 85.3% of standing blocks on cells no panel asks
for; 21.0% of toggles cancelling an earlier toggle; 0 parse failures, 0 default actions, 0 illegal
actions in 951 completions. The single win is the level whose structure is the spawn cell alone.

## 5. rig notes for the next pass

The offline rig drives the REAL `Block3DGameAdapter` and the real renderer, reads the frame through
the same `capture_game_screenshot` path the evaluator uses, and calls the harness's own
`decode_panels` / `cursor_cell` / `pick_cell` / `choose_key`. The frozen model is replaced by a
scripted proposal so that the harness's own contribution is priced with the model's held constant.
`tools/vlm.py:decide` takes `reader=` and `grab=` for exactly this purpose: blinding the panel
reader must collapse this harness onto the parent's answers, and blinding the frame grab must leave
the booked cursor carrying the episode alone. Both ablations are one line.

## 6. re-verification against the final shipped code

The seven single-action arms in §2 were measured before two cosmetic edits landed (the injectable
`reader=`/`grab=` seams in `tools/vlm.py:decide`, and two defaulting expressions rewritten from
`a or b` to a conditional). Three arms were re-run against the final tree and reproduce exactly:

| scripted proposal | wins | decisions | overrides | sighted cursor |
|---|---:|---:|---:|---:|
| `place_block` | 20/20 | 303 | 187 | 286/303 |
| `move_backward` | 20/20 | 309 | 302 | 294/309 |
| `move_left` | 20/20 | 309 | 266 | 294/309 |

The decoder pass was also re-run against the final tree: lattice fit 20/20, decoded projections
20/20, cursor-at-spawn 20/20, and every level fits at (left 40, top 30, side 90, pitch 15) — a fit,
not a constant table, so the same code returns a refusal rather than a wrong mask if the renderer
moves. The adapter's `pre_eval_gate` logic was run over this directory against the live privileged
vocabulary: PASS (no privileged mention in any agent-facing prior, no privileged attribute or
subscript access in `harness.py` or `tools/*.py`, no syntax error).
