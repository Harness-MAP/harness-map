# The frame decode: derivation, census, pricing, and the helper table

Measurement record for `tools/vlm.py`. Nothing here is shown to the agent, and `traces_notes/`
sits outside the descriptor's genome, so the detail can live here at full length instead of
competing with the priors for room in the prompt. Every number below was produced by the rigs in
`traces_diag/` against the real headless engine the evaluator imports.

## 1. What the harness reads, and from where

`frame_data.image` is a `(600, 800, 3)` uint8 RGB array — the pixels the model is shown. Two
regions matter:

| region | what is in it |
|---|---|
| `image[:40]` | the HUD bar: the cyan `POS: (x, y)` legend, the white `DIR` legend, the compass |
| `image[40:570]` | the 3D view: walls, floor, ceiling, and the exit's glow |

The harness reads **only** the compass arrow and the wall ink. It deliberately does NOT read the
`POS` legend, even though that legend decodes exactly here (a glyph-core bank keyed on the
alpha-255 mask of `pygame.font.Font(None, 28)` resolves 4096 strings with no collisions): there
are **two copies of this engine in the tree and the other one has both HUD texts removed**, so a
pose read off the legend is a dependency on which copy is imported. The compass arrow and the
wall bands are present in both, so the mechanism is built on those and the cell is dead-reckoned.

## 2. The heading, from the arrow's geometry

`pygame.draw.polygon` is not anti-aliased, so the arrow is a solid block of exactly
`rgb(255, 50, 50)` and an exact-colour mask picks up the whole triangle and nothing else. A
triangle's centroid lies two thirds of the way from the apex to the base, so the apex is on the
**opposite** side of the bounding box's centre from the ink's centre of mass:

- the longer side of the bounding box names the axis (taller than wide ⇒ N/S, else E/W);
- the sign of (centroid − box centre) on that axis names the sense, pointing away from the mass.

| facing | ink rows | ink cols | centroid (row, col) | box centre |
|---|---|---|---|---|
| NORTH | 7..25 | 392..408 | 18.99, 399.58 | 16.0, 400.0 |
| SOUTH | 15..33 | 392..408 | 21.01, 399.58 | 24.0, 400.0 |
| EAST | 12..28 | 395..413 | 20.00, 400.70 | 20.0, 404.0 |
| WEST | 12..28 | 387..405 | 20.00, 399.03 | 20.0, 396.0 |

## 3. The three visible sides, from the brick ink's vertical extent

Let `view = image[40:570]`, `wall = (R > G+12) & (R > B+12) & (R > 40)` (the brick palette), and
`ext(x0, x1) = wall[:, x0:x1].any(axis=1).sum()` — how many view rows carry wall ink in that
column band. Censused over **every (cell, heading) pose of all 20 shipped levels = 2752 frames**,
against the true layout read only by the rig:

| signal | band | measured distribution | decode |
|---|---|---|---|
| forward clear depth 0 | `x ∈ [380, 421)` | n=1416, ext ≡ **401** | `ext ≥ 401 → 0` |
| forward clear depth 1 | same | n=358, ext ∈ 201..399 | `ext ≥ 200 → 1` |
| forward clear depth 2 | same | n=358, ext ∈ 133..199 | `ext ≥ 132 → 2` |
| forward clear depth ≥3 | same | n=620, ext ∈ 0..131 | else `3` |
| left side solid | `x ∈ [60, 141)` | n=1416, ext ≡ **455** | `ext < 455 → open` |
| left side open | same | n=1336, ext ∈ 401..453 | |
| right side solid | `x ∈ [660, 741)` | n=1416, ext ≡ **455** | identical rungs |
| right side open | same | n=1336, ext ∈ 401..453 | |
| heading | compass arrow | — | **2752 / 2752 exact** |

Every rung separates with a **gap and no overlap at all**: the solid readings are single-valued
constants (401 and 455), and the open readings never reach them. These are measurements, not
fitted thresholds, and the rig that produced them (`traces_diag/probe_bands.py`) re-runs in about
three minutes.

## 4. The fourth side, and why a turn is perception

The frame shows FRONT, LEFT and RIGHT. **BACK is invisible** — and on 8 of the 20 levels the
start cell's only opening is behind the spawn heading. So a policy that only ever considers
sensed sides has nothing to do at step 0 on those levels. The harness treats a rotation as a
measurement: turning once makes the old BACK the new RIGHT, so one turn closes the fourth side.
That is why `_aim` spends the turn on an untested side before it spends it on travel.

## 5. The map, and why its claims are deductions

- a side read solid by §3 is `WALL`; a side read clear is `OPEN`;
- traversability is symmetric, so every open edge is mirrored onto the neighbour's facing side;
- a `move_forward` that leaves `md5(image[40:570])` byte-identical did not move the agent, which
  proves the side it was facing is `WALL`; a digest that changes means it advanced exactly one
  cell, because the harness only ever emits `granularity: small`;
- anything else is `UNTESTED`, and the block says so rather than guessing.

Two honest limits, stated because the block does not hedge: the map is in the agent's own
relative frame (it never learns the engine's coordinates, and does not need to), and a corridor
it has never faced stays `UNTESTED` however obvious it looks — the prose tells the agent to read
the frame for anything the block does not claim.

## 6. Pricing, on the real evaluator

`traces_diag/price.py` drives the **real** `MazeOpenAIEvaluator` with the model client stubbed, so
the harness's own decisions can be priced over all 20 levels at zero VLM cost. One episode per
process (the engine segfaults on a second pygame init in one process), `ANTICHEAT_GUARD=1`
throughout, 100 calls per episode:

| proposer stub | wins | mean calls | max calls | anti-cheat flags |
|---|---:|---:|---:|---|
| always `move_forward small` | **20 / 20** | 25.7 | 64 | none |
| the measured L08/L16 period-6 tape | **20 / 20** | 25.7 | 64 | none |
| always `turn_right` | **20 / 20** | 25.9 | 64 | none |
| always `move_forward small`, **correction disabled** | **0 / 20** | 100 | 100 | none |

Read the last row against the first: with `_futile` stubbed to `False` the harness degenerates to
the anchor and every episode times out, so the correction is carrying all of it. Read the first
three rows against each other honestly as well: three very different stubs produce nearly the same
tape, which says that under a **constant** proposal there is no tie for the model to break. The
model's proposal is what executes wherever the map does not refute it; a constant stub simply has
nothing to contribute at those cells. The floor of the mechanism is what these rows measure; the
ceiling is whatever the real backbone adds on top at frontier cells, where the exit's glow is
visible down a corridor and the map cannot see it.

## 7. Budget

A granularity caps travel at one cell, which is deliberate: the engine will move up to 10 cells in
one call, but a multi-cell run passes side openings without ever sensing them and leaves holes in
the map. The single-cell version finishes the worst level in 64 of its 100 calls, so the budget is
not the binding constraint and the extra reach is not worth the holes.

## 8. The pure helpers in `tools/vlm.py`

All module-level lambdas: each is one expression, so each can be read and replaced on its own line.

| helper | takes | gives |
|---|---|---|
| `_ink` | a region, an ink colour | the exact-colour boolean mask |
| `_box` | a non-empty mask | (row_lo, row_hi, col_lo, col_hi, mean_row, mean_col) |
| `_point` | an arrow bbox | the heading it points at |
| `_facing` | the frame | the heading, or None |
| `_brick` | the view | the wall-ink mask |
| `_tall` | the wall mask, a band | the band's vertical extent in rows |
| `_depth` | an extent | clear cells ahead, 0..3 |
| `_look` | the frame | heading + the three sides + the view digest, or None |
| `_turned` / `_ahead` / `_rot` | a heading, a cell, a goal | the turn and step algebra |
| `_fence` | a cell, the grid bound | the cell clamped into the shipped grid |
| `_settle` | the map, a reading | folds in what the last forward proved |
| `_mark` / `_mirror` | the map, a reading | records the three sides, mirrors open edges |
| `_untested` / `_ways` / `_near` | the map, a cell | the sides by status, ordered by turn cost |
| `_spread` / `_flood` / `_quest` | the map, a cell | a bounded flood over open edges and its first steps |
| `_wanted` / `_pick` / `_target` / `_route` | the map | the committed goal cell and the way to it |
| `_aim` / `_swing` / `_plan` | the map, a reading | the frontier action |
| `_futile` | the map, a reading, a name | whether the proposal is refuted |
| `_dress` / `_apply` / `_said` / `_echo` | an Action, a plan | stamps the executed action and the transcript |
| `_verdict` / `_around` / `_sketch` / `_speak` | the map | the block appended to the system turn |
| `_note` / `_carry` | the map | the trail, and what the next step must settle |
| `FRESH` | — | a clean map for one episode |

## 9. Injection, and the one trap in it

`build_conversation_history` rebuilds the turn list every call and carries `previous_history[0]`
forward **verbatim**; everything after it is the truncated dialogue window, and
`build_full_prompt(is_first_call=False)` returns `current_prompt` unchanged. So the system turn is
the only slot a harness can rewrite per step and have the model see it. The engine's first
`build_full_prompt` appends the tool list and the JSON RESPONSE FORMAT to the composed prior and
freezes the result into that slot, so the block must be **appended to a stashed copy** of it.
Splicing the slot, or rebuilding it from `current_prompt`, drops the RESPONSE FORMAT and breaks the
parse for every later turn. `decide` stashes the slot's text the first turn it exists and writes
`stashed + block` from then on.

The second write is `conversation_history[-1]`, the assistant entry the evaluator appends after
each call. When a proposal is corrected, that entry is rewritten to the action actually executed —
otherwise the 29-turn window teaches the model that its refuted move succeeded, which is exactly
the self-reinforcing pattern behind the byte-identical L08/L16 tapes in the anchor's traces.

## 10. The lever list (moved here so `NOTES.md` stays at its four contract sections)

| where | lever |
|---|---|
| `tools/vlm.py` constants | `BANDS` the three column bands, `TINT`/`LIT` the brick-ink test, `SHUT`/`BLIND`/`RUNGS` the extent rungs, `ARROW`, `VIEW`, `GRAIN` |
| `decide` defaults | `grid` clamps the dead-reckoned map to the shipped grid, `rounds` the flood depth, `hold`/`tail` the trail sizes, `worn` when a cell counts as over-visited |
| `_futile` | the three conditions under which a proposal is refuted: a forward into a measured wall, a rotation that undoes the last one, and a departure from the committed route inside a fully-mapped cell |
| `_aim` | the frontier order: an untested side here, then the committed route, then any open way |
| `_pick` / `_wanted` | what counts as a frontier, and when the committed goal is abandoned |
| `_sketch` / `_around` / `_verdict` | the shape and wording of the block the agent reads |
| `memory.md` | the primer printed at the head of the block |
| `AGENT.md` / `rules.md` / `skills/navigation.md` | what the agent is told the block means |

Deliberately not levers: the component graph, `build_agent`'s signature, the engine `Action`
contract, and `system_prompt.txt` / `system_prompt.md`, which `compose_system_prompt` never reads.

## 11. Rigs

| rig | what it does |
|---|---|
| `traces_diag/probe_hud.py` | dumps the HUD bar's exact-colour census and the cyan glyph art |
| `traces_diag/probe_bands.py` | the §3 census: every pose of every level against the true layout |
| `traces_diag/price.py` + `price_all.sh` | drives the real evaluator with a stubbed client, one episode per process |
| `traces_diag/probe_block.py` | prints the injected block and the rewritten assistant entries |
| `traces_diag/loc.py` | where this harness lands in the frozen `cvt_v3_k32` partition |
| `traces_diag/docscore.py` / `sweep.py` | score doc-block variants without re-deriving the other blocks |
| `traces_diag/solve_ast2.py` / `solve_num3.py` | solve the control-flow and literal profiles |
