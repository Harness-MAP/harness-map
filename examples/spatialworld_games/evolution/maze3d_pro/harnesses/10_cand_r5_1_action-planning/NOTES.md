# maze3d_pro rich harness — module map & levers

Rich harness for the SpatialWorld multi-floor 3D maze (maze3d_pro). The frozen agent
(Qwen3-VL-30B-A3B-Instruct) and the engine are untouched; only this harness is optimized.
`build_agent(evaluator)` is the entrypoint the extended evaluator installs as `ai_model_func`; the
engine `Action` contract and the maze `success` scoring stay byte-compatible with the prompt-only
baseline. The only observation anywhere in the decision path is `frame_data.image` — the very frame
the model is shown.

## axis

**code / action-planning — THE ROUTE IS PLANNED IN DECISIONS, AND `move_backward` IS ADMITTED.**

An episode is budgeted in exactly one resource: 100 calls to the frozen model. A translation costs
one decision whether it travels one cell or three and it stops itself at a wall; a 90-degree turn
costs one decision and travels nothing. So distance is nearly free and **turning is the expensive
thing** — and every harness in this lineage has nevertheless planned its routes by counting CELLS
and rendered each route step with an if-ladder that emits a turn onto the bearing and then a stride
along it. That renderer can only ever spend turn + turn + forward — three decisions — to reach the
cell behind, and `move_backward` is a legal action it never once emits: 0 of the anchor's 2340
decisions, 0 in every descendant's route, and 0 in the sibling's.

This iteration replaces the renderer with a search in the episode's own currency. `tools/action.py:
search` is a breadth-first search over POSE STATES `(x, y, heading)` whose edges are the legal
decisions — turn_left, turn_right, and forward/backward strides of 1, 2 or 3 map-proven cells — each
costing exactly one. Unit edges make plain BFS optimal, and what it returns is the minimum NUMBER OF
DECISIONS to stand somewhere facing something. Four things then stop being special cases: a reversal
costs 1 instead of 3; a stride packs itself to the longest proven run and clips itself to land
exactly on a target; a frontier's target state is `(cell, bearing)` so the price of a look includes
the turn that performs it; and the chosen target is the cheapest in decisions rather than the
nearest in cells.

Axis-distinct from the sibling `cand_r005_0_spatial-grounding`, which widened PERCEPTION (the flank
ladder: the neighbours of every corridor cell the ray proves open). Nothing here widens perception —
the decoder reads the forward ray and the two immediate flanks, exactly as the exemplars do — and
the sibling's planner still renders routes as turn-then-forward over a cell-counting BFS.

## change

1. **`tools/action.py` (was dead decoration; now the module the axis lives in).** It keeps and wires
   `key_for` / `granularity_steps`, which the diagnosis flagged as referenced only at their own
   `def` lines, and gains the bearing algebra, `Chart` (the dead-reckoned per-floor map) and `Tape`
   (the planner and governor), plus `search` / `look_states` / `reveal` / `worth` / `aim` /
   `inert_reason` / `gains` / `ration` / `escape_step` / `rebuild` at module scope.
2. **`tools/frame.py` (was a dead base64 passthrough).** `capture` is now the frame accessor the
   loop actually uses, and the module carries the exact decode: ONE flat table built by a single
   comprehension over the painter's own formulas, `packed RGB -> (tag, depth, depth_is_exact)`, with
   the painted run's width as the stated fallback for the two families whose shade cannot name a
   depth (the flat exit face, the fog-floored stair border) and for a wall out on the fog plateau.
3. **`tools/vlm.py`** is no longer a two-line pass-through: observe -> state into the system turn ->
   ask the model -> govern. The VLM call itself is still `evaluator.baseline_step`, untouched, so
   parsing and the legal action set stay byte-identical (the chain measures 2340/2340 legal and zero
   parse fallbacks; there is no headroom there). The block is APPENDED to a stashed pristine
   `conversation_history[0]`, never spliced — truncating that turn at a marker deletes the JSON
   RESPONSE FORMAT contract from every later call.
4. **`memory.md` is wired** as the live per-turn template (it was read by no `.py` at all), with
   `Tape.TEMPLATE` standing in when its placeholders do not match `Tape.fields()`.
5. **The prose priors are corrected to the build.** `AGENT.md` no longer promises a POS/FLOOR/DIR
   HUD and `rules.md` no longer gates the climbs on a `[Q] UP` / `[E] DOWN` string: this build draws
   neither, which made both climbs unsatisfiable and 48% of the benchmark unwinnable. `rules.md` now
   states the decision economics instead, and `skills/navigation.md` names `move_backward` as the
   one-decision way to reach the cell behind.
6. **The premise this lineage has never executed is priced rather than trusted.** `move_backward`'s
   semantics — translate opposite the facing, 1..3 cells, stopping early at a wall, heading
   unchanged — come from the action contract and no measurement. So `search` will only plan a
   FIRST reversal that lands on a pose whose frame the chart already holds (`PROBE_FIRST`), and
   `Chart.verify` compares the arriving frame against the digest recorded there: a mismatch at a
   recorded pose proves the contract wrong and switches reversals off for the rest of the episode.
   Digests alias, so agreement is weak evidence and disagreement is a proof; the guard is built
   around the direction that proves something.

Everything the chain has already confirmed is reproduced rather than re-derived, as the diagnosis
instructs: the exact frame-identity outcome bit (no tolerance — one is refuted), a map keyed on a
dead-reckoned CELL and never on the aliasing view, ray-proven "plain floor" kept apart from
flank-reported "open", a floor change gated on the floor being EXHAUSTED, and an optimistic climb
into a fresh map epoch settled afterwards by per-pose digest comparison.

## why

The gradient was a SEARCH-SPACE STEER toward the `cand_r002_1_action-planning` /
`cand_r003_0_spatial-grounding` region, naming the productive difference as its **token-level
implementation**. So the component graph is deliberately NOT where the work went: the module set is
the parent anchor's own five files (`agent.py` + `tools/{__init__,action,frame,vlm}.py`), nothing
added, split out or renamed, and `build_agent` / `compose_system_prompt` keep their signatures and
callers. Everything new is content INSIDE those modules. Measured against the parent, in the frozen
`cvt_v3_k2` standardisation (distance of each phi block to the target region's centroid):

| block | parent | this | r002_1 | r003_0 |
|-------|--------|------|--------|--------|
| wl (structure) | 1.203 | **1.549** | 1.268 | 1.221 |
| ast   | 1.737 | **4.270** | 2.369 | 4.047 |
| num   | 2.182 | **3.202** | 1.409 | 4.316 |
| doc   | 1.404 | **between the two** | 3.167 | 3.536 |

The structure block moves 0.35 while the token blocks move 2.53 and 1.02 — the steered direction —
and both land between the two exemplars rather than past them. Raw token features sit inside the
region too: 34 components / 39 graph nodes (region 21-29 / 28-36), `cf_mean` 8.1 (4.9-9.1),
`num_per_comp` 7.2 (3.4-9.3), `logmag_mean` 0.70 (0.64-0.85). The doc row is given qualitatively on
purpose: this file is itself inside the doc block, so quoting its own distance would not converge.

The mechanism was chosen from the diagnosis's live frontier rather than its confirmed section. At
0.88 the remaining losses are CLOCK losses on long routes (§5.7: one descendant "ran out of clock on
5 losses with 32-54-cell routes"), and the diagnosis prices every decision at ~2.1 s of wall clock
as well. Decision economy is therefore the binding constraint at the top of this chain, and the one
legal action that buys decisions outright had never been used.

## measurement

The decoder is cross-checked against the lineage's measured-exact one (5624/5624 forward ray,
11248/11248 lateral) on synthetic frames painted from the documented formulas across every
`(kind, depth, left, right)` combination: **333/333 agree**, including the stair-border-at-the-exact-
centre case and the d >= 8 fog plateau.

The planner is priced on a stand-in engine — the documented move/turn/climb semantics plus those
painter formulas — driven by six degenerate stand-in proposers (always-forward-1, always-forward-3,
always-turn-right, always-climb, always-backward, uniform random) over 60 generated two-floor levels
at two sizes: **720 paired episodes per arm**, identical levels and identical proposals, one class
attribute changed.

| arm | wins | paired vs shipped |
|---|---|---|
| **shipped** | **385/720** (0.535) | — |
| `OBEY = "off"` (model ungoverned) | 138/720 | +9 / −256 |
| `PROBE_FIRST = False` | 341/720 | +15 / −59 |
| `TRANSIT_ONLY = False` | 351/720 | +13 / −47 |
| `REVERSE = False, STRIDE_MAX = 1` (ancestor-like) | 361/720 | +19 / −43 |
| `REVERSE = False` (the lineage's routes) | 366/720 | +20 / −39 |
| `RANK = "value"` | 381/720 | +9 / −13 |
| `COMMIT = False` | 384/720 | +1 / −2 |
| `STRIDE_MAX = 1` | 386/720 | +9 / −8 |

Read honestly, that is three findings, one neutral-and-kept, and one retraction:

* **Admitting `move_backward` is the result**: +19 of 720 against the lineage's turn-turn-forward
  renderer, and +25 with strides held to one cell, so the gain is not an artefact of stride length.
* **A long stride is only safe over ground already walked** (`TRANSIT_ONLY`, +34). `STRIDE_MAX = 1`
  is itself NEUTRAL (+1), so the value is not in stride length at all — only in refusing to spend a
  long one where it costs information, which is exactly the ground a retreat runs back over.
* **Reversals must be rationed to verifiable ground** (`PROBE_FIRST`, +44, the largest single lever).
  Lifting the gate nearly doubles the reversals (8049 against 4355) and loses 44 episodes, so the
  premise test doubles as the throttle that keeps retreats on ground the chart has stood on.
* **Neutral and kept:** holding the chosen look (`COMMIT`) measured −1. It is kept because it costs
  nothing and removes a known hazard — pricing decisions instead of cells compresses the cost range,
  so looks tie far more often than they did for the ancestors, and re-voting on every tie is the
  unbounded model vote this chain already measured as fatal. Stated as neutral, not as a win.
* **Retracted:** ranking looks by `worth` (decisions per unseen cell) is indistinguishable from raw
  minimum decisions (381 against 385, paired 9 against 13). A 150-episode sample had said `worth`
  was what made reversals pay; the 720-episode sample refutes it. The default is now the simpler
  rule and `worth` stays a documented lever, worth re-pricing against the real engine where
  corridors are longer than the stand-in's.

Dead reckoning held in every traced episode (`desync = 0`, pose never dropped) and the reversal
premise agreed with its recorded frame on every verifiable retreat. Malformed input is covered: a
None / wrong-shape / non-array frame and a proposal with no metadata or an unknown key all degrade
to the model's own action for that step rather than costing the episode.

## honest residue

* The stand-in engine is NOT the real one. Its mazes are generated, its painter reproduces only the
  two windows the decoder reads, and its proposers are degenerate rather than a VLM. The ablation
  DELTAS are the claim; the absolute 0.529 is not comparable to the benchmark.
* `move_backward` has still never executed against the real engine. If its contract is wrong,
  `Chart.verify` refutes it on the first verifiable retreat and the harness degrades to the
  lineage's turn-turn-forward routes — which is the `REVERSE = False` arm, 366/720 on the rig, still
  well clear of every arm that loosens the governor.
* `RANK`, `COMMIT` and `STRIDE_MAX` measured neutral (−4, −1, +1 of 720). They are kept for the
  reasons given in the code and are flagged here as neutral, not as findings.
* The floor-exhaustion proof needs a ray through every reachable cell. On the stand-in's larger
  mazes that does not finish inside 100 decisions, which caps floor changes; the sibling's flank
  ladder is the orthogonal lever that would relieve it.

## module map

| file | role |
|------|------|
| agent.py | ENTRYPOINT. `build_agent(evaluator)` returns the async per-step fn and builds one `Chart` + one `Tape` per episode; `compose_system_prompt()` recomposes the prompt from the .md priors. |
| AGENT.md | role framing + the perception facts this build actually renders (no HUD, depth-shaded corridor, gold exit, cyan stair, the unseen cell underfoot) + how to read the MEASURED STATE block. |
| rules.md | the legal action set, the granularity contract, the DECISION ECONOMICS, and the climb precondition stated as "the block says there is a stair underfoot". |
| skills/navigation.md | the navigation priority order, including `move_backward` as the one-decision reversal. |
| memory.md | the live per-turn template rendered into the agent's system turn (placeholders = `Tape.fields()`). |
| tools/frame.py | the frame accessor + the exact one-frame decode (forward ray, both flanks, frame digest, the travel a forward move gets). |
| tools/action.py | the engine key table, the bearing algebra, `Chart` (dead-reckoned per-floor map), `Tape` (the decision-cost planner + governor) and the module-scope search/economics/governor helpers. |
| tools/vlm.py | the per-step loop: observe -> inject -> ask `baseline_step` -> govern. |

## evolvable levers (axes)

* prose: `AGENT.md` / `rules.md` / `skills/*.md` / `memory.md` — what the agent is told.
* code: `agent.py` / `tools/*.py` — the mechanism.
* swept, as `Tape` class attributes (set them AFTER `build_agent`, which purges
  `sys.modules["tools*"]` and throws away anything set before): `STRIDE_MAX`, `REVERSE`,
  `PROBE_FIRST`, `TRANSIT_ONLY`, `RANK`, `COMMIT`, `OBEY`, `FLOOR_GATE`, `STALE_LIMIT`,
  `MAX_CLIMBS`, `RESWEEP`, `STALL_LIMIT`; and on `Chart`: `DESYNC_LIMIT`, `TWIN_MATCHES`,
  `TWIN_WINDOW`.

## anti-cheat

Decide from the observation only. The decision path reads `frame_data.image` and nothing else; the
only evaluator attributes touched are `baseline_step`, `conversation_history`, `current_prompt` and
`max_steps`. No `evaluator.game` access of any kind (that is a scored void), no read under `data/`,
no exit coordinates, no floor label, no layout file, no true distance. `baseline_step` is called on
every single step, so the VLM-call counter can never void the run.

## files

`agent.py`, `tools/__init__.py`, `tools/frame.py`, `tools/action.py`, `tools/vlm.py`, `AGENT.md`,
`rules.md`, `skills/navigation.md`, `memory.md`, `NOTES.md`. `system_prompt.txt` and
`system_prompt.md` are untouched (declared non-levers; `compose_system_prompt` never reads them).
