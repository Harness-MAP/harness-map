# Snake candidate — the pellet warrant, and a second chance to satisfy it

## axis:
Code lever, token-level: `tools/vlm.py` is rewritten and no other module is touched.
`harness.py`, `tools/action.py`, `tools/frame.py`, `tools/__init__.py` and every `.md` prior are
byte-identical to the parent — `compose_system_prompt()` returns the parent's 1624 characters
exactly — so the module wiring (`harness.build_agent` -> `tools.vlm.decide` ->
`evaluator.baseline_step`) is untouched and what moved is the control flow and the constants
INSIDE one module. The parent's `decide` was a one-line pass-through; this one is a warrant the
frozen model is given two chances to satisfy.

## change:
Three public readings meet in `decide`, and a warrant follows from them.

* **Reckoning.** `_recall_trail` keeps an episode-local ledger on the evaluator — head cell,
  heading, trail, per-cell visit counts — integrated from the engine's published opening cell plus
  the keys this module has pressed. The engine's own head is never consulted.
* **The HUD.** `_tally_glyph` decodes the `Score: N` digit the renderer blits into the corner of
  the very screenshot the model is shown, by per-column white-pixel profile. That is the exact
  pellet count, hence the exact body length, so the harness stops guessing the length with a fixed
  window; a reading is accepted only when it does not fall, because the engine's counter never
  goes backwards.
* **The pellet.** `_pellet_cell` snaps the one warm blob on the board to the cube cell whose drawn
  diameter matches the visible fragment. The front end keeps the WHOLE warm mask rather than the
  largest connected blob above a minimum area, and the size gate is one-sided, because when the
  snake is adjacent to the pellet its own disc covers part of it and the fragment is routinely
  SMALLER than the cell's disc.

`_moves_open` then names the moves the engine would actually execute — in the cube, not the 180 it
refuses, not into a body cell — and `_pick_press` ranks them by lattice gap to the pellet, then by
how many continuations the arrival cell keeps, then by least-trodden, then menu order.

The judgement is where this differs from a clamp. An answer inside the warrant and tied with the
top rank is pressed untouched. An answer outside it is **not silently overwritten**:
`_correction_card` states what the harness reckons and reads, `_retry_once` rolls the unusable
user/assistant pair off the end of model-visible history, splices the card into the live per-turn
text channel (`conversation_history[0]["content"][0]["text"]`, written as `base + MARK + card` and
always restored), and the frozen model answers the SAME screenshot once more. Only if that second
answer also fails the warrant does the harness write its own key. The appeal is free: the episode
budget counts calls to `decide`, not calls inside one.

## why:
The parent's diagnosis is unambiguous about the binding constraint and about what is left after it.
`decide` forwarded the model's key unreviewed, so 20/20 episodes walked the head out of the cube by
step 4 and `primary` was 0.0000 with a safe distance-closing move available on 63 of 67 presses.
That half is a code defect. What the diagnosis calls the remaining open problem is AGENCY: the rule
that fixes the score presses the harness's own key on about 72% of decisions, and no
distance-respecting rule can honour more while the model's legal asks close the gap only a quarter
of the time. Both sibling candidates ship a clamp with a de-biasing substitute and neither reads
the pellet, so the pellet-ward rank was unclaimed; and the diagnosis's own unmeasured suggestion —
re-ask once, for free, before substituting — is what this candidate is built around. A substitution
the model never hears about teaches it nothing; an appeal it can satisfy gives the key back.

## measured:
`traces_phi3/warrant_rig.py` drives the real renderer and adapter over 20 seeds and runs the
shipped `decide` unmodified behind a stand-in evaluator that rebuilds the request exactly as the
engine does, so the appeal plumbing is exercised for real; only the language model is a stand-in.
Across four ask models (the live measured mix, uniform, an always-`move_forward` latcher, and one
emitting unparseable names): **primary 1.0000, 0 deaths, 80/80 pellets in every arm**. Reckoned
head == the engine's head **2505/2505**; HUD digit == the engine's score **2505/2505**; pellet
decode by head-pellet distance 640/640 · 622/622 · 504/506 · 726/737 = **1.0000 at the adjacency
frames** that decide whether it eats, 0.9948 overall; the warrant was empty (every move a wall,
the refused 180 or the body) on 0 of 2505 decisions.

Ablations, live ask mix: press the ask (the parent) 0.0000 / 20 deaths · clamp only, pellet
discarded 0.2250 · appeal channel closed 1.0000 but honour 0.245 · appeal ignored 0.402 · appeal
heeded 0.611 · fixed 6-cell body window and no continuation tie-break both neutral at 1.0000.
So the warrant is the score and **the appeal is the agency**: 0.245 -> 0.402 -> 0.611.
`traces_phi3/decodecmp.py` prices four pellet decoders: dropping the minimum-area blob filter is
what buys the adjacency frames: largest-blob + symmetric gate is 0.6750 at distance 1, and all
three of the other combinations are 1.0000 there.

## honest limits:
The offline rig cannot price the appeal's language channel: the heeded arm is an upper bound, the
ignored arm the floor, and the parent's own probe of this channel saw a maximally explicit
directive obeyed on 1 of 4 turns — so expect live honour nearer 0.40 than 0.61. The score claim
does not rest on it (the appeal-closed arm also scores 1.0000). `−HUD` and `−room` measure neutral
because four pellets ends the episode before the body passes six cells. The `.md` priors are left
byte-identical to the parent although the diagnosis shows three of their claims false on this
build; both siblings already took that axis, and the appeal card states the measured facts per
turn instead.

## anti-cheat:
Every input is public. The opening cell is a build constant the parent's own `AGENT.md` already
states; everything after it is the keys this module pressed. The pellet and the pellet count come
from the screenshot the model is shown. Nothing reads `evaluator.game` beyond what the guard
allows, nothing under `SPATIALWORLD/data/`, not `evaluator.last_score`, and nothing draws from the
seeded global `random` stream. `baseline_step` is awaited at least once on every decision, so the
model decides first on every turn and `_vlm_calls` is never zero.

## module map:
| file | state |
|---|---|
| `harness.py` | unchanged. `build_agent` installs the composed prior and returns `tools.vlm.decide`. |
| `AGENT.md`, `rules.md`, `memory.md`, `skills/strategy.md` | unchanged, byte for byte. |
| `tools/action.py` | unchanged — and finally wired: `is_reversal` and `DIRECTION_VECTORS`/`KEY_MAPPING` are now read by `tools/vlm.py` instead of being named only in a docstring. |
| `tools/frame.py` | unchanged (still dormant: `frame_data.image` is current, so no capture wrapper is needed). |
| `tools/vlm.py` | THE CHANGE. Eight components: the ledger, the HUD digit, the pellet, the open set, the rank, the appeal card, the appeal, and `decide`. Tables, pixel windows and the lattice algebra sit at module scope. |
| `traces_phi3/` | the rigs: the live-renderer harness test, the ablation ladder, the decoder comparison, the descriptor probes. Not part of the genome. |

## descriptor aim:
This round's steer named behaviour cell 3 of K0=32 by coordinates, and that cell has exactly ONE
bootstrap founder, so its centroid IS that founder's descriptor point and the three non-text
blocks can be matched outright. `traces_phi3/loc3.py` (the frozen `cvt_v3_k32` partition replayed
offline) and `traces_phi3/graph3.py` / `ast3.py` / `astsolve.py` were used to aim, then to check:
the labelled component graph matches the founder's h1 signature multiset **exactly** (wl 0.0000),
`f_num` matches **exactly** (six in-component literals, four of magnitude one), and `f_AST`'s type
totals match exactly with only `cf_max` and `cf_std` left over (ast 0.0444). So the shipped point
lands in **cell 3 with its whole residue in the text block** and roughly +0.3 of runner-up margin
(the runner-up is c21, then c12 — the parent's own cell). 28 of 28 honest NOTES orders held the
cell, so the basin is wide and the order here was chosen for the diagnostician's benefit rather
than for margin. The aim did not bend the mechanism: what it did decide is that the pixel
tables, the camera re-derivation and the lattice algebra live at module scope — which is where
they belong anyway — leaving each component the decision it names.

## files:
- `tools/vlm.py` — rewritten: the reckoning ledger, the HUD pellet count, the occlusion-tolerant
  pellet decode, the open-move set, the gap-closing rank, the appeal card, the one-turn appeal on
  the live text channel, and the `decide` that arbitrates between them.
- `NOTES.md` — this note. Not shown to the frozen agent.
- Nothing else in the genome is touched.
