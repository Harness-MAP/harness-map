# cand_r002_1_action-planning — the long record

`NOTES.md` is the short version the diagnostician and the next proposer read. This file is the
implementation record: how each fact about the renderer and the engine was measured, what the
cube model is, what was rejected, and every number from the offline rig. It lives under
`traces_notes/` because the descriptor's doc block skips any path whose parts start with `traces`
(`harness_map/qd/descriptors.py::DEFAULT_SKIP_PREFIXES`), so the length of this record cannot move
the candidate in the archive.

## 1. The gradient this answers

`diagnosis.md` (r001) measured the binding constraint and its ceiling:

* the parent's `revisit` rule has the wrong sign — it maximises *distinct configurations stood on*
  (median 4 → 101) while the solution of every level sits **within four quarter turns of the
  episode's own start**, so the walk left that neighbourhood by step 4–12 and never returned
  (steps inside the ≤4 ball: 61.7 % → 8.8 %);
* under that rule an *optimal* ball-covering router reaches **2 of the 12** one-turn configurations
  in 100 moves, and the live run was already at that cap, so no prompt or model change could move
  the primary;
* the whole 0.050 primary was the harness's own step-0 substitution constant;
* the frozen `Instruct` model answers in 22–28 tokens, repeats its own previous answer 49.3 % of
  the time, copies the harness's last played move 65.4 % of the time, and showed **no measurable
  response** to a 14-name per-turn ALLOWED list (refusal ratio 1.09 against chance). The one
  channel with a measured effect was a **static unconditional ban** (z view rotations: 39.5 % of
  picks → 0.4 %);
* `state_callback` stops the episode the instant `cube_solved` is true, so the harness never has
  to *recognise* solved — only walk onto it.

The sibling candidate (`cand_r002_0_spatial-grounding`) answers that gradient the way §5.1/5.2 of
the diagnosis proposes: it keeps the walk inside the near ball and tours it. That is a coverage
policy, and coverage has an arithmetic ceiling — 12/12 of the one-turn shell and ~40 of the 114
two-turn ones in 100 moves, i.e. the five one-turn levels plus whichever two-turn levels happen to
fall in the covered subset. Nothing blind reaches the three-turn (1068) or four-turn (10011)
shells. **This candidate removes the blindness instead of optimising the search.**

## 2. What the frame actually is (measured, not assumed)

Everything below came from running the real evaluator and the real renderer offline with the model
client stubbed (`eval_runner.run_one`, `SDL_VIDEODRIVER=dummy`, `create_openai_response` replaced),
and decoding the captures the agent itself is shown. The engine/renderer source is off-limits to
the proposer and was not read.

**The cube is drawn FACE-ON, not in perspective.** At the home camera pose the 500×500 frame is a
flat `(128,128,128)` backdrop carrying one exact, axis-aligned 3×3 grid of sticker squares: cells
run `[131,209] [211,289] [291,369]` on both axes (79 px squares, 1 px gutter). A dark glyph is
stamped over the centre sticker and a static unfolded-cube legend sits in the bottom-right corner,
which is what fixes the colour of each face: UP yellow, DOWN white, FRONT red, BACK orange, LEFT
blue, RIGHT green. The arithmetic confirms the geometry exactly — a face showing six red and three
green stickers measures `R = 37786` and `G = 19591` painted pixels, and `19591 − 868 = 18723 =
3 × 79²` where 868 is that colour's legend swatch.

So **nine sticker colours are readable for free, every step, with no camera move at all.** A
majority vote over a 6×6 lattice inside each cell returns the sticker; the glyph is a minority of
the centre cell and loses the vote.

**A view rotation is 45°, and only even multiples are readable.** `rotate_cube_y_cw` ×2 brings a
new face square-on; the odd pose in between leaves the cube oblique. An oblique frame is the
dangerous case, not the useless one: it reads a *plausible permutation* of the face in view and
would pass a nine-per-colour check. Two independent measurements separate the two poses sharply:

| pose | worst cell's majority share | the eight probes 6 px outside the grid |
|---|---|---|
| square-on (even) | **0.89** (the glyph costs 0.11 of the centre cell) | all eight backdrop |
| oblique (odd) | **0.44 – 0.50** | the four mid-edge probes are painted |

`frame.read_face` returns both and the harness records a reading only when the gate passes. Over
54 captures on levels 1 / 13 / 18 the gate accepted all 27 square-on poses and refused all 27
oblique ones, with no threshold tuning.

**The camera cycles in four 90° steps per axis.** `y_cw` ×8 and `x_cw` ×8 both return to the home
pose exactly. The y route passes FRONT → LEFT → BACK → RIGHT; the x route FRONT → UP → BACK →
DOWN. `x_acw` ×2 shows DOWN and `y_acw` ×2 shows RIGHT, so every face is two steps from home. The
z axis brings nothing new square-on — it rolls the picture in its own plane — which is why the
parent's one measured-effective prose channel (the static z ban) is kept and restated in the skill.

**Face turns are CUBE-FIXED.** At the pose showing LEFT, `rotate_front_cw` changed exactly one
column of the visible LEFT face (`BBB/BBB/RRR → BBW/BBW/RRW`) instead of rotating the face in view.
A camera pose therefore never renames the faces, and the solution word can be played from wherever
the survey happens to end.

## 3. The cube model, and the two conventions that had to be measured

A sticker is a `(point, normal)` pair with `point ∈ {-1,0,1}³` and `point · normal = 1`; there are
exactly 54 of them. `spin(face, vector)` is one clockwise quarter turn of a face written as
geometry, and `permutation(face)` turns that into a 54-slot permutation. Group invariants come out
right: every quarter turn has order 4, `order(R U) = 105`, and breadth-first search gives the cube
group's own ball sizes **12 / 114 / 1068 / 10011** at depths 1–4.

Two things geometry cannot settle were measured against the frame:

1. **Chirality.** My first pass had FRONT clockwise and the other five faces anticlockwise — a
   sign slip, caught immediately because `rotate_up_cw` brought the RIGHT face's row onto FRONT
   (green) where the model predicted LEFT (blue). With all six corrected to "clockwise seen from
   outside", a 12-move stream on level 1 and two 24-move streams on levels 13 and 18 were replayed
   through the model: **every one of the 60 transitions predicted the decoded front face exactly**,
   and on each level exactly **one** state in the radius-4 ball was consistent with the whole
   stream — level 1 = `D'`, level 13 = `F U L'`, level 18 = `U D D F`. Flipping BACK (the one face
   a front-only observation cannot see) left **zero** consistent states on all three levels, which
   pins it too.
2. **Which nine slots each pose shows, and in which orientation** (`POSE_SLOTS`). Fitted by
   matching each accepted reading against all six faces × eight dihedral orientations of the now
   known configurations. The answer was **unique at every pose** and agreed across all three
   levels: 24/24 accepted readings reproduced. BACK appears twice with different tuples because
   the x route and the y route reach it rolled a quarter turn apart.

`self_test()` re-derives all of this at import against a stored fingerprint — the ball size, the
quarter-turn orders, `order(R U)`, and the seven readings of the measured `D'` configuration — and
`TRUSTED` gates the whole perception path on it. A renderer change makes the harness degrade to a
blind sweep rather than act on a wrong model.

## 4. The turn

```
observe   decode the face in view, file it against the camera pose (ledger), re-believe
think     SOLVE if a configuration is proved · SURVEY while ambiguous · SWEEP if untrusted
inform    splice the brief into conversation_history[0]
delegate  ONE model call through evaluator.baseline_step, its own executor muted
commit    play the agent's pick if the menu allows it, else the planned move; advance the ledger
```

* **Believing.** The home reading alone names the configuration outright for most of the ball:
  there are **5969 distinct front faces among the ball's 11206 states**, median class size 1. When
  it does not, the survey turns the camera and intersects; six readings cover all 54 stickers and
  `assemble` reads the configuration outright, with no reliance on the ball at all — which is the
  path that would still work if a level were scrambled deeper than four turns.
* **Planning.** `solve_word` is a ball lookup inside radius 4, and a short forward search that
  meets the ball outside it, so it proves a solution out to radius 7.
* **The menu.** `menu_of` lists every move that *provably* shortens the proved word — usually one,
  occasionally two. It is the ≤4-item channel the diagnosis asked for (§5.3) in place of the
  measured-inert 14-name ALLOWED list, and honouring any pick from it costs nothing by
  construction.
* **Verifying.** Every kept reading is checked against what the believed configuration predicts.
  A contradiction is the only thing that can distinguish a correct belief from a confident wrong
  one, and it restarts the survey (`faults` in the journal counts them). In the shipped
  configuration this never fired; in a deliberately crippled copy (below) it fired and recovered.
* **The ledger.** The camera pose is tracked from the harness's own emitted view rotations.
  Switching axes away from home would leave a pose the tables cannot name, so `turn_camera`
  returns None there and the survey stops rather than guessing. Both survey loops close, and a
  pick is honoured only when it is on the menu, so the pose is always exact.
* **Kept verbatim from the parent**, because the diagnosis measured all three as correct: the
  `arm`/`disarm` mute around the delegation (one engine action per step), the four-field `SPEC`
  rewrite of `key`/`face`/`direction`/`action_type` (the model's `face` field is the axis letter
  for a view rotation, not the engine's axis face), and the pristine slot-0 splice. `disarm` is now
  in a `finally` so a raising delegate cannot leave the engine muted.

## 5. Measured on the real evaluator and the real renderer

`eval_runner.run_one`, real rubik evaluator, real renderer, `create_openai_response` stubbed, all
20 levels, `max_steps 40`. Five reply streams: `comply` (reads the brief's NEXT ACTION and names
it), `halfcomply` (that half the time, a random face turn otherwise), `front` (always
`rotate_front_cw`), `z` (always the blind view rotation — never once names a legal face turn), and
`randall` (uniform over all eighteen actions).

| arm | solved | total steps over 20 levels | override rate | perception faults |
|---|---|---|---|---|
| comply | **20/20** | 83 | **23 %** (19/83) | 0 |
| halfcomply | **20/20** | 83 | 66 % | 0 |
| front | **20/20** | 83 | 95 % | 0 |
| z | **20/20** | 83 | 100 % | 0 |
| randall | **20/20** | 83 | 100 % | 0 |

Per level the step count is 2 survey steps plus the scramble depth, identical on every arm:
**3 steps** on L1–L5 (depth 1), **4** on L6–L11, L19, L20 (depth 2), **5** on L12–L17 (depth 3),
**6** on L18 (depth 4). The parent spent 1901 steps over the same 20 levels and solved one.

Read the override rate honestly: **the score is mechanism-driven.** The same 20/20 falls out of a
stream that never names a legal face turn, so the agent's compliance buys no extra levels here —
it buys the *difference between 23 % and 100 % overrides*, which is the compliance measurement the
diagnosis asked to be reported next to the primary. 19 of the `comply` arm's 23 % are structural:
step 0 has no conversation history, so there is no brief to comply with on the first turn of each
episode.

Also verified in every episode: `used_rich_harness: true`, `oracle_access` absent (the anti-cheat
guard recorded nothing), one `rotation executed` line per step, no parse failures, no default
actions, and the journal written.

### The two degraded paths, deliberately broken and re-run

* **Ball shortcut disabled** (`hypotheses` forced to return nothing, so the full 16-step survey and
  `assemble` must carry it): **10/10** on levels 1, 6, 13, 18, 20 × {comply, front}, 15–18 steps
  each, 0 faults. This is the path that would carry a scramble deeper than the ball.
* **`TRUSTED` forced False** (perception declared untrustworthy, so the survey never runs and the
  harness falls back to the blind `F F F F R R R R …` sweep): **4/4** on levels 1, 3, 6, 13 with
  the `front` stream, 1–25 steps. It still wins because the sweep keeps sampling front faces and
  the believer resolves one of them — and on L13 the verifier caught **3** wrong beliefs (the sweep
  walks the cube out of the ball, where a front-face match can be a different state) and
  re-surveyed each time. The guard works on the only arm that can exercise it.

## 6. Rejected along the way

* **A belief-space policy with no camera moves at all** — keep the hypothesis set, play the move
  that shortens the way home for the most hypotheses, let the next front reading prune. Simulated
  over 2000 ball states: **39 % never solved inside 100 moves**. It deadlocks in a two-cycle
  whenever the live hypotheses predict identical front faces for the move it wants to play, and
  the front face is blind to a rotation of the FRONT layer itself (the all-red front has 16
  members in the ball). Two camera steps buy what no amount of inference can.
* **Scanning all six faces unconditionally** (16 steps, every episode). Correct but wasteful: the
  front face alone names the configuration most of the time, and the adaptive ladder costs 2.
* **Taking the survey's view rotations outside the step budget** by calling `execute_action` extra
  times inside one decision. The camera turns are state-preserving so it would cost nothing and
  score the same, but it would break the one-engine-action-per-step accounting the diagnosis
  identified as what keeps scoring comparable with the baseline. The survey pays for its steps.
* **A longer prose skill.** The refuted macro/milestone ladder (4.7 kB, ~1150 prompt tokens, which
  bought a relabelled step-0 constant) is deleted, not re-worded. Its replacement states the screen
  facts, what a camera turn is for, how to read the solution word, and the three things that never
  help — in that register, because a static unconditional statement is the one channel r001
  measured as effective.

## 7. Anti-cheat

Every sticker this harness believes in was decoded from the rendered frame the agent is shown, via
`evaluator.capture_game_screenshot()`. Every move it believes it played is a move it emitted
itself. It never touches `evaluator.game` (beyond the sanctioned screenshot path), the cube's
internal arrays, the scramble table, `SpatialWorld/data/`, a solver, the solved flag or the score;
the cube model in `tools/action.py` is built from geometry in this file and validated against
group invariants that carry no level information. The anti-cheat guard recorded nothing in any of
the 114 rig episodes, and the harness delegates to the frozen model on every single step, so the
VLM-call counter is satisfied by design rather than by a token call.

The *scoring* fact the harness leans on is the one the diagnosis pointed at: the engine ends the
episode itself when the cube is solved. Nothing here reads that flag — the harness just walks onto
the configuration and the engine stops.

## 8. Replaying a decision from the trace

`action.think` / `action.choose` are pure functions of the book, and the book is driven by the
decoded readings plus the emitted moves, so a trace replays without the journal:

```python
from tools import action, frame
book = action.new_book()
for png_b64, pick in recorded_turns:                  # frames the agent saw, picks it made
    action.record_reading(book, *frame.read_face(png_b64))
    action.think(book)
    played, honoured = action.choose(book, pick)
    action.note_move(book, played, True)
```
