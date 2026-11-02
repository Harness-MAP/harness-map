# Rubik candidate — read the cube off the screen, then unwind the scramble

## axis: code lever — a PERCEPTION channel plus an exact PLAN, written at token level as plain `def`s

The parent had no perception at all: nothing in it read the frame, so every decision was a guess
over an 18-name vocabulary and the whole policy was a refusal table over the harness's own move
word. This candidate adds the missing channel and replans on top of it.

## change:

`tools/frame.py` now decodes the agent's own observation. The renderer draws the cube **face on** —
one face fills an exact axis-aligned 3x3 grid of 79-pixel sticker squares at origin (131, 131) on
a flat grey backdrop, with a legend in the corner fixing each face's colour — so a majority vote
over a lattice inside each cell returns nine stickers for free, every step, with no camera move.
A 45-degree view rotation leaves the cube oblique, and an oblique frame reads a *plausible but
wrong* permutation of the face in view, so every reading passes a two-part gate (worst cell's
majority share, and eight probes just outside the grid): over 54 real captures it accepted all 27
square-on poses and refused all 27 oblique ones.

`tools/action.py` turns those readings into a configuration and a plan. It carries an exact
54-sticker cube built from geometry (`spin` / `permutation`), advanced only by the move names this
harness itself emitted, and the ball of every configuration within four quarter turns of solved —
the depth the task actually uses. The home reading alone names the configuration outright for most
of that ball (5969 distinct front faces among its 11206 states), and when it does not, a survey
turns the camera two steps at a time and intersects; six readings cover all 54 stickers and the
configuration is read rather than inferred. Once it is named, the shortest word home is proved,
and `menu_of` publishes every move that measurably shortens it — the <=4-item per-turn channel the
diagnosis asked for in place of the 14-name ALLOWED list it measured as inert. Each kept reading is
checked against what the believed configuration predicts, and a contradiction restarts the survey:
that check is the only thing separating a correct belief from a confident wrong one.

Three conventions could not be derived, only measured against the real renderer, and all three now
sit behind an import-time `self_test` that degrades the harness to a blind sweep if it ever fails:
the chirality of the six quarter turns, the fact that face turns are cube-fixed (a camera pose
never renames the faces), and which nine slots each surveyed pose shows. The parent's three proven
pieces of plumbing are kept verbatim — the mute around the delegation, the four-field `SPEC`
rewrite, the pristine slot-0 splice — and one model call per step, one engine action per step, and
the response contract are all unchanged. The refuted macro/milestone skill is deleted rather than
re-worded; its replacement states the screen facts and the two things worth doing with a step.

## why:

The gradient named the binding constraint and measured its ceiling: with the parent's `revisit`
rule an optimal router reaches 2 of the 12 one-turn configurations in 100 moves, and the live run
was already at that cap, so nothing in the prompt or the model could move the primary. But fixing
the search objective only lifts the ceiling to what *blind* search can reach — the one-turn shell
and part of the two-turn one. The three-turn shell has 1068 configurations and the four-turn shell
10011, and level 18 is exactly four turns deep, so coverage cannot reach those levels at any
budget. The way past that ceiling is not a better search, it is **looking at the cube** — which
the frame has been offering all along, since the renderer draws a face square on and the legend
names it. Once the configuration is known the task is trivial: the scramble is at most four quarter
turns, the engine ends the episode by itself the instant the cube is solved, so the harness never
needs to recognise the finish, only walk onto it.

## measured:

Real evaluator, real renderer, model client stubbed, all 20 levels x 5 reply streams (100
episodes, `max_steps 40`): **20/20 solved on every stream**, 83 steps over the 20 levels, **zero**
perception faults, no oracle access recorded, one engine action per step. Step count is 2 survey
steps plus the scramble depth and is identical on every stream: 3 steps on L1-L5, 4 on L6-L11/19/20,
5 on L12-L17, 6 on L18. The parent spent 1901 steps on the same 20 levels and solved one.

Read that with the chain-integrity warning in mind, because it cuts both ways here. **The score is
mechanism-driven**: the same 20/20 falls out of a stream that never once names a legal face turn,
so the agent's compliance buys no extra levels — what it buys is the override rate, which is the
honest compliance measurement: **23 %** when the stream follows the brief (19 of those 20 are
structural: step 0 has no conversation history, so there is no brief to comply with), 66 % at half
compliance, 95-100 % for streams that ignore it. Two deliberately crippled copies were re-run to
price the guards: with the ball shortcut disabled the full 16-step survey carries it (10/10, 15-18
steps); with `TRUSTED` forced false the blind sweep still wins 4/4, and on L13 the verifier caught
3 wrong beliefs and re-surveyed each time.

## token level:

The steer asked to vary the token-level implementation. The parent's components were lambdas with
no branches and no literals — byte-identical to the anchor on both `f_AST` and `f_num` (cf_mean
1.25, cf_max 4, numbers per component 0.12, log-magnitude 0.00). This candidate puts the mechanism
back inside ordinary `def`s that carry their own control flow and their own numbers: cf_mean
**2.34**, cf_max **10**, numbers per component **1.67**, log-magnitude **0.98**, over 58 components
instead of 8 — a qualitative move on exactly that block, with the public entry (`build_agent`,
`compose_system_prompt`) and the `run_episode`-equivalent contract untouched.

## module map

| file | role |
|---|---|
| `agent.py` | unchanged entry: `build_agent(evaluator)` + the sorted-Markdown prompt composer. |
| `AGENT.md` | unchanged role framing and the six face letters. |
| `rules.md` | unchanged move taxonomy. |
| `skills/survey_then_solve.md` | NEW static prior: the screen facts, what a camera turn is for, how to read the solution word, the three things that never help. Replaces the refuted `macro_commitment.md`. |
| `memory.md` | the brief template: phase, what the frame read, distance to solved, the word still to play, NEXT, ALLOWED, last turn. |
| `tools/frame.py` | the 3x3 sticker decoder and the square-on gate. |
| `tools/action.py` | the action vocabulary, the 54-sticker cube, the radius-4 ball, the believer, the survey route, the planner, the menu, the ledger. |
| `tools/vlm.py` | the turn: observe, think, inform, one delegated model call, commit, journal. |
| `traces_journal/` | per-episode journal (phase, pick, played, override, pose, purity, faults) — written at run time, outside the descriptor corpus. |
| `traces_notes/MECHANISM.md` | the long record: how every fact was measured, what was rejected, all rig tables. |

## anti-cheat

Every sticker the harness believes in was decoded from the rendered frame the agent is shown;
every move it believes it played is one it emitted itself. It never reads `evaluator.game` beyond
the sanctioned screenshot path, the cube's internal arrays, the scramble table,
`SpatialWorld/data/`, a solver, the solved flag or the score, and it delegates to the frozen model
on every step. The anti-cheat guard recorded nothing in any of the 114 rig episodes.

## files:
- `tools/frame.py` — the sticker decoder and the oblique-frame gate (new mechanism).
- `tools/action.py` — the cube, the ball, the believer, the survey route, the planner, the menu.
- `tools/vlm.py` — the turn rebuilt around observe/think/inform/commit, with the journal.
- `memory.md` — the brief template; the parent's dead `word`/`stood`/`screen` fields are gone.
- `skills/survey_then_solve.md` — new; replaces `skills/macro_commitment.md` (deleted, refuted).
- `agent.py` — docstrings retargeted; the factory and the composer are unchanged.
- `traces_notes/MECHANISM.md` — the implementation record, outside the agent's prompt.
