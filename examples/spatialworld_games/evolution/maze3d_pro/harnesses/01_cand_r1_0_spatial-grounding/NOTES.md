# maze3d_pro rich harness — module map & levers

Rich harness for the SpatialWorld multi-floor 3D maze (maze3d_pro). The frozen agent
(Qwen3-VL-30B-A3B-Instruct) and the SpatialWorld engine are untouched; only this harness is
optimized. `build_agent(evaluator)` is the entrypoint the extended evaluator installs as
`ai_model_func`; the engine `Action` contract and the maze `success` scoring are byte-compatible
with the prompt-only baseline.

## axis
**Spatial grounding — proprioception from frame differencing** (code lever: a new `tools/grounding.py`
threaded through `tools/vlm.py` + `harness.py`, rendered through the `memory.md` scratchpad, with
`AGENT.md`/`rules.md`/`skills/navigation.md` corrected to the perception facts this build actually
has). The parent's axis was "compose a prompt and delegate"; this one is "measure what the agent's
own actions did to the world, and hand that back every turn".

## change
`tools/grounding.py` is new. Every step it hashes `frame_data.image` and compares it with the
previous frame. An engine action that leaves the frame **byte-identical did nothing** — that single
bit is enough to dead-reckon, with no privileged state at all: heading (a turn is never blocked),
floor (net climbs that actually changed the view), a per-floor local position in cells, and a local
wall map where `(cell, bearing)` is marked WALL the moment a translation toward it comes back
identical and OPEN the moment one succeeds. The same pass runs an exact-colour census for the only
two saturated colours in the game — the gold exit `RGB(255,200,0)` and the cyan stair
`RGB(0,204,204)` — and converts the blob's pixel width into a distance in cells.
`tools/vlm.py` writes all of that into the live system message (`conversation_history[0]`, the only
per-turn text channel the engine has — the block is **appended to a stashed pristine base**, never
spliced in, so the JSON RESPONSE FORMAT contract the parser needs survives), then takes the model's
answer; if the answer is an action the frames have already **proven** inert from the body's current
unchanged state, it says so and asks again (1 re-ask/step, ≤60/episode) and, only if the model
still insists, substitutes the grounding tool's rescue action. The frozen model still chooses every
step and `evaluator.baseline_step` still does every VLM call, parse and fallback.

## why (the gradient, measured — round 1, no diagnosis.md yet)
Replaying the parent's **own recorded answers** through the real evaluator (a stub client feeding
back each level's saved `response_text`, which reproduces each episode's final
state exactly) shows the binding constraint is not planning, it is the absence of any feedback
channel:

| level | parent outcome | actions that left the frame BYTE-IDENTICAL | distinct views in 100 steps |
|---|---|---|---|
| 10 | loss | **99 / 99** | **1** |
| 18 | loss | **99 / 99** | **1** |
|  3 | loss (1 cell from the exit) | 93 / 99 | 5 |
|  1 | loss | 48 / 99 | 3 |
|  4 | loss | 48 / 99 | 3 |
| 17 | win  | 12 / 32 | 12 |
|  6 | win  | 2 / 6  | 5 |

On levels 10 and 18 the parent pressed `move_forward` into the same wall one hundred times and the
screen never changed once. Across all 25 levels it issued `move_forward` 2090 of 2340 times and
`climb_up`/`climb_down` **zero** times, finishing on floor 1 of 2 every single episode. Two measured
causes, both now fixed:

1. **There is no HUD on this build.** The 800x600 render uses 7 brown/grey shades with a max
   channel of 147 and not one bright text pixel; two provably different poses render
   byte-identically. The parent's `AGENT.md` told the agent to read a `POS`/`FLOOR`/`DIR` HUD that
   does not exist, so "where am I" was pure hallucination. It is now supplied by measurement.
2. **The parent's `rules.md` gated both climbs on seeing `[Q] UP` / `[E] DOWN` in a bottom HUD.**
   That prompt does not exist either, so the rule forbade climbing unconditionally — which is
   exactly the 0-climbs-in-2340-actions result. `rules.md` now says to try the climb on a cyan
   stair and let the frame diff answer.

Verified end-to-end through the real evaluator (`build_agent` + `baseline_step`, exactly the
eval_runner path) with a **degenerate stub model that answers the same thing to every single
question**, so the mechanism alone has to do the work. PAIRED against the parent harness on the
same rig, same levels, same stub:

| level | stub | parent harness | this harness |
|---|---|---|---|
|  1 | always `move_forward small` | 59/59 no-ops, 1 view, loss | 3 no-ops, 7 views, **WIN in 12 steps** |
|  3 | always `move_forward small` | 59/59 no-ops, 1 view, loss | 3 no-ops, 6 views, **WIN in 11 steps** |
|  6 | always `move_forward small` | 19/19 no-ops, 1 view, loss | 2 no-ops, 6 views, **WIN in 8 steps** |
| 10 | always `move_forward small` | 59/59 no-ops, 1 view | 11 no-ops, **23 views** |
| 18 | always `move_forward small` | 59/59 no-ops, 1 view | 7 no-ops, **22 views** |
| 13 | always `climb_up`          | 59/59 no-ops, floor 1 | 22 no-ops, 16 views, **reaches floor 2** (12 climbs) |

0 of 6 for the parent, 3 of 6 for this harness, from a model with no intelligence at all. Anti-cheat
gate re-run by hand: PASS (no privileged mention in any agent-facing prior, no privileged attribute
access in any .py, and the code never touches `evaluator.game`).

## live check (real Qwen3-VL-30B-A3B-Instruct, full 100-step episodes)
Two episodes on the parent's two worst levels, run through `eval_runner.py` exactly as the outer
loop will. Both are still LOSSES — this change does not convert 10 or 18 on its own — but the
telemetry says the mechanism is doing what it was built to do, and it says what the next
bottleneck is:

| | parent L10 | this L10 | parent L18 | this L18 |
|---|---|---|---|---|
| distinct views in the episode | **1** | moving all episode | **1** | moving all episode |
| turns the model chose by itself | 0 | 17 | 0 | 11 |
| granularity the model chose | 100% `medium` | 100% `small` | 100% `small` | 100% `small` |
| VLM calls / wall-clock | 100 / 207 s | 142 / 237 s | 100 / 219 s | 144 / 237 s |

Two further live episodes on levels the parent lost from 1 cell away. **Level 3 is a WIN** —
success, 26 steps, 34 VLM calls, 52 s, where the parent burned all 100 steps with 93 of 99 actions
changing nothing and finished 1 cell from the exit. **Level 5 is still a loss**, and it finished 6
cells from the exit where the parent finished 1 — worth stating plainly: the parent's "1 cell away"
on that level is an artefact of never having moved, and a harness that forces real motion can end
an unsuccessful episode further out. `distance_to_exit` is not the score (`success` is), but a
mechanism that converts standing-still into wandering will look worse on that field, so do not read
it as a regression. Live sample so far, all four chosen from the parent's worst episodes: 1 win /
3 losses vs the parent's 0 / 4.

Two things to hand the diagnostician. **(a)** The model is very stubborn about `move_forward`: of
the steps where the harness refused the answer, a first re-ask changed it 2 times out of 44 and a
second re-ask changed it **0 times out of 42** — so `RETRY_LIMIT` was cut from 2 to 1 (pure latency
otherwise) and the harness ends up substituting on ~20% of steps. That substitution rate is the
obedience cost of this change and should be watched; the counterfactual is the parent's 99-of-99
wasted steps on these same levels. **(b)** Still zero climbs from the model even with the corrected
`rules.md`, so the floor-2 half of the task is the obvious next axis — the stub test proves the
climb machinery works (L13 reaches floor 2), the model just never asks for it.

`system_prompt.txt` and `system_prompt.md` are untouched stale copies of the PRE-change prompt;
`compose_system_prompt()` never reads either, so they are inert (and `system_prompt.txt` is excluded
from the evolvable bundle by the adapter).

## module map
| file | role |
|------|------|
| harness.py | ENTRYPOINT. `build_agent(evaluator)` returns the async per-step fn and opens ONE `Grounding` for the episode; `compose_system_prompt()` recomposes the prompt from the .md priors. Wraps the step in a try/except that falls back to the pristine `baseline_step`, so the mechanism can never cost an episode. |
| tools/grounding.py | NEW. Frame-differencing proprioception: view digest, outcome of the last action, exact heading / floor / local position, the proven wall+open map, the exact-colour exit & stair scanner with the `depth = 600/width` ring law, the proven-null set, the rescue action, and the rendering of memory.md. |
| tools/vlm.py | The perceive -> ground -> VLM -> verify step. Injects the block into `conversation_history[0]` (append-to-stashed-base), delegates to `evaluator.baseline_step`, re-asks on a proven-null answer, substitutes the rescue as a last resort. |
| tools/action.py | the maze3d_pro action/key algebra + the engine Action contract (w/s carry granularity 1/2/3; a/d turn; q/e climb). Used by vlm.py to canonicalize the model's action and to build a substituted Action. |
| tools/frame.py | frame capture — reuses `evaluator.capture_game_screenshot` (kept as the fallback pixel source; `frame_data.image` is the primary one and the two were measured to move in exact lockstep). |
| AGENT.md | role framing + the perception facts MEASURED on this build: no HUD at all, the brown depth-ring corridor, the gold exit and cyan stair colours, and the fact that a byte-identical frame means nothing happened. |
| rules.md | the legal action set + the granularity contract + the corrected climb rule (there is no stair prompt; try the climb and read the frame). |
| skills/navigation.md | the navigation-strategy prior, re-ordered around the grounded block: exit-in-sight first, then stairs, never a DEAD action, prefer UNEXPLORED over OPEN, prefer `small`. |
| memory.md | the per-step scratchpad TEMPLATE, now LIVE: `{{SIGHT}} {{HEADING}} {{FLOOR}} {{POS}} {{STEP}} {{LAST}} {{STUCK}} {{REVISIT}} {{BEARINGS}} {{DEAD}} {{TRACE}} {{WASTE}} {{REFUSED}}` are filled by `Grounding.block`. |

## evolvable levers (axes)
- prose: AGENT.md / rules.md / skills/*.md / memory.md — what the agent is told.
- code:  harness.py / tools/*.py — the mechanism (compose, capture, ground, decide, verify).
- tuning knobs inside the new mechanism, all still unexplored: `RETRY_LIMIT` / `MAX_EXTRA_CALLS`
  (tools/vlm.py), `LANE_TOLERANCE` / `ASPECT_TOLERANCE` (tools/grounding.py), the rescue's
  preference order, and whether the harness should also report the *advisory* cross-episode
  `view_nulls` ledger (built but currently not shown).
- CAVEAT for the next proposer: the adapter's `bundle_exclude` drops `memory.md` from the evolvable
  bundle and from `harness_sig`, so editing memory.md alone changes no signature. The same text is
  duplicated as `_FALLBACK_TEMPLATE` in tools/grounding.py (which IS in the bundle) and is used when
  memory.md is absent — keep the two in step, or move the template wholly into the .py.

## what is still open
- The local map goes inexact the moment a `medium`/`large` move succeeds (it stops early at a wall,
  so the distance covered is unknowable). The prose asks for `small`; a mechanism that measures the
  distance actually covered would restore exactness.
- Nothing yet routes toward an unexplored *frontier* across more than one step; the block names the
  turn that aims at each bearing but the agent re-decides every step.
- The cross-floor map is per-floor-visit (a climb starts a fresh local frame), so returning to a
  floor does not re-join its old map.

## anti-cheat
Decide from the observation only. Every field in the grounded block is derived from the pixels the
model is already shown plus the agent's own action history. Nothing reads `evaluator.game` (only
`.screen` is even permitted), the level layout, the exit's position, the floor count, or the true
distance. The pre-eval gate AST-scans the code and text-scans the priors for privileged fields.

## files
harness.py, tools/grounding.py (new), tools/vlm.py, memory.md, AGENT.md, rules.md,
skills/navigation.md, NOTES.md
