# cand_r018_0_spatial-grounding — middleware pipeline

## axis:
Component graph / decision locus (code). Anchor: `prompt -> one model call -> regex`. This harness
wraps the model in an ordered MIDDLEWARE STACK (onion): each stage is a peer object with `pre(ctx)`
(may short-circuit before the model) and `post(ctx, action)` (may rewrite the proposal after it),
all communicating only through a shared per-step `StepContext`.

## change:
New `tools/context.py` builds a StepContext from the observation (cursor, `map:` block cells, message,
adjacent-noun prose, legal set) and carries `notes` (lines stages inject into the prompt) and `trail`
(provenance, written to `reasoning`). New `tools/stages.py` holds five independent stages:
LegalStage (outermost, audits vocabulary), ModalStage (answers `[yn]`/`--More--` without the model),
HazardStage (prose lava/water or `}` glyph -> note + veto), BumpStage (a move that left the hero in
place books (square,dir); repeats vetoed), OrbitStage (immediate reversal onto a square visited >=2x
redirected to the least-visited open neighbour). `agent.py:PipelineHarness.act` runs pre hooks
outer->inner, then the model (reply snapped to the legal set via `action_io.snap`), then post hooks
inner->outer. `prompting.build_messages` gained a `notes` argument. `build_agent` entry point kept.

## why:
Coverage steer: structural novelty vs the anchor's single linear path. Diagnosis names orbits (a),
wall bumps 28% of steps (b), lava deaths (c) and modal-prompt lock (d) — each maps to one stage.

## module map / levers
| file | role | lever |
|---|---|---|
| agent.py | pipeline driver, stack ordering | code |
| tools/context.py | observation -> StepContext | code |
| tools/stages.py | the five stages + `default_stack()` order | code |
| tools/action_io.py | stock regex + legal_actions + snap | code |
| tools/prompting.py | priors + stage notes assembly | code |
| AGENT.md / rules.md / skills/ | empty prose priors | prose |

## files:
agent.py, tools/context.py (new), tools/stages.py (new), tools/action_io.py, tools/prompting.py, NOTES.md
