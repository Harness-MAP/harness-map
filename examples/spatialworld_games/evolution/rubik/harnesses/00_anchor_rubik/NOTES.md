# rubik rich harness — module map & levers

RICH anchor for the SpatialWorld Rubik's Cube game. It reproduces the prompt-only baseline while
exposing the canonical rich-harness surface, so the agentic proposer has real code + prose levers to
evolve. The frozen agent (Qwen3-VL-30B-A3B-Instruct) and the SpatialWorld engine are untouched; only
this harness is optimized. Kept ALONGSIDE `system_prompt.txt` (the prompt-only fallback still runs
when GAME_HARNESS_DIR has no orchestrator.py).

## module map
| file | role |
|------|------|
| orchestrator.py | ENTRYPOINT. `build_agent(evaluator)` returns the async per-step fn; `compose_system_prompt()` recomposes the prompt from the .md priors. Reuses the evaluator's engine handles; returns an engine Action each step. |
| AGENT.md | role framing + verified perception facts (FACE IDENTIFICATION — each face's center letter + color: F/B/L/R/U/D). |
| rules.md | the legal move taxonomy (MOVE TYPES — 90-degree face rotations cw/acw, 45-degree whole-cube view rotations x/y/z) + the brief solving tips that trail it in the source. |
| tools/frame.py | frame capture — reuses `evaluator.capture_game_screenshot`. |
| tools/vlm.py | the perceive->VLM->parse step — reuses `evaluator.baseline_step`. |
| tools/action.py | the Rubik action/key algebra + the engine Action contract (rotate_<face>_<cw/acw> via rotate_face; rotate_cube_<x/y/z>_<cw/acw> via rotate_cube; the full RUBIK_ACTIONS vocabulary + CW/ACW constants). |
| memory.md | a per-step scratchpad TEMPLATE (self-derived state); present as a lever, NOT injected by the anchor. |

## split note (no skills/ subdir)
Unlike the other games, rubik's source `system_prompt.txt` puts its strategy block ("Important tips")
AFTER the move taxonomy ("MOVE TYPES"), whereas `compose_system_prompt()` always emits AGENT -> skills
-> rules in that fixed order. To keep the legal action list where the anchor wants it (rules.md) AND
reproduce the source order byte-for-byte (modulo whitespace), rubik folds its short strategy tips into
the rules.md suffix and ships no separate skills/*.md. A proposer that wants a standalone strategy
lever can carve one out later as long as compose still reproduces the intended prompt.

## evolvable levers (axes)
- prose: AGENT.md / rules.md / memory.md — what the agent is told.
- code:  orchestrator.py / tools/*.py — the mechanism (compose, capture, decide, action algebra, memory).

## anti-cheat
Decide from the observation only. Never wire oracle/privileged state (the cube's full sticker/facelet
array, the scramble string, the solver's move sequence) into the agent's decision path. The pre-eval
gate AST-scans the code and text-scans the priors for privileged fields.

## change
Initial rich anchor: split system_prompt.txt into AGENT.md (role + face identification) + rules.md
(move taxonomy + solving tips) so compose reproduces it in order, and added thin frame/vlm/action
tools that reuse the evaluator, so this scores identically to the prompt-only baseline.
