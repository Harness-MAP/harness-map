# snake rich harness — module map & levers

RICH anchor for the SpatialWorld 3D Snake game (a 4x4x4 grid). It reproduces the prompt-only baseline
while exposing the canonical rich-harness surface, so the agentic proposer has real code + prose
levers to evolve. The frozen agent (Qwen3-VL-30B-A3B-Instruct) and the SpatialWorld engine are
untouched; only this harness is optimized. Kept ALONGSIDE `system_prompt.txt` (the prompt-only
fallback still runs when GAME_HARNESS_DIR has no orchestrator.py).

## module map
| file | role |
|------|------|
| orchestrator.py | ENTRYPOINT. `build_agent(evaluator)` returns the async per-step fn; `compose_system_prompt()` recomposes the prompt from the .md priors. Reuses the evaluator's engine handles; returns an engine Action each step. |
| AGENT.md | role framing + verified perception facts (the 4x4x4 grid, snake spawn, random food, max score 64). |
| rules.md | the legal action set (6 unit-direction moves, no 180 reversal) + control mapping (the on-screen R/L/U/D/W/S letters) + the observation tips (indicators only show for collision-safe moves). |
| skills/strategy.md | the snake-strategy prior (space management / food targeting / body avoidance / wall awareness). |
| tools/frame.py | frame capture — reuses `evaluator.capture_game_screenshot`. |
| tools/vlm.py | the perceive->VLM->parse step — reuses `evaluator.baseline_step`. |
| tools/action.py | the 3D-Snake action/key algebra + the engine Action contract (6 unit moves d/a/w/s/q/e; NO granularity; a 180 reversal of last_move_dir is illegal). |
| memory.md | a per-step scratchpad TEMPLATE (self-derived state); present as a lever, NOT injected by the anchor. |

## evolvable levers (axes)
- prose: AGENT.md / rules.md / skills/*.md / memory.md — what the agent is told.
- code:  orchestrator.py / tools/*.py — the mechanism (compose, capture, decide, action algebra, memory).

## anti-cheat
Decide from the observation only. Never wire oracle/privileged state (the food's grid coordinates, the
snake body's full coordinate list, the RNG seed) into the agent's decision path. The pre-eval gate
AST-scans the code and text-scans the priors for privileged fields; every suggestion must keep the
agent choosing the food from the frame.

## change
Initial rich anchor: split system_prompt.txt into AGENT.md + skills/strategy.md + rules.md (compose
reproduces it in order — AGENT prefix, then the strategy skill, then the movement/control/observation
suffix) and added thin frame/vlm/action tools that reuse the evaluator, so this scores identically to
the prompt-only baseline.
