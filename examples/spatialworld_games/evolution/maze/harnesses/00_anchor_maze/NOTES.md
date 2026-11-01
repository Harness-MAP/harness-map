# maze rich harness — module map & levers

RICH anchor for the SpatialWorld maze game (change #4). It reproduces the prompt-only baseline
while exposing the canonical rich-harness surface, so the agentic proposer has real code + prose
levers to evolve. The frozen agent (Qwen3-VL-30B-A3B-Instruct) and the SpatialWorld engine are
untouched; only this harness is optimized. Kept ALONGSIDE `system_prompt.txt` (the prompt-only
fallback still runs when GAME_HARNESS_DIR has no orchestrator.py).

## module map
| file | role |
|------|------|
| orchestrator.py | ENTRYPOINT. `build_agent(evaluator)` returns the async per-step fn; `compose_system_prompt()` recomposes the prompt from the .md priors. Reuses the evaluator's engine handles; returns an engine Action each step. |
| AGENT.md | role framing + verified perception facts (reading the HUD / compass / walls / exit). |
| rules.md | the legal action set (move_forward / turn_left / turn_right) + the granularity contract. |
| skills/navigation.md | the navigation-strategy prior (open path / wall ahead / avoid spinning). |
| tools/frame.py | frame capture — reuses `evaluator.capture_game_screenshot`. |
| tools/vlm.py | the perceive->VLM->parse step — reuses `evaluator.openai_ai_model`. |
| tools/action.py | the maze action/key algebra + the engine Action contract. |
| memory.md | a per-step scratchpad TEMPLATE (self-derived state); present as a lever, NOT injected by the anchor. |

## evolvable levers (axes)
- prose: AGENT.md / rules.md / skills/*.md / memory.md — what the agent is told.
- code:  orchestrator.py / tools/*.py — the mechanism (compose, capture, decide, action algebra, memory).

## anti-cheat
Decide from the observation only. Never wire oracle/privileged state (the exit's coordinates, the
level's layout file, the true remaining distance) into the agent's decision path. The pre-eval gate
AST-scans the code and text-scans the priors for privileged fields.

## change
Initial rich anchor: split system_prompt.txt into AGENT.md + skills/navigation.md + rules.md
(compose reproduces it in order) and added thin frame/vlm/action tools that reuse the evaluator, so
this scores identically to the prompt-only baseline.
