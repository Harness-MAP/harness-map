# maze3d_pro rich harness — module map & levers

RICH anchor for the SpatialWorld multi-floor 3D maze (maze3d_pro). It reproduces the prompt-only
baseline while exposing the canonical rich-harness surface, so the agentic proposer has real code +
prose levers to evolve. The frozen agent (Qwen3-VL-30B-A3B-Instruct) and the SpatialWorld engine are
untouched; only this harness is optimized. Kept ALONGSIDE `system_prompt.txt` (the prompt-only
fallback still runs when GAME_HARNESS_DIR has no orchestrator.py).

## module map
| file | role |
|------|------|
| orchestrator.py | ENTRYPOINT. `build_agent(evaluator)` returns the async per-step fn; `compose_system_prompt()` recomposes the prompt from the .md priors. Reuses the evaluator's engine handles; returns an engine Action each step. |
| AGENT.md | role framing + verified perception facts (the POS/FLOOR/DIR HUD, walls, exit, stair tiles, the bottom-center [Q] UP / [E] DOWN stair prompts). |
| rules.md | the legal action set (move_forward/backward with granularity, turn_left/right, climb_up/down) + the granularity contract + the closing "analyze the screen" instruction. |
| skills/navigation.md | the multi-floor navigation-strategy prior (exit/stair/open-path priority, avoid spinning, climb between floors). |
| tools/frame.py | frame capture — reuses `evaluator.capture_game_screenshot`. |
| tools/vlm.py | the perceive->VLM->parse step — reuses `evaluator.baseline_step`. |
| tools/action.py | the maze3d_pro action/key algebra + the engine Action contract (w/s carry granularity 1/2/3; a/d turn; q/e climb only on a stair with the matching prompt). |
| memory.md | a per-step scratchpad TEMPLATE (self-derived state); present as a lever, NOT injected by the anchor. |

## evolvable levers (axes)
- prose: AGENT.md / rules.md / skills/*.md / memory.md — what the agent is told.
- code:  orchestrator.py / tools/*.py — the mechanism (compose, capture, decide, action algebra, memory).

## anti-cheat
Decide from the observation only. Never wire oracle/privileged state (the exit's floor/coordinates,
the level's layout file, the stair graph, the true remaining distance) into the agent's decision path.
The pre-eval gate AST-scans the code and text-scans the priors for privileged fields.

## change
Initial rich anchor: split system_prompt.txt into AGENT.md + skills/navigation.md + rules.md (compose
reproduces it in order — AGENT prefix with role + visual elements, then the navigation skill, then the
available-actions/granularity/closing suffix) and added thin frame/vlm/action tools that reuse the
evaluator, so this scores identically to the prompt-only baseline.
