# block3d rich harness — module map & levers

RICH anchor for the SpatialWorld Block3D game. It reproduces the prompt-only baseline while exposing
the canonical rich-harness surface, so the agentic proposer has real code + prose levers to evolve.
The frozen agent (Qwen3-VL-30B-A3B-Instruct) and the SpatialWorld engine are untouched; only this
harness is optimized. Kept ALONGSIDE `system_prompt.txt` (the prompt-only fallback still runs when
GAME_HARNESS_DIR has no harness.py).

## module map
| file | role |
|------|------|
| harness.py | ENTRYPOINT. `build_agent(evaluator)` returns the async per-step fn; `compose_system_prompt()` recomposes the prompt from the .md priors. Reuses the evaluator's engine handles; returns an engine Action each step. |
| AGENT.md | role framing + verified perception facts (the 6x6x6 grid, cursor start, target-in-three-views, scoring). |
| rules.md | the legal action set (6 cursor moves + place_block toggle) + control mapping + the observation tips (the three orthographic views / block+cursor coloring). |
| skills/building.md | the build-strategy prior (view analysis / error correction). |
| tools/frame.py | frame capture — reuses `evaluator.capture_game_screenshot`. |
| tools/vlm.py | the perceive->VLM->parse step — reuses `evaluator.baseline_step`. |
| tools/action.py | the Block3D action/key algebra + the engine Action contract (6 unit cursor moves + place_block; NO granularity). |
| memory.md | a per-step scratchpad TEMPLATE (self-derived state); present as a lever, NOT injected by the anchor. |

## evolvable levers (axes)
- prose: AGENT.md / rules.md / skills/*.md / memory.md — what the agent is told.
- code:  harness.py / tools/*.py — the mechanism (compose, capture, decide, action algebra, memory).

## anti-cheat
Decide from the observation only. Never wire oracle/privileged state (the target structure's cell
list, the solution layout file, the true remaining blocks) into the agent's decision path. The
pre-eval gate AST-scans the code and text-scans the priors for privileged fields.

## change
Initial rich anchor: split system_prompt.txt into AGENT.md + skills/building.md + rules.md (compose
reproduces it in order — AGENT prefix, then the strategy skill, then the movement/control/observation
suffix) and added thin frame/vlm/action tools that reuse the evaluator, so this scores identically to
the prompt-only baseline.
