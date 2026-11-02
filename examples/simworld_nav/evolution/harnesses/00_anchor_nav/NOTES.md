# NOTES.md — SimWorld-MMNav anchor (module map · evolvable levers · contract)

Proposer/diagnostician read this FIRST. It is proposer-facing only — never shown to the frozen
agent. `## change:` at the bottom is where each candidate records the one edit it made.

## What this harness is
The scaffolding around a frozen **Qwen3-VL-30B-Instruct** driving a quadruped through a UE5 city,
one VLM call per step. `run_episode` wires 7 modules through a per-step perceive→plan→act→commit
loop. The weights never change — **this harness is what evolves.** See `AGENT.md` (role + verified
perception facts) and `rules.md` (action set, sanctioned inputs, anti-cheat).

## Entrypoint contract — KEEP INTACT (a changed signature / broken import scores 0)
```python
# agent.py
def run_episode(env, task_spec, vlm_clients, trace_dir=None) -> EpisodeResult
#   returns EpisodeResult(success, subtask_success, steps, actions, llm_calls, notes, observations)
```
`evaluator.py` does `spec_from_file_location(...); exec_module(...); harness.run_episode(...)`, then
scores from the env's TRUE post-rollout state → `score.json` (success, subtask_success[], subtask_sr).
SR (all 4 legs) is the hard bar; **SubtaskSR** (fraction of ORIENTATION→MOVE_ALONG_ROAD→
TURN_AT_INTERSECTION→REACH_DESTINATION legs passed) is the graded, discriminating signal.

## Module map (the CODE levers — this is where behaviour lives)
| file | class | loc | role |
|---|---|---|---|
| `agent.py` | — | ~640 | ENTRYPOINT + orchestrator: the per-step loop, action-code↔env-step map, stuck detector |
| `modules/spatial_memory.py` | SpatialMemory | 149 | honest odometry + step / subtask-boundary bookkeeping (self-derived state, never oracle) |
| `modules/prompt_assembler.py` | PromptAssembler | 135 | builds the chat-completions messages; the LIVE system/user text comes from `perception_template()` here (NOT from AGENT.md/rules.md) |
| `modules/perceiver.py` | Perceiver | 126 | one VLM call: RGB(+landmark hint) → JSON scene description |
| `modules/planner.py` | Planner | 181 | one VLM call: description → action-code list |
| `modules/affordance_estimator.py` | AffordanceEstimator | 43 | gate which actions are physically meaningful (stub in the anchor) |
| `modules/active_perception.py` | ActivePerception | 595 | proactive look-scan bursts before committing a turn/arrival (heaviest evolved lever) |
| `modules/completion_checker.py` | CompletionChecker | 759 | last-line authority on the `-1` subtask commit (heaviest evolved lever) |

## Two evolvable axes (for attribution)
- **CODE levers** — `agent.py` + `modules/*.py`: the mechanism (control flow, VLM call+parse+
  fallback, look-scan tactics, commit-timing, memory maintenance, the live prompt text in
  `prompt_assembler.perception_template`). **This is the productive surface** — every score move in
  the lineage came from a code edit.
- **PROSE docs** — `AGENT.md` / `rules.md`: the task contract + the anti-cheat rule + the audit
  vocabulary. `agent.py` does NOT inject these into the agent prompt, so editing them changes the
  contract/audit view, not the agent's live instructions. To change what the agent is TOLD, edit
  `perception_template()` (in `prompt_assembler.py`) and the planner/perceiver prompt strings.
- **NOT a lever** — `system_prompt.txt` (redundant reference; the harness never reads it), `NOTES.md`
  (this change-log). `__init__.py` pins `modules/` as a package — keep it.

## Anti-cheat (enforced pre-eval + post-hoc — see rules.md)
Navigate from **relative perception + odometry + the cardinal compass ONLY**. Reading any privileged
field — `agent_location`, `target_location`, `gt.json`, `ideal_path`, `current_instruction`, the
analog `_agent_rotation` yaw — is an automatic FAIL (`validate_harness.py` AST lint + the read-audit).
`env.unwrapped.current_subtask` is a structural index, not an oracle. There is no shortcut: you
cannot compute a heading to the goal from coordinates you may not read.

## change:
**anchor** — the honest H_0 skeleton, unchanged. No edit; this is the baseline every method forks from.
