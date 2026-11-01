## module map

| file | components | role in the loop |
|---|---|---|
| `harness.py` | `_read`, `compose_system_prompt`, `build_agent` | entrypoint + prompt composition; returns the per-step `agent(frame_data)` the evaluator installs |
| `AGENT.md` | — | role framing + the verified HUD/perception facts |
| `rules.md` | — | legal action set + granularity contract |
| `skills/navigation.md` | — | corridor strategy, anti-spin rule |
| `memory.md` | — | the working-memory sheet, a live `str.format` template |
| `tools/action.py` | `key_for`, `granularity_steps` | the action/key algebra and the cell count per granularity |
| `tools/frame.py` | `capture` | base64 frame grab via the evaluator's own screenshot path |
| `tools/vlm.py` | `decide` | the ledger + the single `baseline_step` delegation |
