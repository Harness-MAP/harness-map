## lever table

| lever | kind | where | currently |
|---|---|---|---|
| `grain` | threshold | `decide` keyword default | 2 — digest stride over the frame |
| `qbin` | threshold | `decide` keyword default | 24 — pixel bucket width, the drift tolerance |
| `stall_cap` | threshold | `decide` keyword default | 6 — ceiling on the unchanged-frame streak |
| `_CUES` | prose | `tools/vlm.py` module level | three rungs at streak 1 / 3 / 5 |
| `_SHEET` | prose | `memory.md` | four fields: step, last key, streak, cue |
| graft point | code | `decide` | message 0, rebuilt from a stashed base every step |
| settle rule | code | `decide` | digest equality, scaled by `_FROZEN` |
| label normalisation | code | `key_for` | `str().strip().lower()` before the table lookup |
| `_MIN_CELLS` | threshold | `tools/action.py` module level | 1 — the cautious granularity fall-back |
| prompt composition order | prose | `compose_system_prompt` | AGENT.md, skills/*.md, rules.md |

## what this iteration deliberately did NOT touch
- the component graph (same nodes, same edges, same role labels, same main module)
- `AGENT.md`, `rules.md`, `skills/navigation.md` — the shared prose priors are the anchor's, byte for byte
- the `run_episode`-equivalent entry point `build_agent(evaluator)` and the engine `Action` contract
- `system_prompt.txt` / `system_prompt.md`, which `compose_system_prompt` never reads
