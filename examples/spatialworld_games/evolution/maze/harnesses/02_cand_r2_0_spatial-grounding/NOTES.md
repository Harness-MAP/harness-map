## module map and levers

| file | components | role | levers here |
|---|---|---|---|
| `agent.py` | `_read`, `compose_system_prompt`, `build_agent` | entrypoint, prompt composition, agent install | composition order |
| `tools/vlm.py` | `decide` | frame digest, settlement ledger, message-0 graft, one frozen-VLM call | `grain`=2, `qbin`=24, `stall_cap`=6, `_CUES` rungs 1/3/5, graft point, settle rule |
| `tools/action.py` | `key_for`, `granularity_steps` | action/key algebra, cells per granularity | label normalisation, `_MIN_CELLS`=1 |
| `tools/frame.py` | `capture` | evaluator screenshot helper | — |
| `AGENT.md`, `rules.md`, `skills/navigation.md` | — | role framing, legal actions, corridor prior | the anchor's, byte for byte |
| `memory.md` | — | the self-derived sheet | its four fields and their wording |

Untouched on purpose: the component graph; the `build_agent(evaluator)` entry point and the engine `Action` contract; `system_prompt.txt` / `system_prompt.md`, which `compose_system_prompt` never reads. Longer tables live in `traces_notes/`.

## axis: token-level implementation — control flow, thresholds and wording inside the existing components

## change:

`tools/vlm.py`'s `decide` grew from a one-line delegation into a frame-digest settlement ledger around the same single `baseline_step` call. Each step it md5s a drift-tolerant digest of `frame_data.image` (every `grain`-th pixel, values bucketed by `qbin`), compares it with the previous step's digest, and settles the key it returned last step: an unchanged digest means the key bought nothing — a forward the wall refused, or a turn the engine swallowed. `stall` is the streak of such findings, capped at `stall_cap` and reset the instant the frame moves; it selects a cue from `_CUES`, and step / last key / streak / cue render through `memory.md` (now a live `str.format` template) and are grafted onto message 0. The graft writes `stashed base + sheet`, never truncate-then-append, so the AVAILABLE ACTIONS list and the JSON RESPONSE FORMAT the engine folded into message 0 on the first call stay in front of the model all episode. Nothing is said on step 1. Three smaller moves finish the pass: `key_for` normalises the label before the table lookup, `granularity_steps` takes its fall-back cell count from a named constant, and `capture` is total rather than partial. The `sys.modules` purge left `build_agent` for a module-level lambda.

## why:

The gradient was an archive-coverage steer, not a score diagnosis: cell 20 of K0=32 was empty, the move asked for was 4 -> 20, and the decomposition named the token axis dominant (||dstructure||=0.930 vs ||dtoken||=2.348). So the graph is held fixed — same components, same edges, same role labels, same main module — and the whole budget is spent on control flow, thresholds and wording inside the bodies. The mechanism makes that honest rather than cosmetic: a maze agent's worst step is the one it repeats, and an identical digest after a key is an exact one-bit fact the harness derived itself. Grafting message 0 is the only way to put that fact in front of a frozen model whose system turn the engine froze at step 1.

## files:

- `tools/vlm.py` — the ledger, the module-level `_MARK` / `_FROZEN` / `_BET` / `_STASH` / `_GRAFT` / `_CUES` plumbing, `_SHEET` loaded from `memory.md`.
- `agent.py` — module-level `_PURGE`, `build_agent` slimmed to the path guard.
- `tools/action.py` — `key_for` normalisation, `_MIN_CELLS`.
- `tools/frame.py` — `capture` total.
- `memory.md` — commented sketch replaced by the live sheet template.
- `NOTES.md` — this record.

Full module map, the long lever table, the honesty notes and the offline stub rig that produced the validation result live in `traces_notes/`.
