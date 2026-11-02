# naive anchor — module map and evolvable levers

**BALROG's published `naive` agent, behaviourally unchanged, in the harness_map rich-harness
skeleton.** This is the anchor every arm starts from, so it deliberately contains NO hand-written
capability: every point of measured improvement is attributable to the search, not to the author.

Measured: BALROG naive on the official 8, Qwen3-VL-30B-A3B-Instruct, n=40 -> **2.5% ± 2.47%**.

## Module map

| file | role | kind |
|---|---|---|
| `agent.py` | entrypoint `build_agent()`, the per-step `act()` loop | orchestrator |
| `tools/prompting.py` | message assembly + `compose_priors()` over the prose levers | code lever |
| `tools/action_io.py` | model reply -> action string | code lever |
| `AGENT.md` | task framing | prose lever — **EMPTY** |
| `rules.md` | output contract / constraints | prose lever — **EMPTY** |
| `memory.md` | scratchpad template | prose lever — **EMPTY** |
| `skills/` | composable priors, one concern per file | prose levers — **EMPTY (no files)** |

## Why the prose files exist but are empty

The composition MECHANISM is live: `compose_priors()` reads these files every step and appends
whatever they hold. They ship empty, so the composed block is `''` and the prompt is identical to
NaiveAgent's. Shipping the mechanism (rather than omitting it) means a proposer's edit to any prose
lever takes effect immediately — otherwise those files would be inert and every prose-only
candidate would be an undetectable NOOP.

## Known headroom (measured on this anchor, NOT fixed here)

Deliberately left for the search to find:
1. **Degenerate looping** — `north x100`, `north x50 / south x50`; the agent has no memory of where
   it has been (`max_text_history=16` only).
2. **Death before play** — Quest-Easy dies 5/5 at steps 4-6; Quest-Medium 4/5 at step 25.
3. **Illegal actions are silently coerced** — `check_action_validity` rewrites an unlisted action to
   `north` and the step is spent. Naive's own rate is low (2%, mostly `attack east`), but any
   candidate that starts emitting `far <dir>` will pay heavily; the legal set is per-task.

## Invariants a candidate must preserve

1. `build_agent(client_factory, prompt_builder, config=None)` returning `.act(obs, prev_action=None)` + `.reset()`.
2. `.act` returns the LLMResponse shape BALROG expects (`completion` = one legal action).
3. No privileged reads (env internals, .des file, goal coords, success flag).
4. BALROG itself is never edited — only this directory.
