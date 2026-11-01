"""tools/vlm.py — the perceive -> MEASURE -> state -> ask -> govern step.

The VLM call itself stays the evaluator's own pristine per-step fn (`evaluator.baseline_step`): it
captures the frame, maintains the rolling 29-turn image+text history, calls the model through
`evaluator.client`, parses the JSON action and falls back to a safe default, returning an engine
`Action(type="key_press", ...)`. Leaving it untouched is what keeps parsing, the legal action set and
the engine contract byte-identical to the baseline's — this lineage measures 2340/2340 legal parsed
actions and zero parse fallbacks on the anchor, so there is no headroom there to spend an iter on.

What this file adds is the loop around that call; tools/plan.py owns all of the state:

    read the frame exactly  ->  state what was measured in the agent's own system turn
                            ->  ask the frozen model  ->  govern its proposal  ->  engine Action

THE INJECTION CHANNEL. `evaluator.current_prompt` is consumed by `build_full_prompt` on the FIRST
call only; from then on `build_conversation_history` reuses `previous_history[0]` verbatim, so the
system turn is the one and only per-turn text channel into the agent. It is rewritten as
`stashed_base + block`, where the stash is the whole first-call system turn (it already carries the
tool list and the JSON RESPONSE FORMAT). It is APPENDED to, never spliced or truncated: cutting that
turn at a marker deletes the response-format contract from every later call and breaks the parser.

NB `evaluator.baseline_step` is the UNPATCHED per-step fn that eval_runner stashes before it
reassigns `evaluator.openai_ai_model = build_agent(evaluator)`. Delegating here rather than to
`openai_ai_model` (now this very agent) is what keeps the reuse recursion-free, and it is called on
EVERY step, so the anti-cheat guard's VLM-call counter can never void the run.
"""
from __future__ import annotations


def _inject_memory(evaluator, pilot) -> None:
    """Append the pilot's measured-state block to the agent's system turn. A no-op on the first
    call, where the history is still empty and `current_prompt` carries the prior instead."""
    history = getattr(evaluator, "conversation_history", None)
    if not history:
        return
    try:
        slot = history[0]["content"][0]
    except Exception:
        return
    if not isinstance(slot, dict) or slot.get("type") != "input_text":
        return
    if pilot.base_prompt is None:
        pilot.base_prompt = slot.get("text") or ""          # the whole first-call system turn
    block = pilot.render_memory()
    slot["text"] = pilot.base_prompt + ("\n\n" + block if block else "")


async def decide(evaluator, frame_data, pilot=None):
    """One decision step -> engine Action. The frozen model is asked every step; its proposal is
    governed only against what the frame has already been measured to prove."""
    step = getattr(evaluator, "baseline_step", None) or evaluator.openai_ai_model
    if pilot is None:
        return await step(frame_data)
    try:
        pilot.observe(getattr(frame_data, "image", None))
        _inject_memory(evaluator, pilot)
    except Exception:
        pass                     # the governor is an optimisation; never cost the step over it
    proposal = await step(frame_data)
    try:
        return pilot.apply(proposal)
    except Exception:
        return proposal
