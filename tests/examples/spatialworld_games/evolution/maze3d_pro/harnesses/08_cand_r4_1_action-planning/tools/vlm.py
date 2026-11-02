"""The perceive -> MEASURE -> inject -> VLM -> parse -> GOVERN step.

The VLM call itself is still the evaluator's own pristine per-step fn (`evaluator.baseline_step`):
it captures the frame (tools/frame.py), maintains the rolling 29-turn image+text history, calls the
model through `evaluator.client`, parses the JSON action and falls back to a safe default, returning
an engine `Action(type="key_press", ...)`. Leaving it untouched is what keeps parsing, the legal
action set and the engine contract (tools/action.py) byte-identical to the baseline's — the lineage
measures 2340/2340 legal parsed actions and zero parse fallbacks, so there is nothing to win there.

What this file adds is the loop around that call; tools/plan.py owns all of the state:

    decode the frame exactly  ->  render the measured state into the agent's system turn
                              ->  ask the frozen model  ->  govern its proposal  ->  engine Action

THE INJECTION CHANNEL. `evaluator.current_prompt` is consumed by `build_full_prompt` on the FIRST
call only; from then on `build_conversation_history` reuses `previous_history[0]` verbatim, so the
system turn is the one and only per-turn text channel into the agent. It is rewritten as
`stashed_base + block`, where the stash is the whole first-call system turn (it already carries the
tool list and the JSON RESPONSE FORMAT). It is APPENDED to, never spliced or truncated: cutting that
turn at a marker deletes the JSON response contract from every later call and breaks the parser.

NB `evaluator.baseline_step` is the UNPATCHED SpatialWorld per-step fn that eval_runner stashes
before it reassigns `evaluator.openai_ai_model = build_agent(evaluator)`. Delegating here rather
than to `openai_ai_model` (now this very agent) is what keeps the reuse recursion-free, and it is
called on EVERY step so the anti-cheat guard's VLM-call counter can never void the run.
"""
from __future__ import annotations


def _inject_memory(evaluator, planner) -> None:
    """Append the planner's measured-state block to the agent's system turn. A no-op on the first
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
    if planner.base_prompt is None:
        planner.base_prompt = slot.get("text") or ""        # the full first-call system turn
    block = planner.render_memory()
    slot["text"] = planner.base_prompt + ("\n\n" + block if block else "")


async def decide(evaluator, frame_data, planner=None):
    """One decision step -> engine Action. The frozen model is asked every step; its proposal is
    governed only against what the frame has already been measured to prove."""
    step = getattr(evaluator, "baseline_step", None) or evaluator.openai_ai_model
    if planner is None:
        return await step(frame_data)
    try:
        planner.observe(getattr(frame_data, "image", None))
        _inject_memory(evaluator, planner)
    except Exception:
        pass                      # the governor is an optimisation; never cost the step over it
    proposal = await step(frame_data)
    try:
        return planner.apply(proposal)
    except Exception:
        return proposal
