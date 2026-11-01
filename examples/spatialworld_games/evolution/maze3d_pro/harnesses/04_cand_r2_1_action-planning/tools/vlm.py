"""The perceive -> VLM -> parse -> GOVERN step.

The VLM call itself is still the evaluator's own pristine per-step fn (`evaluator.baseline_step`),
which captures the frame (see tools/frame.py), builds the 29-turn image+text history, calls the VLM
via `evaluator.client`, parses the JSON action and falls back to a safe default — returning an
engine `Action(type="key_press", ...)`. That is left untouched, so parsing, the legal action set and
the engine contract (see tools/action.py) are byte-identical to the baseline's.

What this file adds is the CLOSED LOOP around that call (tools/plan.py owns the state machine):

    observe the frame  ->  inject the working-memory block  ->  ask the frozen model
                       ->  govern its proposal  ->  return the engine Action

The frozen model is asked on EVERY step and obeyed unless its proposal is one the frames already
prove cannot do anything (a translation this exact view refused, or the undo of the floor change
just made). Those are re-aimed onto the next step of a consistently-handed escape sweep.

THE INJECTION CHANNEL. `evaluator.current_prompt` is read by `build_full_prompt` on the FIRST call
only: from then on `build_conversation_history` re-uses `previous_history[0]` verbatim, so the system
turn is the one and only per-turn channel into the agent. It is rewritten as `stashed_base + block`
— stashed_base being the whole first-call system turn, which already carries the tool list and the
JSON RESPONSE FORMAT. It is never spliced or truncated: cutting it at a marker would delete the JSON
response contract from every later turn and break parsing.

NB `evaluator.baseline_step` is the UNPATCHED SpatialWorld per-step fn that eval_runner stashes
before it reassigns `evaluator.openai_ai_model = build_agent(evaluator)`. Delegating here rather
than to `openai_ai_model` (now this very agent) is what keeps the reuse recursion-free.
"""
from __future__ import annotations


def _inject_memory(evaluator, planner) -> None:
    """Append the planner's working-memory block to the agent's system turn. No-op on the first
    call (the history is still empty; `current_prompt` carries the prior there)."""
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
    """One decision step -> engine Action. Reuses the evaluator's pristine VLM step + JSON parser,
    then governs the proposal against what the frames have already proven."""
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
