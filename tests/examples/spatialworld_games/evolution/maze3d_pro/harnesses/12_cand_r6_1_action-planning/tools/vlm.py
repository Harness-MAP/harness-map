"""The perceive -> MEASURE -> state -> ask -> GOVERN step.

The VLM call itself is still the evaluator's own pristine per-step fn (`evaluator.baseline_step`),
which captures the frame (see tools/frame.py), builds the rolling image+text history, calls the VLM
via `evaluator.client`, parses the JSON action and falls back to a safe default — returning an
engine `Action(type="key_press", ...)`. That is left untouched, so parsing, the legal action set and
the engine contract (see tools/action.py) stay byte-identical to the baseline's, and the frozen
model is asked on EVERY step: the harness displaces proposals, it never replaces the call.

What this file adds is the closed loop around that call (`tools/plan.py` owns the state machine):

    read the frame as geometry  ->  settle the last decision  ->  state the measurements back
                                ->  ask the frozen model      ->  govern the proposal

THE INJECTION CHANNEL. `evaluator.current_prompt` is read by `build_full_prompt` on the FIRST call
only: from then on `build_conversation_history` re-uses `previous_history[0]` by reference, so the
system turn is the one and only per-turn channel into the agent. It is rewritten as
`stashed_base + block`, where `stashed_base` is the WHOLE first-call system turn — which already
carries the tool list and the JSON RESPONSE FORMAT. It is never spliced and never truncated at a
marker: cutting it would delete the response contract from every later turn and break parsing.
Nothing is injected on the first call, so turn 1 is byte-identical to the baseline's.

NB `evaluator.baseline_step` is the UNPATCHED SpatialWorld per-step fn that the runner stashes
before it reassigns `evaluator.openai_ai_model = build_agent(evaluator)`. Delegating here rather
than to `openai_ai_model` (now this very agent) is what keeps the reuse recursion-free.
"""
from __future__ import annotations


def inject_memory(evaluator, pilot):
    """Append the pilot's working-memory block to the agent's system turn, in place.

    A no-op on the first call, when the history is still empty and `current_prompt` is carrying the
    prior there by itself."""
    history = getattr(evaluator, "conversation_history", None)
    if not history:
        return
    try:
        slot = history[0]["content"][0]
    except Exception:
        return
    if not isinstance(slot, dict) or "text" not in slot:
        return
    if pilot.base_prompt is None:
        pilot.base_prompt = slot.get("text") or ""      # the full first-call system turn
    block = pilot.render_memory()
    slot["text"] = pilot.base_prompt + ("\n\n" + block if block else "")


async def decide(evaluator, frame_data, pilot=None):
    """One decision step -> engine Action. Reuses the evaluator's pristine VLM step + JSON parser,
    then governs the proposal against the route the frames have paid for."""
    step = getattr(evaluator, "baseline_step", None) or evaluator.openai_ai_model
    if pilot is None:
        return await step(frame_data)
    try:
        pilot.observe(getattr(frame_data, "image", None))
        inject_memory(evaluator, pilot)
    except Exception:
        pass                   # the governor is an optimisation; never cost the step over it
    proposal = await step(frame_data)
    try:
        return pilot.apply(proposal)
    except Exception:
        return proposal
